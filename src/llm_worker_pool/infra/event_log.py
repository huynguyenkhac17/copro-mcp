"""Append-only JSONL event log: one line per delegate_task invocation (or failed tier).

Unlike stats.json, which every MCP process overwrites with its own counters, this file
is shared: several server instances (Claude Code, Gemini CLI...) append to it, and the
separate `worker-metrics` exporter folds it into Prometheus counters. Each event is
written with a single os.write() on an O_APPEND descriptor, which POSIX keeps atomic
for regular files, so concurrent writers never interleave half-lines.

Never contains task/context/response text unless debug_log_content is on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows: rotation races are then only best effort
    fcntl = None  # type: ignore[assignment]

# Plain logging (not infra.logging.get_logger): infra.logging imports this module.
_logger = logging.getLogger(__name__)

# Label = everything before the first ':', '(', ' ' or ';' - "fallback_used:groq after ..."
# and "output_truncated_at_max_tokens (num_predict=...)" both collapse to their code.
_CODE_RE = re.compile(r"^[^:( ;]+")


def default_event_log_path() -> Path:
    return Path(os.path.expanduser("~")) / ".llm_worker_pool" / "events.jsonl"


def default_eval_latest_path() -> Path:
    return default_event_log_path().with_name("eval_latest.json")


def configured_path(value: str | None, default: Path) -> Path:
    return Path(os.path.expanduser(value)) if value else default


def warning_code(warning: str) -> str:
    """Low-cardinality label for a free-text warning (paths/models stripped)."""
    match = _CODE_RE.match(warning.strip())
    code = match.group(0) if match else "unknown"
    # thinking_consumed_<n>_of_<m>_output_tokens embeds numbers in the code itself.
    if code.startswith("thinking_consumed_"):
        return "thinking_consumed"
    return code


_HTTP_STATUS_RE = re.compile(r"\b(?:status|error)\s+([1-5]\d\d)\b", re.IGNORECASE)


def error_code(error: str) -> str:
    """Low-cardinality label for a provider error (free text with bodies/keys inside)."""
    lowered = error.lower()
    if "did not respond within" in lowered or "timeout" in lowered:
        return "timeout"
    if "exceeds max_prompt_tokens" in lowered:
        return "prompt_too_large"
    if "circuit" in lowered or "quarantin" in lowered:
        return "circuit_open"
    match = _HTTP_STATUS_RE.search(error)
    if match:
        return f"http_{match.group(1)}"
    return "error"


class EventLog:
    def __init__(self, path: Path, max_bytes: int = 50_000_000) -> None:
        self._path = path
        self._max_bytes = max_bytes

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: dict[str, Any]) -> None:
        """Best effort, like StatsCollector._persist: a full disk or a permissions
        problem must never fail the delegate_task call it is recording."""
        record = {"ts": round(time.time(), 3), **event}
        line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError as exc:
            _logger.warning("event log write failed: %s", exc)

    def _rotate_if_needed(self) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._max_bytes:
            return
        # Two writers can both see an oversized file. Without the lock the second one
        # would rotate the fresh file over the first one's `.1` and drop those events,
        # so the size is re-checked while holding it. The exporter notices the new
        # inode, finishes the `.1` copy, then reads the new file from offset 0.
        lock_path = self._path.with_name(self._path.name + ".lock")
        with open(lock_path, "a") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if self._path.stat().st_size >= self._max_bytes:
                    os.replace(self._path, self._path.with_name(self._path.name + ".1"))
            except FileNotFoundError:
                pass
