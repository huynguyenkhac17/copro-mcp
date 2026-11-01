"""Key quota / quarantine state shared by every server process on the machine.

Each Claude Code session starts its own MCP server process, and each used to keep the
quota tracker and circuit breaker in memory only: a key one session had just seen 429
(or drained per Groq's x-ratelimit headers) still looked fresh to every other session,
which then spent its own 429 finding out. This file carries what one process learns
about a key to the others:

- per key: `blocked_until` (a 429's Retry-After/retryDelay, a per-day quota), the last
  x-ratelimit token/request counts with their reset time, and the breaker's
  `quarantined_until`;
- per provider: `model_blocked_until` (the model hung on repeated calls).

Not shared: in-flight reservations (they belong to the process that holds them) and the
counters behind `worker-stats`. Conversations (`continuation_id`) never go to disk: they
hold code, and live for an hour at most.

Keys are stored as a truncated SHA-256, never in clear. Times are wall-clock epoch
seconds, since time.monotonic() means nothing in another process after a reboot.

Writes are read-modify-write under a lock on a side file (flock, or msvcrt.locking on
Windows; non-blocking, given up after 0.25 s) and land via os.replace (like
stats.json), so a reader never sees a half-written file. Every failure is swallowed: shared state is an optimisation, a locked or unwritable file only means
each process is on its own again, as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)

_VERSION = 1

# Fields holding a wall-clock deadline; an entry whose deadlines have all passed is
# dropped on the next write so the file does not grow with every key ever seen.
_DEADLINES = ("blocked_until", "quarantined_until", "tokens_reset_at", "requests_reset_at")


def default_shared_state_path() -> Path:
    return Path(os.path.expanduser("~")) / ".llm_worker_pool" / "key_state.json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class SharedKeyState:
    def __init__(
        self,
        path: Path,
        *,
        min_refresh_seconds: float = 0.2,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._min_refresh = min_refresh_seconds
        self._wall = wall
        self._data: dict[str, Any] = {}
        self._text: str | None = None
        self._checked_at = float("-inf")
        self.generation = 0
        """Bumped whenever the content read from disk changes, so a tracker can skip
        merging when nothing new arrived."""

    def now(self) -> float:
        return self._wall()

    # --- reading ---------------------------------------------------------------

    def provider(self, name: str) -> dict[str, Any]:
        """This provider's section as last read (re-read at most every
        min_refresh_seconds, and only when the file changed). Entries of the wrong
        shape (a hand-edited file) are left out, so callers can trust the types."""
        self._refresh()
        providers = self._data.get("providers")
        section = providers.get(name) if isinstance(providers, dict) else None
        if not isinstance(section, dict):
            return {}
        clean: dict[str, Any] = {}
        if _is_number(section.get("model_blocked_until")):
            clean["model_blocked_until"] = section["model_blocked_until"]
        keys = section.get("keys")
        if isinstance(keys, dict):
            clean["keys"] = {
                kid: {f: v for f, v in entry.items() if v is None or _is_number(v)}
                for kid, entry in keys.items()
                if isinstance(entry, dict)
            }
        return clean

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self._checked_at < self._min_refresh:
            return
        self._checked_at = now
        # Compared by content, not mtime: two same-size writes within the filesystem's
        # timestamp granularity (a few ms) would look unchanged. The file is small.
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return
        if text == self._text:
            return
        data = self._parse(text)
        if data is not None:
            self._text = text
            self._data = data
            self.generation += 1

    def _load(self) -> dict[str, Any] | None:
        try:
            return self._parse(self._path.read_text(encoding="utf-8"))
        except OSError:
            return None

    @staticmethod
    def _parse(text: str) -> dict[str, Any] | None:
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("version") != _VERSION:
            return None
        return data

    # --- writing ---------------------------------------------------------------

    def update(self, provider: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        """Apply `mutate` to this provider's section of the freshest file content."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._path.with_name(self._path.name + ".lock")
            with open(lock_path, "a+") as lock:
                _lock(lock)
                try:
                    data = self._apply(self._load(), provider, mutate)
                except (AttributeError, TypeError, ValueError, KeyError):
                    # The file has the wrong shape (hand-edited, other version of this
                    # code): start over rather than never being able to write again.
                    data = self._apply(None, provider, mutate)
                self._text = self._write(data)
                self._data = data
                self.generation += 1
        except Exception:  # noqa: BLE001 - see module docstring
            _logger.debug("shared key state not written", exc_info=True)

    def _apply(
        self, data: dict[str, Any] | None, provider: str, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        data = data or {"version": _VERSION, "providers": {}}
        mutate(data.setdefault("providers", {}).setdefault(provider, {}))
        self._prune(data)
        return data

    def _prune(self, data: dict[str, Any]) -> None:
        now = self._wall()
        for name in list(data["providers"]):
            section = data["providers"][name]
            keys = section.get("keys", {})
            for kid in list(keys):
                if all(keys[kid].get(field, 0) <= now for field in _DEADLINES):
                    del keys[kid]
            if section.get("model_blocked_until", 0) <= now:
                section.pop("model_blocked_until", None)
            if not keys:
                section.pop("keys", None)
            if not section:
                del data["providers"][name]

    def _write(self, data: dict[str, Any]) -> str:
        text = json.dumps(data, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=".key_state-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_name = handle.name
        try:
            _replace(temp_name, self._path)
        except OSError:
            os.unlink(temp_name)
            raise
        return text


def _lock(handle: Any, timeout: float = 0.25) -> None:
    """Exclusive lock held until `handle` is closed; OSError after `timeout` seconds.

    Non-blocking attempts with short sleeps: this runs on the server's event loop,
    and a blocking lock would freeze every other request of the session while it
    waits. msvcrt's LK_LOCK sleeps a full second per retry - 3 processes writing at
    once measured 1-7 s stalls (2026-11-02). Giving up only skips one write, which
    update() already tolerates."""
    if fcntl is None and msvcrt is None:
        return
    deadline = time.monotonic() + timeout
    delay = 0.002
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.02)


def _replace(source: str, target: Path, attempts: int = 5) -> None:
    """os.replace, retried briefly: on Windows it fails while another process has the
    target open, which a reader does for a moment on every refresh."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 * (attempt + 1))
