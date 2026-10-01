"""Stats collection for the local AI gateway.

Counters live in the server process, which is long-lived but not immortal: restart
the MCP server and they reset. `snapshot_path` optionally mirrors each update to a
JSON file so tools outside this process (the `worker-stats` CLI, the /worker slash
command) can read the numbers without going through the MCP tool.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, Field


class TokenTotals(BaseModel):
    """Real consumption, as opposed to the cache-hit savings tracked separately.

    Reasoning models bill `thinking` inside the total while reporting it in neither
    prompt nor completion, so it is tracked on its own - it is usually the single
    largest line item and the easiest one to not notice.
    """

    prompt: int = 0
    completion: int = 0
    thinking: int = 0
    total: int = 0


class BudgetStats(BaseModel):
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    cached_count: int = 0
    tokens_saved_by_cache: int = 0
    truncated_count: int = 0
    tokens: TokenTotals = Field(default_factory=TokenTotals)


class ProviderStats(BaseModel):
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    cached_count: int = 0
    tokens_saved_by_cache: int = 0
    truncated_count: int = 0
    tokens: TokenTotals = Field(default_factory=TokenTotals)
    by_budget: dict[str, BudgetStats] = Field(default_factory=dict)


class GatewayStats(BaseModel):
    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_success: int = 0
    total_failures: int = 0
    total_cached: int = 0
    total_tokens_saved_by_cache: int = 0
    total_truncated: int = 0
    """Answers cut off at the output ceiling. Non-zero means callers received
    incomplete results, so it belongs next to the failure count rather than buried."""

    tokens: TokenTotals = Field(default_factory=TokenTotals)
    providers: dict[str, ProviderStats] = Field(default_factory=dict)


def default_snapshot_path() -> Path:
    return Path(os.path.expanduser("~")) / ".llm_worker_pool" / "stats.json"


class StatsCollector:
    def __init__(self, snapshot_path: Path | None = None) -> None:
        self._start_time = time.monotonic()
        self._stats = GatewayStats()
        self._snapshot_path = snapshot_path

    def record_success(
        self,
        *,
        provider: str,
        budget: str,
        cached: bool,
        tokens_used: int | None,
        latency_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        thinking_tokens: int | None = None,
        truncated: bool = False,
    ) -> None:
        self._stats.total_requests += 1
        self._stats.total_success += 1

        if provider not in self._stats.providers:
            self._stats.providers[provider] = ProviderStats()
        p_stats = self._stats.providers[provider]

        p_stats.success_count += 1
        p_stats.total_latency_ms += latency_ms

        if budget not in p_stats.by_budget:
            p_stats.by_budget[budget] = BudgetStats()
        b_stats = p_stats.by_budget[budget]

        b_stats.success_count += 1
        b_stats.total_latency_ms += latency_ms

        if truncated:
            self._stats.total_truncated += 1
            p_stats.truncated_count += 1
            b_stats.truncated_count += 1

        if cached:
            self._stats.total_cached += 1
            p_stats.cached_count += 1
            b_stats.cached_count += 1
            if tokens_used:
                self._stats.total_tokens_saved_by_cache += tokens_used
                p_stats.tokens_saved_by_cache += tokens_used
                b_stats.tokens_saved_by_cache += tokens_used
        else:
            # A cache hit costs nothing upstream, so only real calls add to spend.
            for totals in (self._stats.tokens, p_stats.tokens, b_stats.tokens):
                totals.prompt += prompt_tokens or 0
                totals.completion += completion_tokens or 0
                totals.thinking += thinking_tokens or 0
                totals.total += tokens_used or 0

        self._persist()

    def record_failure(self, *, provider: str, budget: str) -> None:
        self._stats.total_requests += 1
        self._stats.total_failures += 1

        if provider not in self._stats.providers:
            self._stats.providers[provider] = ProviderStats()
        p_stats = self._stats.providers[provider]
        p_stats.failure_count += 1

        if budget not in p_stats.by_budget:
            p_stats.by_budget[budget] = BudgetStats()
        b_stats = p_stats.by_budget[budget]
        b_stats.failure_count += 1

        self._persist()

    def snapshot(self) -> GatewayStats:
        self._stats.uptime_seconds = time.monotonic() - self._start_time
        return self._stats.model_copy(deep=True)

    def _persist(self) -> None:
        """Mirror the snapshot to disk, atomically and without ever raising.

        Stats are a convenience; a full disk or a locked file must never turn a
        successful delegation into an error, so every failure here is swallowed.
        The write goes to a temp file in the same directory and is then renamed,
        so a reader can never observe a half-written file.
        """
        if self._snapshot_path is None:
            return
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.snapshot().model_dump_json(indent=2)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._snapshot_path.parent,
                prefix=".stats-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                temp_name = handle.name
            os.replace(temp_name, self._snapshot_path)
        except OSError:
            pass
