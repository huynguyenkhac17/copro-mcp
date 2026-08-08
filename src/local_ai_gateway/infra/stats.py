"""Stats collection for the local AI gateway."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class BudgetStats(BaseModel):
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    cached_count: int = 0
    tokens_saved_by_cache: int = 0


class ProviderStats(BaseModel):
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    cached_count: int = 0
    tokens_saved_by_cache: int = 0
    by_budget: dict[str, BudgetStats] = Field(default_factory=dict)


class GatewayStats(BaseModel):
    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_success: int = 0
    total_failures: int = 0
    total_cached: int = 0
    total_tokens_saved_by_cache: int = 0
    providers: dict[str, ProviderStats] = Field(default_factory=dict)


class StatsCollector:
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._stats = GatewayStats()

    def record_success(
        self,
        *,
        provider: str,
        budget: str,
        cached: bool,
        tokens_used: int | None,
        latency_ms: float,
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

        if cached:
            self._stats.total_cached += 1
            p_stats.cached_count += 1
            b_stats.cached_count += 1
            if tokens_used:
                self._stats.total_tokens_saved_by_cache += tokens_used
                p_stats.tokens_saved_by_cache += tokens_used
                b_stats.tokens_saved_by_cache += tokens_used

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

    def snapshot(self) -> GatewayStats:
        self._stats.uptime_seconds = time.monotonic() - self._start_time
        return self._stats.model_copy(deep=True)
