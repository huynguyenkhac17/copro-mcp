from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from llm_worker_pool.cli import render
from llm_worker_pool.infra.cache import InMemoryTTLCache
from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker
from llm_worker_pool.infra.stats import GatewayStats, StatsCollector
from llm_worker_pool.providers.caching_provider import CachingProvider
from llm_worker_pool.providers.concurrency_provider import ConcurrencyProvider
from llm_worker_pool.providers.openai_compatible_provider import OpenAICompatibleProvider
from llm_worker_pool.server import _breaker_stats


def _wrapped_provider(breaker: KeyCircuitBreaker | None):
    inner = OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=["sk-one", "sk-two"],
        http_client=httpx.AsyncClient(),
        circuit_breaker=breaker,
    )
    return CachingProvider(ConcurrencyProvider(inner, 2), InMemoryTTLCache(max_entries=4, ttl_seconds=60))


def test_breaker_stats_reach_through_provider_wrappers():
    breaker = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0)
    breaker.record_failure("sk-two", 429)
    providers = {"gemini": _wrapped_provider(breaker), "nobreaker": _wrapped_provider(None)}

    stats = _breaker_stats(providers)

    assert list(stats) == ["gemini"]
    assert stats["gemini"]["key_1"]["is_quarantined"] is True
    assert "sk-two" not in repr(stats)


def test_snapshot_and_disk_mirror_include_breaker_state(tmp_path: Path):
    path = tmp_path / "stats.json"
    breaker = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0)
    breaker.record_failure("sk-one", 503)
    providers = {"groq": _wrapped_provider(breaker)}
    collector = StatsCollector(snapshot_path=path, breaker_source=lambda: _breaker_stats(providers))

    collector.record_failure(provider="groq", budget="L1")

    assert collector.snapshot().circuit_breakers["groq"]["key_0"].total_quarantines == 1
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["circuit_breakers"]["groq"]["key_0"]["is_quarantined"] is True


def test_old_snapshot_without_breakers_still_loads():
    assert GatewayStats.model_validate({"total_requests": 3}).circuit_breakers == {}


def test_cli_judges_quarantine_by_deadline_not_stale_flag():
    now = time.time()
    stats = GatewayStats.model_validate({
        "providers": {"gemini": {"success_count": 1}},
        "circuit_breakers": {"gemini": {
            "key_0": {"total_successes": 4},
            "key_1": {"total_failures": 3, "total_quarantines": 1,
                      "is_quarantined": True, "quarantined_until": now + 90},
            # Written as quarantined, but the deadline has passed since.
            "key_2": {"total_failures": 3, "total_quarantines": 1,
                      "is_quarantined": True, "quarantined_until": now - 5},
        }},
    })

    text = render(stats, now=now)

    assert "gemini/key_1" in text and "khóa còn 1m30s" in text
    assert "gemini/key_2" in text and text.split("gemini/key_2")[1].split("\n")[0].endswith("mở")
    assert "gemini/key_0" not in text


def test_cli_healthy_pool_prints_one_line():
    stats = GatewayStats.model_validate({
        "providers": {"groq": {"success_count": 1}},
        "circuit_breakers": {"groq": {"key_0": {"total_successes": 1}}},
    })
    assert "Chưa key nào lỗi." in render(stats)


def test_failing_breaker_source_never_breaks_recording(tmp_path: Path):
    def broken():
        raise RuntimeError("provider gone")

    collector = StatsCollector(snapshot_path=tmp_path / "stats.json", breaker_source=broken)

    collector.record_failure(provider="groq", budget="L1")

    assert collector.snapshot().circuit_breakers == {}
    assert json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))["total_failures"] == 1
