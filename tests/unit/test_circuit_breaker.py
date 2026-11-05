from __future__ import annotations

import time
import pytest

from copro.infra.circuit_breaker import KeyCircuitBreaker


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_circuit_breaker_disabled():
    cb = KeyCircuitBreaker(enabled=False, failure_threshold=2)
    cb.record_failure("key-1", 429)
    cb.record_failure("key-1", 429)
    cb.record_failure("key-1", 429)
    assert cb.is_available("key-1") is True
    assert cb.filter_available_keys(["key-1"]) == ["key-1"]


def test_quarantines_after_threshold_failures():
    cb = KeyCircuitBreaker(failure_threshold=3, quarantine_seconds=60.0)
    assert cb.is_available("key-1") is True

    # 1st failure
    assert cb.record_failure("key-1", 429) is False
    assert cb.is_available("key-1") is True

    # Non-severe failure like 400 doesn't increment
    assert cb.record_failure("key-1", 400) is False
    assert cb.is_available("key-1") is True

    # 2nd severe failure (503)
    assert cb.record_failure("key-1", 503) is False
    assert cb.is_available("key-1") is True

    # 3rd severe failure (403) -> quarantined!
    assert cb.record_failure("key-1", 403) is True
    assert cb.is_available("key-1") is False


def test_success_resets_consecutive_failures():
    cb = KeyCircuitBreaker(failure_threshold=2, quarantine_seconds=60.0)
    cb.record_failure("key-1", 429)
    assert cb._get_state("key-1").consecutive_failures == 1

    cb.record_success("key-1")
    assert cb._get_state("key-1").consecutive_failures == 0
    assert cb.is_available("key-1") is True


def test_filter_available_keys_skips_quarantined():
    cb = KeyCircuitBreaker(failure_threshold=2, quarantine_seconds=60.0)
    keys = ["key-1", "key-2", "key-3"]

    cb.record_failure("key-2", 429)
    cb.record_failure("key-2", 429)

    available = cb.filter_available_keys(keys)
    assert available == ["key-1", "key-3"]


def test_filter_available_keys_fallback_when_all_quarantined():
    clock = FakeClock()
    cb = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0, clock=clock)
    keys = ["key-1", "key-2"]

    cb.record_failure("key-1", 429)
    clock.now += 1.0  # key-2 fails a second later
    cb.record_failure("key-2", 429)

    # When all quarantined, returns the one with earliest expiry (key-1)
    available = cb.filter_available_keys(keys)
    assert available == ["key-1"]


def test_quarantine_expires():
    # A fake clock, not time.sleep: on Windows before Python 3.13 time.monotonic ticks
    # every ~15.6 ms, so sleeping 60 ms past a 50 ms quarantine could read as 47 ms.
    clock = FakeClock()
    cb = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0, clock=clock)
    cb.record_failure("key-1", 429)
    assert cb.is_available("key-1") is False

    clock.now += 59.9
    assert cb.is_available("key-1") is False
    clock.now += 0.2
    assert cb.is_available("key-1") is True


def test_get_stats():
    cb = KeyCircuitBreaker(failure_threshold=2, quarantine_seconds=60.0)
    cb.record_success("key-1")
    cb.record_failure("key-2", 429)
    cb.record_failure("key-2", 429)

    stats = cb.get_stats()
    assert "key-1" in stats
    assert stats["key-1"]["total_successes"] == 1
    assert stats["key-1"]["is_quarantined"] is False

    assert "key-2" in stats
    assert stats["key-2"]["total_quarantines"] == 1
    assert stats["key-2"]["is_quarantined"] is True


def test_stats_by_position_hides_keys_and_covers_unused_ones():
    cb = KeyCircuitBreaker(failure_threshold=2, quarantine_seconds=60.0)
    for _ in range(2):
        cb.record_failure("sk-secret-b", 429)

    before = time.time()
    stats = cb.stats_by_position(["sk-secret-a", "sk-secret-b"])

    assert list(stats) == ["key_0", "key_1"]
    assert "sk-secret" not in repr(stats)
    assert stats["key_0"]["total_failures"] == 0 and stats["key_0"]["quarantined_until"] == 0.0
    assert stats["key_1"]["is_quarantined"] is True
    assert stats["key_1"]["total_quarantines"] == 1
    assert before + 55 < stats["key_1"]["quarantined_until"] < time.time() + 61
