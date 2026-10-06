from __future__ import annotations

import time
import pytest

from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker


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
    cb = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0)
    keys = ["key-1", "key-2"]

    cb.record_failure("key-1", 429)
    # Simulate key-2 failing a bit later
    time.sleep(0.01)
    cb.record_failure("key-2", 429)

    # When all quarantined, returns the one with earliest expiry (key-1)
    available = cb.filter_available_keys(keys)
    assert available == ["key-1"]


def test_quarantine_expires():
    cb = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=0.05)
    cb.record_failure("key-1", 429)
    assert cb.is_available("key-1") is False

    time.sleep(0.06)
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
