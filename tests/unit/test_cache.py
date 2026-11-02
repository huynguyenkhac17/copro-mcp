from __future__ import annotations

import time

from copro.infra.cache import InMemoryTTLCache


def test_get_set_roundtrip():
    cache: InMemoryTTLCache[str] = InMemoryTTLCache(max_entries=10)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_missing_key_returns_none():
    cache: InMemoryTTLCache[str] = InMemoryTTLCache(max_entries=10)
    assert cache.get("does-not-exist") is None


def test_ttl_expiry_evicts_entry():
    cache: InMemoryTTLCache[str] = InMemoryTTLCache(max_entries=10, ttl_seconds=0.05)
    cache.set("k", "v")
    assert cache.get("k") == "v"

    time.sleep(0.1)

    assert cache.get("k") is None


def test_no_ttl_never_expires():
    cache: InMemoryTTLCache[str] = InMemoryTTLCache(max_entries=10, ttl_seconds=None)
    cache.set("k", "v")
    time.sleep(0.05)
    assert cache.get("k") == "v"


def test_lru_eviction_drops_least_recently_used():
    cache: InMemoryTTLCache[str] = InMemoryTTLCache(max_entries=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.get("a")  # "a" was just accessed -> "b" becomes the LRU
    cache.set("c", "3")  # exceeds max_entries -> evicts LRU ("b")

    assert cache.get("a") == "1"
    assert cache.get("b") is None
    assert cache.get("c") == "3"
