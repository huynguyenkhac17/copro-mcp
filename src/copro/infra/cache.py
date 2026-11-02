"""Pure in-memory cache (TTL + LRU), with no external framework/service
dependencies - used generally for any value needing caching in the server
process (e.g. ProviderResponse in CachingProvider).

Not persistent across server restarts - acceptable for v1, simplicity is
the top priority for a single long-running process cache.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class InMemoryTTLCache(Generic[T]):
    def __init__(self, max_entries: int = 256, ttl_seconds: float | None = None) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None

        inserted_at, value = entry
        if self._ttl_seconds is not None and time.monotonic() - inserted_at > self._ttl_seconds:
            del self._store[key]
            return None

        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)
