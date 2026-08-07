"""Cache in-memory thuan (TTL + LRU), khong phu thuoc framework/service
ben ngoai - dung chung cho bat ky gia tri nao can cache trong tien trinh
server (vd ProviderResponse trong CachingProvider).

Khong ben vung qua lan restart server - chap nhan duoc o v1, don gian hon
la uu tien hang dau cho mot cache trong tien trinh dai han duy nhat.
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
