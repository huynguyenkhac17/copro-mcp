"""Backoff wrapper cho cac loi transient khi goi provider."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from local_ai_gateway.domain.exceptions import (
    ProviderTimeoutError,
    ProviderUnavailableError,
)

T = TypeVar("T")

_RETRYABLE_EXCEPTIONS = (ProviderUnavailableError, ProviderTimeoutError)


async def with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    backoff_base_seconds: float,
) -> T:
    """Goi func(), retry toi da max_attempts lan neu gap loi transient.

    Backoff: backoff_base_seconds * 2**attempt + jitter nho. Loi khong
    transient (ConfigurationError, InvalidResponseError, ...) khong duoc
    bat o day - se raise thang len tren ngay lan dau.
    """

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            delay = backoff_base_seconds * (2**attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
