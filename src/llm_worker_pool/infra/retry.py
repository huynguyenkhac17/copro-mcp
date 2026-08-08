"""Backoff wrapper for transient errors when calling the provider."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from llm_worker_pool.domain.exceptions import (
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
    """Call func(), retry up to max_attempts times if a transient error occurs.

    Backoff: backoff_base_seconds * 2**attempt + small jitter. Non-transient
    errors (ConfigurationError, InvalidResponseError, ...) are not
    caught here - they will be raised immediately on the first failure.
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
