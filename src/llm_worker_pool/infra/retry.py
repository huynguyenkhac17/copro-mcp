"""Backoff wrapper for transient errors when calling the provider."""

from __future__ import annotations

import asyncio
import random
import time
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
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Call func(), retry up to max_attempts times if a transient error occurs.

    Backoff: the error's `retry_after` when the provider said how long it is blocked
    (a 429 with Retry-After/retryDelay), else backoff_base_seconds * 2**attempt; plus
    small jitter. A wait that would end at or after `deadline` (a `clock()` value) is
    not slept: the last error is raised at once, so the fallback chain moves on instead
    of four blind retries burning the tier's time on a key known to be locked.
    Non-transient errors (ConfigurationError, InvalidResponseError, ...) are not
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
            hinted = getattr(exc, "retry_after", None)
            base = hinted if hinted is not None else backoff_base_seconds * (2**attempt)
            delay = base + random.uniform(0, 0.1)
            if deadline is not None and clock() + delay >= deadline:
                break
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
