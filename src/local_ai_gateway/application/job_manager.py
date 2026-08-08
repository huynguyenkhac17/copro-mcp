"""Background job manager for async call_local tasks."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from enum import Enum
from typing import Any

from local_ai_gateway.application.dto import CallLocalResult
from local_ai_gateway.domain.exceptions import ConfigurationError
from local_ai_gateway.infra.cache import InMemoryTTLCache


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobManager:
    def __init__(self, ttl_cache: InMemoryTTLCache[CallLocalResult]) -> None:
        self._cache = ttl_cache
        # asyncio only holds a WEAK reference to tasks created via create_task - without
        # keeping our own strong reference here, the task can be garbage-collected mid-run
        # and the job would silently vanish. See asyncio docs "Important" note on create_task.
        self._background_tasks: set[asyncio.Task[None]] = set()

    def submit(self, coro: Coroutine[Any, Any, CallLocalResult]) -> str:
        job_id = str(uuid.uuid4())

        initial_result = CallLocalResult(
            success=True,
            status=JobStatus.RUNNING.value,  # type: ignore[arg-type]
            job_id=job_id,
        )
        self._cache.set(job_id, initial_result)

        async def _run() -> None:
            try:
                result = await coro
                result.job_id = job_id
                result.status = JobStatus.DONE.value  # type: ignore[arg-type]
                self._cache.set(job_id, result)
            except Exception as exc:
                failed_result = CallLocalResult(
                    success=False,
                    status=JobStatus.FAILED.value,  # type: ignore[arg-type]
                    job_id=job_id,
                    error=str(exc)
                )
                self._cache.set(job_id, failed_result)

        task = asyncio.create_task(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return job_id

    def get(self, job_id: str) -> CallLocalResult:
        result = self._cache.get(job_id)
        if result is None:
            raise ConfigurationError(f"Job ID '{job_id}' does not exist or has expired.")
        return result
