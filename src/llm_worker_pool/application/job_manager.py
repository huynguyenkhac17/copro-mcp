"""Background job manager for async delegate_task tasks."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from enum import Enum
from typing import Any

from llm_worker_pool.application import progress
from llm_worker_pool.application.dto import DelegateTaskResult
from llm_worker_pool.domain.exceptions import ConfigurationError
from llm_worker_pool.infra.cache import InMemoryTTLCache


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobManager:
    def __init__(self, ttl_cache: InMemoryTTLCache[DelegateTaskResult]) -> None:
        self._cache = ttl_cache
        # asyncio only holds a WEAK reference to tasks created via create_task - without
        # keeping our own strong reference here, the task can be garbage-collected mid-run
        # and the job would silently vanish. See asyncio docs "Important" note on create_task.
        self._background_tasks: set[asyncio.Task[None]] = set()

    def submit(self, coro: Coroutine[Any, Any, DelegateTaskResult]) -> str:
        job_id = str(uuid.uuid4())

        initial_result = DelegateTaskResult(
            success=True,
            status=JobStatus.RUNNING.value,  # type: ignore[arg-type]
            job_id=job_id,
        )
        self._cache.set(job_id, initial_result)

        async def _record_progress(value: float, total: float | None, message: str) -> None:
            # The MCP request that submitted the job has already been answered, so the
            # progress goes into the status check_job_status returns.
            current = self._cache.get(job_id)
            if current is not None and current.status == JobStatus.RUNNING.value:
                current.progress = format_progress(value, total, message)

        async def _run() -> None:
            try:
                with progress.reporting_to(_record_progress):
                    result = await coro
                result.job_id = job_id
                result.status = JobStatus.DONE.value  # type: ignore[arg-type]
                self._cache.set(job_id, result)
            except Exception as exc:
                failed_result = DelegateTaskResult(
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

    def get(self, job_id: str) -> DelegateTaskResult:
        result = self._cache.get(job_id)
        if result is None:
            raise ConfigurationError(f"Job ID '{job_id}' does not exist or has expired.")
        return result


def format_progress(value: float, total: float | None, message: str) -> str:
    count = f"{value:g}/{total:g}" if total is not None else f"{value:g}"
    return f"[{count}] {message}"
