"""Progress of a long delegate_task, for the caller to see while it waits.

A synchronous call reports through MCP `notifications/progress` (Claude Code sends a
progress token with every tool call - checked on 2.1.312); an async job writes the
latest step into its status, read by check_job_status. Both go through one reporter
held in a ContextVar, so it reaches the map/batch/arm calls that asyncio.gather runs
as child tasks without threading a parameter through every layer.

MCP requires the progress value to increase with every notification, so there is a
single counter per request:
- outside an orchestrator phase, the use case notes each provider tier it tries
  (`note`), counting up with no known total;
- an orchestrator announces N steps (`begin`) and ticks them off as they finish
  (`advance`); the use-case notes of its child calls are dropped meanwhile, since
  several of them run at once and would make the count meaningless.

Reporting never fails the task: a sink error (the client went away, the request is
already answered) is swallowed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")

ProgressSink = Callable[[float, "float | None", str], Awaitable[None]]

_logger = logging.getLogger(__name__)


class ProgressReporter:
    def __init__(self, sink: ProgressSink) -> None:
        self._sink = sink
        self._value = 0.0
        self._total: float | None = None
        self._phase = False
        self._sent: float | None = None

    async def note(self, message: str) -> None:
        """A step of a single call (e.g. the tier being tried). Ignored inside a phase."""
        if self._phase:
            return
        self._value += 1
        await self._emit(message)

    async def begin(self, steps: int, message: str) -> None:
        """Start a phase of `steps` steps, each finished with `advance`."""
        self._phase = True
        self._total = self._value + steps
        await self._emit(message)

    async def advance(self, message: str) -> None:
        self._value += 1
        if self._total is not None and self._value > self._total:
            self._total = self._value
        await self._emit(message)

    async def _emit(self, message: str) -> None:
        if self._sent is not None and self._value <= self._sent:
            return  # e.g. `begin` right after a note: MCP wants every value to grow
        self._sent = self._value
        try:
            await self._sink(self._value, self._total, message)
        except Exception:  # noqa: BLE001 - progress is best effort
            _logger.debug("progress report dropped", exc_info=True)


_current: ContextVar[ProgressReporter | None] = ContextVar("progress_reporter", default=None)


@contextmanager
def reporting_to(sink: ProgressSink | None) -> Iterator[ProgressReporter | None]:
    """Report the progress of everything awaited inside this block to `sink`."""
    reporter = ProgressReporter(sink) if sink is not None else None
    token = _current.set(reporter)
    try:
        yield reporter
    finally:
        _current.reset(token)


def current() -> ProgressReporter | None:
    return _current.get()


async def note(message: str) -> None:
    reporter = _current.get()
    if reporter is not None:
        await reporter.note(message)


async def begin(steps: int, message: str) -> None:
    reporter = _current.get()
    if reporter is not None:
        await reporter.begin(steps, message)


async def advance(message: str) -> None:
    reporter = _current.get()
    if reporter is not None:
        await reporter.advance(message)


def counter(template: str, total: int) -> Callable[[], str]:
    """Labels for `step`: template.format(done=1..total, total=total), in finishing order."""
    done = 0

    def label() -> str:
        nonlocal done
        done += 1
        return template.format(done=done, total=total)

    return label


async def step(work: Awaitable[T], label: Callable[[], str]) -> T:
    """Await `work`, then tick the current phase on (also when it raised)."""
    try:
        result = await work
    except Exception:
        await advance(label())
        raise
    await advance(label())
    return result
