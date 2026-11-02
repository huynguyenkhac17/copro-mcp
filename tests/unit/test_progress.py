from __future__ import annotations

import asyncio
import json

import pytest

from copro.application import progress
from copro.application.dto import DelegateTaskRequest, DelegateTaskResult
from copro.application.job_manager import JobManager
from copro.application.map_reduce_orchestrator import MapReduceOrchestrator
from copro.domain.chunker import Chunker
from copro.domain.exceptions import ProviderUnavailableError
from copro.domain.file_context import FileContextResolver
from copro.domain.models import BudgetProfile, GenerationOptions
from copro.infra.cache import InMemoryTTLCache
from tests.unit.test_delegate_task_use_case import (
    FakeProvider,
    NamedFakeProvider,
    _chain_use_case,
    _config,
    _ok,
    _use_case,
)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[float, float | None, str]] = []

    async def __call__(self, value: float, total: float | None, message: str) -> None:
        self.events.append((value, total, message))

    @property
    def values(self) -> list[float]:
        return [value for value, _, _ in self.events]


@pytest.mark.asyncio
async def test_notes_count_up_without_a_total_and_phase_mutes_them():
    sink = Recorder()
    with progress.reporting_to(sink):
        await progress.note("calling groq")
        await progress.note("calling lite")
    assert sink.events == [(1, None, "calling groq"), (2, None, "calling lite")]

    sink = Recorder()
    with progress.reporting_to(sink):
        await progress.begin(2, "two steps")
        await progress.note("muted inside the phase")
        await progress.advance("one")
        await progress.advance("two")
    assert sink.events == [(0, 2, "two steps"), (1, 2, "one"), (2, 2, "two")]


@pytest.mark.asyncio
async def test_without_a_reporter_everything_is_a_no_op():
    await progress.note("x")
    await progress.begin(3, "x")
    assert await progress.step(asyncio.sleep(0, result=7), progress.counter("{done}", 1)) == 7


@pytest.mark.asyncio
async def test_a_failing_sink_never_fails_the_task():
    async def broken(value, total, message):
        raise RuntimeError("client went away")

    with progress.reporting_to(broken):
        await progress.note("still fine")
        assert await progress.step(asyncio.sleep(0, result="ok"), lambda: "done") == "ok"


@pytest.mark.asyncio
async def test_step_ticks_on_failure_too_and_reaches_child_tasks():
    sink = Recorder()

    async def boom():
        raise ValueError("x")

    with progress.reporting_to(sink):
        await progress.begin(2, "start")
        label = progress.counter("{done}/{total}", 2)
        results = await asyncio.gather(
            progress.step(asyncio.sleep(0, result=1), label),
            progress.step(boom(), label),
            return_exceptions=True,
        )

    assert results[0] == 1 and isinstance(results[1], ValueError)
    assert [m for _, _, m in sink.events] == ["start", "1/2", "2/2"]
    assert sink.values == sorted(sink.values)


@pytest.mark.asyncio
async def test_fallback_chain_reports_each_tier_it_tries():
    gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("status 503: overloaded")])
    lite = NamedFakeProvider("gemini-lite", responses=[_ok("gemini-lite")])
    use_case = _chain_use_case({"gemini": gemini, "gemini-lite": lite}, {"gemini": ["gemini-lite"]})

    sink = Recorder()
    with progress.reporting_to(sink):
        result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success
    messages = [m for _, _, m in sink.events]
    assert messages[0] == "calling gemini"
    assert messages[1].startswith("calling gemini-lite after gemini: status 503")
    assert sink.values == [1, 2]


@pytest.mark.asyncio
async def test_map_reduce_reports_every_part_then_the_reduce():
    provider = FakeProvider()
    profile = BudgetProfile(name="L2", description="", options=GenerationOptions(num_ctx=400))
    use_case = _use_case(provider, _config(), {"L2": profile})
    orchestrator = MapReduceOrchestrator(use_case, Chunker(), FileContextResolver(allowed_roots=[]))

    sink = Recorder()
    with progress.reporting_to(sink):
        result = await orchestrator.execute(
            DelegateTaskRequest(task="summarize", budget="L2", context="word " * 2000)
        )

    assert result.success
    parts = provider.call_count - 1
    assert parts >= 2
    messages = [m for _, _, m in sink.events]
    assert messages[0] == f"map-reduce: {parts} parts"
    assert messages[1:-1] == [f"map {i}/{parts} done" for i in range(1, parts + 1)]
    assert messages[-1] == f"reduce of {parts} extracts done"
    assert sink.events[-1][:2] == (parts + 1, parts + 1)


@pytest.mark.asyncio
async def test_async_job_exposes_its_latest_step_until_done():
    manager = JobManager(InMemoryTTLCache(max_entries=10, ttl_seconds=60))
    gate = asyncio.Event()

    async def job() -> DelegateTaskResult:
        await progress.begin(2, "start")
        await progress.advance("half way")
        await gate.wait()
        return DelegateTaskResult(success=True, raw_text="ok")

    job_id = manager.submit(job())
    for _ in range(5):
        await asyncio.sleep(0)
    running = manager.get(job_id)
    assert running.status == "running"
    assert running.progress == "[1/2] half way"

    gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    done = manager.get(job_id)
    assert done.status == "done" and done.progress is None


@pytest.mark.asyncio
async def test_mcp_client_receives_progress_notifications(monkeypatch):
    from mcp.client.client import Client

    from copro import server as server_module

    def fake_build(config, stats_collector):
        gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("503")])
        lite = NamedFakeProvider("gemini-lite", responses=[_ok("gemini-lite")])
        use_case = _chain_use_case({"gemini": gemini, "gemini-lite": lite}, {"gemini": ["gemini-lite"]})
        return use_case, FileContextResolver(allowed_roots=[])

    monkeypatch.setattr(server_module, "_build_use_case", fake_build)
    server = server_module.build_server()

    seen: list[tuple[float, float | None, str | None]] = []

    async def on_progress(value, total, message):
        seen.append((value, total, message))

    async with Client(server) as client:
        call = await client.call_tool(
            "delegate_task", {"task": "t", "budget": "L2", "provider": "gemini"},
            progress_callback=on_progress,
        )

    assert json.loads(call.content[0].text)["provider"] == "gemini-lite"
    assert [m for _, _, m in seen][:1] == ["calling gemini"]
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_a_value_that_does_not_grow_is_not_sent():
    sink = Recorder()
    with progress.reporting_to(sink):
        await progress.note("calling groq")
        await progress.begin(2, "phase after a note")
        await progress.advance("one")

    assert sink.events == [(1, None, "calling groq"), (2, 3, "one")]
