from __future__ import annotations

import pytest

from copro.domain.exceptions import InvalidResponseError, ProviderUnavailableError
from copro.infra import retry
from copro.infra.retry import with_backoff


class FakeTime:
    """Clock + sleep pair: sleeping advances the clock, nothing really waits."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(retry.asyncio, "sleep", fake.sleep)
    monkeypatch.setattr(retry.random, "uniform", lambda a, b: 0.0)
    return fake


def _failing(errors: list[Exception], result: str = "ok"):
    calls = 0

    async def func() -> str:
        nonlocal calls
        calls += 1
        if errors:
            raise errors.pop(0)
        return result

    func.calls = lambda: calls  # type: ignore[attr-defined]
    return func


@pytest.mark.asyncio
async def test_blind_backoff_without_hint(fake_time):
    func = _failing([ProviderUnavailableError("503"), ProviderUnavailableError("503")])
    assert await with_backoff(func, max_attempts=4, backoff_base_seconds=0.5, clock=fake_time) == "ok"
    assert fake_time.sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_sleeps_the_hinted_retry_after(fake_time):
    func = _failing([ProviderUnavailableError("429", retry_after=7.0)])
    assert await with_backoff(
        func, max_attempts=4, backoff_base_seconds=0.5, deadline=fake_time.now + 30, clock=fake_time
    ) == "ok"
    assert fake_time.sleeps == [7.0]


@pytest.mark.asyncio
async def test_zero_hint_retries_at_once_on_another_key(fake_time):
    func = _failing([ProviderUnavailableError("429", retry_after=0.0)])
    assert await with_backoff(func, max_attempts=4, backoff_base_seconds=0.5, clock=fake_time) == "ok"
    assert fake_time.sleeps == [0.0]


@pytest.mark.asyncio
async def test_gives_up_at_once_when_the_lock_outlasts_the_deadline(fake_time):
    error = ProviderUnavailableError("429", retry_after=40.0)
    func = _failing([error])
    with pytest.raises(ProviderUnavailableError) as raised:
        await with_backoff(
            func, max_attempts=4, backoff_base_seconds=0.5, deadline=fake_time.now + 20, clock=fake_time
        )
    assert raised.value is error
    assert func.calls() == 1
    assert fake_time.sleeps == []


@pytest.mark.asyncio
async def test_blind_backoff_also_stops_at_the_deadline(fake_time):
    func = _failing([ProviderUnavailableError("503")] * 4)
    with pytest.raises(ProviderUnavailableError):
        await with_backoff(
            func, max_attempts=4, backoff_base_seconds=0.5, deadline=fake_time.now + 1.2, clock=fake_time
        )
    # 0.5 fits; the next 1.0 would end past the deadline.
    assert fake_time.sleeps == [0.5]
    assert func.calls() == 2


@pytest.mark.asyncio
async def test_non_transient_error_is_not_retried(fake_time):
    func = _failing([InvalidResponseError("400")])
    with pytest.raises(InvalidResponseError):
        await with_backoff(func, max_attempts=4, backoff_base_seconds=0.5, clock=fake_time)
    assert func.calls() == 1
