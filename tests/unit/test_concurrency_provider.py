from __future__ import annotations

import asyncio

import pytest

from llm_worker_pool.domain.models import GenerationOptions
from llm_worker_pool.providers.concurrency_provider import ConcurrencyProvider
from tests.unit.test_delegate_task_use_case import FakeProvider


@pytest.mark.asyncio
async def test_max_parallel_one_runs_calls_sequentially():
    inner = FakeProvider(sleep_seconds=0.1)
    provider = ConcurrencyProvider(inner, max_parallel=1)

    await asyncio.gather(
        provider.generate("a", GenerationOptions()),
        provider.generate("b", GenerationOptions()),
        provider.generate("c", GenerationOptions()),
    )

    assert inner.max_concurrent_calls == 1


@pytest.mark.asyncio
async def test_higher_max_parallel_allows_overlap():
    inner = FakeProvider(sleep_seconds=0.1)
    provider = ConcurrencyProvider(inner, max_parallel=3)

    await asyncio.gather(
        provider.generate("a", GenerationOptions()),
        provider.generate("b", GenerationOptions()),
        provider.generate("c", GenerationOptions()),
    )

    assert inner.max_concurrent_calls == 3


@pytest.mark.asyncio
async def test_delegates_provider_and_model_name():
    inner = FakeProvider()
    provider = ConcurrencyProvider(inner, max_parallel=1)

    assert provider.provider_name == inner.provider_name
    assert provider.model_name == inner.model_name
