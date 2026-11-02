from __future__ import annotations

import pytest

from copro.domain.models import GenerationOptions, ProviderResponse
from copro.infra.cache import InMemoryTTLCache
from copro.providers.caching_provider import CachingProvider
from tests.unit.test_delegate_task_use_case import FakeProvider


def _options(**overrides) -> GenerationOptions:
    return GenerationOptions(**overrides)


@pytest.mark.asyncio
async def test_second_identical_call_hits_cache():
    inner = FakeProvider()
    cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(max_entries=10)
    provider = CachingProvider(inner, cache)

    options = _options()
    first = await provider.generate("cung mot prompt", options)
    second = await provider.generate("cung mot prompt", options)

    assert inner.call_count == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.raw_text == first.raw_text


@pytest.mark.asyncio
async def test_different_prompt_is_cache_miss():
    inner = FakeProvider()
    cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(max_entries=10)
    provider = CachingProvider(inner, cache)

    options = _options()
    await provider.generate("prompt A", options)
    await provider.generate("prompt B", options)

    assert inner.call_count == 2


@pytest.mark.asyncio
async def test_different_options_is_cache_miss():
    inner = FakeProvider()
    cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(max_entries=10)
    provider = CachingProvider(inner, cache)

    await provider.generate("cung mot prompt", _options(temperature=0.1))
    await provider.generate("cung mot prompt", _options(temperature=0.9))

    assert inner.call_count == 2


@pytest.mark.asyncio
async def test_delegates_provider_and_model_name():
    inner = FakeProvider()
    cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(max_entries=10)
    provider = CachingProvider(inner, cache)

    assert provider.provider_name == inner.provider_name
    assert provider.model_name == inner.model_name
