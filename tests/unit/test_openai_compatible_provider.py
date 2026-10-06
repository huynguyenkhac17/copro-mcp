from __future__ import annotations

import httpx
import pytest

from llm_worker_pool.domain.models import GenerationOptions
from llm_worker_pool.providers.openai_compatible_provider import OpenAICompatibleProvider


def _make_provider(api_keys: list[str], handler) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=None)
    return OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=api_keys,
        http_client=client,
    )


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        },
    )


@pytest.mark.asyncio
async def test_single_key_reused_every_call():
    seen_auth_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers["Authorization"])
        return _ok_response(request)

    provider = _make_provider(["key-a"], handler)

    await provider.generate("p1", GenerationOptions())
    await provider.generate("p2", GenerationOptions())

    assert seen_auth_headers == ["Bearer key-a", "Bearer key-a"]


@pytest.mark.asyncio
async def test_multiple_keys_round_robin_in_order():
    seen_auth_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers["Authorization"])
        return _ok_response(request)

    provider = _make_provider(["key-a", "key-b"], handler)

    await provider.generate("p1", GenerationOptions())
    await provider.generate("p2", GenerationOptions())
    await provider.generate("p3", GenerationOptions())
    await provider.generate("p4", GenerationOptions())

    assert seen_auth_headers == [
        "Bearer key-a",
        "Bearer key-b",
        "Bearer key-a",
        "Bearer key-b",
    ]


@pytest.mark.asyncio
async def test_circuit_breaker_skips_quarantined_key():
    from llm_worker_pool.domain.exceptions import ProviderUnavailableError
    from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker

    cb = KeyCircuitBreaker(failure_threshold=1, quarantine_seconds=60.0)

    seen_auth_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        seen_auth_headers.append(auth)
        if auth == "Bearer key-a":
            return httpx.Response(429, text="Rate limit exceeded")
        return _ok_response(request)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=None)
    provider = OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=["key-a", "key-b"],
        http_client=client,
        circuit_breaker=cb,
    )

    # 1st call picks key-a -> 429 -> quarantines key-a
    with pytest.raises(ProviderUnavailableError):
        await provider.generate("p1", GenerationOptions())

    # 2nd and 3rd calls should skip quarantined key-a and use key-b
    res2 = await provider.generate("p2", GenerationOptions())
    assert res2.raw_text == "hi"

    res3 = await provider.generate("p3", GenerationOptions())
    assert res3.raw_text == "hi"

    assert seen_auth_headers == [
        "Bearer key-a",
        "Bearer key-b",
        "Bearer key-b",
    ]


@pytest.mark.asyncio
async def test_per_key_concurrency_parallel_execution():
    import asyncio

    running: set[str] = set()
    max_running_per_key: dict[str, int] = {"key-a": 0, "key-b": 0}

    async def async_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        key = auth.replace("Bearer ", "")
        running.add(key)
        await asyncio.sleep(0.05)
        running.remove(key)
        return _ok_response(request)

    transport = httpx.MockTransport(async_handler)
    client = httpx.AsyncClient(transport=transport, timeout=None)
    provider = OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=["key-a", "key-b"],
        http_client=client,
        per_key_concurrency=1,
    )

    # Run 2 tasks simultaneously
    t1 = asyncio.create_task(provider.generate("p1", GenerationOptions()))
    t2 = asyncio.create_task(provider.generate("p2", GenerationOptions()))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1.raw_text == "hi"
    assert r2.raw_text == "hi"
