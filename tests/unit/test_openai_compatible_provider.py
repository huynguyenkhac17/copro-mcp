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
