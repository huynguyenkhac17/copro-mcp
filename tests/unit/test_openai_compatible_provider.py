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


@pytest.mark.asyncio
async def test_prompt_over_max_prompt_tokens_is_refused_before_any_request():
    from llm_worker_pool.domain.exceptions import RequestTooLargeError
    from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker

    calls: list[httpx.Request] = []
    cb = KeyCircuitBreaker(failure_threshold=1)
    provider = OpenAICompatibleProvider(
        provider_name="groq",
        endpoint="https://example.invalid/v1",
        model="m",
        api_keys=["key-a"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: calls.append(r) or _ok_response(r))),
        circuit_breaker=cb,
        max_prompt_tokens=100,
    )

    with pytest.raises(RequestTooLargeError):
        await provider.generate("x" * 4000, GenerationOptions())  # ~1000 tokens

    assert calls == []
    assert cb.is_available("key-a")  # a local refusal must not quarantine the key

    await provider.generate("short prompt", GenerationOptions())
    assert len(calls) == 1


def test_each_provider_gets_its_own_circuit_breaker(monkeypatch):
    from llm_worker_pool.config.settings import AppConfig, ProviderInstanceConfig
    from llm_worker_pool.providers.registry import ProviderFactory

    monkeypatch.setenv("SHARED_TEST_KEY", "k")
    cfg = AppConfig()
    cfg.cache.enabled = False
    spec = dict(type="openai_compatible", endpoint="https://example.invalid/v1",
                api_key_envs=["SHARED_TEST_KEY"], per_key_concurrency=1)
    cfg.providers = {"flash": ProviderInstanceConfig(model="flash", **spec),
                     "lite": ProviderInstanceConfig(model="lite", **spec)}

    providers = ProviderFactory(cfg).create_all()
    flash_cb, lite_cb = providers["flash"].circuit_breaker, providers["lite"].circuit_breaker

    for _ in range(cfg.circuit_breaker.failure_threshold):
        flash_cb.record_failure("k", 503)

    assert not flash_cb.is_available("k")
    assert lite_cb.is_available("k")  # flash overload must not lock the key for lite


def test_strict_mode_only_for_strict_compatible_schemas():
    # Regression: strict=True with a loose schema made Groq answer HTTP 400 for every
    # json_schema request (found by worker-eval, 2026-10-30).
    from llm_worker_pool.providers.openai_compatible_provider import _is_strict_compatible

    loose = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
    strict = {**loose, "additionalProperties": False}
    nested_loose = {
        "type": "object", "required": ["items"], "additionalProperties": False,
        "properties": {"items": {"type": "array", "items": loose}},
    }
    partial_required = {**strict, "properties": {"a": {}, "b": {}}}

    assert _is_strict_compatible(strict) is True
    assert _is_strict_compatible(loose) is False
    assert _is_strict_compatible(nested_loose) is False
    assert _is_strict_compatible(partial_required) is False


# --- quota-aware key selection (infra/quota_tracker.py) ---------------------------

import asyncio

from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker
from llm_worker_pool.infra.quota_tracker import QuotaTracker


def _quota_provider(api_keys, handler, **tracker_kwargs) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
    return OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=api_keys,
        http_client=client,
        circuit_breaker=KeyCircuitBreaker(failure_threshold=3, quarantine_seconds=60),
        quota_tracker=QuotaTracker(**tracker_kwargs),
    )


@pytest.mark.asyncio
async def test_key_reported_out_of_tokens_is_skipped():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        seen.append(auth)
        remaining = "0" if auth == "Bearer key-a" else "8000"
        return httpx.Response(
            200,
            headers={"x-ratelimit-remaining-tokens": remaining, "x-ratelimit-reset-tokens": "30s"},
            json={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
        )

    provider = _quota_provider(["key-a", "key-b"], handler)
    for _ in range(3):
        await provider.generate("p", GenerationOptions())

    # key-a answered once, reported 0 tokens left, and is not picked again.
    assert seen == ["Bearer key-a", "Bearer key-b", "Bearer key-b"]


@pytest.mark.asyncio
async def test_429_with_retry_delay_parks_the_key():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        seen.append(auth)
        if auth == "Bearer key-a":
            return httpx.Response(429, text='{"error": {"details": [{"retryDelay": "40s"}]}}')
        return _ok_response(request)

    provider = _quota_provider(["key-a", "key-b"], handler)
    with pytest.raises(Exception):
        await provider.generate("p", GenerationOptions())
    await provider.generate("p", GenerationOptions())
    await provider.generate("p", GenerationOptions())

    assert seen == ["Bearer key-a", "Bearer key-b", "Bearer key-b"]


@pytest.mark.asyncio
async def test_unready_when_every_key_is_parked_and_last_tier_still_sends():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text='{"error": {"details": [{"retryDelay": "40s"}]}}')

    provider = _quota_provider(["key-a"], handler)
    with pytest.raises(Exception):
        await provider.generate("p", GenerationOptions())

    reason = provider.unready_reason("p", GenerationOptions())
    assert reason is not None and reason.startswith("quota_exhausted_locally")
    assert "key-a" not in reason

    # Called directly (= as the last tier), it still goes out instead of refusing.
    with pytest.raises(Exception):
        await provider.generate("p", GenerationOptions())
    assert calls == 2


@pytest.mark.asyncio
async def test_cancelled_hang_is_recorded_and_blocks_model():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return _ok_response(request)

    provider = _quota_provider(["key-a"], handler, timeouts_before_block=2)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(provider.generate("p", GenerationOptions()), timeout=0.05)

    assert provider.circuit_breaker.get_stats()  # breaker saw the failures
    reason = provider.unready_reason("p", GenerationOptions())
    assert reason is not None and reason.startswith("model_unresponsive_locally")
    # Reservations were released even though the call was cancelled.
    assert provider.quota_tracker.can_serve("key-a", 1)


@pytest.mark.asyncio
async def test_without_tracker_provider_is_always_ready():
    provider = _make_provider(["key-a"], _ok_response)
    assert provider.unready_reason("p", GenerationOptions()) is None


@pytest.mark.asyncio
async def test_quota_is_reserved_while_waiting_on_a_busy_key():
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return _ok_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
    tracker = QuotaTracker()
    provider = OpenAICompatibleProvider(
        provider_name="fake",
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_keys=["key-a"],
        http_client=client,
        per_key_concurrency=1,
        quota_tracker=tracker,
    )
    prompt = "x" * 2400  # ~600 tokens each
    first = asyncio.create_task(provider.generate(prompt, GenerationOptions()))
    await asyncio.sleep(0)
    second = asyncio.create_task(provider.generate(prompt, GenerationOptions()))
    await asyncio.sleep(0.01)

    # The second call is queued on the semaphore, yet its tokens already count.
    assert tracker._state("key-a").reserved_requests == 2

    release.set()
    await asyncio.gather(first, second)
    assert tracker._state("key-a").reserved_requests == 0
    assert tracker._state("key-a").reserved_tokens == 0
