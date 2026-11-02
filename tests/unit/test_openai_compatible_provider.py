from __future__ import annotations

import httpx
import pytest

from llm_worker_pool.domain.exceptions import ProviderUnavailableError
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


@pytest.mark.asyncio
async def test_prompt_cap_counts_punctuation_heavy_text_higher():
    # 2026-11-01: JSON a worker had to repair came back ~27% bigger on groq than
    # len/4, and the repair call (prompt + its system prompt) drew a 413.
    from llm_worker_pool.domain.exceptions import RequestTooLargeError

    calls: list[httpx.Request] = []
    provider = OpenAICompatibleProvider(
        provider_name="groq",
        endpoint="https://example.invalid/v1",
        model="m",
        api_keys=["key-a"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: calls.append(r) or _ok_response(r))),
        max_prompt_tokens=1000,
    )
    payload = '{"a": [1, 2], "b": {}}, ' * 160  # 3840 chars: len/4 = 960, groq-like ~1300
    with pytest.raises(RequestTooLargeError):
        await provider.generate(payload, GenerationOptions())
    await provider.generate("word " * 700, GenerationOptions())  # 3500 chars of letters: ~700
    assert len(calls) == 1


def test_prompt_estimate_tracks_measured_groq_ratios():
    from llm_worker_pool.domain.token_estimate import estimate_prompt_tokens, estimate_tokens

    code = "def total(price, qty):\n    return price * qty\n" * 100
    vietnamese = "Trả lời bằng tiếng Việt, trừ khi người dùng yêu cầu khác. " * 50
    json_text = '{"id": "review-orders-l3", "scores": {"bug_recall": 1.0}}, ' * 50
    # Indented code stays near len/4 (groq measured 0.86-1.06x of it).
    assert 0.8 * estimate_tokens(code) <= estimate_prompt_tokens(code) <= 1.2 * estimate_tokens(code)
    # Non-ASCII and punctuation weigh more than len/4 gives them.
    assert estimate_prompt_tokens(vietnamese) > 1.15 * estimate_tokens(vietnamese)
    assert estimate_prompt_tokens(json_text) > 1.15 * estimate_tokens(json_text)
    assert estimate_prompt_tokens("") == 0


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


def test_factory_records_providers_skipped_for_missing_keys(monkeypatch):
    from llm_worker_pool.config.settings import AppConfig, ProviderInstanceConfig
    from llm_worker_pool.providers.registry import ProviderFactory

    monkeypatch.delenv("UNSET_TEST_KEY", raising=False)
    cfg = AppConfig()
    cfg.providers = {"cloud": ProviderInstanceConfig(
        type="openai_compatible", endpoint="https://example.invalid/v1", model="m",
        api_key_envs=["UNSET_TEST_KEY"],
    )}

    factory = ProviderFactory(cfg)
    assert factory.create_all() == {}
    assert "UNSET_TEST_KEY" in factory.skipped["cloud"]


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
async def test_unready_reason_names_quarantine_not_quota_after_connection_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("certificate verify failed")

    provider = _quota_provider(["key-a", "key-b"], handler)
    for _ in range(6):  # failure_threshold=3 per key, keys picked round-robin
        with pytest.raises(ProviderUnavailableError):
            await provider.generate("p", GenerationOptions())

    reason = provider.unready_reason("p", GenerationOptions())
    assert reason is not None and reason.startswith("keys_quarantined_locally")
    assert "quota" not in reason
    assert "earliest key free in" in reason
    assert "key-a" not in reason


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


@pytest.mark.asyncio
async def test_429_reports_the_wait_until_some_key_is_free():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] == "Bearer key-a":
            return httpx.Response(429, headers={"retry-after": "20"}, text="rate limited")
        return httpx.Response(429, headers={"retry-after": "8"}, text="rate limited")

    provider = _quota_provider(["key-a", "key-b"], handler)
    with pytest.raises(ProviderUnavailableError) as first:
        await provider.generate("p", GenerationOptions())
    # key-b is still free, so a retry need not wait.
    assert first.value.retry_after == 0.0

    with pytest.raises(ProviderUnavailableError) as second:
        await provider.generate("p", GenerationOptions())
    # Both parked now: the earlier of the two locks.
    assert second.value.retry_after == pytest.approx(8, abs=0.5)


@pytest.mark.asyncio
async def test_429_without_a_hint_leaves_retry_after_unset():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    provider = _quota_provider(["key-a"], handler)
    with pytest.raises(ProviderUnavailableError) as raised:
        await provider.generate("p", GenerationOptions())
    assert raised.value.retry_after is None


@pytest.mark.asyncio
async def test_429_retry_after_header_used_without_a_tracker():
    provider = _make_provider(
        ["key-a"], lambda request: httpx.Response(429, headers={"retry-after": "5"}, text="x")
    )
    with pytest.raises(ProviderUnavailableError) as raised:
        await provider.generate("p", GenerationOptions())
    assert raised.value.retry_after == 5.0


async def _payload_sent(options: GenerationOptions, **provider_kwargs) -> dict:
    import json

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _ok_response(request)

    provider = OpenAICompatibleProvider(
        provider_name="fake", endpoint="https://example.invalid/v1", model="m", api_keys=["k"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **provider_kwargs,
    )
    await provider.generate("p", options)
    return sent[0]


_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


@pytest.mark.asyncio
async def test_default_payload_keeps_reasoning_effort_and_json_schema():
    payload = await _payload_sent(GenerationOptions(reasoning_effort="low", format=_SCHEMA))
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_vendor_switches_shape_the_payload():
    # Mistral rejects reasoning_effort values other than high/none; DeepSeek only
    # knows json_object; some OpenRouter free models refuse response_format.
    options = GenerationOptions(reasoning_effort="low", format=_SCHEMA)
    payload = await _payload_sent(options, send_reasoning_effort=False, structured_output="json_object")
    assert "reasoning_effort" not in payload
    assert payload["response_format"] == {"type": "json_object"}

    payload = await _payload_sent(options, structured_output="none")
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_extra_body_adds_vendor_fields_without_overriding_the_adapter():
    payload = await _payload_sent(
        GenerationOptions(reasoning_effort="low"),
        extra_body={"thinking": {"type": "disabled"}, "model": "should-not-win"},
    )
    assert payload["thinking"] == {"type": "disabled"} and payload["model"] == "m"


def test_no_reasoning_knob_when_the_field_is_not_sent():
    provider = OpenAICompatibleProvider(
        provider_name="fake", endpoint="https://example.invalid/v1", model="m", api_keys=["k"],
        send_reasoning_effort=False,
    )
    assert provider.thinking_option is None
