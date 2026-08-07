from __future__ import annotations

import asyncio
import logging

import pytest

from local_ai_gateway.application.call_local_use_case import CallLocalUseCase
from local_ai_gateway.application.dto import CallLocalRequest
from local_ai_gateway.config.settings import (
    AppConfig,
    ConcurrencyConfig,
    RetryConfig,
)
from local_ai_gateway.domain.exceptions import ConfigurationError, ProviderUnavailableError
from local_ai_gateway.domain.models import BudgetProfile, GenerationOptions, ProviderResponse
from local_ai_gateway.infra.logging import InvocationLogger


class FakeProvider:
    """Provider gia lap LLMProvider de test use case ma khong can Ollama that."""

    def __init__(
        self,
        responses: list[ProviderResponse | Exception] | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._responses = list(responses or [])
        self._sleep_seconds = sleep_seconds
        self.call_count = 0
        self.concurrent_calls = 0
        self.max_concurrent_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        self.call_count += 1
        self.concurrent_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.concurrent_calls)
        try:
            if self._sleep_seconds:
                await asyncio.sleep(self._sleep_seconds)
            if self._responses:
                item = self._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return ProviderResponse(
                raw_text="default response",
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_ms=1.0,
            )
        finally:
            self.concurrent_calls -= 1


def _profile(timeout_seconds: float | None = None) -> BudgetProfile:
    return BudgetProfile(
        name="L2",
        description="test budget",
        options=GenerationOptions(num_ctx=8192),
        timeout_seconds=timeout_seconds,
    )


def _config(max_attempts: int = 1, max_parallel: int = 1) -> AppConfig:
    return AppConfig(
        retry=RetryConfig(max_attempts=max_attempts, backoff_base_seconds=0.01),
        concurrency=ConcurrencyConfig(max_parallel_requests=max_parallel),
        timeout_seconds=5.0,
    )


def _use_case(provider: FakeProvider, config: AppConfig, profiles: dict[str, BudgetProfile]) -> CallLocalUseCase:
    return CallLocalUseCase(
        provider=provider,
        config=config,
        budget_profiles=profiles,
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
    )


@pytest.mark.asyncio
async def test_success_text():
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="hello", provider_name="fake", model_name="fake-model", latency_ms=5.0
            )
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    result = await use_case.execute(CallLocalRequest(task="say hi", budget="L2"))

    assert result.success is True
    assert result.content == "hello"
    assert result.raw_text == "hello"
    assert result.warnings == []


@pytest.mark.asyncio
async def test_success_json_output():
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text='{"a": 1}', provider_name="fake", model_name="fake-model", latency_ms=5.0
            )
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    result = await use_case.execute(
        CallLocalRequest(task="return json", budget="L2", output_format="json")
    )

    assert result.success is True
    assert result.content == {"a": 1}


@pytest.mark.asyncio
async def test_unknown_budget_raises_configuration_error():
    provider = FakeProvider()
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    with pytest.raises(ConfigurationError):
        await use_case.execute(CallLocalRequest(task="task", budget="L99"))


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds():
    provider = FakeProvider(
        responses=[
            ProviderUnavailableError("tam thoi khong ket noi duoc"),
            ProviderResponse(
                raw_text="ok", provider_name="fake", model_name="fake-model", latency_ms=1.0
            ),
        ]
    )
    use_case = _use_case(provider, _config(max_attempts=2), {"L2": _profile()})

    result = await use_case.execute(CallLocalRequest(task="task", budget="L2"))

    assert result.success is True
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_transient_error_exhausts_retries_returns_failure_result():
    provider = FakeProvider(
        responses=[
            ProviderUnavailableError("loi 1"),
            ProviderUnavailableError("loi 2"),
        ]
    )
    use_case = _use_case(provider, _config(max_attempts=2), {"L2": _profile()})

    result = await use_case.execute(CallLocalRequest(task="task", budget="L2"))

    assert result.success is False
    assert result.error is not None
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_timeout_returns_failure_result_without_raising():
    provider = FakeProvider(sleep_seconds=0.3)
    use_case = _use_case(provider, _config(), {"L2": _profile(timeout_seconds=0.05)})

    result = await use_case.execute(CallLocalRequest(task="task", budget="L2"))

    assert result.success is False
    assert "khong tra loi" in (result.error or "")


@pytest.mark.asyncio
async def test_concurrency_semaphore_limits_parallel_provider_calls():
    provider = FakeProvider(sleep_seconds=0.1)
    use_case = _use_case(provider, _config(max_parallel=1), {"L2": _profile()})

    await asyncio.gather(
        use_case.execute(CallLocalRequest(task="a", budget="L2")),
        use_case.execute(CallLocalRequest(task="b", budget="L2")),
        use_case.execute(CallLocalRequest(task="c", budget="L2")),
    )

    assert provider.max_concurrent_calls == 1
