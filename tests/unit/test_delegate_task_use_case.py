from __future__ import annotations

import asyncio
import logging

import pytest

from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest
from llm_worker_pool.config.settings import (
    AppConfig,
    RetryConfig,
)
from llm_worker_pool.domain.exceptions import ConfigurationError, ProviderUnavailableError
from llm_worker_pool.domain.file_context import FileContextResolver
from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions, ProviderResponse
from llm_worker_pool.infra.logging import InvocationLogger


class FakeProvider:
    """Mock provider for testing without real Ollama."""

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
        self.last_prompt: str | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        self.call_count += 1
        self.last_prompt = prompt
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


def _config(max_attempts: int = 1) -> AppConfig:
    return AppConfig(
        retry=RetryConfig(max_attempts=max_attempts, backoff_base_seconds=0.01),
        timeout_seconds=5.0,
    )


def _use_case(
    provider: FakeProvider,
    config: AppConfig,
    profiles: dict[str, BudgetProfile],
    file_context_resolver: FileContextResolver | None = None,
) -> DelegateTaskUseCase:
    return DelegateTaskUseCase(
        providers={"fake": provider},
        default_provider="fake",
        config=config,
        budget_profiles=profiles,
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
        file_context_resolver=file_context_resolver,
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

    result = await use_case.execute(DelegateTaskRequest(task="say hi", budget="L2"))

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
        DelegateTaskRequest(task="return json", budget="L2", output_format="json")
    )

    assert result.success is True
    assert result.content == {"a": 1}


@pytest.mark.asyncio
async def test_unknown_budget_raises_configuration_error():
    provider = FakeProvider()
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    with pytest.raises(ConfigurationError):
        await use_case.execute(DelegateTaskRequest(task="task", budget="L99"))


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds():
    provider = FakeProvider(
        responses=[
            ProviderUnavailableError("temporarily unable to connect"),
            ProviderResponse(
                raw_text="ok", provider_name="fake", model_name="fake-model", latency_ms=1.0
            ),
        ]
    )
    use_case = _use_case(provider, _config(max_attempts=2), {"L2": _profile()})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_transient_error_exhausts_retries_returns_failure_result():
    provider = FakeProvider(
        responses=[
            ProviderUnavailableError("error 1"),
            ProviderUnavailableError("error 2"),
        ]
    )
    use_case = _use_case(provider, _config(max_attempts=2), {"L2": _profile()})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is False
    assert result.error is not None
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_timeout_returns_failure_result_without_raising():
    provider = FakeProvider(sleep_seconds=0.3)
    use_case = _use_case(provider, _config(), {"L2": _profile(timeout_seconds=0.05)})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is False
    assert "did not respond" in (result.error or "")


@pytest.mark.asyncio
async def test_empty_response_from_thinking_exhaustion_retries_with_thinking_disabled():
    profile = _profile()  # think=None (model default, i.e. thinking left on)
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                tokens_used=profile.options.num_predict,
            ),
            ProviderResponse(
                raw_text="actual answer",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
            ),
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": profile})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert result.content == "actual answer"
    assert provider.call_count == 2
    assert "retried_with_thinking_disabled_after_empty_response" in result.warnings


@pytest.mark.asyncio
async def test_empty_response_with_thinking_already_disabled_does_not_retry():
    profile = _profile()
    profile = profile.model_copy(update={"options": profile.options.model_copy(update={"think": False})})
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                tokens_used=profile.options.num_predict,
            ),
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": profile})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert provider.call_count == 1
    assert "empty_response_possible_thinking_exhaustion" in result.warnings


@pytest.mark.asyncio
async def test_file_paths_are_merged_into_context_sent_to_provider(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("content from file")
    resolver = FileContextResolver(allowed_roots=[tmp_path])

    provider = FakeProvider()
    use_case = _use_case(provider, _config(), {"L2": _profile()}, file_context_resolver=resolver)

    result = await use_case.execute(
        DelegateTaskRequest(task="summarize", budget="L2", file_paths=[str(f)])
    )

    assert result.success is True
    assert "content from file" in (provider.last_prompt or "")
    assert result.warnings == []


@pytest.mark.asyncio
async def test_file_paths_without_configured_resolver_warns_disabled():
    provider = FakeProvider()
    use_case = _use_case(provider, _config(), {"L2": _profile()})  # resolver mac dinh: allowed_roots=[]

    result = await use_case.execute(
        DelegateTaskRequest(task="summarize", budget="L2", file_paths=["D:/not/allowed/read.txt"])
    )

    assert result.success is True
    assert any(w.startswith("file_access_disabled") for w in result.warnings)
