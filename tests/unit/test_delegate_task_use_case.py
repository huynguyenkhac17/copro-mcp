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


class ReasoningProvider(FakeProvider):
    """A cloud-style backend: limits reasoning via reasoning_effort, not `think`."""

    @property
    def thinking_option(self) -> str | None:
        return "reasoning_effort"


@pytest.mark.asyncio
async def test_truncated_answer_is_flagged_instead_of_passing_as_complete():
    """A cut-off answer used to come back success=True with no warnings at all."""
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="an answer that stops mid-sen",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                finish_reason="length",
                prompt_tokens=100,
                completion_tokens=512,
                tokens_used=612,
            ),
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert result.truncated is True
    assert any("output_truncated_at_max_tokens" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_thinking_tokens_are_derived_and_reported():
    """Reasoning is billed inside the total but absent from prompt and completion."""
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="short answer",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                prompt_tokens=100,
                completion_tokens=40,
                tokens_used=1140,
            ),
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.thinking_tokens == 1000
    assert any("thinking_consumed_1000" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_large_prompt_alone_does_not_look_like_thinking_exhaustion():
    """The old guard compared a prompt+thinking+completion total against a cap on
    completion alone, so any big prompt satisfied it automatically."""
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                raw_text="a perfectly complete answer",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                prompt_tokens=50_000,
                completion_tokens=12,
                tokens_used=50_012,
                finish_reason="stop",
            ),
        ]
    )
    use_case = _use_case(provider, _config(), {"L2": _profile()})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert provider.call_count == 1
    assert result.warnings == []


@pytest.mark.asyncio
async def test_retry_lowers_reasoning_effort_for_cloud_backends():
    """`think=False` is not on the OpenAI-compatible wire, so retrying with only
    that set re-sent an identical request. The retry must move the knob the
    backend actually reads."""
    empty = ProviderResponse(
        raw_text="",
        provider_name="fake",
        model_name="fake-model",
        latency_ms=5.0,
        completion_tokens=0,
        tokens_used=4096,
        finish_reason="length",
    )
    answered = ProviderResponse(
        raw_text="actual answer",
        provider_name="fake",
        model_name="fake-model",
        latency_ms=5.0,
        completion_tokens=20,
        tokens_used=120,
    )
    provider = ReasoningProvider(responses=[empty, answered])
    profile = _profile()
    profile = profile.model_copy(
        update={"options": profile.options.model_copy(update={"reasoning_effort": "high"})}
    )
    use_case = _use_case(provider, _config(), {"L2": profile})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert provider.call_count == 2
    assert result.content == "actual answer"


@pytest.mark.asyncio
async def test_no_retry_when_reasoning_effort_already_at_floor():
    """"low" is as far down as Gemini's shim allows - "none" is a hard 400."""
    provider = ReasoningProvider(
        responses=[
            ProviderResponse(
                raw_text="",
                provider_name="fake",
                model_name="fake-model",
                latency_ms=5.0,
                completion_tokens=0,
                tokens_used=4096,
                finish_reason="length",
            ),
        ]
    )
    profile = _profile()
    profile = profile.model_copy(
        update={"options": profile.options.model_copy(update={"reasoning_effort": "low"})}
    )
    use_case = _use_case(provider, _config(), {"L2": profile})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert provider.call_count == 1
    assert "empty_response_possible_thinking_exhaustion" in result.warnings


class NamedFakeProvider(FakeProvider):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name


def _ok(name: str) -> ProviderResponse:
    return ProviderResponse(raw_text=f"from {name}", provider_name=name, model_name="m", latency_ms=1.0)


def _chain_use_case(
    providers: dict[str, FakeProvider],
    fallbacks: dict[str, list[str]],
    default_provider: str = "gemini",
    profile: BudgetProfile | None = None,
) -> DelegateTaskUseCase:
    config = _config()
    config.fallbacks = fallbacks
    return DelegateTaskUseCase(
        providers=providers,
        default_provider=default_provider,
        config=config,
        budget_profiles={"L2": profile or _profile()},
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
    )


@pytest.mark.asyncio
async def test_overloaded_provider_falls_back_to_next_tier():
    gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("status 503:\n  overloaded")])
    lite = NamedFakeProvider("gemini-lite", responses=[_ok("gemini-lite")])
    groq = NamedFakeProvider("groq")
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite, "groq": groq},
        {"gemini": ["gemini-lite", "groq"]},
    )

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert result.raw_text == "from gemini-lite"
    assert groq.call_count == 0
    assert any(
        w.startswith("fallback_used:gemini-lite after gemini: status 503: overloaded")
        for w in result.warnings
    )


@pytest.mark.asyncio
async def test_chain_reaches_last_tier_and_skips_unconfigured_fallback():
    gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("503")])
    groq = NamedFakeProvider("groq", responses=[_ok("groq")])
    # gemini-lite is named in the chain but absent (e.g. skipped for missing keys)
    use_case = _chain_use_case({"gemini": gemini, "groq": groq}, {"gemini": ["gemini-lite", "groq"]})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert result.provider == "groq"


@pytest.mark.asyncio
async def test_timeout_on_primary_falls_back():
    gemini = NamedFakeProvider("gemini", sleep_seconds=0.3)
    groq = NamedFakeProvider("groq", responses=[_ok("groq")])
    use_case = _chain_use_case(
        {"gemini": gemini, "groq": groq},
        {"gemini": ["groq"]},
        profile=_profile(timeout_seconds=0.05),
    )

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert result.provider == "groq"


@pytest.mark.asyncio
async def test_all_tiers_failing_reports_every_error():
    gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("503 overloaded")])
    groq = NamedFakeProvider("groq", responses=[ProviderUnavailableError("429 quota")])
    use_case = _chain_use_case({"gemini": gemini, "groq": groq}, {"gemini": ["groq"]})

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is False
    assert result.error == "All providers failed: gemini: 503 overloaded; groq: 429 quota"


@pytest.mark.asyncio
async def test_provider_without_chain_never_falls_back():
    ollama = NamedFakeProvider("ollama", responses=[ProviderUnavailableError("ollama down")])
    groq = NamedFakeProvider("groq")
    use_case = _chain_use_case(
        {"ollama": ollama, "groq": groq},
        {"gemini": ["groq"]},
        default_provider="ollama",
    )

    result = await use_case.execute(DelegateTaskRequest(task="secret", budget="L2"))

    assert result.success is False
    assert result.error == "ollama down"
    assert groq.call_count == 0


class RecordingProvider(NamedFakeProvider):
    """Remembers the options of every call, to assert what each tier was sent."""

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.seen_options: list[GenerationOptions] = []

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        self.seen_options.append(options)
        return await super().generate(prompt, options)


def _l3_profile(timeout_seconds: float, effort: str | None = "medium") -> BudgetProfile:
    return BudgetProfile(
        name="L2",
        description="test budget",
        options=GenerationOptions(num_ctx=8192, reasoning_effort=effort),
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_tier_timeout_cuts_a_hanging_primary_short():
    """2026-10-06: flash-latest accepted the call and never answered; the fallback
    must start after `tier_timeout_seconds`, not after the whole budget."""
    gemini = NamedFakeProvider("gemini", sleep_seconds=5.0)
    lite = NamedFakeProvider("gemini-lite", responses=[_ok("gemini-lite")])
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=10.0),
    )
    use_case._config.tier_timeout_seconds = 0.05

    started = asyncio.get_running_loop().time()
    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))
    elapsed = asyncio.get_running_loop().time() - started

    assert result.success is True and result.provider == "gemini-lite"
    assert elapsed < 2.0  # not the 10s budget, nor gemini's 5s hang
    assert any("did not respond within 0.05s" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_last_tier_keeps_the_full_budget_timeout():
    gemini = NamedFakeProvider("gemini", responses=[ProviderUnavailableError("503")])
    lite = NamedFakeProvider("gemini-lite", sleep_seconds=0.2, responses=[_ok("gemini-lite")])
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=5.0),
    )
    use_case._config.tier_timeout_seconds = 0.05  # would kill lite if it applied to the last tier

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True and result.provider == "gemini-lite"


@pytest.mark.asyncio
async def test_unavailable_later_tier_does_not_count_as_a_fallback():
    """A chain entry skipped at startup is not a rescue, so the primary keeps the full budget."""
    gemini = NamedFakeProvider("gemini", sleep_seconds=0.2, responses=[_ok("gemini")])
    use_case = _chain_use_case(
        {"gemini": gemini},
        {"gemini": ["gemini-lite"]},  # gemini-lite not configured
        profile=_l3_profile(timeout_seconds=5.0),
    )
    use_case._config.tier_timeout_seconds = 0.05

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True and result.provider == "gemini"


@pytest.mark.asyncio
async def test_fallback_tier_gets_lowered_reasoning_effort():
    gemini = RecordingProvider("gemini", responses=[ProviderUnavailableError("503")])
    lite = RecordingProvider("gemini-lite", responses=[_ok("gemini-lite")])
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=5.0, effort="medium"),
    )
    use_case._config.fallback_reasoning_effort = "low"

    result = await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))

    assert result.success is True
    assert gemini.seen_options[0].reasoning_effort == "medium"  # primary untouched
    assert lite.seen_options[0].reasoning_effort == "low"


@pytest.mark.asyncio
async def test_fallback_never_raises_reasoning_effort_and_defaults_off():
    gemini = RecordingProvider("gemini", responses=[ProviderUnavailableError("503")])
    lite = RecordingProvider("gemini-lite", responses=[_ok("gemini-lite")])
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=5.0, effort="medium"),
    )
    # feature off (None) -> budget value kept. Set explicitly: AppConfig() in tests also
    # reads config/default.yaml, which turns the feature on.
    use_case._config.fallback_reasoning_effort = None
    await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))
    assert lite.seen_options[0].reasoning_effort == "medium"

    gemini2 = RecordingProvider("gemini", responses=[ProviderUnavailableError("503")])
    lite2 = RecordingProvider("gemini-lite", responses=[_ok("gemini-lite")])
    use_case2 = _chain_use_case(
        {"gemini": gemini2, "gemini-lite": lite2},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=5.0, effort="low"),
    )
    use_case2._config.fallback_reasoning_effort = "low"
    await use_case2.execute(DelegateTaskRequest(task="task", budget="L2"))
    assert lite2.seen_options[0].reasoning_effort == "low"  # already low, unchanged


class ExhaustedThenSlowProvider(RecordingProvider):
    """First call: thinking used up the whole budget (empty answer); retry: hangs."""

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        self.seen_options.append(options)
        if len(self.seen_options) == 1:
            return ProviderResponse(
                raw_text="", provider_name=self.provider_name, model_name="m",
                latency_ms=1.0, finish_reason="length",
            )
        await asyncio.sleep(5.0)
        return _ok(self.provider_name)


@pytest.mark.asyncio
async def test_thinking_retry_on_a_non_last_tier_keeps_the_tier_cap():
    """Review 2026-10-06: the thinking-exhaustion retry used the full budget timeout."""
    gemini = ExhaustedThenSlowProvider("gemini")
    lite = NamedFakeProvider("gemini-lite")
    use_case = _chain_use_case(
        {"gemini": gemini, "gemini-lite": lite},
        {"gemini": ["gemini-lite"]},
        profile=_l3_profile(timeout_seconds=10.0, effort="medium"),
    )
    use_case._config.tier_timeout_seconds = 0.05

    started = asyncio.get_running_loop().time()
    await use_case.execute(DelegateTaskRequest(task="task", budget="L2"))
    elapsed = asyncio.get_running_loop().time() - started

    assert len(gemini.seen_options) == 2  # the retry did happen
    assert elapsed < 2.0  # and was cut at the tier cap, not the 10s budget
