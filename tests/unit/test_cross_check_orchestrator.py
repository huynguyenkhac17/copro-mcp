from __future__ import annotations

import json
import logging

import pytest

from llm_worker_pool.application.cross_check_orchestrator import CrossCheckOrchestrator
from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest
from llm_worker_pool.config.settings import AppConfig, RetryConfig
from llm_worker_pool.domain.exceptions import ProviderUnavailableError, RequestTooLargeError
from llm_worker_pool.domain.file_context import FileContextResolver
from llm_worker_pool.domain.models import (
    BudgetProfile,
    GenerationOptions,
    ProviderResponse,
    TaskProfile,
)
from llm_worker_pool.infra.logging import InvocationLogger

_SCHEMA = {"type": "object", "required": ["findings"], "properties": {"findings": {"type": "array"}}}


class ScriptedProvider:
    """Answers with findings for whichever files appear in the prompt."""

    def __init__(self, name: str, findings_by_file: dict[str, list[dict]] | None = None, error: Exception | None = None,
                 max_prompt_tokens: int | None = None):
        self._name = name
        self._findings = findings_by_file or {}
        self._error = error
        self._max_prompt_tokens = max_prompt_tokens  # what OpenAICompatibleProvider refuses
        self.prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return f"{self._name}-model"

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        if self._max_prompt_tokens is not None and len(prompt) // 4 > self._max_prompt_tokens:
            raise RequestTooLargeError(f"{self._name}: prompt exceeds max_prompt_tokens; skipped")
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        found = [f for path, fs in self._findings.items() if f"--- file: {path} ---" in prompt for f in fs]
        return ProviderResponse(
            raw_text=json.dumps({"findings": found}),
            provider_name=self._name,
            model_name=self.model_name,
            latency_ms=5.0,
            prompt_tokens=10,
            completion_tokens=5,
            tokens_used=15,
        )

    async def aclose(self) -> None:
        pass


def _setup(tmp_path, providers: dict[str, ScriptedProvider], *, groq_limit: int | None = None,
           arms=("groq", "lite"), files: dict[str, str] | None = None, deny_list=None):
    config = AppConfig(retry=RetryConfig(max_attempts=1, backoff_base_seconds=0.01), timeout_seconds=5.0)
    config.budget_providers = {}
    config.budget_fallbacks = {}
    config.fallbacks = {"groq": ["lite"], "lite": ["groq"]}  # must be ignored by arms
    if "groq" in config.providers:
        config.providers["groq"] = config.providers["groq"].model_copy(update={"max_prompt_tokens": groq_limit})
    review = TaskProfile(
        name="review", output_format="json", json_schema=_SCHEMA, default_budget="L3",
        validators=["json_repair", "citations"], cross_check=True, cross_check_providers=list(arms),
    )
    use_case = DelegateTaskUseCase(
        providers=providers,
        default_provider="lite",
        config=config,
        budget_profiles={
            "L2": BudgetProfile(name="L2", description="", options=GenerationOptions(num_ctx=65536)),
            "L3": BudgetProfile(name="L3", description="", options=GenerationOptions(num_ctx=65536)),
        },
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
        file_context_resolver=FileContextResolver(allowed_roots=[tmp_path], deny_list=deny_list),
        task_profiles={"review": review},
    )
    paths = []
    for name, body in (files or {"orders.py": "a\nb\ntotal = price * qty\nd\ne\nf\n"}).items():
        path = tmp_path / name
        path.write_text(body)
        paths.append(str(path))
    return CrossCheckOrchestrator(use_case), paths


def _f(file: str, line: int, quote: str = "", severity: str = "medium") -> dict:
    return {"file": file, "line": line, "severity": severity, "claim": "bug", "evidence_quote": quote}


@pytest.mark.asyncio
async def test_findings_from_both_providers_are_merged(tmp_path):
    full = str(tmp_path / "orders.py")
    groq = ScriptedProvider("groq", {full: [_f(full, 3, "total = price * qty"), _f(full, 6)]})
    lite = ScriptedProvider("lite", {full: [_f("orders.py", 4, "total = price * qty", "high")]})
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite})

    request = DelegateTaskRequest(task="review", task_type="review", file_paths=paths)
    assert orch.wants_cross_check(request)
    result = await orch.execute(request)

    assert result.success, result.error
    findings = result.content["findings"]
    assert findings[0]["agreed_by"] == ["groq", "lite"]
    assert findings[0]["severity"] == "high"
    assert findings[0]["citation"] == "ok"
    assert result.cross_check["findings_total"] == 2
    assert result.cross_check["findings_agreed"] == 1
    assert result.provider == "groq+lite"
    assert result.tokens_used == 30
    assert json.loads(result.raw_text) == result.content


@pytest.mark.asyncio
async def test_failed_arm_does_not_fall_back_and_result_is_partial(tmp_path):
    full = str(tmp_path / "orders.py")
    groq = ScriptedProvider("groq", error=ProviderUnavailableError("503 overloaded"))
    lite = ScriptedProvider("lite", {full: [_f(full, 3, "total = price * qty")]})
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite})

    result = await orch.execute(DelegateTaskRequest(task="review", task_type="review", file_paths=paths))

    assert result.success
    assert result.provider == "lite"
    assert len(lite.prompts) == 1  # the groq arm did not fall back onto lite
    assert any(w.startswith("cross_check_partial:groq failed") for w in result.warnings)
    assert result.cross_check["arms"]["groq"]["success"] is False


@pytest.mark.asyncio
async def test_every_arm_failing_is_a_failure(tmp_path):
    groq = ScriptedProvider("groq", error=ProviderUnavailableError("503"))
    lite = ScriptedProvider("lite", error=ProviderUnavailableError("429"))
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite})

    result = await orch.execute(DelegateTaskRequest(task="review", task_type="review", file_paths=paths))

    assert result.success is False
    assert "groq" in result.error and "lite" in result.error


@pytest.mark.asyncio
async def test_arm_with_prompt_cap_gets_files_in_batches(tmp_path):
    files = {f"f{i}.py": ("x = 1\n" * 300) for i in range(3)}  # ~450 tokens each
    groq = ScriptedProvider("groq")
    lite = ScriptedProvider("lite")
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite}, groq_limit=1700, files=files)

    result = await orch.execute(DelegateTaskRequest(task="review", task_type="review", file_paths=paths))

    assert result.success, result.error
    assert len(lite.prompts) == 1  # no cap: everything at once
    assert len(groq.prompts) >= 2  # capped: split by whole files
    for prompt in groq.prompts:
        assert prompt.count("--- file: ") <= 1
    assert result.cross_check["arms"]["groq"]["batches"] == len(groq.prompts)


@pytest.mark.asyncio
async def test_cross_check_false_or_text_output_opts_out(tmp_path):
    lite = ScriptedProvider("lite")
    orch, paths = _setup(tmp_path, {"groq": ScriptedProvider("groq"), "lite": lite})

    assert not orch.wants_cross_check(
        DelegateTaskRequest(task="r", task_type="review", file_paths=paths, cross_check=False)
    )
    result = await orch.execute(
        DelegateTaskRequest(task="r", task_type="review", file_paths=paths, output_format="text")
    )
    assert "cross_check_skipped:needs_json_findings" in result.warnings


@pytest.mark.asyncio
async def test_single_available_provider_degrades_to_one_call(tmp_path):
    lite = ScriptedProvider("lite")
    orch, paths = _setup(tmp_path, {"lite": lite})

    result = await orch.execute(DelegateTaskRequest(task="r", task_type="review", file_paths=paths))

    assert result.success
    assert "cross_check_skipped:fewer_than_two_providers" in result.warnings
    assert len(lite.prompts) == 1


@pytest.mark.asyncio
async def test_other_task_types_do_not_cross_check(tmp_path):
    orch, _ = _setup(tmp_path, {"lite": ScriptedProvider("lite")})
    assert not orch.wants_cross_check(DelegateTaskRequest(task="r"))


@pytest.mark.asyncio
async def test_merged_review_is_one_conversation_arms_are_not(tmp_path):
    from llm_worker_pool.application.conversation_store import ConversationStore

    full = str(tmp_path / "orders.py")
    groq = ScriptedProvider("groq", {full: [_f(full, 3, "total = price * qty")]})
    lite = ScriptedProvider("lite", {full: [_f(full, 3, "total = price * qty")]})
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite})
    store = ConversationStore()
    orch._use_case._conversations = store

    result = await orch.execute(DelegateTaskRequest(task="review", task_type="review", file_paths=paths))

    assert result.continuation_id
    conversation = store.get(result.continuation_id)
    assert conversation.turns[0].answer == result.raw_text
    assert "total = price * qty" in conversation.context
    assert len(store._cache._store) == 1


# --- one provider, batched (execute_single) -----------------------------------------


@pytest.mark.asyncio
async def test_single_provider_review_over_its_cap_goes_in_batches(tmp_path):
    # 2026-11-01, review-large-l3: groq skipped the whole 15k-token prompt and the
    # gemini-lite fallback found 0.33-0.50 of the bugs.
    files = {f"f{i}.py": ("x = 1\n" * 300) for i in range(3)}  # ~450 tokens each
    full = {name: str(tmp_path / name) for name in files}
    groq = ScriptedProvider("groq", {full["f0.py"]: [_f(full["f0.py"], 2, "x = 1")],
                                     full["f2.py"]: [_f(full["f2.py"], 5, "x = 1")]})
    lite = ScriptedProvider("lite")
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite}, groq_limit=1700, files=files)

    request = DelegateTaskRequest(task="review", task_type="review", file_paths=paths, provider="groq", cross_check=False)
    assert not orch.wants_cross_check(request) and orch.wants_batching(request)
    result = await orch.execute_single(request)

    assert result.success, result.error
    assert len(groq.prompts) >= 2 and not lite.prompts
    assert all(p.count("--- file: ") <= 1 for p in groq.prompts)
    assert f"review_batched:{len(groq.prompts)}" in result.warnings
    assert [(f["file"], f["line"]) for f in result.content["findings"]] == [(full["f0.py"], 2), (full["f2.py"], 5)]
    assert all(f["citation"] == "ok" for f in result.content["findings"])
    assert (result.provider, result.cross_check) == ("groq", None)
    assert result.tokens_used == 15 * len(groq.prompts)


@pytest.mark.asyncio
async def test_a_batch_the_provider_refuses_falls_back_alone(tmp_path):
    files = {"small.py": "x = 1\n" * 50, "big.py": "y = 2\n" * 1500}  # ~75 / ~2250 tokens
    big = str(tmp_path / "big.py")
    groq = ScriptedProvider("groq", max_prompt_tokens=1700)
    lite = ScriptedProvider("lite", {big: [_f(big, 7, "y = 2")]})
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite}, groq_limit=1700, files=files)

    result = await orch.execute_single(
        DelegateTaskRequest(task="review", task_type="review", file_paths=paths, provider="groq")
    )

    assert result.success, result.error
    assert len(groq.prompts) == 1  # big.py alone is over the cap: skipped before any call
    assert len(lite.prompts) == 1 and "big.py" in lite.prompts[0] and "small.py" not in lite.prompts[0]
    assert result.content["findings"][0]["agreed_by"] == ["lite"]
    assert result.provider == "groq+lite"
    # Dropped before the chain ran, so lite answered as a first try, not a fallback.
    assert any(w.startswith("prompt_cap_skip:groq") for w in result.warnings)
    assert not any(w.startswith("fallback_used") for w in result.warnings)


@pytest.mark.asyncio
async def test_every_batch_failing_is_a_failure(tmp_path):
    files = {f"f{i}.py": ("x = 1\n" * 300) for i in range(3)}
    groq = ScriptedProvider("groq", error=ProviderUnavailableError("503"))
    lite = ScriptedProvider("lite", error=ProviderUnavailableError("429"))
    orch, paths = _setup(tmp_path, {"groq": groq, "lite": lite}, groq_limit=1700, files=files)

    result = await orch.execute_single(
        DelegateTaskRequest(task="review", task_type="review", file_paths=paths, provider="groq")
    )

    assert result.success is False
    assert result.error.startswith("Every one of")
    assert "review_batch_failed:1/" in result.error


@pytest.mark.asyncio
async def test_review_that_fits_is_one_call_and_keeps_file_warnings(tmp_path):
    groq = ScriptedProvider("groq")
    from llm_worker_pool.domain.guardrails import PathDenyList

    orch, paths = _setup(tmp_path, {"groq": groq, "lite": ScriptedProvider("lite")}, groq_limit=1700,
                         deny_list=PathDenyList())
    secret = tmp_path / ".env"
    secret.write_text("KEY=1\n")

    result = await orch.execute_single(DelegateTaskRequest(
        task="review", task_type="review", file_paths=[*paths, str(secret)], provider="groq",
    ))

    assert result.success, result.error
    assert len(groq.prompts) == 1
    assert not any(w.startswith("review_batched") for w in result.warnings)
    denied = [w for w in result.warnings if w.startswith("file_denied_sensitive:")]
    assert denied and set(denied) <= set(result.guardrail_actions)


@pytest.mark.asyncio
async def test_only_one_arm_up_still_batches_for_its_cap(tmp_path):
    files = {f"f{i}.py": ("x = 1\n" * 300) for i in range(3)}
    groq = ScriptedProvider("groq")
    orch, paths = _setup(tmp_path, {"groq": groq}, groq_limit=1700, files=files)
    orch._use_case._default_provider = "groq"

    result = await orch.execute(DelegateTaskRequest(task="r", task_type="review", file_paths=paths))

    assert result.success, result.error
    assert "cross_check_skipped:fewer_than_two_providers" in result.warnings
    assert len(groq.prompts) >= 2
    assert any(w.startswith("review_batched:") for w in result.warnings)


@pytest.mark.asyncio
async def test_batching_needs_the_findings_contract(tmp_path):
    groq = ScriptedProvider("groq")
    orch, paths = _setup(tmp_path, {"groq": groq}, groq_limit=1700)
    assert not orch.wants_batching(DelegateTaskRequest(task="r", file_paths=paths))
    assert not orch.wants_batching(
        DelegateTaskRequest(task="r", task_type="review", file_paths=paths, output_format="text")
    )


@pytest.mark.asyncio
async def test_batched_review_is_one_conversation_turn(tmp_path):
    from llm_worker_pool.application.conversation_store import ConversationStore

    files = {f"f{i}.py": ("x = 1\n" * 300) for i in range(3)}
    orch, paths = _setup(tmp_path, {"groq": ScriptedProvider("groq")}, groq_limit=1700, files=files)
    store = ConversationStore()
    orch._use_case._conversations = store

    result = await orch.execute_single(
        DelegateTaskRequest(task="review", task_type="review", file_paths=paths, provider="groq")
    )

    assert result.continuation_id
    assert len(store._cache._store) == 1
    assert all(f"f{i}.py" in store.get(result.continuation_id).context for i in range(3))
