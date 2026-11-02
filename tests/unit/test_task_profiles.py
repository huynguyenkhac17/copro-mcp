"""DelegateTaskUseCase with task_type profiles, guardrails and validators."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llm_worker_pool.application.dto import DelegateTaskRequest
from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.config.settings import ProviderInstanceConfig, load_task_profiles
from llm_worker_pool.domain.exceptions import ConfigurationError, NoUsableContextError
from llm_worker_pool.domain.file_context import FileContextResolver
from llm_worker_pool.domain.guardrails import PathDenyList, SecretScanner
from llm_worker_pool.domain.models import (
    BudgetProfile,
    GenerationOptions,
    ProviderResponse,
    TaskProfile,
)
from llm_worker_pool.infra.logging import InvocationLogger

from tests.unit.test_delegate_task_use_case import NamedFakeProvider, _config

FAKE_GROQ_KEY = "gsk_" + "z" * 40


class RecordingProvider(NamedFakeProvider):
    def __init__(self, name: str, texts: list[str] | None = None) -> None:
        super().__init__(
            name,
            responses=[
                ProviderResponse(raw_text=t, provider_name=name, model_name="m", latency_ms=1.0)
                for t in texts or []
            ],
        )
        self.prompts: list[str] = []
        self.options: list[GenerationOptions] = []

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        self.prompts.append(prompt)
        self.options.append(options)
        return await super().generate(prompt, options)


def _budget(name: str) -> BudgetProfile:
    return BudgetProfile(name=name, description="", options=GenerationOptions(num_ctx=8192, system_prompt=f"sys-{name}"))


REVIEW = TaskProfile(
    name="review",
    system_addon="REVIEW CONTRACT",
    default_budget="L3",
    output_format="json",
    json_schema={"type": "object", "required": ["findings"]},
    validators=["json_repair", "citations"],
)


def _use_case(
    providers: dict[str, RecordingProvider],
    *,
    tmp_path: Path | None = None,
    scanner: bool = True,
    fallbacks: dict[str, list[str]] | None = None,
    default: str = "groq",
    local_types: dict[str, str] | None = None,
) -> DelegateTaskUseCase:
    config = _config()
    config.fallbacks = fallbacks or {}
    for name, type_ in (local_types or {}).items():
        config.providers[name] = ProviderInstanceConfig(type=type_, endpoint="http://x", model="m")
    return DelegateTaskUseCase(
        providers=providers,
        default_provider=default,
        config=config,
        budget_profiles={"L1": _budget("L1"), "L2": _budget("L2"), "L3": _budget("L3")},
        invocation_logger=InvocationLogger(logging.getLogger("test")),
        file_context_resolver=FileContextResolver(
            allowed_roots=[tmp_path] if tmp_path else [], deny_list=PathDenyList()
        ),
        task_profiles={"review": REVIEW},
        secret_scanner=SecretScanner() if scanner else None,
        local_providers=["ollama"],
    )


@pytest.mark.asyncio
async def test_task_type_none_matches_previous_behaviour():
    with_harness = RecordingProvider("groq", ["answer"])
    without = RecordingProvider("groq", ["answer"])
    request = DelegateTaskRequest(task="t", context="plain context")

    a = await _use_case({"groq": with_harness}).execute(request)
    plain = DelegateTaskUseCase(
        providers={"groq": without}, default_provider="groq", config=_config(),
        budget_profiles={"L2": _budget("L2")},
        invocation_logger=InvocationLogger(logging.getLogger("test")),
    )
    b = await plain.execute(request)

    assert a.model_dump() == b.model_dump()
    assert with_harness.prompts == without.prompts
    assert with_harness.options == without.options
    assert a.task_type is None and a.guardrail_actions == [] and a.verification is None


@pytest.mark.asyncio
async def test_task_profile_fills_unset_fields_only():
    groq = RecordingProvider("groq", ['{"findings": []}', "plain"])
    use_case = _use_case({"groq": groq})

    result = await use_case.execute(DelegateTaskRequest(task="t", task_type="review"))
    assert result.budget_used == "L3"
    assert groq.options[0].system_prompt.startswith("sys-L3\n\nREVIEW CONTRACT")
    assert groq.options[0].format == {"type": "object", "required": ["findings"]}
    assert result.content == {"findings": []}

    explicit = await use_case.execute(
        DelegateTaskRequest(task="t", task_type="review", budget="L1", output_format="text")
    )
    assert explicit.budget_used == "L1"
    assert groq.options[1].format is None
    assert explicit.content == "plain"


@pytest.mark.asyncio
async def test_unknown_task_type_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="Task type 'nope'"):
        await _use_case({"groq": RecordingProvider("groq")}).execute(
            DelegateTaskRequest(task="t", task_type="nope")
        )


@pytest.mark.asyncio
async def test_json_repair_succeeds_on_second_call():
    groq = RecordingProvider("groq", ["Sure! findings: none", '{"findings": []}'])
    result = await _use_case({"groq": groq}).execute(DelegateTaskRequest(task="t", task_type="review"))

    assert groq.call_count == 2
    assert "Sure! findings: none" in groq.prompts[1]
    assert result.content == {"findings": []}
    assert "json_repaired" in result.warnings
    assert not any(w.startswith("json_parse_failed") for w in result.warnings)


@pytest.mark.asyncio
async def test_json_repair_still_broken_keeps_original_warning():
    groq = RecordingProvider("groq", ["not json", "still not json"])
    result = await _use_case({"groq": groq}).execute(DelegateTaskRequest(task="t", task_type="review"))

    assert groq.call_count == 2
    assert "json_parse_failed" in result.warnings
    assert "json_repaired" not in result.warnings
    assert result.raw_text == "not json"


@pytest.mark.asyncio
async def test_json_without_repair_validator_is_not_retried():
    groq = RecordingProvider("groq", ["not json"])
    result = await _use_case({"groq": groq}).execute(
        DelegateTaskRequest(task="t", output_format="json")
    )
    assert groq.call_count == 1 and "json_parse_failed" in result.warnings


@pytest.mark.asyncio
async def test_citations_are_verified_against_files(tmp_path: Path):
    src = tmp_path / "orders.py"
    src.write_text("a = 1\nb = []\nreturn b\n")
    answer = (
        '{"findings": ['
        '{"file": "orders.py", "line": 2, "evidence_quote": "b = []"},'
        '{"file": "orders.py", "line": 99, "evidence_quote": "x"},'
        '{"file": "ghost.py", "line": 1}]}'
    )
    groq = RecordingProvider("groq", [answer])
    result = await _use_case({"groq": groq}, tmp_path=tmp_path).execute(
        DelegateTaskRequest(task="review", task_type="review", file_paths=[str(src)])
    )

    statuses = {(c["file"], c["line"]): c["status"] for c in result.verification["citations"]}
    assert statuses == {
        ("orders.py", 2): "ok",
        ("orders.py", 99): "line_out_of_range",
        ("ghost.py", 1): "file_not_in_context",
    }
    assert "citation_unverified:ghost.py:1 (file_not_in_context)" in result.warnings


@pytest.mark.asyncio
async def test_sensitive_file_is_denied_and_reported(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_GROQ_KEY}\n")
    notes = tmp_path / "notes.txt"
    notes.write_text("plain notes\n")
    groq = RecordingProvider("groq", ["ok"])
    result = await _use_case({"groq": groq}, tmp_path=tmp_path).execute(
        DelegateTaskRequest(task="read it", file_paths=[str(env), str(notes)])
    )
    assert FAKE_GROQ_KEY not in groq.prompts[0]
    assert "plain notes" in groq.prompts[0]
    assert result.guardrail_actions == [f"file_denied_sensitive:{env}"]


@pytest.mark.asyncio
async def test_only_a_sensitive_file_refuses_before_any_call(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_GROQ_KEY}\n")
    groq = RecordingProvider("groq", ["ok"])
    with pytest.raises(NoUsableContextError, match="file_denied_sensitive"):
        await _use_case({"groq": groq}, tmp_path=tmp_path).execute(
            DelegateTaskRequest(task="read it", file_paths=[str(env)])
        )
    assert groq.prompts == []


@pytest.mark.asyncio
async def test_secrets_redacted_before_cloud_call():
    groq = RecordingProvider("groq", ["done"])
    result = await _use_case({"groq": groq}).execute(
        DelegateTaskRequest(task=f"use {FAKE_GROQ_KEY}", context=f"key={FAKE_GROQ_KEY}")
    )
    assert FAKE_GROQ_KEY not in groq.prompts[0]
    assert groq.prompts[0].count("[REDACTED:groq_api_key]") == 2
    assert result.guardrail_actions == ["secret_redacted:groq_api_key x2"]
    assert "secret_redacted:groq_api_key x2" in result.warnings


@pytest.mark.asyncio
async def test_no_redaction_when_whole_chain_is_ollama():
    ollama = RecordingProvider("ollama", ["done"])
    result = await _use_case({"ollama": ollama}, default="ollama").execute(
        DelegateTaskRequest(task="t", context=f"key={FAKE_GROQ_KEY}")
    )
    assert FAKE_GROQ_KEY in ollama.prompts[0]
    assert result.guardrail_actions == []


@pytest.mark.asyncio
async def test_local_provider_detected_by_type():
    box = RecordingProvider("gpu-box", ["done"])
    await _use_case({"gpu-box": box}, default="gpu-box", local_types={"gpu-box": "ollama"}).execute(
        DelegateTaskRequest(task="t", context=f"key={FAKE_GROQ_KEY}")
    )
    assert FAKE_GROQ_KEY in box.prompts[0]


@pytest.mark.asyncio
async def test_redacts_when_any_fallback_tier_is_cloud():
    ollama = RecordingProvider("ollama", ["done"])
    groq = RecordingProvider("groq")
    await _use_case(
        {"ollama": ollama, "groq": groq}, default="ollama", fallbacks={"ollama": ["groq"]}
    ).execute(DelegateTaskRequest(task="t", context=f"key={FAKE_GROQ_KEY}"))
    assert FAKE_GROQ_KEY not in ollama.prompts[0]


@pytest.mark.asyncio
async def test_secret_echoed_in_response_is_flagged():
    groq = RecordingProvider("groq", [f"the key is {FAKE_GROQ_KEY}"])
    result = await _use_case({"groq": groq}).execute(DelegateTaskRequest(task="t"))
    assert "secret_in_response:groq_api_key x1" in result.guardrail_actions
    assert "secret_in_response:groq_api_key x1" in result.warnings


@pytest.mark.asyncio
async def test_scanner_disabled_leaves_text_alone():
    groq = RecordingProvider("groq", [f"echo {FAKE_GROQ_KEY}"])
    result = await _use_case({"groq": groq}, scanner=False).execute(
        DelegateTaskRequest(task="t", context=FAKE_GROQ_KEY)
    )
    assert FAKE_GROQ_KEY in groq.prompts[0]
    assert result.guardrail_actions == []


def test_shipped_task_profiles_load_and_reference_real_budgets():
    profiles = load_task_profiles()
    assert {"review", "summarize", "extract", "classify", "explain", "translate", "generate"} <= set(profiles)
    for p in profiles.values():
        assert p.default_budget in (None, "L1", "L2", "L3")
        assert set(p.validators) <= {"json_repair", "citations"}
    assert profiles["review"].json_schema["properties"]["findings"]["items"]["required"] == [
        "file", "line", "severity", "claim", "evidence_quote",
    ]


def test_missing_task_profiles_file_is_empty(tmp_path: Path):
    assert load_task_profiles(tmp_path / "nope.yaml") == {}
