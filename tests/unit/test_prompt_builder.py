from copro.domain.models import BudgetProfile, GenerationOptions
from copro.domain.prompt_builder import PromptBuilder


def _profile(num_ctx: int = 100, system_prompt: str | None = "Ban la trợ lý.") -> BudgetProfile:
    return BudgetProfile(
        name="L_test",
        description="test",
        options=GenerationOptions(system_prompt=system_prompt, num_ctx=num_ctx),
    )


def test_build_without_context_uses_task_as_prompt():
    builder = PromptBuilder()
    built = builder.build(task="lam gi do", context=None, profile=_profile())

    assert built.user_prompt == "lam gi do"
    assert built.system_prompt.startswith("Ban la trợ lý.")
    assert "Never invent facts" in built.system_prompt
    assert built.warnings == []


def test_build_appends_context_when_present():
    builder = PromptBuilder()
    built = builder.build(task="tom tat", context="noi dung file", profile=_profile())

    assert "tom tat" in built.user_prompt
    assert "noi dung file" in built.user_prompt
    assert built.warnings == []


def test_context_truncated_when_exceeds_budget_and_warns():
    builder = PromptBuilder()
    # num_ctx=100 -> max_context_chars ~ 100*4*0.75 = 300
    long_context = "x" * 5000
    built = builder.build(task="task", context=long_context, profile=_profile(num_ctx=100))

    assert "context_truncated" in built.warnings
    assert len(built.user_prompt) < len(long_context)


def test_json_output_format_appends_instruction():
    builder = PromptBuilder()
    built = builder.build(
        task="task", context=None, profile=_profile(), output_format="json"
    )

    assert built.system_prompt is not None
    assert "JSON" in built.system_prompt


def test_json_schema_is_embedded_in_system_prompt():
    builder = PromptBuilder()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    built = builder.build(
        task="task",
        context=None,
        profile=_profile(),
        output_format="json",
        json_schema=schema,
    )

    assert "summary" in built.system_prompt
