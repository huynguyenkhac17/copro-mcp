from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llm_worker_pool.application.conversation_store import ConversationStore, Turn, render_history
from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest
from llm_worker_pool.domain.exceptions import ConfigurationError
from llm_worker_pool.domain.file_context import FileContextResolver
from llm_worker_pool.domain.guardrails import SecretScanner
from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions, ProviderResponse, TaskProfile
from llm_worker_pool.infra.logging import InvocationLogger
from tests.unit.test_delegate_task_use_case import FakeProvider, _config


def _answer(text: str, truncated: bool = False) -> ProviderResponse:
    return ProviderResponse(
        raw_text=text, provider_name="fake", model_name="fake-model", latency_ms=1.0, truncated=truncated
    )


def _use_case(provider, *, store=None, tmp_path: Path | None = None, num_ctx: int = 8192, **kwargs):
    profiles = {
        name: BudgetProfile(name=name, description="", options=GenerationOptions(num_ctx=num_ctx))
        for name in ("L1", "L2", "L3")
    }
    return DelegateTaskUseCase(
        providers={"fake": provider},
        default_provider="fake",
        config=_config(),
        budget_profiles=profiles,
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
        file_context_resolver=FileContextResolver(allowed_roots=[tmp_path] if tmp_path else []),
        conversation_store=store if store is not None else ConversationStore(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_follow_up_replays_context_and_previous_turn_without_rereading(tmp_path: Path):
    source = tmp_path / "orders.py"
    source.write_text("FILE_BODY_ORDERS")
    provider = FakeProvider([_answer("first part", truncated=True), _answer("second part")])
    use_case = _use_case(provider, tmp_path=tmp_path)

    first = await use_case.execute(
        DelegateTaskRequest(task="explain orders", budget="L3", file_paths=[str(source)])
    )
    assert first.success and first.continuation_id
    source.write_text("CHANGED_ON_DISK")  # the follow-up must reuse what turn 1 saw

    second = await use_case.execute(
        DelegateTaskRequest(task="continue", continuation_id=first.continuation_id)
    )

    assert second.success and second.raw_text == "second part"
    prompt = provider.last_prompt
    assert "FILE_BODY_ORDERS" in prompt and "CHANGED_ON_DISK" not in prompt
    assert "explain orders" in prompt and "first part" in prompt
    assert prompt.index("first part") < prompt.index("without repeating it):\ncontinue")
    assert second.budget_used == "L3"  # inherited, not the DTO default L2
    assert second.continuation_id == first.continuation_id


@pytest.mark.asyncio
async def test_explicit_settings_beat_inherited_ones():
    provider = FakeProvider([_answer("a"), _answer("b")])
    use_case = _use_case(provider)
    first = await use_case.execute(DelegateTaskRequest(task="t1", budget="L3"))
    second = await use_case.execute(
        DelegateTaskRequest(task="t2", budget="L1", continuation_id=first.continuation_id)
    )
    assert second.budget_used == "L1"


@pytest.mark.asyncio
async def test_inherited_task_type_keeps_its_profile_but_not_its_default_budget():
    profile = TaskProfile(name="summarize", default_budget="L1", system_addon="ADDON")
    provider = FakeProvider([_answer("a"), _answer("b")])
    use_case = _use_case(provider, task_profiles={"summarize": profile})
    first = await use_case.execute(DelegateTaskRequest(task="t1", task_type="summarize", budget="L3"))
    second = await use_case.execute(DelegateTaskRequest(task="t2", continuation_id=first.continuation_id))
    assert second.task_type == "summarize"
    assert second.budget_used == "L3"


@pytest.mark.asyncio
async def test_unknown_continuation_is_a_clear_error():
    use_case = _use_case(FakeProvider())
    with pytest.raises(ConfigurationError, match="continuation_not_found"):
        await use_case.execute(DelegateTaskRequest(task="continue", continuation_id="nope"))


@pytest.mark.asyncio
async def test_failed_follow_up_leaves_the_conversation_unchanged():
    from llm_worker_pool.domain.exceptions import ProviderUnavailableError

    store = ConversationStore()
    provider = FakeProvider([_answer("a"), ProviderUnavailableError("down"), _answer("c")])
    use_case = _use_case(provider, store=store)
    first = await use_case.execute(DelegateTaskRequest(task="t1"))

    failed = await use_case.execute(DelegateTaskRequest(task="t2", continuation_id=first.continuation_id))
    assert not failed.success and failed.continuation_id == first.continuation_id
    assert len(store.get(first.continuation_id).turns) == 1

    retried = await use_case.execute(DelegateTaskRequest(task="t2", continuation_id=first.continuation_id))
    assert retried.success and len(store.get(first.continuation_id).turns) == 2


@pytest.mark.asyncio
async def test_history_is_trimmed_oldest_first_to_fit_the_window():
    # num_ctx 400 -> 300 tokens of room; each turn below is ~100 tokens.
    provider = FakeProvider([_answer(f"ANSWER_{i}_" + "x" * 360) for i in range(4)] + [_answer("done")])
    use_case = _use_case(provider, num_ctx=400)
    cid = None
    for i in range(4):
        result = await use_case.execute(DelegateTaskRequest(task=f"t{i}", continuation_id=cid))
        cid = result.continuation_id

    result = await use_case.execute(DelegateTaskRequest(task="last", continuation_id=cid))

    assert "ANSWER_3_" in provider.last_prompt and "ANSWER_0_" not in provider.last_prompt
    assert any(w.startswith("continuation_history_trimmed:") for w in result.warnings)


@pytest.mark.asyncio
async def test_secrets_in_history_are_redacted_before_a_cloud_call():
    secret = "AIza" + "B" * 35
    provider = FakeProvider([_answer(f"the key is {secret}"), _answer("ok")])
    use_case = _use_case(provider, secret_scanner=SecretScanner())
    first = await use_case.execute(DelegateTaskRequest(task="t1"))
    await use_case.execute(DelegateTaskRequest(task="t2", continuation_id=first.continuation_id))
    assert secret not in provider.last_prompt


@pytest.mark.asyncio
async def test_no_store_means_no_continuation_id():
    use_case = DelegateTaskUseCase(
        providers={"fake": FakeProvider()},
        default_provider="fake",
        config=_config(),
        budget_profiles={"L2": BudgetProfile(name="L2", description="", options=GenerationOptions())},
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
    )
    result = await use_case.execute(DelegateTaskRequest(task="t"))
    assert result.success and result.continuation_id is None


def test_store_caps_turns_and_expires():
    store = ConversationStore(max_turns=2, ttl_seconds=0)
    cid = store.append(None, context=None, settings={}, task="a", answer="1")
    assert store.get(cid) is None  # ttl 0: already expired

    store = ConversationStore(max_turns=2)
    cid = store.append(None, context=None, settings={}, task="a", answer="1")
    for task in ("b", "c"):
        store.append(cid, context=None, settings={}, task=task, answer="x")
    assert [t.task for t in store.get(cid).turns] == ["b", "c"]


def test_render_history_without_room_sends_the_task_alone():
    prompt, dropped = render_history((Turn("a", "1"),), "new", max_tokens=0)
    assert prompt == "new" and dropped == 1


@pytest.mark.asyncio
async def test_resending_the_same_files_does_not_duplicate_the_context(tmp_path: Path):
    source = tmp_path / "a.py"
    source.write_text("ONLY_ONCE_BODY")
    provider = FakeProvider([_answer("1"), _answer("2"), _answer("3")])
    use_case = _use_case(provider, tmp_path=tmp_path)
    cid = None
    for _ in range(3):
        result = await use_case.execute(
            DelegateTaskRequest(task="t", file_paths=[str(source)], continuation_id=cid)
        )
        cid = result.continuation_id
    assert provider.last_prompt.count("ONLY_ONCE_BODY") == 1


@pytest.mark.asyncio
async def test_cross_check_on_a_follow_up_is_reported_as_skipped():
    provider = FakeProvider([_answer("1"), _answer("2")])
    use_case = _use_case(provider)
    first = await use_case.execute(DelegateTaskRequest(task="t1"))
    second = await use_case.execute(
        DelegateTaskRequest(task="t2", cross_check=True, continuation_id=first.continuation_id)
    )
    assert "cross_check_skipped:follow_up_runs_on_one_provider" in second.warnings
