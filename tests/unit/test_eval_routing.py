from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_worker_pool.application.dto import DelegateTaskResult
from llm_worker_pool.config.settings import load_task_profiles
from llm_worker_pool.evals.dataset import EvalCase, ExpectedFinding
from llm_worker_pool.evals.routing import gate, render_routing_yaml, routing_table
from llm_worker_pool.evals.runner import main
from llm_worker_pool.evals.scoring import score_case


def _report(summary: dict, stamp: str = "T1") -> dict:
    return {
        "generated_at": stamp,
        "meta": {"harness": "on"},
        "summary": {
            arm: {"overall": {}, "by_task_type": by_type} for arm, by_type in summary.items()
        },
    }


def _stats(success: float, latency: float = 1000, cases: int = 4, **metrics) -> dict:
    return {"cases": cases, "success": success, "latency_p50_ms": latency, **metrics}


def test_best_primary_metric_wins_among_providers_that_answer_enough():
    report = _report({
        "groq": {"review": _stats(1.0, 2000, bug_recall=0.8), "extract": _stats(1.0, 500, field_accuracy=0.9)},
        "gemini-lite": {"review": _stats(1.0, 9000, bug_recall=0.95), "extract": _stats(0.5, 400, field_accuracy=1.0)},
        "xcheck": {"review": _stats(1.0, bug_recall=1.0)},  # composite arm: never preferred
    })
    table = routing_table(report, min_success=0.8)
    assert table["review"]["provider"] == "gemini-lite"
    assert table["review"]["others"] == {"groq": 0.8}
    # gemini-lite extracts better but answered only half its cases.
    assert table["extract"]["provider"] == "groq"


def test_tie_goes_to_the_faster_provider():
    report = _report({
        "a": {"classify": _stats(1.0, 3000, field_accuracy=1.0)},
        "b": {"classify": _stats(1.0, 800, field_accuracy=1.0)},
    })
    assert routing_table(report)["classify"]["provider"] == "b"


def test_routing_yaml_is_ascii_and_loads_into_task_profiles(tmp_path: Path):
    report = _report({"groq": {"review": _stats(1.0, bug_recall=0.9)}})
    text = render_routing_yaml(routing_table(report), report)
    text.encode("ascii")
    assert yaml.safe_load(text) == {"preferred_provider": {"review": "groq"}}

    (tmp_path / "task_profiles.yaml").write_text(
        "task_types:\n  review: {}\n  extract: {preferred_provider: gemini}\n", encoding="utf-8"
    )
    (tmp_path / "routing_from_eval.yaml").write_text(
        "preferred_provider:\n  review: groq\n  extract: groq\n", encoding="ascii"
    )
    profiles = load_task_profiles(tmp_path / "task_profiles.yaml")
    assert profiles["review"].preferred_provider == "groq"
    assert profiles["extract"].preferred_provider == "gemini"  # hand-written wins


def test_gate_flags_a_quality_drop_but_only_notes_lower_availability():
    base = _report({"groq": {"review": _stats(1.0, bug_recall=1.0), "extract": _stats(1.0, field_accuracy=0.9)}})
    now = _report({"groq": {"review": _stats(0.5, bug_recall=1.0), "extract": _stats(1.0, field_accuracy=0.7)}})
    regressions, notes = gate(base, now, tolerance=0.05)
    assert regressions == ["groq/extract: field_accuracy 0.90 -> 0.70"]
    assert any("success 1.00 -> 0.50" in n for n in notes)


def test_gate_cli_exit_code(tmp_path: Path):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    base.write_text(json.dumps(_report({"groq": {"review": _stats(1.0, bug_recall=1.0)}})))
    new.write_text(json.dumps(_report({"groq": {"review": _stats(1.0, bug_recall=0.5)}})))
    with pytest.raises(SystemExit) as exc:
        main(["--from-report", str(new), "--gate", str(base)])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        main(["--from-report", str(base), "--gate", str(base)])
    assert exc.value.code == 0


def test_failed_call_is_not_scored_as_zero_quality():
    # 2026-10-30: two 429s scored as bug_recall 0 looked like a harness regression.
    case = EvalCase(
        id="r", task="review", task_type="review", budget="L3",
        expected_findings=[ExpectedFinding(file="orders.py", line=3)],
    )
    scores = score_case(case, DelegateTaskResult(success=False, error="429"), None)
    assert scores["success"] == 0.0
    assert "bug_recall" not in scores


@pytest.mark.asyncio
async def test_preferred_provider_routes_a_task_type_unless_the_request_names_one():
    import logging

    from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
    from llm_worker_pool.application.dto import DelegateTaskRequest
    from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions, TaskProfile
    from llm_worker_pool.infra.logging import InvocationLogger
    from tests.unit.test_delegate_task_use_case import FakeProvider, _config

    default, measured = FakeProvider(), FakeProvider()
    use_case = DelegateTaskUseCase(
        providers={"default": default, "measured": measured},
        default_provider="default",
        config=_config(),
        budget_profiles={"L2": BudgetProfile(name="L2", description="", options=GenerationOptions())},
        invocation_logger=InvocationLogger(logging.getLogger("test.invocation")),
        task_profiles={
            "extract": TaskProfile(name="extract", preferred_provider="measured"),
            "classify": TaskProfile(name="classify", preferred_provider="not-configured"),
        },
    )

    await use_case.execute(DelegateTaskRequest(task="t", task_type="extract"))
    assert (measured.call_count, default.call_count) == (1, 0)
    await use_case.execute(DelegateTaskRequest(task="t", task_type="extract", provider="default"))
    assert default.call_count == 1
    await use_case.execute(DelegateTaskRequest(task="t", task_type="classify"))  # unknown -> budget/default
    assert default.call_count == 2


def test_gate_compares_on_a_metric_both_reports_measured():
    # Baseline judged (faithfulness), new run --no-judge: must_contain is shared.
    base = _report({"groq": {"summarize": _stats(1.0, faithfulness=0.9, must_contain=1.0)}})
    now = _report({"groq": {"summarize": _stats(1.0, must_contain=0.5)}})
    regressions, _ = gate(base, now)
    assert regressions == ["groq/summarize: must_contain 1.00 -> 0.50"]
    _, notes = gate(base, _report({"groq": {"summarize": _stats(0.0)}}))
    assert any("not gated" in n for n in notes)


def test_emit_routing_creates_the_target_directory(tmp_path: Path):
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_report({"groq": {"review": _stats(1.0, bug_recall=0.9)}})))
    out = tmp_path / "new" / "dir" / "routing.yaml"
    with pytest.raises(SystemExit) as exc:
        main(["--from-report", str(report), "--emit-routing", str(out)])
    assert exc.value.code == 0 and out.exists()
