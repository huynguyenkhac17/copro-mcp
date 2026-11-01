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
    assert table["review"]["others"] == {"groq": {"value": 0.8, "success": 1.0}}
    assert table["review"]["paired_cases"] is None  # no per-case rows: summary means
    # gemini-lite extracts better but answered only half its cases.
    assert table["extract"]["provider"] == "groq"


def _row(arm: str, case: str, task_type: str = "review", **scores) -> dict:
    return {"id": case, "arm": arm, "task_type": task_type, "scores": scores}


def test_providers_are_compared_on_the_cases_both_answered():
    # 2026-10-31: groq skipped the large review (prompt over max_prompt_tokens) that
    # gemini-lite answered at 0.50, so groq's summary mean (1.00) beat 0.90 unfairly.
    report = _report({
        "groq": {"review": _stats(0.8, 900, cases=5, bug_recall=1.0)},
        "gemini-lite": {"review": _stats(1.0, 1800, cases=5, bug_recall=0.9)},
    })
    report["rows"] = [
        *(_row(arm, f"small-{i}", success=1.0, bug_recall=1.0) for arm in ("groq", "gemini-lite") for i in range(4)),
        _row("groq", "large", success=0.0),
        _row("gemini-lite", "large", success=1.0, bug_recall=0.5),
    ]
    row = routing_table(report)["review"]
    # Tied at 1.00 on the 4 shared cases; gemini-lite answered all 5.
    assert row["provider"] == "gemini-lite"
    assert (row["value"], row["paired_cases"]) == (1.0, 4)
    assert row["others"] == {"groq": {"value": 1.0, "success": 0.8}}
    render_routing_yaml(routing_table(report), report).encode("ascii")


def test_paired_comparison_still_prefers_the_better_answers():
    report = _report({
        "a": {"extract": _stats(1.0, 500, field_accuracy=0.8)},
        "b": {"extract": _stats(1.0, 900, field_accuracy=0.85)},
    })
    report["rows"] = [
        _row("a", "c1", "extract", field_accuracy=1.0), _row("b", "c1", "extract", field_accuracy=1.0),
        _row("a", "c2", "extract", field_accuracy=0.6), _row("b", "c2", "extract", field_accuracy=0.7),
    ]
    assert routing_table(report)["extract"]["provider"] == "b"


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


def _repeat_row(arm: str, case: str, repeat: int, task_type: str = "review", **scores) -> dict:
    return {**_row(arm, case, task_type, **scores), "repeat": repeat}


def test_repeated_cases_count_once_at_their_mean_and_report_spread():
    report = _report({
        "a": {"review": _stats(1.0, 900, bug_recall=0.75)},
        "b": {"review": _stats(1.0, 900, bug_recall=0.75)},
    })
    report["rows"] = [
        _repeat_row("a", "c1", 0, bug_recall=1.0), _repeat_row("a", "c1", 1, bug_recall=0.5),
        _repeat_row("a", "c2", 0, bug_recall=1.0), _repeat_row("a", "c2", 1, bug_recall=1.0),
        _repeat_row("b", "c1", 0, bug_recall=0.5), _repeat_row("b", "c1", 1, bug_recall=0.5),
        _repeat_row("b", "c2", 0, bug_recall=1.0), _repeat_row("b", "c2", 1, bug_recall=1.0),
    ]
    row = routing_table(report)["review"]
    # a: c1 mean 0.75, c2 1.0 -> 0.875 (a later repeat must not overwrite an earlier one).
    assert (row["provider"], row["value"], row["paired_cases"]) == ("a", 0.875, 2)
    assert row["spread"] == 0.25  # repeat means 1.0 vs 0.75
    assert row["others"]["b"]["value"] == 0.75


def test_default_provider_is_kept_unless_beaten_by_more_than_the_margin():
    report = _report({
        "groq": {"review": _stats(1.0, 900, bug_recall=0.90), "extract": _stats(1.0, 500, field_accuracy=0.70)},
        "gemini-lite": {"review": _stats(1.0, 900, bug_recall=0.93), "extract": _stats(1.0, 500, field_accuracy=0.90)},
    })
    table = routing_table(report, defaults={"review": "groq", "extract": "groq"}, margin=0.05)
    assert (table["review"]["provider"], table["review"]["switch"]) == ("groq", False)
    assert table["review"]["held"] == {"best": "gemini-lite", "lead": 0.03, "required": 0.05}
    assert table["review"]["others"] == {"gemini-lite": {"value": 0.93, "success": 1.0}}
    assert (table["extract"]["provider"], table["extract"]["switch"]) == ("gemini-lite", True)

    text = render_routing_yaml(table, report)
    text.encode("ascii")
    assert yaml.safe_load(text) == {"preferred_provider": {"extract": "gemini-lite"}}
    assert "kept default: gemini-lite led by 0.03" in text


def test_a_lead_inside_the_run_to_run_spread_does_not_switch():
    # Lead 0.25 beats the 0.05 margin, but the default itself moved 0.50 between runs.
    report = _report({
        "groq": {"review": _stats(1.0, bug_recall=0.5)},
        "gemini-lite": {"review": _stats(1.0, bug_recall=0.75)},
    })
    report["rows"] = [
        _repeat_row("groq", "c1", 0, bug_recall=0.75), _repeat_row("groq", "c1", 1, bug_recall=0.25),
        _repeat_row("gemini-lite", "c1", 0, bug_recall=0.75), _repeat_row("gemini-lite", "c1", 1, bug_recall=0.75),
    ]
    row = routing_table(report, defaults={"review": "groq"}, margin=0.05)["review"]
    assert (row["provider"], row["switch"]) == ("groq", False)
    assert row["held"]["required"] == 0.5


def test_winning_default_or_unmeasured_default_needs_no_margin():
    report = _report({
        "groq": {"review": _stats(1.0, bug_recall=0.9)},
        "gemini-lite": {"review": _stats(1.0, bug_recall=0.8)},
    })
    row = routing_table(report, defaults={"review": "groq"}, margin=0.5)["review"]
    assert (row["provider"], row["switch"], row["held"]) == ("groq", False, None)
    # Default never measured (or not qualified): the best measured provider is written.
    row = routing_table(report, defaults={"review": "ollama"}, margin=0.5)["review"]
    assert (row["provider"], row["switch"]) == ("groq", True)
    assert yaml.safe_load(render_routing_yaml({}, report)) == {"preferred_provider": None}


def test_default_providers_ignore_the_eval_routing_file(tmp_path: Path, monkeypatch):
    from llm_worker_pool.config.settings import AppConfig
    from llm_worker_pool.evals.runner import default_providers

    (tmp_path / "task_profiles.yaml").write_text(
        "task_types:\n  review: {default_budget: L3}\n  extract: {preferred_provider: gemini}\n"
        "  summarize: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "routing_from_eval.yaml").write_text("preferred_provider:\n  review: gemini-lite\n", encoding="ascii")
    monkeypatch.setenv("LLM_WORKER_POOL_TASK_PROFILES_CONFIG", str(tmp_path / "task_profiles.yaml"))
    config = AppConfig.model_construct(budget_providers={"L3": "groq"}, default_provider="ollama")
    assert default_providers(config) == {"review": "groq", "extract": "gemini", "summarize": "ollama"}


@pytest.mark.asyncio
async def test_repeat_runs_every_case_n_times_per_arm(tmp_path: Path, monkeypatch):
    import argparse

    from llm_worker_pool.evals import runner

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"id": i, "task": "t", "task_type": "classify", "budget": "L1"}) for i in ("x", "y")),
        encoding="utf-8",
    )

    class _UseCase:
        def available_providers(self):
            return ["groq"]

    async def fake_run_case(use_case, judge, case, arm, harness, sem, repeat=0):
        return {"id": case.id, "task_type": case.task_type, "budget": case.budget, "arm": arm,
                "repeat": repeat, "model": "m", "latency_ms": 1.0, "prompt_tokens": 0,
                "completion_tokens": 0, "thinking_tokens": 0, "scores": {"success": 1.0}}

    monkeypatch.setattr(runner, "_build_use_case", lambda config: _UseCase())
    monkeypatch.setattr(runner, "_run_case", fake_run_case)
    args = argparse.Namespace(
        dataset=[str(dataset)], case=None, task_type=None, harness="on", judge=False,
        judge_provider=None, concurrency=1, arm=["groq"], repeat=3, label=None,
    )
    report = await runner.run(args)
    assert sorted((r["id"], r["repeat"]) for r in report["rows"]) == [
        (i, n) for i in ("x", "y") for n in range(3)
    ]
    assert report["meta"]["repeat"] == 3
    report["rows"] = [{**r, "provider": "groq", "warnings": [], "error": None, "judge_errors": {}} for r in report["rows"]]
    md = runner.render_markdown(report)
    assert "x 3 repeats" in md
    # Every run is numbered from #1, the first one included.
    assert all(f"| x#{n} |" in md for n in (1, 2, 3)) and "| x |" not in md
