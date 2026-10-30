"""`worker-eval`: run the eval dataset through DelegateTaskUseCase and write a report.

Each --arm forces one provider (fallbacks off, so the score belongs to that model), or
`routed` keeps the real routing + fallback chains to measure what callers actually get.
--harness on passes each case's task_type (config/task_profiles.yaml); off sends the
request exactly as a caller without task_type would - that is the baseline.

    worker-eval --arm groq --arm gemini-lite                 # baseline
    worker-eval --arm groq --harness on --compare evals/reports/<baseline>.json
    worker-eval --compare A.json B.json                      # diff two reports only

Outputs evals/reports/<timestamp>.{json,md} and ~/.llm_worker_pool/eval_latest.json
(read by `worker-metrics` as llm_worker_eval_score).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.config.settings import AppConfig
from llm_worker_pool.domain.validators import split_context_files
from llm_worker_pool.evals.dataset import EvalCase, load_cases
from llm_worker_pool.evals.scoring import mean, percentile, score_case
from llm_worker_pool.infra.event_log import configured_path, default_eval_latest_path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTED_ARM = "routed"
XCHECK_ARM = "xcheck"
"""Real routing through the cross-check orchestrator (needs --harness on: it is the
task profile that turns cross-checking on and names the providers)."""
_SPECIAL_ARMS = (ROUTED_ARM, XCHECK_ARM)

SUMMARY_METRICS = (
    "success", "faithfulness", "factual_correctness", "answer_relevancy", "context_precision",
    "context_recall", "bug_recall", "bug_precision", "agreement_rate", "citation_valid_rate", "json_valid", "json_schema_valid",
    "field_accuracy", "must_contain", "no_leak", "secret_guarded", "file_denied", "fallback", "truncated",
)


def _arm_config(arm: str) -> AppConfig:
    config = AppConfig()
    config.cache.enabled = False  # measure the model, not the cache
    config.metrics.enabled = False  # keep eval traffic out of production metrics
    roots = list(config.file_access.allowed_roots)
    if str(REPO_ROOT) not in roots:
        roots.append(str(REPO_ROOT))
    config.file_access.allowed_roots = roots
    if arm not in _SPECIAL_ARMS:
        config.fallbacks = {}
        config.budget_fallbacks = {}
    return config


def _build_use_case(config: AppConfig):
    # Same wiring as the MCP server, so the eval exercises the real code path.
    from llm_worker_pool.infra.stats import StatsCollector
    from llm_worker_pool.server import _build_use_case as build

    use_case, _resolver = build(config, StatsCollector())
    return use_case


def _request(case: EvalCase, arm: str, harness: bool) -> DelegateTaskRequest:
    fields: dict[str, Any] = {
        "task": case.task,
        "provider": None if arm in _SPECIAL_ARMS else arm,
        "context": case.context,
        "file_paths": [str(REPO_ROOT / p) for p in case.file_paths] if case.file_paths else None,
        "json_schema": case.json_schema,
    }
    # Forward only what the case states: a default sent explicitly would override the
    # task profile's own (e.g. review's output_format=json) once the harness is on.
    for name in ("budget", "output_format"):
        if name in case.model_fields_set:
            fields[name] = getattr(case, name)
    if harness:
        if "task_type" not in DelegateTaskRequest.model_fields:
            raise SystemExit("--harness on needs DelegateTaskRequest.task_type (not implemented yet)")
        fields["task_type"] = case.task_type
    return DelegateTaskRequest(**fields)


async def _resolve_context(use_case, request: DelegateTaskRequest) -> str | None:
    if hasattr(use_case, "resolve_context"):
        context, _ = await use_case.resolve_context(request)
        return context
    # Pre-harness code (baseline runs against an older checkout) has no resolve_context.
    from llm_worker_pool.domain.file_context import resolve_request_context

    context, _ = await resolve_request_context(
        use_case._file_context_resolver, context=request.context,
        file_paths=request.file_paths, query=request.task,
    )
    return context


def _is_rag(case: EvalCase) -> bool:
    return any((REPO_ROOT / p).is_dir() for p in case.file_paths or [])


async def _run_case(use_case, judge, case: EvalCase, arm: str, harness: bool, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        request = _request(case, arm, harness)
        execute = use_case.execute
        if arm == XCHECK_ARM:
            from llm_worker_pool.application.cross_check_orchestrator import CrossCheckOrchestrator

            orchestrator = CrossCheckOrchestrator(use_case)
            if orchestrator.wants_cross_check(request):
                execute = orchestrator.execute
        started = time.monotonic()
        context = await _resolve_context(use_case, request)
        try:
            result = await execute(request)
        except Exception as exc:  # noqa: BLE001 - a broken case must not end the run
            result = DelegateTaskResult(success=False, error=f"{type(exc).__name__}: {exc}")
        wall_ms = (time.monotonic() - started) * 1000

    scores = score_case(case, result, context)
    judge_errors: dict[str, str] = {}
    if judge is not None and result.success and result.raw_text.strip():
        files = split_context_files(context)
        contexts = list(files.values()) or ([context] if context else [])
        llm_scores, judge_errors = await judge.score(case, result.raw_text, contexts, _is_rag(case))
        scores.update(llm_scores)

    return {
        "id": case.id,
        "task_type": case.task_type,
        "budget": case.budget,
        "arm": arm,
        "provider": result.provider,
        "model": result.model,
        "latency_ms": round(result.latency_ms or wall_ms, 1),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "thinking_tokens": result.thinking_tokens,
        "scores": scores,
        "judge_errors": judge_errors,
        "warnings": result.warnings,
        "error": result.error,
        "answer": result.raw_text or "",
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": len(rows)}
    for metric in SUMMARY_METRICS:
        values = [r["scores"][metric] for r in rows if metric in r["scores"]]
        if values:
            out[metric] = round(mean(values), 4)
            out[f"{metric}_n"] = len(values)
    latencies = [r["latency_ms"] for r in rows if r["scores"].get("success")]
    out["latency_p50_ms"] = percentile(latencies, 0.5)
    out["latency_p95_ms"] = percentile(latencies, 0.95)
    for kind in ("prompt", "completion", "thinking"):
        out[f"{kind}_tokens"] = sum(r[f"{kind}_tokens"] or 0 for r in rows)
    return out


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_type = {
            t: _aggregate([r for r in arm_rows if r["task_type"] == t])
            for t in dict.fromkeys(r["task_type"] for r in arm_rows)
        }
        by_arm[arm] = {"overall": _aggregate(arm_rows), "by_task_type": by_type}
    return by_arm


def _eval_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flattened means for the llm_worker_eval_score gauge."""
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for r in rows:
        for metric, value in r["scores"].items():
            groups.setdefault((metric, r["arm"], r["budget"], r["task_type"]), []).append(value)
    return [
        {"metric": m, "provider": a, "budget": b, "task_type": t, "value": round(mean(v), 4)}
        for (m, a, b, t), v in sorted(groups.items())
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}" if value >= 100 else f"{value:.2f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# worker-eval {report['generated_at']}", ""]
    meta = report["meta"]
    lines.append(f"- harness: **{meta['harness']}**, arms: {', '.join(meta['arms'])}, cases: {meta['cases']}")
    if meta.get("note"):
        lines.append(f"- note: {meta['note']}")
    judge = meta.get("judge")
    if judge:
        lines.append(f"- judge: `{judge['provider']}` ({judge['model']}), reasoning_effort={judge['reasoning_effort']}")
        for arm, model in meta["arm_models"].items():
            if model and model == judge["model"]:
                lines.append(f"  - WARNING: arm `{arm}` answers with the judge's own model ({model}) - "
                             "self-preference risk, compare its LLM scores with care")
        if judge.get("note"):
            lines.append(f"- {judge['note']}")
    else:
        lines.append("- judge: disabled (deterministic metrics only)")
    lines.append("")

    cols = ["arm", *[m for m in SUMMARY_METRICS], "latency_p50_ms", "latency_p95_ms", "prompt_tokens", "thinking_tokens"]
    present = [c for c in cols if c == "arm" or any(c in s["overall"] for s in report["summary"].values())]
    lines += ["## Overall", "", "| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
    for arm, s in report["summary"].items():
        lines.append("| " + " | ".join([arm, *[_fmt(s["overall"].get(c)) for c in present[1:]]]) + " |")

    lines += ["", "## By task type", ""]
    for arm, s in report["summary"].items():
        lines += [f"### {arm}", "", "| task_type | cases | key metrics |", "|---|---|---|"]
        for t, agg in s["by_task_type"].items():
            keys = [f"{m}={_fmt(agg[m])}" for m in SUMMARY_METRICS if m in agg and m not in ("fallback", "truncated")]
            lines.append(f"| {t} | {agg['cases']} | {', '.join(keys)} |")
        lines.append("")

    lines += ["## Cases", "", "| arm | case | answered by | latency ms | scores | notes |", "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        scores = ", ".join(f"{k}={_fmt(v)}" for k, v in r["scores"].items() if k != "success")
        notes = r["error"] or "; ".join(sorted({w.split(":")[0].split(" ")[0] for w in r["warnings"]}))
        if r["judge_errors"]:
            notes += f" judge_errors={list(r['judge_errors'])}"
        lines.append(f"| {r['arm']} | {r['id']} | {r['provider']} | {_fmt(r['latency_ms'])} | {scores} | {notes[:160]} |")
    return "\n".join(lines) + "\n"


def render_compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    lines = [
        f"# worker-eval compare: {old['generated_at']} (harness={old['meta']['harness']}) -> "
        f"{new['generated_at']} (harness={new['meta']['harness']})",
        "",
    ]
    for arm in new["summary"]:
        if arm not in old["summary"]:
            continue
        a, b = old["summary"][arm]["overall"], new["summary"][arm]["overall"]
        lines += [f"## {arm}", "", "| metric | before | after | delta |", "|---|---|---|---|"]
        for m in [*SUMMARY_METRICS, "latency_p50_ms", "latency_p95_ms", "prompt_tokens", "thinking_tokens"]:
            if m in a or m in b:
                va, vb = a.get(m), b.get(m)
                delta = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
                sign = "+" if delta is not None and delta > 0 else ""
                lines.append(f"| {m} | {_fmt(va)} | {_fmt(vb)} | {sign}{_fmt(delta) if delta is not None else '-'} |")
        lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases([Path(p) for p in args.dataset])
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if args.task_type:
        cases = [c for c in cases if c.task_type in set(args.task_type)]
    harness = args.harness == "on"

    judge = None
    judge_meta = None
    base_config = AppConfig()
    if args.judge:
        from llm_worker_pool.evals.judge import Judge

        judge = Judge(base_config, provider=args.judge_provider)
        judge_meta = {
            "provider": judge.provider, "model": judge.model,
            "reasoning_effort": base_config.eval.judge_reasoning_effort, "note": judge.embeddings_note,
        }

    sem = asyncio.Semaphore(args.concurrency or base_config.eval.max_concurrency)
    rows: list[dict[str, Any]] = []
    arm_models: dict[str, str | None] = {}
    try:
        for arm in args.arm:
            use_case = _build_use_case(_arm_config(arm))
            if arm not in _SPECIAL_ARMS and arm not in use_case.available_providers():
                print(f"skipping arm {arm!r}: provider not available", file=sys.stderr)
                continue
            print(f"arm {arm}: {len(cases)} cases", file=sys.stderr)
            arm_rows = await asyncio.gather(
                *(_run_case(use_case, judge, c, arm, harness, sem) for c in cases)
            )
            rows.extend(arm_rows)
            arm_models[arm] = next((r["model"] for r in arm_rows if r["model"]), None)
        judge_stats = judge.breaker_stats() if judge else None
    finally:
        if judge is not None:
            await judge.aclose()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "generated_at": stamp,
        "meta": {
            "harness": args.harness, "arms": list(args.arm), "arm_models": arm_models,
            "cases": len(cases), "dataset": args.dataset, "judge": judge_meta,
            "judge_key_stats": judge_stats, "label": args.label,
        },
        "summary": _summaries(rows),
        "scores": _eval_scores(rows),
        "rows": rows,
    }


def _write_report(report: dict[str, Any], out_dir: Path, eval_latest: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = report["generated_at"] + (f"-{report['meta']['label']}" if report["meta"].get("label") else "")
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render_markdown(report), encoding="utf-8")
    update_eval_latest(eval_latest, report)
    return json_path


def update_eval_latest(path: Path, report: dict[str, Any]) -> None:
    """Merge this run's scores into eval_latest.json by (metric, provider, budget,
    task_type), so a run over a subset (--case / --task-type) only replaces what it
    measured instead of wiping the other gauges."""
    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("scores") or []
    except (OSError, ValueError, AttributeError):
        previous = []
    key = lambda s: (s.get("metric"), s.get("provider"), s.get("budget"), s.get("task_type"))  # noqa: E731
    merged = {key(s): s for s in previous if isinstance(s, dict)}
    for s in report["scores"]:
        merged[key(s)] = {**s, "generated_at": report["generated_at"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"generated_at": report["generated_at"], "harness": report["meta"]["harness"],
                    "judge": report["meta"]["judge"], "scores": list(merged.values())}),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="worker-eval", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", default=None,
                        help="JSONL file or directory (default: evals/datasets)")
    parser.add_argument("--arm", action="append", default=[],
                        help=f"provider to force (repeatable), '{ROUTED_ARM}' for real routing, "
                        f"or '{XCHECK_ARM}' for routing through cross-check (with --harness on)")
    parser.add_argument("--harness", choices=("off", "on"), default="off")
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="deterministic metrics only")
    parser.add_argument("--judge-provider", default=None, help="override eval.judge_provider")
    parser.add_argument("--case", action="append", help="only these case ids")
    parser.add_argument("--task-type", action="append", help="only these task types")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--label", default=None, help="suffix for the report file name")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "evals" / "reports")
    parser.add_argument("--compare", action="append", default=[],
                        help="baseline report .json; given twice with no --arm, just diff the two")
    args = parser.parse_args(argv)
    args.dataset = args.dataset or [str(REPO_ROOT / "evals" / "datasets")]

    if len(args.compare) == 2 and not args.arm:
        old, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        print(render_compare(old, new))
        return
    if not args.arm:
        parser.error("give at least one --arm (or two --compare reports)")

    eval_latest = configured_path(AppConfig().metrics.eval_latest_path, default_eval_latest_path())
    report = asyncio.run(run(args))
    json_path = _write_report(report, args.out_dir, eval_latest)
    print(render_markdown(report))
    print(f"report: {json_path} (+ .md)", file=sys.stderr)

    if args.compare:
        old = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        compare_md = render_compare(old, report)
        json_path.with_name(json_path.stem + "-compare.md").write_text(compare_md, encoding="utf-8")
        print(compare_md)


if __name__ == "__main__":
    main()
