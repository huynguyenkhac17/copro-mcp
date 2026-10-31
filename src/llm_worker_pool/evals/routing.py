"""Turn a worker-eval report into decisions: which provider to prefer per task type
(`--emit-routing`, RouteLLM-style but from our own measurements) and whether a change
made things worse (`--gate`).

Both judge each task type by ONE primary metric - the thing that kind of work is for -
computed over answered cases only (scoring.py leaves failed calls out of quality
metrics). Availability is weighed separately through `success`, so a run that hit a
429 storm does not read as a quality regression.
"""

from __future__ import annotations

from typing import Any

# First metric present in a summary wins; must_contain is the judge-free stand-in.
PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "review": ("bug_recall",),
    "extract": ("field_accuracy",),
    "classify": ("field_accuracy", "must_contain"),
    "summarize": ("faithfulness", "must_contain"),
    "explain": ("faithfulness", "must_contain"),
    "translate": ("factual_correctness", "must_contain"),
    "generate": ("must_contain",),
}
_DEFAULT_PRIMARY = ("must_contain", "faithfulness")
# Arms that are not a single provider (routing / cross-check) cannot be "preferred".
_COMPOSITE_ARMS = frozenset({"routed", "xcheck"})


def primary_metric(task_type: str, *summaries: dict[str, Any]) -> str | None:
    """First candidate metric measured in every given summary (a baseline run with
    the judge and a new one without still share must_contain)."""
    for metric in PRIMARY_METRICS.get(task_type, _DEFAULT_PRIMARY):
        if all(summary.get(metric) is not None for summary in summaries):
            return metric
    return None


def _paired_means(
    rows: list[dict[str, Any]], task_type: str, arms: list[str], metric: str
) -> tuple[dict[str, float], int] | None:
    """Each arm's mean `metric` over the cases EVERY arm answered. Summary means are
    over each arm's own answered cases, so an arm that failed the hard case would
    look better than one that answered it imperfectly (2026-10-31: groq skipped the
    15k-token review that gemini-lite scored 0.50 on, then won review at 1.00)."""
    per_arm = {
        arm: {
            r["id"]: r["scores"][metric]
            for r in rows
            if r.get("arm") == arm and r.get("task_type") == task_type and metric in r.get("scores", {})
        }
        for arm in arms
    }
    common = set.intersection(*(set(scores) for scores in per_arm.values()))
    if not common:
        return None
    means = {arm: sum(scores[i] for i in common) / len(common) for arm, scores in per_arm.items()}
    return means, len(common)


def routing_table(report: dict[str, Any], *, min_success: float = 0.8) -> dict[str, dict[str, Any]]:
    """{task_type: {provider, metric, value, success, latency_p50_ms, cases, paired_cases,
    others}}. A provider qualifies for a task type when it answered at least
    `min_success` of its cases. Qualified providers are compared on the primary metric
    over the cases all of them answered (when the report has per-case rows), then on
    success, then on lower latency."""
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for arm, summary in report.get("summary", {}).items():
        if arm in _COMPOSITE_ARMS:
            continue
        for task_type, stats in summary.get("by_task_type", {}).items():
            by_type.setdefault(task_type, []).append((arm, stats))

    rows = report.get("rows") or []
    table: dict[str, dict[str, Any]] = {}
    for task_type, entries in by_type.items():
        qualified = [(arm, s) for arm, s in entries if (s.get("success") or 0) >= min_success]
        metric = primary_metric(task_type, *(s for _, s in qualified)) if qualified else None
        if metric is None:
            continue
        paired = _paired_means(rows, task_type, [arm for arm, _ in qualified], metric) if rows else None
        candidates = []
        for arm, stats in qualified:
            value = paired[0][arm] if paired else stats[metric]
            latency = stats.get("latency_p50_ms")
            key = (value, stats["success"], -(latency if latency is not None else float("inf")))
            candidates.append((key, arm, stats))
        candidates.sort(reverse=True)
        (value, _, _), arm, stats = candidates[0]
        table[task_type] = {
            "provider": arm,
            "metric": metric,
            "value": value,
            "success": stats["success"],
            "latency_p50_ms": stats.get("latency_p50_ms"),
            "cases": stats.get("cases"),
            "paired_cases": paired[1] if paired else None,
            "others": {
                other: {"value": round(k[0], 4), "success": s["success"]}
                for k, other, s in candidates[1:]
            },
        }
    return table


def render_routing_yaml(table: dict[str, dict[str, Any]], report: dict[str, Any]) -> str:
    """config/routing_from_eval.yaml - pure ASCII (the YAML loaders use the platform
    encoding, cp1252 on Windows)."""
    lines = [
        "# Generated by `worker-eval --emit-routing` - review the evidence, then commit.",
        f"# Source report: {report.get('generated_at')} (harness={report.get('meta', {}).get('harness')}),",
        "# primary metric per task type over the cases every qualified provider",
        "# answered (paired=N), then success, then latency; a provider needed",
        "# success >= min_success to qualify. A preferred_provider written in",
        "# task_profiles.yaml beats this file.",
        "preferred_provider:",
    ]
    for task_type in sorted(table):
        row = table[task_type]
        others = ", ".join(
            f"{k} {v['value']:.2f} (success {v['success']:.2f})" for k, v in row["others"].items()
        ) or "no other qualified arm"
        latency = row["latency_p50_ms"]
        paired = row["paired_cases"]
        lines.append(
            f"  # {row['metric']}={row['value']:.2f} (paired={'-' if paired is None else paired}), "
            f"success={row['success']:.2f}, p50={'-' if latency is None else f'{latency:.0f}ms'}, "
            f"cases={row['cases']}; vs {others}"
        )
        lines.append(f"  {task_type}: {row['provider']}")
    text = "\n".join(lines) + "\n"
    return text.encode("ascii", "replace").decode("ascii")


def gate(
    baseline: dict[str, Any], current: dict[str, Any], *, tolerance: float = 0.05
) -> tuple[list[str], list[str]]:
    """(regressions, notes). A regression: for an arm and task type measured in both
    reports, the primary metric dropped by more than `tolerance`. A success drop is
    only a note - quota errors come and go between runs."""
    regressions: list[str] = []
    notes: list[str] = []
    for arm, now in current.get("summary", {}).items():
        before = baseline.get("summary", {}).get(arm)
        if before is None:
            notes.append(f"{arm}: not in the baseline, not gated")
            continue
        for task_type, stats in now.get("by_task_type", {}).items():
            old = before.get("by_task_type", {}).get(task_type)
            if old is None:
                continue
            metric = primary_metric(task_type, old, stats)
            if metric is None:
                if primary_metric(task_type, old) is not None:
                    notes.append(f"{arm}/{task_type}: no metric measured in both reports, not gated")
                continue
            drop = old[metric] - stats[metric]
            line = f"{arm}/{task_type}: {metric} {old[metric]:.2f} -> {stats[metric]:.2f}"
            if drop > tolerance:
                regressions.append(line)
            old_success, new_success = old.get("success"), stats.get("success")
            if old_success is not None and new_success is not None and old_success - new_success > tolerance:
                notes.append(f"{arm}/{task_type}: success {old_success:.2f} -> {new_success:.2f}")
    return regressions, notes
