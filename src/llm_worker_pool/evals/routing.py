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


def _case_scores(
    rows: list[dict[str, Any]], task_type: str, arm: str, metric: str
) -> dict[str, list[tuple[int, float]]]:
    """{case id: [(repeat, score), ...]} over the arm's answered runs of each case."""
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("arm") == arm and r.get("task_type") == task_type and metric in r.get("scores", {}):
            out.setdefault(r["id"], []).append((r.get("repeat", 0), r["scores"][metric]))
    return out


def _paired_means(
    rows: list[dict[str, Any]], task_type: str, arms: list[str], metric: str
) -> tuple[dict[str, float], int, dict[str, float]] | None:
    """(means, paired cases, spreads). Each arm's mean `metric` over the cases EVERY
    arm answered. Summary means are over each arm's own answered cases, so an arm that
    failed the hard case would look better than one that answered it imperfectly
    (2026-10-31: groq skipped the 15k-token review that gemini-lite scored 0.50 on,
    then won review at 1.00). A case run several times (--repeat) counts once, at the
    mean of its answered runs. The spread is how far that arm's per-repeat mean over
    the paired cases moved between runs (0 with a single repeat)."""
    per_arm = {arm: _case_scores(rows, task_type, arm, metric) for arm in arms}
    common = set.intersection(*(set(scores) for scores in per_arm.values()))
    if not common:
        return None
    means: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for arm, scores in per_arm.items():
        means[arm] = sum(sum(v for _, v in scores[i]) / len(scores[i]) for i in common) / len(common)
        by_repeat: dict[int, list[float]] = {}
        for i in common:
            for repeat, value in scores[i]:
                by_repeat.setdefault(repeat, []).append(value)
        # Only repeats that answered every paired case are comparable run to run.
        run_means = [sum(v) / len(v) for v in by_repeat.values() if len(v) == len(common)]
        spreads[arm] = max(run_means) - min(run_means) if len(run_means) > 1 else 0.0
    return means, len(common), spreads


def routing_table(
    report: dict[str, Any],
    *,
    min_success: float = 0.8,
    defaults: dict[str, str] | None = None,
    margin: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """{task_type: {provider, metric, value, success, latency_p50_ms, cases, paired_cases,
    spread, others, switch}}. A provider qualifies for a task type when it answered at
    least `min_success` of its cases. Qualified providers are compared on the primary
    metric over the cases all of them answered (when the report has per-case rows),
    then on success, then on lower latency.

    `defaults` is {task_type: provider callers get today}. When that provider qualified
    and lost, the winner must lead it on the metric by more than `margin` and by more
    than either one's run-to-run spread; otherwise the row keeps the default with
    switch=False (two judged runs on 2026-10-31 flipped review/classify/generate)."""
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
        spreads = paired[2] if paired else {}
        best = candidates[0]
        chosen, held = best, None
        default = (defaults or {}).get(task_type)
        incumbent = next((c for c in candidates if c[1] == default), None)
        if incumbent is not None and incumbent is not best:
            lead = best[0][0] - incumbent[0][0]
            required = max(margin, spreads.get(best[1], 0.0), spreads.get(default, 0.0))
            if lead <= required:
                chosen = incumbent
                held = {"best": best[1], "lead": round(lead, 4), "required": round(required, 4)}
        (value, _, _), arm, stats = chosen
        table[task_type] = {
            "provider": arm,
            "metric": metric,
            "value": value,
            "success": stats["success"],
            "latency_p50_ms": stats.get("latency_p50_ms"),
            "cases": stats.get("cases"),
            "paired_cases": paired[1] if paired else None,
            "spread": spreads.get(arm),
            "others": {
                other: {"value": round(k[0], 4), "success": s["success"]}
                for k, other, s in candidates
                if other != arm
            },
            # False: the default stays (no measured reason to move), so nothing is written.
            "switch": arm != default,
            "held": held,
            "default": default,
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
        "# task_profiles.yaml beats this file. A task type whose default provider",
        "# was not beaten by more than the switch margin and the run-to-run spread",
        "# stays commented out (it keeps the default).",
        f"# Repeats per case: {report.get('meta', {}).get('repeat', 1)}.",
        "preferred_provider:",
    ]
    for task_type in sorted(table):
        row = table[task_type]
        others = ", ".join(
            f"{k} {v['value']:.2f} (success {v['success']:.2f})" for k, v in row["others"].items()
        ) or "no other qualified arm"
        latency = row["latency_p50_ms"]
        paired = row["paired_cases"]
        spread = row.get("spread")
        lines.append(
            f"  # {row['metric']}={row['value']:.2f} (paired={'-' if paired is None else paired}"
            f"{'' if not spread else f', spread={spread:.2f}'}), "
            f"success={row['success']:.2f}, p50={'-' if latency is None else f'{latency:.0f}ms'}, "
            f"cases={row['cases']}; vs {others}"
        )
        held = row.get("held")
        if held:
            lines.append(
                f"  # kept default: {held['best']} led by {held['lead']:.2f}, "
                f"needed > {held['required']:.2f}"
            )
        if row.get("switch", True):
            lines.append(f"  {task_type}: {row['provider']}")
        else:
            lines.append(f"  # {task_type}: {row['provider']}  (default, not written)")
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
