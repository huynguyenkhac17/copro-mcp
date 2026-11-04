"""Command-line view over the gateway's stats snapshot.

The MCP server keeps its counters in memory, so reading them through the
`get_gateway_stats` tool means paying for a model round-trip just to look at
numbers. The server mirrors every update to a JSON file; this reads that file
directly, which is what makes `/worker` in Claude Code cost nothing.

    copro stats            # human-readable table
    copro stats --json     # raw snapshot, for piping into jq
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from copro._legacy import migrate_user_dir
from copro.infra.stats import GatewayStats, default_snapshot_path


def _fmt_int(value: int) -> str:
    """Thousands separators, because six-digit token counts are unreadable raw."""
    return f"{value:,}"


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _render_breakers(stats: GatewayStats, now: float) -> list[str]:
    """Only keys that have failed at least once: a healthy pool prints one line."""
    lines = ["", "  CIRCUIT BREAKERS (keys by position in the config)"]
    rows = []
    for provider, keys in sorted(stats.circuit_breakers.items()):
        for label, k in keys.items():
            if not k.total_failures:
                continue
            # The snapshot is written per request, so is_quarantined may be stale;
            # the wall-clock deadline is not.
            left = k.quarantined_until - now
            state = f"locked for {_fmt_duration(left)}" if left > 0 else "open"
            rows.append(
                f"  {provider + '/' + label:<22}{k.total_failures:>6}{k.total_quarantines:>8}"
                f"{k.total_successes:>8}  {state}"
            )
    if not rows:
        lines.append("  No key has failed.")
        return lines
    lines.append(f"  {'':<22}{'fails':>6}{'locks':>8}{'ok':>8}  state")
    return lines + rows


def render(stats: GatewayStats, now: float | None = None) -> str:
    now = time.time() if now is None else now
    lines: list[str] = []
    add = lines.append

    add("=" * 66)
    add("COPRO - usage")
    add("=" * 66)
    add(f"  Uptime           {_fmt_duration(stats.uptime_seconds)}")
    add(
        f"  Requests         {stats.total_requests}  "
        f"(ok {stats.total_success}, failed {stats.total_failures})"
    )
    add(f"  Cache hit        {stats.total_cached}")

    if stats.total_truncated:
        add(
            f"  ! Truncated      {stats.total_truncated}  "
            "<- incomplete answers: raise the budget or split the task"
        )
    else:
        add("  Truncated        0")

    tokens = stats.tokens
    add("")
    add("  TOKENS SPENT (cache hits excluded)")
    add(f"    prompt         {_fmt_int(tokens.prompt):>12}")
    add(f"    completion     {_fmt_int(tokens.completion):>12}")

    if tokens.total:
        share = tokens.thinking / tokens.total * 100
        flag = "  <- high" if share >= 40 else ""
        add(f"    thinking       {_fmt_int(tokens.thinking):>12}  ({share:.0f}% of total){flag}")
    else:
        add(f"    thinking       {_fmt_int(tokens.thinking):>12}")

    add(f"    total          {_fmt_int(tokens.total):>12}")
    add(f"  Tokens saved by cache  {_fmt_int(stats.total_tokens_saved_by_cache)}")

    if not stats.providers:
        add("")
        add("  No calls recorded yet.")
        return "\n".join(lines)

    add("")
    add("  BY PROVIDER / BUDGET")
    add(f"  {'':<22}{'ok':>6}{'fail':>6}{'trunc':>6}{'tokens':>12}{'ms/call':>10}")
    for name, p in sorted(stats.providers.items()):
        avg = p.total_latency_ms / p.success_count if p.success_count else 0.0
        add(
            f"  {name:<22}{p.success_count:>6}{p.failure_count:>6}"
            f"{p.truncated_count:>6}{_fmt_int(p.tokens.total):>12}{avg:>10.0f}"
        )
        for budget, b in sorted(p.by_budget.items()):
            b_avg = b.total_latency_ms / b.success_count if b.success_count else 0.0
            add(
                f"    └ {budget:<18}{b.success_count:>6}{b.failure_count:>6}"
                f"{b.truncated_count:>6}{_fmt_int(b.tokens.total):>12}{b_avg:>10.0f}"
            )

    if stats.circuit_breakers:
        lines.extend(_render_breakers(stats, now))

    return "\n".join(lines)


def _force_utf8_stdout() -> None:
    """Windows consoles default to a legacy codepage (cp1252 here), which cannot
    encode the box-drawing characters and would crash on the first one. Reconfigure
    rather than rely on the caller exporting PYTHONIOENCODING."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    # Reads the stats file without loading AppConfig, so migrate here too.
    migrate_user_dir(default_snapshot_path().parent)
    parser = argparse.ArgumentParser(
        prog="copro stats", description="Show Copro usage statistics."
    )
    parser.add_argument("--json", action="store_true", help="print the raw JSON snapshot")
    parser.add_argument(
        "--path",
        type=Path,
        default=default_snapshot_path(),
        help="stats file (default: ~/.copro/stats.json)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        # Not an error worth a stack trace: the server simply has not run since the
        # snapshot feature was added, or has never been started at all.
        print(
            f"No stats file at {args.path}\n"
            "The MCP server has not run yet (it writes this file on every call).\n"
            "Call delegate_task once, then try again."
        )
        return 0

    try:
        raw = json.loads(args.path.read_text(encoding="utf-8"))
        stats = GatewayStats.model_validate(raw)
    except (OSError, ValueError) as exc:
        print(f"Cannot read the stats file {args.path}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(raw, ensure_ascii=False, indent=2))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
