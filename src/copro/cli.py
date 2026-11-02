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
    lines = ["", "  CIRCUIT BREAKER (key theo vị trí trong cấu hình)"]
    rows = []
    for provider, keys in sorted(stats.circuit_breakers.items()):
        for label, k in keys.items():
            if not k.total_failures:
                continue
            # The snapshot is written per request, so is_quarantined may be stale;
            # the wall-clock deadline is not.
            left = k.quarantined_until - now
            state = f"khóa còn {_fmt_duration(left)}" if left > 0 else "mở"
            rows.append(
                f"  {provider + '/' + label:<22}{k.total_failures:>6}{k.total_quarantines:>8}"
                f"{k.total_successes:>8}  {state}"
            )
    if not rows:
        lines.append("  Chưa key nào lỗi.")
        return lines
    lines.append(f"  {'':<22}{'lỗi':>6}{'bị khóa':>8}{'ok':>8}  trạng thái")
    return lines + rows


def render(stats: GatewayStats, now: float | None = None) -> str:
    now = time.time() if now is None else now
    lines: list[str] = []
    add = lines.append

    add("=" * 66)
    add("COPRO — thống kê")
    add("=" * 66)
    add(f"  Uptime           {_fmt_duration(stats.uptime_seconds)}")
    add(
        f"  Requests         {stats.total_requests}  "
        f"(thành công {stats.total_success}, lỗi {stats.total_failures})"
    )
    add(f"  Cache hit        {stats.total_cached}")

    if stats.total_truncated:
        add(
            f"  ⚠ Bị cắt output  {stats.total_truncated}  "
            "← câu trả lời không trọn vẹn, nâng budget hoặc chia nhỏ task"
        )
    else:
        add("  Bị cắt output    0")

    tokens = stats.tokens
    add("")
    add("  TOKEN ĐÃ TIÊU (không tính cache hit)")
    add(f"    prompt         {_fmt_int(tokens.prompt):>12}")
    add(f"    completion     {_fmt_int(tokens.completion):>12}")

    if tokens.total:
        share = tokens.thinking / tokens.total * 100
        flag = "  ← đáng chú ý" if share >= 40 else ""
        add(f"    thinking       {_fmt_int(tokens.thinking):>12}  ({share:.0f}% tổng){flag}")
    else:
        add(f"    thinking       {_fmt_int(tokens.thinking):>12}")

    add(f"    tổng           {_fmt_int(tokens.total):>12}")
    add(f"  Token tiết kiệm nhờ cache  {_fmt_int(stats.total_tokens_saved_by_cache)}")

    if not stats.providers:
        add("")
        add("  Chưa có lời gọi nào được ghi nhận.")
        return "\n".join(lines)

    add("")
    add("  THEO PROVIDER / BUDGET")
    add(f"  {'':<22}{'gọi':>6}{'lỗi':>6}{'cắt':>6}{'token':>12}{'ms/lần':>10}")
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
    encode Vietnamese and would crash on the first accented character. Reconfigure
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
        prog="copro stats", description="Xem thống kê của Copro."
    )
    parser.add_argument("--json", action="store_true", help="in nguyên snapshot JSON")
    parser.add_argument(
        "--path",
        type=Path,
        default=default_snapshot_path(),
        help="đường dẫn file stats (mặc định: ~/.copro/stats.json)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        # Not an error worth a stack trace: the server simply has not run since the
        # snapshot feature was added, or has never been started at all.
        print(
            f"Chưa có file thống kê tại {args.path}\n"
            "MCP server chưa chạy lần nào kể từ khi bật tính năng ghi snapshot.\n"
            "Gọi delegate_task một lần rồi thử lại."
        )
        return 0

    try:
        raw = json.loads(args.path.read_text(encoding="utf-8"))
        stats = GatewayStats.model_validate(raw)
    except (OSError, ValueError) as exc:
        print(f"Không đọc được file thống kê {args.path}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(raw, ensure_ascii=False, indent=2))
    else:
        print(render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
