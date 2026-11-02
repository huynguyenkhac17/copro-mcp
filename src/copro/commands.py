"""The `copro` command: one entry point for the server and its tools.

    copro serve      # the MCP server over stdio (same as `copro-mcp`)
    copro stats      # usage table from the stats snapshot (--json for the raw file)
    copro metrics    # Prometheus exporter over the event log ([metrics] extra)
    copro eval       # eval runner ([eval] extra)

Each subcommand takes its own options: `copro eval --help`.
"""

from __future__ import annotations

import sys

from copro import __version__

_SUBCOMMANDS = {
    "serve": "run the MCP server over stdio",
    "stats": "show usage statistics",
    "metrics": "serve Prometheus metrics",
    "eval": "run the eval dataset",
}


def _usage() -> str:
    lines = ["usage: copro <command> [options]", "", "commands:"]
    lines += [f"  {name:<8} {text}" for name, text in _SUBCOMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0] == "--version":
        print(f"copro {__version__}")
        return 0
    command, rest = args[0], args[1:]
    if command == "serve":
        if rest:
            print("copro serve takes no options", file=sys.stderr)
            return 2
        from copro.server import main as serve

        serve()
        return 0
    if command == "stats":
        from copro.cli import main as stats

        return stats(rest)
    if command == "metrics":
        from copro.metrics_exporter import main as metrics

        metrics(rest)
        return 0
    if command == "eval":
        from copro.evals.runner import main as run_eval

        result = run_eval(rest)
        return result if isinstance(result, int) else 0
    print(f"copro: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
    return 2


def run() -> None:
    raise SystemExit(main())
