"""Compatibility with the project's old name, llm-worker-pool (until 0.4.x ends).

Three things carried the old name on installs that predate 0.4.0:
- settings in LLM_WORKER_POOL_* environment variables (and in .env files);
- the per-user directory ~/.llm_worker_pool (keys .env, stats, event log, key state);
- the commands llm-worker-pool, worker-stats, worker-metrics and worker-eval.

All three keep working, each with a note on stderr saying what to rename. stderr, not
stdout: the server speaks MCP over stdout.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path

LEGACY_ENV_PREFIX = "LLM_WORKER_POOL_"
ENV_PREFIX = "COPRO_"
LEGACY_USER_DIR_NAME = ".llm_worker_pool"


def _warn(message: str) -> None:
    print(f"copro: {message}", file=sys.stderr)


def copy_legacy_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy each LLM_WORKER_POOL_X to COPRO_X unless COPRO_X is already set, so the new
    name always wins. Returns the old names that were copied."""
    env = os.environ if environ is None else environ
    copied = []
    for name in sorted(env):
        if not name.startswith(LEGACY_ENV_PREFIX):
            continue
        new_name = ENV_PREFIX + name[len(LEGACY_ENV_PREFIX):]
        if new_name not in env:
            env[new_name] = env[name]
            copied.append(name)
    if copied:
        _warn(
            f"read {len(copied)} setting(s) from the old LLM_WORKER_POOL_ prefix "
            f"({', '.join(copied)}); rename them to COPRO_ (the old prefix goes after 0.4.x)"
        )
    return copied


def migrate_user_dir(user_dir: Path, legacy_dir: Path | None = None) -> bool:
    """Copy ~/.llm_worker_pool into ~/.copro the first time, when only the old one
    exists. Copy, not move: a server of the old version may still be running and
    writing there. Lock files are left behind. Returns True when it copied."""
    legacy = legacy_dir if legacy_dir is not None else user_dir.parent / LEGACY_USER_DIR_NAME
    if user_dir.exists() or not legacy.is_dir():
        return False
    # Copy into a directory of our own, then rename it into place in one step: several
    # processes may start at once (Claude Code, the metrics exporter), and none of them
    # may ever see, or delete, a half-copied ~/.copro made by another.
    staging = user_dir.with_name(f"{user_dir.name}.migrating-{os.getpid()}")
    try:
        shutil.copytree(legacy, staging, ignore=shutil.ignore_patterns("*.lock"))
        os.rename(staging, user_dir)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if user_dir.exists():
            return False  # another process finished the copy first
        # Never fatal: the server starts with empty stats, and keys can still come from
        # the real environment.
        _warn(f"could not copy {legacy} to {user_dir}: {exc}")
        return False
    _warn(f"copied {legacy} to {user_dir}; the old directory is no longer read")
    return True


def _alias(old: str, new: str, target: Callable[[], object]) -> Callable[[], None]:
    def run() -> None:
        _warn(f"`{old}` is the old name of `{new}` and goes after 0.4.x")
        result = target()
        if isinstance(result, int):
            raise SystemExit(result)

    return run


def _server() -> object:
    from copro.server import main

    return main()


def _stats() -> object:
    from copro.cli import main

    return main()


def _metrics() -> object:
    from copro.metrics_exporter import main

    return main()


def _eval() -> object:
    from copro.evals.runner import main

    return main()


llm_worker_pool = _alias("llm-worker-pool", "copro-mcp", _server)
worker_stats = _alias("worker-stats", "copro stats", _stats)
worker_metrics = _alias("worker-metrics", "copro metrics", _metrics)
worker_eval = _alias("worker-eval", "copro eval", _eval)
