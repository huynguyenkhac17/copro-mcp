"""The rename from llm-worker-pool to copro: old names keep working until 0.4.x ends."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from copro import _legacy, commands

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_old_env_prefix_is_copied_to_the_new_one(capsys):
    env = {"LLM_WORKER_POOL_DEFAULT_PROVIDER": "groq", "LLM_WORKER_POOL_RAG__ENABLED": "true"}

    copied = _legacy.copy_legacy_env(env)

    assert env["COPRO_DEFAULT_PROVIDER"] == "groq"
    assert env["COPRO_RAG__ENABLED"] == "true"
    assert copied == ["LLM_WORKER_POOL_DEFAULT_PROVIDER", "LLM_WORKER_POOL_RAG__ENABLED"]
    assert "rename them to COPRO_" in capsys.readouterr().err


def test_a_new_name_already_set_wins_over_the_old_one(capsys):
    env = {"LLM_WORKER_POOL_DEFAULT_PROVIDER": "groq", "COPRO_DEFAULT_PROVIDER": "gemini-lite"}

    assert _legacy.copy_legacy_env(env) == []
    assert env["COPRO_DEFAULT_PROVIDER"] == "gemini-lite"
    assert capsys.readouterr().err == ""


def test_the_old_prefix_reaches_appconfig(monkeypatch):
    from copro.config.settings import AppConfig

    monkeypatch.delenv("COPRO_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("LLM_WORKER_POOL_TIMEOUT_SECONDS", "42")
    _legacy.copy_legacy_env()
    try:
        assert AppConfig().timeout_seconds == 42
    finally:
        monkeypatch.delenv("COPRO_TIMEOUT_SECONDS", raising=False)


def test_the_old_user_dir_is_copied_once_without_lock_files(tmp_path):
    legacy = tmp_path / ".llm_worker_pool"
    legacy.mkdir()
    (legacy / ".env").write_text("GROQ_API_KEY=x\n", encoding="utf-8")
    (legacy / "stats.json").write_text("{}", encoding="utf-8")
    (legacy / "key_state.json.lock").write_text("", encoding="utf-8")
    new = tmp_path / ".copro"

    assert _legacy.migrate_user_dir(new) is True
    assert (new / ".env").read_text(encoding="utf-8") == "GROQ_API_KEY=x\n"
    assert (new / "stats.json").exists()
    assert not (new / "key_state.json.lock").exists()
    assert legacy.is_dir()  # left in place for an old server that may still run

    (legacy / "stats.json").write_text('{"newer": 1}', encoding="utf-8")
    assert _legacy.migrate_user_dir(new) is False
    assert (new / "stats.json").read_text(encoding="utf-8") == "{}"


def test_nothing_to_migrate_without_the_old_dir(tmp_path):
    assert _legacy.migrate_user_dir(tmp_path / ".copro") is False
    assert not (tmp_path / ".copro").exists()


def test_a_failed_copy_leaves_no_half_made_dir(tmp_path, monkeypatch, capsys):
    legacy = tmp_path / ".llm_worker_pool"
    legacy.mkdir()
    new = tmp_path / ".copro"

    def boom(src, dst, ignore=None):
        Path(dst).mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(_legacy.shutil, "copytree", boom)

    assert _legacy.migrate_user_dir(new) is False
    assert not new.exists()
    assert "disk full" in capsys.readouterr().err


def test_old_command_names_still_run_with_a_note(monkeypatch, capsys):
    calls = []
    alias = _legacy._alias("worker-stats", "copro stats", lambda: calls.append(1) or 3)

    with pytest.raises(SystemExit) as exit_info:
        alias()

    assert calls == [1]
    assert exit_info.value.code == 3
    assert "`worker-stats` is the old name of `copro stats`" in capsys.readouterr().err


def test_every_old_and_new_command_is_declared():
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]

    assert scripts["copro"] == "copro.commands:run"
    assert scripts["copro-mcp"] == "copro.server:main"
    for old in ("llm-worker-pool", "worker-stats", "worker-metrics", "worker-eval"):
        module, attr = scripts[old].split(":")
        assert module == "copro._legacy"
        assert callable(getattr(_legacy, attr))


def test_copro_without_a_command_prints_usage(capsys):
    assert commands.main([]) == 0
    out = capsys.readouterr().out
    for name in ("serve", "stats", "metrics", "eval"):
        assert name in out


def test_copro_rejects_an_unknown_command(capsys):
    assert commands.main(["stat"]) == 2
    assert "unknown command 'stat'" in capsys.readouterr().err


def test_copro_stats_passes_its_options_through(tmp_path, capsys):
    assert commands.main(["stats", "--path", str(tmp_path / "none.json")]) == 0
    assert "none.json" in capsys.readouterr().out


def test_copro_version(capsys):
    from copro import __version__

    assert commands.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"copro {__version__}"


def test_package_version_matches_pyproject():
    from copro import __version__

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"]


def test_a_process_losing_the_migration_race_keeps_the_winners_copy(tmp_path, monkeypatch, capsys):
    legacy = tmp_path / ".llm_worker_pool"
    legacy.mkdir()
    (legacy / "stats.json").write_text("{}", encoding="utf-8")
    new = tmp_path / ".copro"
    real_copytree = _legacy.shutil.copytree

    def other_process_wins(src, dst, ignore=None):
        # Another server finishes its migration while this one is still copying.
        real_copytree(src, new)
        (new / "events.jsonl").write_text("written by the winner\n", encoding="utf-8")
        return real_copytree(src, dst, ignore=ignore)

    monkeypatch.setattr(_legacy.shutil, "copytree", other_process_wins)

    assert _legacy.migrate_user_dir(new) is False
    assert (new / "events.jsonl").read_text(encoding="utf-8") == "written by the winner\n"
    assert [p.name for p in tmp_path.iterdir() if "migrating" in p.name] == []
    assert capsys.readouterr().err == ""
