r"""The server must start from a wheel install (`uv tool install`, `uvx`), not only
from a source checkout: 2026-11-02 a wheel install failed on startup with
"Budgets config file not found: ...\Lib\config\budgets.yaml"."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import yaml

from copro.config import settings

_YAML = ("default.yaml", "budgets.yaml", "task_profiles.yaml", "routing_from_eval.yaml")


def test_yaml_defaults_live_inside_the_package_and_are_declared_as_package_data():
    package = resources.files("copro")
    for name in _YAML:
        assert (package / "config" / name).is_file(), name
    pyproject = Path(settings.__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "config/*.yaml" in data["tool"]["setuptools"]["package-data"]["copro"]


def test_shipped_defaults_read_no_files_until_the_user_says_where():
    raw = yaml.safe_load((settings.CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    assert raw["file_access"]["allowed_roots"] == []


def test_env_files_checkout_first_then_user_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "USER_DIR", tmp_path / "user")
    files = settings.env_files()
    assert files[-1] == tmp_path / "user" / ".env"
    assert files[0] == Path(settings.__file__).resolve().parents[3] / ".env"  # tests run from a checkout

    monkeypatch.setattr(settings, "_source_checkout_root", lambda: None)  # an installed wheel
    assert settings.env_files() == (tmp_path / "user" / ".env",)


def test_new_vendors_are_wired_into_the_fallback_chains_and_ascii():
    text = (settings.CONFIG_DIR / "default.yaml").read_text(encoding="utf-8")
    text.encode("ascii")  # the YAML loader uses the platform encoding (cp1252 on Windows)
    raw = yaml.safe_load(text)
    for vendor in ("mistral", "deepseek", "openrouter"):
        assert vendor in raw["providers"]
        assert all(vendor in chain for name, chain in raw["fallbacks"].items() if name != vendor)
    assert raw["default_provider"] in raw["providers"]
    assert raw["rag"]["enabled"] is False  # needs Ollama and the [rag] extra


def test_server_starts_without_the_rag_extra(monkeypatch, caplog):
    import importlib.util

    from copro import server
    from copro.config.settings import AppConfig
    from copro.infra.stats import StatsCollector

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "faiss" else real_find_spec(name, *a))
    config = AppConfig()
    config.rag.enabled = True
    use_case, _ = server._build_use_case(config, StatsCollector())
    assert use_case._rag_indexer is None
    assert "faiss/numpy are not installed" in caplog.text
