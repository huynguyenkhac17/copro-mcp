"""The server must start from a wheel install (`uv tool install`, `uvx`), not only
from a source checkout: 2026-11-02 a wheel install failed on startup with
"Budgets config file not found: ...\Lib\config\budgets.yaml"."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import yaml

from llm_worker_pool.config import settings

_YAML = ("default.yaml", "budgets.yaml", "task_profiles.yaml", "routing_from_eval.yaml")


def test_yaml_defaults_live_inside_the_package_and_are_declared_as_package_data():
    package = resources.files("llm_worker_pool")
    for name in _YAML:
        assert (package / "config" / name).is_file(), name
    pyproject = Path(settings.__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "config/*.yaml" in data["tool"]["setuptools"]["package-data"]["llm_worker_pool"]


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
