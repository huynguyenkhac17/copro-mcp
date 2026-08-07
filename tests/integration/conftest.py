"""Skip toan bo integration test neu khong co Ollama dang chay that."""

from __future__ import annotations

import httpx
import pytest

from local_ai_gateway.config.settings import AppConfig


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return AppConfig()


@pytest.fixture(scope="session", autouse=True)
def _require_ollama_running(app_config: AppConfig) -> None:
    try:
        response = httpx.get(f"{app_config.ollama.endpoint}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama khong san sang tai {app_config.ollama.endpoint}: {exc}")
