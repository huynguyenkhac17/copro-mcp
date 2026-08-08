"""Skip all integration tests if Ollama is not running."""

from __future__ import annotations

import httpx
import pytest

from local_ai_gateway.config.settings import AppConfig


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return AppConfig()


@pytest.fixture(scope="session", autouse=True)
def _require_ollama_running(app_config: AppConfig) -> None:
    ollama_config = app_config.providers.get("ollama")
    if ollama_config is None:
        pytest.skip("No 'ollama' entry configured in providers")

    try:
        response = httpx.get(f"{ollama_config.endpoint}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama is not ready at {ollama_config.endpoint}: {exc}")
