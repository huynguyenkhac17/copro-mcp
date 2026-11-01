"""Skip all integration tests if Ollama is not running or lacks the configured model
(a box that only pulled nomic-embed-text for RAG is not an Ollama test bed)."""

from __future__ import annotations

import httpx
import pytest

from llm_worker_pool.config.settings import AppConfig


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
    pulled = {m.get("name") for m in response.json().get("models", [])}
    if ollama_config.model not in pulled:
        pytest.skip(f"Ollama model {ollama_config.model!r} is not pulled (`ollama pull {ollama_config.model}`)")
