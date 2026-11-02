"""Integration test: calls real Ollama with the model configured in default.yaml
(currently qwen2.5-coder:7b). Skipped automatically if Ollama is not running (see conftest).
"""

from __future__ import annotations

import pytest

from copro.config.settings import AppConfig
from copro.domain.models import GenerationOptions
from copro.domain.output_parser import OutputParser
from copro.providers.ollama_provider import OllamaProvider


@pytest.fixture
def provider(app_config: AppConfig) -> OllamaProvider:
    ollama_config = app_config.providers["ollama"]
    return OllamaProvider(endpoint=ollama_config.endpoint, model=ollama_config.model)


@pytest.mark.asyncio
async def test_generate_plain_text(provider: OllamaProvider):
    options = GenerationOptions(
        system_prompt="Answer with exactly 1 word, do not explain.",
        temperature=0.0,
        num_predict=16,
        num_ctx=2048,
        seed=42,
        # think=false is a no-op for non-thinking models like qwen2.5-coder, and
        # required for thinking models so a small num_predict isn't consumed by reasoning.
        think=False,
    )

    response = await provider.generate("What is the capital of Vietnam?", options)

    assert response.raw_text.strip() != ""
    assert response.provider_name == "ollama"
    assert response.model_name == provider.model_name
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_generate_json_format_round_trip(provider: OllamaProvider):
    options = GenerationOptions(
        system_prompt=(
            "You MUST reply with exactly 1 JSON object in the format "
            '{"answer": <number>}, without any other text.'
        ),
        temperature=0.0,
        num_predict=32,
        num_ctx=2048,
        seed=42,
        format="json",
        think=False,
    )

    response = await provider.generate("What is 2 + 2?", options)
    parsed = OutputParser().parse(response.raw_text)

    assert "json_parse_failed" not in parsed.warnings
    assert isinstance(parsed.content, dict)
    assert "answer" in parsed.content
