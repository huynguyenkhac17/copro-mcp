"""Integration test: goi Ollama that voi model cau hinh trong default.yaml
(hien la qwen3.5:9b). Bi skip tu dong neu Ollama khong chay (xem conftest).
"""

from __future__ import annotations

import pytest

from local_ai_gateway.config.settings import AppConfig
from local_ai_gateway.domain.models import GenerationOptions
from local_ai_gateway.domain.output_parser import OutputParser
from local_ai_gateway.providers.ollama_provider import OllamaProvider


@pytest.fixture
def provider(app_config: AppConfig) -> OllamaProvider:
    return OllamaProvider(endpoint=app_config.ollama.endpoint, model=app_config.ollama.model)


@pytest.mark.asyncio
async def test_generate_plain_text(provider: OllamaProvider):
    options = GenerationOptions(
        system_prompt="Tra loi bang dung 1 tu, khong giai thich.",
        temperature=0.0,
        num_predict=16,
        num_ctx=2048,
        seed=42,
        # qwen3.5 la model thinking - tat thinking de tra loi truc tiep,
        # neu khong num_predict nho se bi tieu het cho phan suy nghi.
        think=False,
    )

    response = await provider.generate("Thu do cua Viet Nam la gi?", options)

    assert response.raw_text.strip() != ""
    assert response.provider_name == "ollama"
    assert response.model_name == provider.model_name
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_generate_json_format_round_trip(provider: OllamaProvider):
    options = GenerationOptions(
        system_prompt=(
            "Ban PHAI tra loi bang duy nhat 1 JSON object dang "
            '{"answer": <so>}, khong kem chu nao khac.'
        ),
        temperature=0.0,
        num_predict=32,
        num_ctx=2048,
        seed=42,
        format="json",
        think=False,
    )

    response = await provider.generate("2 + 2 bang may?", options)
    parsed = OutputParser().parse(response.raw_text)

    assert "json_parse_failed" not in parsed.warnings
    assert isinstance(parsed.content, dict)
    assert "answer" in parsed.content
