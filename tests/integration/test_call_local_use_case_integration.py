"""Integration test: chay toan bo pipeline CallLocalUseCase voi Ollama that
qua ca 3 budget (L1/L2/L3) va ca 2 output_format. Bi skip tu dong neu
Ollama khong chay (xem conftest).
"""

from __future__ import annotations

import logging

import pytest

from local_ai_gateway.application.call_local_use_case import CallLocalUseCase
from local_ai_gateway.application.dto import CallLocalRequest
from local_ai_gateway.config.settings import AppConfig, load_budget_profiles
from local_ai_gateway.infra.logging import InvocationLogger
from local_ai_gateway.providers.registry import ProviderFactory


@pytest.fixture
def use_case(app_config: AppConfig) -> CallLocalUseCase:
    provider = ProviderFactory(app_config).create()
    budget_profiles = load_budget_profiles()
    return CallLocalUseCase(
        provider=provider,
        config=app_config,
        budget_profiles=budget_profiles,
        invocation_logger=InvocationLogger(logging.getLogger("test.integration.invocation")),
    )


@pytest.mark.parametrize("budget", ["L1", "L2", "L3"])
@pytest.mark.asyncio
async def test_call_local_text_round_trip(use_case: CallLocalUseCase, budget: str):
    result = await use_case.execute(
        CallLocalRequest(task="Tra loi ngan gon: 1 + 1 bang may?", budget=budget)
    )

    assert result.success is True, result.error
    assert result.raw_text.strip() != ""
    assert result.provider == "ollama"
    assert result.budget_used == budget


@pytest.mark.asyncio
async def test_call_local_json_round_trip(use_case: CallLocalUseCase):
    schema = {
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    result = await use_case.execute(
        CallLocalRequest(
            task='Tom tat cau sau trong dung 1 truong "summary": "Con meo ngoi tren ghe."',
            budget="L1",
            output_format="json",
            json_schema=schema,
        )
    )

    assert result.success is True, result.error
    if "json_parse_failed" not in result.warnings:
        assert isinstance(result.content, dict)
