"""Integration test: run full CallLocalUseCase pipeline with real Ollama
across 3 budgets (L1/L2/L3) and 2 output_formats. Automatically skipped if
Ollama is not running (see conftest).
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
    providers = ProviderFactory(app_config).create_all()
    budget_profiles = load_budget_profiles()
    return CallLocalUseCase(
        providers=providers,
        default_provider=app_config.default_provider,
        config=app_config,
        budget_profiles=budget_profiles,
        invocation_logger=InvocationLogger(logging.getLogger("test.integration.invocation")),
    )


@pytest.mark.parametrize("budget", ["L1", "L2", "L3"])
@pytest.mark.asyncio
async def test_call_local_text_round_trip(use_case: CallLocalUseCase, budget: str):
    result = await use_case.execute(
        CallLocalRequest(task="Answer briefly: what is 1 + 1?", budget=budget, provider="ollama")
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
            task='Summarize the following sentence in exactly 1 "summary" field: "The cat sits on the chair."',
            budget="L1",
            output_format="json",
            json_schema=schema,
        )
    )

    assert result.success is True, result.error
    if "json_parse_failed" not in result.warnings:
        assert isinstance(result.content, dict)
