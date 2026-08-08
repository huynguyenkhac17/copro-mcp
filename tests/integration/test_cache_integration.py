"""Integration test: cache response and read files directly via file_paths,
run with real Ollama. Automatically skipped if Ollama is not running (see conftest).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from local_ai_gateway.application.call_local_use_case import CallLocalUseCase
from local_ai_gateway.application.dto import CallLocalRequest
from local_ai_gateway.config.settings import AppConfig, load_budget_profiles
from local_ai_gateway.domain.file_context import FileContextResolver
from local_ai_gateway.infra.logging import InvocationLogger
from local_ai_gateway.providers.registry import ProviderFactory


@pytest.mark.asyncio
async def test_repeated_call_with_same_file_hits_cache(app_config: AppConfig, tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("The black cat is sleeping on the sofa.")

    providers = ProviderFactory(app_config).create_all()
    use_case = CallLocalUseCase(
        providers=providers,
        default_provider=app_config.default_provider,
        config=app_config,
        budget_profiles=load_budget_profiles(),
        invocation_logger=InvocationLogger(logging.getLogger("test.integration.cache")),
        file_context_resolver=FileContextResolver(allowed_roots=[tmp_path]),
    )

    request = CallLocalRequest(
        task="Summarize the file content in exactly 1 short sentence.",
        budget="L1",
        file_paths=[str(f)],
    )

    first = await use_case.execute(request)
    second = await use_case.execute(request)

    assert first.success is True, first.error
    assert second.success is True, second.error
    assert first.cached is False
    assert second.cached is True
    assert second.latency_ms == 0.0
