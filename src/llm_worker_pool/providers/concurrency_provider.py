"""Decorator (composition) wrapping any LLMProvider to limit how many
generate() calls run in parallel for that specific provider instance.

Local models (Ollama) are VRAM-bound and need a low limit; cloud API
providers can usually tolerate much higher parallelism. Configuring this
per-provider (instead of one global semaphore) lets map-reduce chunks
targeting a cloud provider run concurrently while local calls stay
sequential.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_worker_pool.domain.interfaces import LLMProvider
from llm_worker_pool.domain.models import GenerationOptions, ProviderResponse


class ConcurrencyProvider:
    def __init__(self, inner: LLMProvider, max_parallel: int) -> None:
        self._inner = inner
        self._semaphore = asyncio.Semaphore(max_parallel)

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def thinking_option(self) -> str | None:
        return getattr(self._inner, "thinking_option", "think")

    def unready_reason(self, prompt: str, options: GenerationOptions) -> str | None:
        check = getattr(self._inner, "unready_reason", None)
        return check(prompt, options) if check else None

    def breaker_stats(self) -> dict[str, Any] | None:
        check = getattr(self._inner, "breaker_stats", None)
        return check() if check else None

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        async with self._semaphore:
            return await self._inner.generate(prompt, options)

    async def aclose(self) -> None:
        await self._inner.aclose()
