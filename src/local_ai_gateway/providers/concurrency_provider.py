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

from local_ai_gateway.domain.interfaces import LLMProvider
from local_ai_gateway.domain.models import GenerationOptions, ProviderResponse


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

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        async with self._semaphore:
            return await self._inner.generate(prompt, options)

    async def aclose(self) -> None:
        await self._inner.aclose()
