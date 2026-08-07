"""Decorator (composition) wrapping any LLMProvider to cache response
by (provider, model, final prompt, options). Does not change application/
domain/mcp layers - it's just an additional wrap step in ProviderFactory.

The prompt hashed here is the ALREADY built prompt (after PromptBuilder +
FileContextResolver), so the cache automatically invalidates when file/context
content changes - no need to track mtime separately.
"""

from __future__ import annotations

import hashlib
import json

from local_ai_gateway.domain.interfaces import LLMProvider
from local_ai_gateway.domain.models import GenerationOptions, ProviderResponse
from local_ai_gateway.infra.cache import InMemoryTTLCache


class CachingProvider:
    def __init__(self, inner: LLMProvider, cache: InMemoryTTLCache[ProviderResponse]) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        key = self._make_key(prompt, options)

        cached = self._cache.get(key)
        if cached is not None:
            return cached.model_copy(update={"latency_ms": 0.0, "from_cache": True})

        response = await self._inner.generate(prompt, options)
        self._cache.set(key, response)
        return response

    def _make_key(self, prompt: str, options: GenerationOptions) -> str:
        payload = {
            "provider": self._inner.provider_name,
            "model": self._inner.model_name,
            "prompt": prompt,
            "options": options.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
