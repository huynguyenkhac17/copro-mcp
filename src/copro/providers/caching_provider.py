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
from typing import Any

from copro.domain.interfaces import LLMProvider
from copro.domain.models import GenerationOptions, ProviderResponse
from copro.infra.cache import InMemoryTTLCache


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

    @property
    def thinking_option(self) -> str | None:
        return getattr(self._inner, "thinking_option", "think")

    def unready_reason(self, prompt: str, options: GenerationOptions) -> str | None:
        # A cached answer costs no quota, so a saturated provider is still "ready" for it.
        if self._cache.get(self._make_key(prompt, options)) is not None:
            return None
        check = getattr(self._inner, "unready_reason", None)
        return check(prompt, options) if check else None

    def breaker_stats(self) -> dict[str, Any] | None:
        check = getattr(self._inner, "breaker_stats", None)
        return check() if check else None

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        key = self._make_key(prompt, options)

        cached = self._cache.get(key)
        if cached is not None:
            return cached.model_copy(update={"latency_ms": 0.0, "from_cache": True})

        response = await self._inner.generate(prompt, options)
        self._cache.set(key, response)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()

    def _make_key(self, prompt: str, options: GenerationOptions) -> str:
        payload = {
            "provider": self._inner.provider_name,
            "model": self._inner.model_name,
            "prompt": prompt,
            "options": options.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
