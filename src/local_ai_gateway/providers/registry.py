"""Registry: maps provider name (from config) to the LLMProvider
implementing class.

To add a new provider (vLLM, OpenAI, Gemini, ...):
    1. Write a class in providers/ implementing LLMProvider.generate(...)
    2. Register that class in _REGISTRY (or call register_provider externally)
    3. Add the corresponding config block in default.yaml

No changes needed in application/domain/mcp layers.
"""

from __future__ import annotations

from collections.abc import Callable

from local_ai_gateway.config.settings import AppConfig
from local_ai_gateway.domain.exceptions import ConfigurationError
from local_ai_gateway.domain.interfaces import LLMProvider
from local_ai_gateway.domain.models import ProviderResponse
from local_ai_gateway.infra.cache import InMemoryTTLCache
from local_ai_gateway.providers.caching_provider import CachingProvider
from local_ai_gateway.providers.ollama_provider import OllamaProvider

ProviderBuilder = Callable[[AppConfig], LLMProvider]

_REGISTRY: dict[str, ProviderBuilder] = {
    "ollama": lambda config: OllamaProvider(
        endpoint=config.ollama.endpoint,
        model=config.ollama.model,
    ),
}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    _REGISTRY[name] = builder


class ProviderFactory:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def create(self) -> LLMProvider:
        builder = _REGISTRY.get(self._config.provider)
        if builder is None:
            available = ", ".join(sorted(_REGISTRY)) or "(no registered providers)"
            raise ConfigurationError(
                f"Provider '{self._config.provider}' is not registered. "
                f"Available providers: {available}"
            )
        provider = builder(self._config)

        if self._config.cache.enabled:
            cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(
                max_entries=self._config.cache.max_entries,
                ttl_seconds=self._config.cache.ttl_seconds,
            )
            provider = CachingProvider(provider, cache)

        return provider
