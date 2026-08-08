"""Registry: maps provider type (from config) to the LLMProvider
implementing class.

To add a new provider type (vLLM, OpenAI, Gemini, ...):
    1. Write a class in providers/ implementing LLMProvider.generate(...)
    2. Write a builder function and register it in _REGISTRY
    3. Add the corresponding config block in default.yaml

No changes needed in application/domain/mcp layers.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from local_ai_gateway.config.settings import AppConfig, ProviderInstanceConfig
from local_ai_gateway.domain.exceptions import ConfigurationError
from local_ai_gateway.domain.interfaces import LLMProvider
from local_ai_gateway.domain.models import ProviderResponse
from local_ai_gateway.infra.cache import InMemoryTTLCache
from local_ai_gateway.infra.logging import get_logger
from local_ai_gateway.providers.caching_provider import CachingProvider
from local_ai_gateway.providers.ollama_provider import OllamaProvider
from local_ai_gateway.providers.openai_compatible_provider import OpenAICompatibleProvider

_logger = get_logger(__name__)

ProviderBuilder = Callable[[str, ProviderInstanceConfig], LLMProvider]

def build_ollama(name: str, config: ProviderInstanceConfig) -> LLMProvider:
    return OllamaProvider(
        endpoint=config.endpoint,
        model=config.model,
    )

def build_openai_compatible(name: str, config: ProviderInstanceConfig) -> LLMProvider:
    api_key = ""
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ConfigurationError(
                f"Provider '{name}' requires environment variable '{config.api_key_env}' which is not set or empty."
            )
    return OpenAICompatibleProvider(
        provider_name=name,
        endpoint=config.endpoint,
        model=config.model,
        api_key=api_key,
    )

_REGISTRY: dict[str, ProviderBuilder] = {
    "ollama": build_ollama,
    "openai_compatible": build_openai_compatible,
}


def register_provider(type_name: str, builder: ProviderBuilder) -> None:
    _REGISTRY[type_name] = builder


class ProviderFactory:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def create_all(self) -> dict[str, LLMProvider]:
        providers: dict[str, LLMProvider] = {}
        for name, instance_config in self._config.providers.items():
            builder = _REGISTRY.get(instance_config.type)
            if builder is None:
                available = ", ".join(sorted(_REGISTRY)) or "(no registered provider types)"
                raise ConfigurationError(
                    f"Provider '{name}' has unknown type '{instance_config.type}'. "
                    f"Available types: {available}"
                )

            try:
                provider = builder(name, instance_config)
            except ConfigurationError as exc:
                # A provider missing its credentials (e.g. unset api_key_env) should not
                # take down the whole server - skip it, other providers stay usable.
                _logger.warning("Skipping provider '%s': %s", name, exc)
                continue

            if self._config.cache.enabled:
                cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(
                    max_entries=self._config.cache.max_entries,
                    ttl_seconds=self._config.cache.ttl_seconds,
                )
                provider = CachingProvider(provider, cache)

            providers[name] = provider
            
        return providers
