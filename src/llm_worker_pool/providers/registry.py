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

from llm_worker_pool.config.settings import AppConfig, ProviderInstanceConfig
from llm_worker_pool.domain.exceptions import ConfigurationError
from llm_worker_pool.domain.interfaces import LLMProvider
from llm_worker_pool.domain.models import ProviderResponse
from llm_worker_pool.infra.cache import InMemoryTTLCache
from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker
from llm_worker_pool.infra.logging import get_logger
from llm_worker_pool.infra.quota_tracker import QuotaTracker
from llm_worker_pool.providers.caching_provider import CachingProvider
from llm_worker_pool.providers.concurrency_provider import ConcurrencyProvider
from llm_worker_pool.providers.ollama_provider import OllamaProvider
from llm_worker_pool.providers.openai_compatible_provider import OpenAICompatibleProvider

_logger = get_logger(__name__)

# Local models are VRAM-bound (default 1 = sequential); cloud/API-based
# provider types default to a higher parallelism since they have no such
# constraint. Explicit `concurrency:` in config always wins over this.
_DEFAULT_CONCURRENCY_BY_TYPE: dict[str, int] = {"ollama": 1}
_DEFAULT_CONCURRENCY_FALLBACK = 5

ProviderBuilder = Callable[..., LLMProvider]

def build_ollama(name: str, config: ProviderInstanceConfig, **kwargs) -> LLMProvider:
    return OllamaProvider(
        endpoint=config.endpoint,
        model=config.model,
    )

def _api_key_env_vars(config: ProviderInstanceConfig) -> list[str]:
    return [*(config.api_key_envs or []), *([config.api_key_env] if config.api_key_env else [])]


def resolve_api_keys(config: ProviderInstanceConfig) -> list[str]:
    """Distinct non-empty key values named by api_key_envs/api_key_env, in order.
    Also used by `worker-eval` to build its judge client from a provider entry."""
    api_keys: list[str] = []
    for env_var in _api_key_env_vars(config):
        value = os.environ.get(env_var, "")
        if value and value not in api_keys:
            api_keys.append(value)
    return api_keys


def build_openai_compatible(
    name: str,
    config: ProviderInstanceConfig,
    circuit_breaker: KeyCircuitBreaker | None = None,
    quota_tracker: QuotaTracker | None = None,
    **kwargs,
) -> LLMProvider:
    env_vars = _api_key_env_vars(config)
    api_keys = resolve_api_keys(config)

    if not api_keys:
        checked = ", ".join(env_vars) or "(none configured)"
        raise ConfigurationError(
            f"Provider '{name}' requires at least one of these environment variables "
            f"to be set and non-empty: {checked}"
        )

    return OpenAICompatibleProvider(
        provider_name=name,
        endpoint=config.endpoint,
        model=config.model,
        api_keys=api_keys,
        circuit_breaker=circuit_breaker,
        per_key_concurrency=config.per_key_concurrency,
        max_prompt_tokens=config.max_prompt_tokens,
        quota_tracker=quota_tracker,
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

    def _new_circuit_breaker(self) -> KeyCircuitBreaker:
        # One breaker PER PROVIDER, not one shared by the pool. gemini and gemini-lite
        # use the same API keys, but 503 "model overloaded" and free-tier 429 quotas are
        # per model: with a shared breaker, flash failing 3x on a key quarantined that
        # key for flash-lite too - exactly when the chain needed lite to take over.
        cb = self._config.circuit_breaker
        return KeyCircuitBreaker(
            failure_threshold=cb.failure_threshold,
            quarantine_seconds=cb.quarantine_seconds,
            enabled=cb.enabled,
        )

    def _new_quota_tracker(self) -> QuotaTracker:
        # Per provider for the same reason as the breaker: quotas are per model.
        q = self._config.quota
        return QuotaTracker(
            enabled=q.enabled,
            safety_margin_tokens=q.safety_margin_tokens,
            timeouts_before_block=q.timeouts_before_block,
            timeout_block_seconds=q.timeout_block_seconds,
            daily_quota_block_seconds=q.daily_quota_block_seconds,
        )

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
                provider = builder(
                    name,
                    instance_config,
                    circuit_breaker=self._new_circuit_breaker(),
                    quota_tracker=self._new_quota_tracker(),
                )
            except TypeError:
                # A custom builder that does not take these kwargs runs without either -
                # say so, rather than silently losing key quarantine and quota checks.
                _logger.warning(
                    "Provider '%s': builder takes no circuit_breaker/quota_tracker; "
                    "running without key quarantine and quota checks",
                    name,
                )
                try:
                    provider = builder(name, instance_config)
                except ConfigurationError as exc:
                    _logger.warning("Skipping provider '%s': %s", name, exc)
                    continue
            except ConfigurationError as exc:
                # A provider missing its credentials (e.g. unset api_key_env) should not
                # take down the whole server - skip it, other providers stay usable.
                _logger.warning("Skipping provider '%s': %s", name, exc)
                continue

            if instance_config.per_key_concurrency is not None:
                # Per-key concurrency is handled directly by the provider instance
                pass
            else:
                max_parallel = instance_config.concurrency
                if max_parallel is None:
                    max_parallel = _DEFAULT_CONCURRENCY_BY_TYPE.get(
                        instance_config.type, _DEFAULT_CONCURRENCY_FALLBACK
                    )
                provider = ConcurrencyProvider(provider, max_parallel)

            if self._config.cache.enabled:
                cache: InMemoryTTLCache[ProviderResponse] = InMemoryTTLCache(
                    max_entries=self._config.cache.max_entries,
                    ttl_seconds=self._config.cache.ttl_seconds,
                )
                provider = CachingProvider(provider, cache)

            providers[name] = provider
            
        return providers
