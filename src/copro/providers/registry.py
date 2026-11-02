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

from copro.config.settings import AppConfig, ProviderInstanceConfig
from copro.domain.exceptions import ConfigurationError
from copro.domain.interfaces import LLMProvider
from copro.domain.models import ProviderResponse
from copro.infra.cache import InMemoryTTLCache
from copro.infra.circuit_breaker import KeyCircuitBreaker
from copro.infra.logging import get_logger
from copro.infra.event_log import configured_path
from copro.infra.quota_tracker import QuotaTracker
from copro.infra.shared_key_state import SharedKeyState, default_shared_state_path
from copro.providers.caching_provider import CachingProvider
from copro.providers.concurrency_provider import ConcurrencyProvider
from copro.providers.ollama_provider import OllamaProvider
from copro.providers.openai_compatible_provider import OpenAICompatibleProvider

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
    Also used by `copro eval` to build its judge client from a provider entry."""
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
        send_reasoning_effort=config.send_reasoning_effort,
        structured_output=config.structured_output,
        extra_body=config.extra_body,
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
        q = config.quota
        # One file for every provider of this process; each writes under its own name.
        self._shared = (
            SharedKeyState(configured_path(q.shared_state_path, default_shared_state_path()))
            if q.shared_state
            else None
        )
        self.skipped: dict[str, str] = {}
        """Providers left out by the last create_all(), with why (e.g. no API key set)."""

    def _new_circuit_breaker(self, name: str) -> KeyCircuitBreaker:
        # One breaker PER PROVIDER, not one shared by the pool. gemini and gemini-lite
        # use the same API keys, but 503 "model overloaded" and free-tier 429 quotas are
        # per model: with a shared breaker, flash failing 3x on a key quarantined that
        # key for flash-lite too - exactly when the chain needed lite to take over.
        cb = self._config.circuit_breaker
        return KeyCircuitBreaker(
            failure_threshold=cb.failure_threshold,
            quarantine_seconds=cb.quarantine_seconds,
            enabled=cb.enabled,
            shared=self._shared,
            scope=name,
        )

    def _new_quota_tracker(self, name: str) -> QuotaTracker:
        # Per provider for the same reason as the breaker: quotas are per model.
        q = self._config.quota
        return QuotaTracker(
            enabled=q.enabled,
            safety_margin_tokens=q.safety_margin_tokens,
            timeouts_before_block=q.timeouts_before_block,
            timeout_block_seconds=q.timeout_block_seconds,
            daily_quota_block_seconds=q.daily_quota_block_seconds,
            shared=self._shared,
            scope=name,
        )

    def create_all(self) -> dict[str, LLMProvider]:
        providers: dict[str, LLMProvider] = {}
        self.skipped = {}
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
                    circuit_breaker=self._new_circuit_breaker(name),
                    quota_tracker=self._new_quota_tracker(name),
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
                    self.skipped[name] = str(exc)
                    continue
            except ConfigurationError as exc:
                # A provider missing its credentials (e.g. unset api_key_env) should not
                # take down the whole server - skip it, other providers stay usable.
                _logger.warning("Skipping provider '%s': %s", name, exc)
                self.skipped[name] = str(exc)
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
