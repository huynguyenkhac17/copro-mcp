"""Registry: anh xa ten provider (tu config) sang class implement
LLMProvider.

Them provider moi (vLLM, OpenAI, Gemini, ...):
    1. Viet 1 class trong providers/ implement LLMProvider.generate(...)
    2. Dang ky class do o _REGISTRY (hoac goi register_provider tu ngoai)
    3. Them block config tuong ung trong default.yaml

Khong can sua application/domain/mcp layer.
"""

from __future__ import annotations

from collections.abc import Callable

from local_ai_gateway.config.settings import AppConfig
from local_ai_gateway.domain.exceptions import ConfigurationError
from local_ai_gateway.domain.interfaces import LLMProvider
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
            available = ", ".join(sorted(_REGISTRY)) or "(khong co provider nao dang ky)"
            raise ConfigurationError(
                f"Provider '{self._config.provider}' chua duoc dang ky. "
                f"Cac provider hien co: {available}"
            )
        return builder(self._config)
