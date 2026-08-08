"""Interface (Protocol) that every provider must follow.

This is the only boundary between the application layer and specific LLM backends
(Ollama, vLLM, OpenAI, Gemini, ...). The application layer only knows
LLMProvider, and never imports a specific provider directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from local_ai_gateway.domain.models import GenerationOptions, ProviderResponse


@runtime_checkable
class LLMProvider(Protocol):
    """A backend generating text from prompt + generation options.

    Transient errors (connection failures, timeouts) must be raised as
    ProviderUnavailableError / ProviderTimeoutError (see domain.exceptions),
    without leaking HTTP/SDK library specific exceptions outside the provider.
    """

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate(
        self, prompt: str, options: GenerationOptions
    ) -> ProviderResponse: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A backend for generating vector embeddings."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...
