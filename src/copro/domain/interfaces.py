"""Interface (Protocol) that every provider must follow.

This is the only boundary between the application layer and specific LLM backends
(Ollama, vLLM, OpenAI, Gemini, ...). The application layer only knows
LLMProvider, and never imports a specific provider directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from copro.domain.models import GenerationOptions, ProviderResponse


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

    @property
    def thinking_option(self) -> str | None:
        """Which GenerationOptions field this backend actually honours to limit
        internal reasoning: "think" (Ollama's boolean), "reasoning_effort" (the
        OpenAI-compatible string), or None for backends with no such control.

        The application layer needs this to decide whether retrying a
        thinking-exhausted call can change anything. Assuming the wrong knob is
        how the retry became a no-op for cloud providers: it re-sent `think=False`,
        a field the OpenAI-compatible wire format does not carry, producing a
        byte-identical request at full cost.

        Optional - callers read it defensively and fall back to "think".
        """
        ...

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
