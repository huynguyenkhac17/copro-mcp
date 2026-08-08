"""Provider calling Ollama via HTTP (/api/chat).

This is the only SPECIFIC implementation of LLMProvider currently available. There is no
business logic here other than mapping GenerationOptions <-> Ollama API
and transforming httpx errors into domain exceptions.
"""

from __future__ import annotations

import time

import httpx

from llm_worker_pool.domain.exceptions import (
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_worker_pool.domain.models import GenerationOptions, ProviderResponse

_PROVIDER_NAME = "ollama"


class OllamaProvider:
    def __init__(
        self,
        endpoint: str,
        model: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        # timeout=None: waiting for timeout is the responsibility of the application
        # layer (asyncio.wait_for per budget), not the provider's.
        self._client = http_client or httpx.AsyncClient(timeout=None)
        self._owns_client = http_client is None

    @property
    def provider_name(self) -> str:
        return _PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if options.system_prompt:
            messages.append({"role": "system", "content": options.system_prompt})
        messages.append({"role": "user", "content": prompt})

        ollama_options: dict[str, object] = {
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "repeat_penalty": options.repeat_penalty,
            "num_predict": options.num_predict,
            "num_ctx": options.num_ctx,
        }
        if options.seed is not None:
            ollama_options["seed"] = options.seed
        if options.stop:
            ollama_options["stop"] = options.stop

        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": ollama_options,
        }
        if options.format is not None:
            payload["format"] = options.format
        if options.think is not None:
            payload["think"] = options.think

        start = time.monotonic()
        try:
            response = await self._client.post(f"{self._endpoint}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self._endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"HTTP error when calling Ollama: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Ollama returned status {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"Ollama returned non-JSON body: {exc}") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise InvalidResponseError(
                f"Ollama response missing 'message.content' field: {data}"
            ) from exc

        return ProviderResponse(
            raw_text=content,
            provider_name=_PROVIDER_NAME,
            model_name=self._model,
            latency_ms=latency_ms,
            tokens_used=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
