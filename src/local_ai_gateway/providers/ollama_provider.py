"""Provider goi Ollama qua HTTP (/api/chat).

Day la implementation CU THE duy nhat cua LLMProvider hien co. Khong co
logic nghiep vu nao o day ngoai viec map GenerationOptions <-> Ollama API
va bien loi httpx thanh exception domain.
"""

from __future__ import annotations

import time

import httpx

from local_ai_gateway.domain.exceptions import (
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from local_ai_gateway.domain.models import GenerationOptions, ProviderResponse

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
        # timeout=None: viec cho timeout la trach nhiem cua application
        # layer (asyncio.wait_for theo tung budget), khong phai cua provider.
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
                f"Khong ket noi duoc Ollama tai {self._endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Loi HTTP khi goi Ollama: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Ollama tra ve status {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"Ollama tra ve body khong phai JSON: {exc}") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise InvalidResponseError(
                f"Response Ollama thieu truong 'message.content': {data}"
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
