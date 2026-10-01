"""OpenAI-compatible provider via httpx.

Can be used for Gemini (via its OpenAI-compatible endpoint), OpenAI, Groq, etc.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import httpx

from llm_worker_pool.domain.exceptions import (
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_worker_pool.domain.models import GenerationOptions, ProviderResponse


class OpenAICompatibleProvider:
    def __init__(
        self,
        provider_name: str,
        endpoint: str,
        model: str,
        api_keys: list[str],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        # itertools.cycle().__next__() has no internal await point, so calling
        # it from multiple concurrent coroutines on the same event loop thread
        # cannot interleave mid-call - no lock needed.
        self._key_cycle = itertools.cycle(api_keys)
        self._client = http_client or httpx.AsyncClient(timeout=None)
        self._owns_client = http_client is None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def thinking_option(self) -> str | None:
        return "reasoning_effort"

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        start = time.monotonic()

        messages = []
        if options.system_prompt:
            messages.append({"role": "system", "content": options.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.num_predict,
        }

        if options.stop:
            payload["stop"] = options.stop

        # Caps internal reasoning so it cannot eat the whole max_tokens budget before
        # the model writes a single visible character. Omitted entirely when unset -
        # sending null is not the same as not sending the field.
        if options.reasoning_effort:
            payload["reasoning_effort"] = options.reasoning_effort

        # Not sent: real OpenAI supports "seed", but Gemini's OpenAI-compat shim
        # rejects it with a hard 400 ("Unknown name seed") - since "OpenAI-compatible"
        # backends vary in which fields they actually implement, skip anything that
        # isn't universally safe rather than special-casing per backend.

        if options.format:
            if isinstance(options.format, dict):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.format,
                        "strict": True,
                    },
                }
            elif options.format == "json":
                payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {next(self._key_cycle)}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self._endpoint}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to {self._provider_name} at {self._endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{self._provider_name} timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"HTTP error when calling {self._provider_name}: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code != 200:
            if response.status_code >= 500 or response.status_code == 429:
                raise ProviderUnavailableError(
                    f"{self._provider_name} returned status {response.status_code}: {response.text[:500]}"
                )
            else:
                raise InvalidResponseError(
                    f"{self._provider_name} returned error {response.status_code}: {response.text[:500]}"
                )

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"{self._provider_name} returned non-JSON body: {exc}") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]

            # Handle Gemini's response format which may include extra_content with thinking state
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = message

            finish_reason = choice.get("finish_reason")

            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            # Reasoning models bill thinking inside total_tokens but omit it from both
            # prompt_tokens and completion_tokens, so the three never add up. Keeping the
            # parts lets the caller recover the thinking spend by subtraction.
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(
                f"{self._provider_name} response missing expected fields: {data}"
            ) from exc

        return ProviderResponse(
            raw_text=content or "",
            provider_name=self._provider_name,
            model_name=self._model,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            from_cache=False,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
