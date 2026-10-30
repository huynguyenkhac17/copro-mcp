"""OpenAI-compatible provider via httpx.

Can be used for Gemini (via its OpenAI-compatible endpoint), OpenAI, Groq, etc.
Supports per-key round-robin, key-level circuit breaking, and per-key concurrency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from llm_worker_pool.domain.exceptions import (
    InvalidResponseError,
    ProviderTimeoutError,
    RequestTooLargeError,
    ProviderUnavailableError,
)
from llm_worker_pool.domain.models import GenerationOptions, ProviderResponse
from llm_worker_pool.domain.token_estimate import estimate_tokens
from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker


class OpenAICompatibleProvider:
    def __init__(
        self,
        provider_name: str,
        endpoint: str,
        model: str,
        api_keys: list[str],
        http_client: httpx.AsyncClient | None = None,
        circuit_breaker: KeyCircuitBreaker | None = None,
        per_key_concurrency: int | None = None,
        max_prompt_tokens: int | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_keys = list(api_keys)
        self._key_index = 0
        self._circuit_breaker = circuit_breaker
        self._per_key_concurrency = per_key_concurrency
        self._max_prompt_tokens = max_prompt_tokens
        self._key_semaphores: dict[str, asyncio.Semaphore] = {}
        if per_key_concurrency and per_key_concurrency > 0:
            for k in self._api_keys:
                self._key_semaphores[k] = asyncio.Semaphore(per_key_concurrency)
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

    @property
    def circuit_breaker(self) -> KeyCircuitBreaker | None:
        return self._circuit_breaker

    def _select_key(self) -> tuple[str, asyncio.Semaphore | None]:
        candidates = self._api_keys
        if self._circuit_breaker:
            candidates = self._circuit_breaker.filter_available_keys(self._api_keys)

        if not candidates:
            candidates = self._api_keys

        # If per-key semaphores are enabled, try finding a key that isn't locked (free slot available)
        if self._key_semaphores:
            start_idx = self._key_index % len(candidates)
            for i in range(len(candidates)):
                idx = (start_idx + i) % len(candidates)
                candidate_key = candidates[idx]
                sem = self._key_semaphores.get(candidate_key)
                if sem and not sem.locked():
                    self._key_index = (idx + 1) % len(candidates)
                    return candidate_key, sem

        # Fallback to standard round-robin
        idx = self._key_index % len(candidates)
        selected = candidates[idx]
        self._key_index = (idx + 1) % len(candidates)
        sem = self._key_semaphores.get(selected)
        return selected, sem

    async def generate(self, prompt: str, options: GenerationOptions) -> ProviderResponse:
        if self._max_prompt_tokens is not None:
            estimated = estimate_tokens((options.system_prompt or "") + prompt)
            if estimated > self._max_prompt_tokens:
                # Checked before a key is picked: the request never leaves the machine,
                # so it neither spends quota nor counts against the key's circuit breaker.
                raise RequestTooLargeError(
                    f"{self._provider_name}: prompt ~{estimated} tokens exceeds "
                    f"max_prompt_tokens={self._max_prompt_tokens}; skipped"
                )
        key, sem = self._select_key()
        if sem:
            async with sem:
                return await self._execute_generate(prompt, options, key)
        return await self._execute_generate(prompt, options, key)

    async def _execute_generate(
        self, prompt: str, options: GenerationOptions, key: str
    ) -> ProviderResponse:
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

        if options.reasoning_effort:
            payload["reasoning_effort"] = options.reasoning_effort

        if options.format:
            if isinstance(options.format, dict):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": options.format,
                        # Strict mode only for schemas that already satisfy its rules;
                        # Groq answers any other schema with HTTP 400 (found by
                        # worker-eval 2026-10-30: every json_schema call to groq failed).
                        # The answer is still validated by OutputParser either way.
                        "strict": _is_strict_compatible(options.format),
                    },
                }
            elif options.format == "json":
                payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self._endpoint}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.ConnectError as exc:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key)
            raise ProviderUnavailableError(
                f"Could not connect to {self._provider_name} at {self._endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key)
            raise ProviderTimeoutError(f"{self._provider_name} timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key)
            raise ProviderUnavailableError(f"HTTP error when calling {self._provider_name}: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code != 200:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key, response.status_code)
            if response.status_code >= 500 or response.status_code == 429:
                raise ProviderUnavailableError(
                    f"{self._provider_name} returned status {response.status_code}: {response.text[:500]}"
                )
            else:
                raise InvalidResponseError(
                    f"{self._provider_name} returned error {response.status_code}: {response.text[:500]}"
                )

        if self._circuit_breaker:
            self._circuit_breaker.record_success(key)

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"{self._provider_name} returned non-JSON body: {exc}") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]

            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = message

            finish_reason = choice.get("finish_reason")

            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
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


def _is_strict_compatible(schema: Any) -> bool:
    """OpenAI-style strict structured output: every object lists all of its properties
    as required and sets additionalProperties: false."""
    if isinstance(schema, list):
        return all(_is_strict_compatible(item) for item in schema)
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required") or []) != set(props):
            return False
    return all(_is_strict_compatible(value) for value in schema.values() if isinstance(value, (dict, list)))
