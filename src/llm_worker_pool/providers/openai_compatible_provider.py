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
from llm_worker_pool.domain.token_estimate import estimate_prompt_tokens
from llm_worker_pool.infra.circuit_breaker import KeyCircuitBreaker
from llm_worker_pool.infra.quota_tracker import QuotaTracker, parse_retry_after


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
        quota_tracker: QuotaTracker | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_keys = list(api_keys)
        self._key_index = 0
        self._circuit_breaker = circuit_breaker
        self._per_key_concurrency = per_key_concurrency
        self._max_prompt_tokens = max_prompt_tokens
        self._quota = quota_tracker
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

    @property
    def quota_tracker(self) -> QuotaTracker | None:
        return self._quota

    def breaker_stats(self) -> dict[str, Any] | None:
        if self._circuit_breaker is None:
            return None
        return self._circuit_breaker.stats_by_position(self._api_keys)

    def _estimate(self, prompt: str, options: GenerationOptions) -> int:
        return estimate_prompt_tokens((options.system_prompt or "") + prompt)

    def _servable_keys(self, est_tokens: int) -> list[str]:
        keys = self._api_keys
        if self._circuit_breaker:
            keys = [k for k in keys if self._circuit_breaker.is_available(k)]
        if self._quota:
            keys = [k for k in keys if self._quota.can_serve(k, est_tokens)]
        return keys

    def unready_reason(self, prompt: str, options: GenerationOptions) -> str | None:
        """Why this provider should be skipped right now, or None if it can take the call.

        Asked by the fallback loop before a tier that still has a fallback behind it, so a
        tier known to be saturated costs nothing instead of a 429 or a full tier timeout.
        The last tier is always attempted regardless (see `_select_key`)."""
        if self._quota is None or not self._quota.enabled:
            return None
        blocked = self._quota.model_blocked_for()
        if blocked > 0:
            return (
                f"model_unresponsive_locally: {self._provider_name} skipped for another "
                f"{blocked:.0f}s after repeated timeouts"
            )
        if self._servable_keys(self._estimate(prompt, options)):
            return None
        for key in self._api_keys:
            self._quota.note_skip(key)
        wait = self._quota.next_available_in(self._api_keys)
        hint = f"; earliest key free in {wait:.0f}s" if wait > 0 else ""
        return (
            f"quota_exhausted_locally: no {self._provider_name} key has quota left "
            f"for this request{hint}"
        )

    def _select_key(self, est_tokens: int = 0) -> tuple[str, asyncio.Semaphore | None]:
        candidates = self._api_keys
        if self._circuit_breaker:
            candidates = self._circuit_breaker.filter_available_keys(self._api_keys)

        if not candidates:
            candidates = self._api_keys

        if self._quota:
            # Prefer keys with quota left; when none has any (this provider is the last
            # tier, or the estimate is pessimistic), fall back to the breaker's choice
            # rather than refusing outright.
            with_quota = [k for k in candidates if self._quota.can_serve(k, est_tokens)]
            candidates = with_quota or candidates

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
            estimated = self._estimate(prompt, options)
            if estimated > self._max_prompt_tokens:
                # Checked before a key is picked: the request never leaves the machine,
                # so it neither spends quota nor counts against the key's circuit breaker.
                raise RequestTooLargeError(
                    f"{self._provider_name}: prompt ~{estimated} tokens exceeds "
                    f"max_prompt_tokens={self._max_prompt_tokens}; skipped"
                )
        est_tokens = self._estimate(prompt, options)
        key, sem = self._select_key(est_tokens)
        # Reserved right after the pick, with no await in between - in particular BEFORE
        # waiting on the key's semaphore. Reserving only once the semaphore is held let
        # every caller queued on a busy key read its quota as still untouched.
        if self._quota:
            self._quota.reserve(key, est_tokens)
        try:
            if sem:
                async with sem:
                    return await self._execute_generate(prompt, options, key)
            return await self._execute_generate(prompt, options, key)
        finally:
            if self._quota:
                self._quota.release(key, est_tokens)

    def _record_timeout(self, key: str) -> None:
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(key)
        if self._quota:
            self._quota.record_timeout()

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
        except asyncio.CancelledError:
            # The tier deadline (asyncio.wait_for in the use case) cancels a call that
            # never answered. That is this provider's commonest failure - a model that
            # accepts the request and hangs - and before this it was never recorded,
            # so every request waited out the full tier timeout on it again.
            self._record_timeout(key)
            raise
        except httpx.ConnectError as exc:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key)
            raise ProviderUnavailableError(
                f"Could not connect to {self._provider_name} at {self._endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            self._record_timeout(key)
            raise ProviderTimeoutError(f"{self._provider_name} timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key)
            raise ProviderUnavailableError(f"HTTP error when calling {self._provider_name}: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        retry_after: float | None = None
        if self._quota:
            self._quota.observe_headers(key, response.headers)
        if response.status_code == 429 and self._quota and self._quota.enabled:
            blocked = self._quota.observe_rate_limited(key, response.headers, response.text)
            if blocked is not None:
                # Across all keys, not just this one: the retry picks another key, so it
                # need only wait for the first key that is free.
                retry_after = self._quota.wait_until_servable(
                    self._api_keys, self._estimate(prompt, options)
                )
        elif response.status_code == 429:
            retry_after = parse_retry_after(response.headers.get("retry-after"))

        if response.status_code != 200:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(key, response.status_code)
            if response.status_code >= 500 or response.status_code == 429:
                raise ProviderUnavailableError(
                    f"{self._provider_name} returned status {response.status_code}: {response.text[:500]}",
                    retry_after=retry_after,
                )
            else:
                raise InvalidResponseError(
                    f"{self._provider_name} returned error {response.status_code}: {response.text[:500]}"
                )

        if self._circuit_breaker:
            self._circuit_breaker.record_success(key)
        if self._quota:
            self._quota.record_success()

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
