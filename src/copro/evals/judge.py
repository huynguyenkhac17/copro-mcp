"""RAGAS judge: LLM-scored metrics over (task, answer, retrieved contexts, reference).

The judge reaches an OpenAI-compatible endpoint (a provider entry from default.yaml,
`eval.judge_provider`) through the openai client. Requests go through a transport that
rotates the provider's API keys and quarantines a key after repeated 429/403/5xx, using
the same KeyCircuitBreaker as the server; the openai client's own retries then land on
the next key instead of hammering the exhausted one.

Needs the [eval] extra (ragas, openai).
"""

from __future__ import annotations

import asyncio
import itertools
import warnings
from typing import Any

import httpx

from copro.config.settings import AppConfig
from copro.domain.exceptions import ConfigurationError
from copro.evals.dataset import EvalCase
from copro.infra.circuit_breaker import KeyCircuitBreaker
from copro.providers.registry import resolve_api_keys

_QUARANTINE_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
NO_FAITHFULNESS_TASK_TYPES = frozenset({"review", "generate"})
"""A bug claim or freshly written code is not a statement the context "supports", so
RAGAS faithfulness scores them near 0 regardless of quality. These task types are
measured by bug_recall / citation_valid_rate / must_contain instead."""


class RotatingKeyTransport(httpx.AsyncBaseTransport):
    def __init__(self, keys: list[str], breaker: KeyCircuitBreaker) -> None:
        self._keys = keys
        self._breaker = breaker
        self._cycle = itertools.cycle(range(len(keys)))
        self._inner = httpx.AsyncHTTPTransport()

    def _next_key(self) -> str:
        available = set(self._breaker.filter_available_keys(self._keys))
        for _ in range(len(self._keys)):
            key = self._keys[next(self._cycle)]
            if key in available:
                return key
        return next(iter(available))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = self._next_key()
        request.headers["Authorization"] = f"Bearer {key}"
        try:
            response = await self._inner.handle_async_request(request)
        except httpx.TransportError:
            self._breaker.record_failure(key)
            raise
        if response.status_code in _QUARANTINE_STATUSES:
            self._breaker.record_failure(key, response.status_code)
        elif response.status_code < 400:
            self._breaker.record_success(key)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class Judge:
    """Builds the RAGAS metrics once and scores cases with whichever apply."""

    def __init__(
        self, config: AppConfig, embeddings_ok: bool | None = None, provider: str | None = None
    ) -> None:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        eval_cfg = config.eval
        judge_provider = provider or eval_cfg.judge_provider
        provider_cfg = config.providers.get(judge_provider)
        if provider_cfg is None or provider_cfg.type != "openai_compatible":
            raise ConfigurationError(
                f"judge provider '{judge_provider}' must name an openai_compatible provider"
            )
        keys = resolve_api_keys(provider_cfg)
        if not keys:
            raise ConfigurationError(f"judge provider '{judge_provider}' has no API keys set")

        self._sem = asyncio.Semaphore(max(1, eval_cfg.max_concurrency))
        self.provider = judge_provider
        self.model = provider_cfg.model
        cb = config.circuit_breaker
        self._breaker = KeyCircuitBreaker(cb.failure_threshold, cb.quarantine_seconds, cb.enabled)
        # Bounded on purpose: an overloaded model that accepts calls and never answers
        # (gemini-flash on 2026-10-30) would otherwise stall the whole run.
        self._http = httpx.AsyncClient(
            transport=RotatingKeyTransport(keys, self._breaker),
            timeout=eval_cfg.judge_timeout_seconds,
        )
        client = AsyncOpenAI(
            base_url=provider_cfg.endpoint, api_key="rotated-by-transport",
            http_client=self._http, max_retries=eval_cfg.judge_max_retries,
        )
        model_args: dict[str, Any] = {"temperature": 0.0, "max_tokens": 8192}
        if eval_cfg.judge_reasoning_effort:
            model_args["reasoning_effort"] = eval_cfg.judge_reasoning_effort
        llm = llm_factory(provider_cfg.model, provider="openai", client=client, **model_args)

        self._metrics: dict[str, Any] = {
            "faithfulness": Faithfulness(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
        self.embeddings_note = None
        if embeddings_ok is None:
            embeddings_ok = _endpoint_up(eval_cfg.embedding_endpoint)
        if embeddings_ok:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from ragas.embeddings.base import embedding_factory

                embeddings = embedding_factory(
                    "openai", model=eval_cfg.embedding_model,
                    client=AsyncOpenAI(base_url=eval_cfg.embedding_endpoint, api_key="ollama"),
                )
            self._metrics["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embeddings)
        else:
            self.embeddings_note = (
                f"answer_relevancy skipped: embedding endpoint {eval_cfg.embedding_endpoint} unreachable"
            )

    async def score(
        self, case: EvalCase, response: str, contexts: list[str], is_rag: bool
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Returns (scores, errors) - a metric that throws is recorded, not fatal."""
        jobs: dict[str, Any] = {}
        m = self._metrics
        if contexts and case.task_type not in NO_FAITHFULNESS_TASK_TYPES:
            jobs["faithfulness"] = m["faithfulness"].ascore(
                user_input=case.task, response=response, retrieved_contexts=contexts
            )
        if "answer_relevancy" in m:
            jobs["answer_relevancy"] = m["answer_relevancy"].ascore(user_input=case.task, response=response)
        if case.reference:
            jobs["factual_correctness"] = m["factual_correctness"].ascore(
                response=response, reference=case.reference
            )
            if is_rag and contexts:
                jobs["context_precision"] = m["context_precision"].ascore(
                    user_input=case.task, reference=case.reference, retrieved_contexts=contexts
                )
                jobs["context_recall"] = m["context_recall"].ascore(
                    user_input=case.task, retrieved_contexts=contexts, reference=case.reference
                )

        async def limited(coro: Any) -> Any:
            async with self._sem:
                return await coro

        names = list(jobs)
        results = await asyncio.gather(*(limited(j) for j in jobs.values()), return_exceptions=True)
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                errors[name] = f"{type(result).__name__}: {str(result)[:200]}"
                continue
            value = getattr(result, "value", result)
            if isinstance(value, (int, float)) and value == value:  # drop NaN
                scores[name] = float(value)
            else:
                errors[name] = f"non-numeric score: {value!r}"[:200]
        return scores, errors

    def breaker_stats(self) -> dict[str, Any]:
        stats = self._breaker.get_stats()
        # Keys are secrets: report them by position only.
        return {f"key_{i}": v for i, v in enumerate(stats.values())}

    async def aclose(self) -> None:
        await self._http.aclose()


def _endpoint_up(base_url: str) -> bool:
    try:
        return httpx.get(base_url.rstrip("/") + "/models", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False
