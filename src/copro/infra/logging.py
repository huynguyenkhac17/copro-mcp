"""Structured logging for the entire gateway.

IMPORTANT: The server uses MCP stdio transport - stdout is reserved exclusively for
the MCP protocol. All logs MUST be written to stderr, never to stdout.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from copro.infra.event_log import EventLog, error_code, warning_code
from copro.infra.stats import StatsCollector

EventSink = Callable[[dict[str, Any]], None]
"""Optional extra destination for invocation events (hook for a future tracer such as
a self-hosted Langfuse). Receives the same dict the event log gets."""

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class InvocationLogger:
    """Log metadata of each delegate_task invocation (timestamp via log formatter,
    budget, provider, model, latency, tokens, success/failure) - does not leak
    user content (task/context/response) unless debug_log_content=True.
    """

    def __init__(
        self,
        logger: logging.Logger,
        debug_log_content: bool = False,
        stats_collector: StatsCollector | None = None,
        event_log: EventLog | None = None,
        sinks: list[EventSink] | None = None,
    ) -> None:
        self._logger = logger
        self._debug_log_content = debug_log_content
        self._stats_collector = stats_collector
        self._event_log = event_log
        self._sinks = list(sinks or [])

    def log_success(
        self,
        *,
        budget: str,
        provider: str,
        model: str,
        latency_ms: float,
        tokens_used: int | None,
        cached: bool,
        warnings: list[str],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        thinking_tokens: int | None = None,
        truncated: bool = False,
        task: str | None = None,
        raw_response: str | None = None,
        task_type: str | None = None,
        requested_provider: str | None = None,
        fallback_chain: list[str] | None = None,
        guardrail_actions: list[str] | None = None,
    ) -> None:
        if self._stats_collector:
            self._stats_collector.record_success(
                provider=provider,
                budget=budget,
                cached=cached,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=thinking_tokens,
                truncated=truncated,
            )

        fields: dict[str, Any] = {
            "event": "delegate_task",
            "success": True,
            "budget": budget,
            "provider": provider,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tokens_used": tokens_used,
            "thinking_tokens": thinking_tokens,
            "truncated": truncated,
            "warnings": warnings,
        }
        fields.update(self._content_fields(task, raw_response))
        self._logger.info(fields)

        self._emit(
            {
                "outcome": "success",
                "provider": provider,
                "model": model,
                "budget": budget,
                "task_type": task_type,
                "requested_provider": requested_provider or provider,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
                "tokens_used": tokens_used,
                "cached": cached,
                "truncated": truncated,
                "fallback_chain": fallback_chain or [provider],
                "warning_codes": sorted({warning_code(w) for w in warnings}),
                "guardrail_actions": [warning_code(a) for a in guardrail_actions or []],
            },
            task,
            raw_response,
        )

    def log_failure(
        self,
        *,
        budget: str,
        provider: str,
        model: str,
        latency_ms: float,
        error: str,
        task: str | None = None,
        task_type: str | None = None,
    ) -> None:
        if self._stats_collector:
            self._stats_collector.record_failure(provider=provider, budget=budget)

        fields: dict[str, Any] = {
            "event": "delegate_task",
            "success": False,
            "budget": budget,
            "provider": provider,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "error": error,
        }
        fields.update(self._content_fields(task, None))
        self._logger.error(fields)

        self._emit(
            {
                "outcome": "failure",
                "provider": provider,
                "model": model,
                "budget": budget,
                "task_type": task_type,
                "latency_ms": round(latency_ms, 1),
                "error_code": error_code(error),
            },
            task,
            None,
        )

    def _emit(self, event: dict[str, Any], task: str | None, raw_response: str | None) -> None:
        if self._debug_log_content:
            event = {**event, "task": task, "raw_response": raw_response}
        if self._event_log:
            self._event_log.append(event)
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 - a broken sink must not fail the call
                self._logger.warning("event sink failed: %s", exc)

    def _content_fields(self, task: str | None, raw_response: str | None) -> dict[str, Any]:
        if self._debug_log_content:
            return {"task": task, "raw_response": raw_response}
        return {
            "task_len": len(task) if task else 0,
            "response_len": len(raw_response) if raw_response else 0,
        }
