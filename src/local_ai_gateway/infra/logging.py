"""Structured logging cho toan bo gateway.

QUAN TRONG: server dung MCP stdio transport - stdout danh rieng cho giao
thuc MCP. Moi log PHAI ghi ra stderr, khong bao gio ra stdout.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

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
    """Ghi metadata cua moi lan goi call_local (timestamp qua log formatter,
    budget, provider, model, latency, tokens, success/failure) - khong leak
    noi dung user (task/context/response) tru khi debug_log_content=True.
    """

    def __init__(self, logger: logging.Logger, debug_log_content: bool = False) -> None:
        self._logger = logger
        self._debug_log_content = debug_log_content

    def log_success(
        self,
        *,
        budget: str,
        provider: str,
        model: str,
        latency_ms: float,
        tokens_used: int | None,
        warnings: list[str],
        task: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "event": "call_local",
            "success": True,
            "budget": budget,
            "provider": provider,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tokens_used": tokens_used,
            "warnings": warnings,
        }
        fields.update(self._content_fields(task, raw_response))
        self._logger.info(fields)

    def log_failure(
        self,
        *,
        budget: str,
        provider: str,
        model: str,
        latency_ms: float,
        error: str,
        task: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "event": "call_local",
            "success": False,
            "budget": budget,
            "provider": provider,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "error": error,
        }
        fields.update(self._content_fields(task, None))
        self._logger.error(fields)

    def _content_fields(self, task: str | None, raw_response: str | None) -> dict[str, Any]:
        if self._debug_log_content:
            return {"task": task, "raw_response": raw_response}
        return {
            "task_len": len(task) if task else 0,
            "response_len": len(raw_response) if raw_response else 0,
        }
