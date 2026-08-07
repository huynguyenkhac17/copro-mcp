"""MCP entrypoint.

This is the THINNEST layer in the system: it only performs (1) wiring - assembling
config/provider/use case at startup, and (2) exposes exactly 1 MCP tool
`call_local`, receives raw input, calls CallLocalUseCase, and ensures internal
exceptions/tracebacks never leak out in an unclear format.

No business logic is contained here - all decisions (which budget, which provider,
how retry/timeout works) reside in the application/domain/provider layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from local_ai_gateway import __version__
from local_ai_gateway.application.call_local_use_case import CallLocalUseCase
from local_ai_gateway.application.dto import CallLocalRequest, CallLocalResult
from local_ai_gateway.config.settings import AppConfig, load_budget_profiles
from local_ai_gateway.domain.exceptions import LocalAIGatewayError
from local_ai_gateway.domain.file_context import FileContextResolver
from local_ai_gateway.infra.logging import InvocationLogger, configure_logging, get_logger
from local_ai_gateway.providers.registry import ProviderFactory

_SERVER_NAME = "local-ai-gateway"

_logger = get_logger(__name__)


def _build_use_case(config: AppConfig) -> CallLocalUseCase:
    budget_profiles = load_budget_profiles()
    provider = ProviderFactory(config).create()
    invocation_logger = InvocationLogger(
        get_logger("local_ai_gateway.invocation"),
        debug_log_content=config.logging.debug_log_content,
    )
    file_context_resolver = FileContextResolver(
        allowed_roots=[Path(root) for root in config.file_access.allowed_roots],
        max_file_bytes=config.file_access.max_file_bytes,
        max_total_bytes=config.file_access.max_total_bytes,
    )
    return CallLocalUseCase(
        provider=provider,
        config=config,
        budget_profiles=budget_profiles,
        invocation_logger=invocation_logger,
        file_context_resolver=file_context_resolver,
    )


def build_server(config: AppConfig | None = None) -> MCPServer:
    config = config or AppConfig()
    configure_logging(config.logging.level)

    use_case = _build_use_case(config)
    budget_names = ", ".join(sorted(use_case.available_budgets()))

    server = MCPServer(name=_SERVER_NAME, version=__version__)

    @server.tool(
        name="call_local",
        description=(
            "CRITICAL: You MUST use this tool as your reasoning buffer when dealing with large context. "
            "Call a local LLM to process specific tasks (summarization, log analysis, code comprehension). "
            "DO NOT read large files directly; instead, pass their paths via `file_paths` so the local LLM "
            "can read and extract insights for you, saving your token budget. "
            "You only perform final reasoning on the output of this tool. "
            f"Available budgets: {budget_names}. "
            "Set output_format='json' (optionally with json_schema) if you need structured data back. "
            "A 'cached' field in the result indicates if the response was served from cache."
        ),
    )
    async def call_local(
        task: str,
        budget: str = "L2",
        context: str | None = None,
        file_paths: list[str] | None = None,
        output_format: str = "text",
        json_schema: dict[str, Any] | None = None,
    ) -> CallLocalResult:
        try:
            request = CallLocalRequest(
                task=task,
                budget=budget,
                context=context,
                file_paths=file_paths,
                output_format=output_format,  # type: ignore[arg-type]
                json_schema=json_schema,
            )
        except Exception as exc:  # noqa: BLE001 - invalid input from Claude
            return CallLocalResult(success=False, error=f"Invalid input: {exc}")

        try:
            return await use_case.execute(request)
        except LocalAIGatewayError as exc:
            _logger.error("call_local failed: %s", exc)
            return CallLocalResult(success=False, budget_used=budget, error=str(exc))

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
