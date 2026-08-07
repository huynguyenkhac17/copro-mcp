"""MCP entrypoint.

Day la lop MONG NHAT trong he thong: chi lam nhiem vu (1) wiring - lap rap
config/provider/use case luc khoi dong, va (2) expose dung 1 tool MCP
`call_local`, nhan input tho, goi CallLocalUseCase, va dam bao khong bao
gio de exception/traceback noi bo lo ra ngoai duoi dang khong ro rang.

Khong chua logic nghiep vu - moi quyet dinh (budget nao, provider nao,
retry/timeout ra sao) deu nam o application/domain/provider layer.
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
            "Goi mot LLM chay local (hien tai qua Ollama) de xu ly mot tac vu "
            "cu the - dung nhu mot 'chuyen gia' phu, KHONG thay the vai tro "
            "quyet dinh cuoi cung cua Claude. Dung khi can nen context lon "
            "(doc nhieu file, tom tat, tim bug candidate) hoac tac vu du "
            f"'nang' de dang goi ra ngoai. Budget hien co: {budget_names}. "
            "output_format='json' se yeu cau model tra ve JSON (co the kem "
            "json_schema de goi y cau truc). Neu can model doc noi dung mot "
            "so file, uu tien dung file_paths (server tu doc truc tiep) thay "
            "vi paste noi dung vao context - tiet kiem token hon; can server "
            "duoc cau hinh cho phep thu muc do (xem warning "
            "'file_access_disabled' neu chua bat). Ket qua co truong 'cached' "
            "bao co phai tra loi da co san tu lan goi truoc hay khong."
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
        except Exception as exc:  # noqa: BLE001 - input tu Claude sai dinh dang
            return CallLocalResult(success=False, error=f"Input khong hop le: {exc}")

        try:
            return await use_case.execute(request)
        except LocalAIGatewayError as exc:
            _logger.error("call_local that bai: %s", exc)
            return CallLocalResult(success=False, budget_used=budget, error=str(exc))

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
