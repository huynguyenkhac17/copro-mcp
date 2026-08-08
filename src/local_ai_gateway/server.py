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
from local_ai_gateway.application.job_manager import JobManager
from local_ai_gateway.application.map_reduce_orchestrator import MapReduceOrchestrator
from local_ai_gateway.config.settings import AppConfig, load_budget_profiles
from local_ai_gateway.domain.chunker import Chunker
from local_ai_gateway.domain.exceptions import LocalAIGatewayError
from local_ai_gateway.domain.file_context import FileContextResolver
from local_ai_gateway.domain.rag_indexer import RagIndexer
from local_ai_gateway.infra.cache import InMemoryTTLCache
from local_ai_gateway.infra.logging import InvocationLogger, configure_logging, get_logger
from local_ai_gateway.infra.stats import GatewayStats, StatsCollector
from local_ai_gateway.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from local_ai_gateway.providers.registry import ProviderFactory

_SERVER_NAME = "local-ai-gateway"

_logger = get_logger(__name__)


def _build_use_case(
    config: AppConfig, stats_collector: StatsCollector
) -> tuple[CallLocalUseCase, FileContextResolver]:
    budget_profiles = load_budget_profiles()
    providers = ProviderFactory(config).create_all()
    invocation_logger = InvocationLogger(
        get_logger("local_ai_gateway.invocation"),
        debug_log_content=config.logging.debug_log_content,
        stats_collector=stats_collector,
    )
    file_context_resolver = FileContextResolver(
        allowed_roots=[Path(root) for root in config.file_access.allowed_roots],
        max_file_bytes=config.file_access.max_file_bytes,
        max_total_bytes=config.file_access.max_total_bytes,
    )
    
    rag_indexer = None
    if config.rag.enabled:
        chunker = Chunker()
        # Fallback to default endpoint logic from config.providers if needed
        # Assuming the Ollama instance is local and uses default port
        default_ollama = config.providers.get(config.default_provider)
        endpoint = default_ollama.endpoint if default_ollama else "http://localhost:11434"
        embed_provider = OllamaEmbeddingProvider(endpoint=endpoint, model=config.rag.embedding_model)
        rag_indexer = RagIndexer(config=config.rag, embedding_provider=embed_provider, chunker=chunker)
    
    use_case = CallLocalUseCase(
        providers=providers,
        default_provider=config.default_provider,
        config=config,
        budget_profiles=budget_profiles,
        invocation_logger=invocation_logger,
        file_context_resolver=file_context_resolver,
        rag_indexer=rag_indexer,
    )
    
    return use_case, file_context_resolver


def build_server(config: AppConfig | None = None) -> MCPServer:
    config = config or AppConfig()
    configure_logging(config.logging.level)

    stats_collector = StatsCollector()
    use_case, file_context_resolver = _build_use_case(config, stats_collector)
    
    chunker = Chunker()
    map_reduce_orchestrator = MapReduceOrchestrator(use_case, chunker, file_context_resolver)
    
    jobs_cache: InMemoryTTLCache[CallLocalResult] = InMemoryTTLCache(
        max_entries=config.jobs.max_entries,
        ttl_seconds=config.jobs.ttl_seconds,
    )
    job_manager = JobManager(jobs_cache)
    
    budget_names = ", ".join(sorted(use_case.available_budgets()))
    provider_names = ", ".join(sorted(use_case.available_providers()))

    server = MCPServer(name=_SERVER_NAME, version=__version__)

    @server.tool(
        name="call_local",
        description=(
            "Delegates a specific sub-task to a worker LLM (local or cloud) to save your own "
            "context/token budget. You (Claude) remain the supervisor: read the tool's output, "
            "verify it, and do the final reasoning yourself - never blindly trust the content. "
            "DO NOT read large files yourself first; pass their paths via `file_paths` so the worker "
            "reads them directly (server-side, does not consume your tokens). "
            "\n\n"
            "DECIDE WHETHER TO CALL THIS TOOL AT ALL, using this policy:\n"
            "1. Trivial task (fits easily in your own context, quick lookup) -> just do it yourself, "
            "do not call this tool.\n"
            "2. Deterministic/exact task (exact count, exact numeric filter, regex-matchable pattern, "
            "line-based search) -> do NOT use any LLM (local or cloud, this tool included) - use "
            "Grep/Read/Bash instead. LLMs, even strong cloud ones, are unreliable at exact extraction "
            "over long repetitive data (empirically confirmed: qwen3.5:9b failed a simple 'find values "
            "above threshold in a list' task even via map_reduce).\n"
            "3. Task requires architecture/design judgment, trade-off evaluation, or planning -> do NOT "
            "delegate, reason about it yourself - that is your job as supervisor.\n"
            "4. Heavy but mechanical and tolerant of minor imprecision (summarize files, describe what "
            "code does, first-pass bug-candidate scan, log analysis) -> call this tool with "
            "provider='ollama' (local, free, default).\n"
            "5. Heavy AND needs stronger reasoning/instruction-following than the local model reliably "
            "gives (nuanced semantic classification, complex multi-step extraction) -> call this tool "
            "with provider='gemini'.\n"
            f"Available providers: {provider_names} (default: {config.default_provider}). "
            f"Available budgets: {budget_names} - pick the smallest one that plausibly fits the task. "
            "Set output_format='json' (optionally with json_schema) if you need structured data back. "
            "A 'cached' field in the result indicates if the response was served from cache. "
            "Use async_mode=true for tasks expected to take a long time to prevent timeout; check "
            "status later via check_job_status, do not poll in a tight loop. "
            "Use map_reduce=true to chunk very large context and process it in multiple calls - only "
            "for tasks tolerant of imprecision (see policy #2), never for exact/numeric extraction."
        ),
    )
    async def call_local(
        task: str,
        provider: str | None = None,
        budget: str = "L2",
        context: str | None = None,
        file_paths: list[str] | None = None,
        output_format: str = "text",
        json_schema: dict[str, Any] | None = None,
        async_mode: bool = False,
        map_reduce: bool = False,
    ) -> CallLocalResult:
        try:
            request = CallLocalRequest(
                task=task,
                provider=provider,
                budget=budget,
                context=context,
                file_paths=file_paths,
                output_format=output_format,  # type: ignore[arg-type]
                json_schema=json_schema,
            )
        except Exception as exc:  # noqa: BLE001 - invalid input from Claude
            return CallLocalResult(success=False, error=f"Invalid input: {exc}")

        if map_reduce:
            coro = map_reduce_orchestrator.execute(request)
        else:
            coro = use_case.execute(request)

        if async_mode:
            job_id = job_manager.submit(coro)
            return CallLocalResult(
                success=True,
                status="running",
                job_id=job_id,
                budget_used=budget,
            )
        else:
            try:
                return await coro
            except LocalAIGatewayError as exc:
                _logger.error("call_local failed: %s", exc)
                return CallLocalResult(success=False, budget_used=budget, error=str(exc))

    @server.tool(
        name="check_job_status",
        description=(
            "Check the status of a call_local task previously submitted with async_mode=true. "
            "DO NOT call this in a tight loop - do other work and check back later "
            "(seconds or tens of seconds depending on the budget). If status is still 'running', "
            "it is not done yet, try again later."
        ),
    )
    async def check_job_status(job_id: str) -> CallLocalResult:
        try:
            return job_manager.get(job_id)
        except LocalAIGatewayError as exc:
            return CallLocalResult(success=False, status="failed", error=str(exc))

    @server.tool(
        name="get_gateway_stats",
        description=(
            "View LLM Worker Pool usage statistics: cache hits, estimated "
            "tokens saved by cache, error counts, and a breakdown by provider/budget."
        ),
    )
    async def get_gateway_stats() -> GatewayStats:
        return stats_collector.snapshot()

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
