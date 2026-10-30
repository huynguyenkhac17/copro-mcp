"""MCP entrypoint.

This is the THINNEST layer in the system: it only performs (1) wiring - assembling
config/provider/use case at startup, and (2) exposes exactly 1 MCP tool
`delegate_task`, receives raw input, calls DelegateTaskUseCase, and ensures internal
exceptions/tracebacks never leak out in an unclear format.

No business logic is contained here - all decisions (which budget, which provider,
how retry/timeout works) reside in the application/domain/provider layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from llm_worker_pool import __version__
from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.application.job_manager import JobManager
from llm_worker_pool.application.cross_check_orchestrator import CrossCheckOrchestrator
from llm_worker_pool.application.map_reduce_orchestrator import MapReduceOrchestrator
from llm_worker_pool.config.settings import AppConfig, load_budget_profiles, load_task_profiles
from llm_worker_pool.domain.chunker import Chunker
from llm_worker_pool.domain.exceptions import LocalAIGatewayError
from llm_worker_pool.domain.file_context import FileContextResolver
from llm_worker_pool.domain.guardrails import DEFAULT_DENY_GLOBS, PathDenyList, SecretScanner
from llm_worker_pool.infra.url_fetcher import UrlContextFetcher
from llm_worker_pool.domain.rag_indexer import RagIndexer
from llm_worker_pool.infra.cache import InMemoryTTLCache
from llm_worker_pool.infra.event_log import EventLog, configured_path, default_event_log_path
from llm_worker_pool.infra.logging import InvocationLogger, configure_logging, get_logger
from llm_worker_pool.infra.stats import GatewayStats, StatsCollector, default_snapshot_path
from llm_worker_pool.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from llm_worker_pool.providers.registry import ProviderFactory

_SERVER_NAME = "llm-worker-pool"

_logger = get_logger(__name__)


def _build_use_case(
    config: AppConfig, stats_collector: StatsCollector
) -> tuple[DelegateTaskUseCase, FileContextResolver]:
    budget_profiles = load_budget_profiles()
    providers = ProviderFactory(config).create_all()
    event_log = None
    if config.metrics.enabled:
        event_log = EventLog(
            configured_path(config.metrics.event_log_path, default_event_log_path()),
            max_bytes=config.metrics.max_bytes,
        )
    invocation_logger = InvocationLogger(
        get_logger("llm_worker_pool.invocation"),
        debug_log_content=config.logging.debug_log_content,
        stats_collector=stats_collector,
        event_log=event_log,
    )
    guardrails = config.guardrails
    deny_globs = [
        *(DEFAULT_DENY_GLOBS if guardrails.deny_globs is None else guardrails.deny_globs),
        *guardrails.extra_deny_globs,
    ]
    file_context_resolver = FileContextResolver(
        allowed_roots=[Path(root) for root in config.file_access.allowed_roots],
        max_file_bytes=config.file_access.max_file_bytes,
        max_total_bytes=config.file_access.max_total_bytes,
        deny_list=PathDenyList(deny_globs) if deny_globs else None,
        max_dir_files=config.file_access.max_dir_files,
        max_dir_depth=config.file_access.max_dir_depth,
        max_dir_entries=config.file_access.max_dir_entries,
    )
    
    url_fetcher = None
    if config.url_fetch.enabled:
        url_fetcher = UrlContextFetcher(
            max_bytes_per_url=config.url_fetch.max_bytes_per_url,
            max_total_chars=config.url_fetch.max_total_chars,
            timeout_seconds=config.url_fetch.timeout_seconds,
            allow_private_hosts=config.url_fetch.allow_private_hosts,
            repo_max_file_bytes=config.url_fetch.repo_max_file_bytes,
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
    
    use_case = DelegateTaskUseCase(
        providers=providers,
        default_provider=config.default_provider,
        config=config,
        budget_profiles=budget_profiles,
        invocation_logger=invocation_logger,
        file_context_resolver=file_context_resolver,
        rag_indexer=rag_indexer,
        url_fetcher=url_fetcher,
        task_profiles=load_task_profiles(),
        secret_scanner=(
            SecretScanner(guardrails.extra_patterns) if guardrails.redact_secrets else None
        ),
        local_providers=guardrails.local_providers,
    )
    
    return use_case, file_context_resolver


def build_server(config: AppConfig | None = None) -> MCPServer:
    config = config or AppConfig()
    configure_logging(config.logging.level)

    # Mirrored to disk so `worker-stats` / the /worker slash command can read the
    # counters without an MCP round-trip through the model.
    stats_collector = StatsCollector(snapshot_path=default_snapshot_path())
    use_case, file_context_resolver = _build_use_case(config, stats_collector)
    
    chunker = Chunker()
    map_reduce_orchestrator = MapReduceOrchestrator(use_case, chunker, file_context_resolver)
    cross_check_orchestrator = CrossCheckOrchestrator(use_case)
    
    jobs_cache: InMemoryTTLCache[DelegateTaskResult] = InMemoryTTLCache(
        max_entries=config.jobs.max_entries,
        ttl_seconds=config.jobs.ttl_seconds,
    )
    job_manager = JobManager(jobs_cache)
    
    budget_routing = ", ".join(
        f"{b}->{p}" for b, p in sorted(config.budget_providers.items())
    ) or "none"
    budget_names = ", ".join(sorted(use_case.available_budgets()))
    provider_names = ", ".join(sorted(use_case.available_providers()))
    task_types = "; ".join(
        f"{name} ({p.description})" if p.description else name
        for name, p in use_case.available_task_types().items()
    ) or "none configured"

    server = MCPServer(name=_SERVER_NAME, version=__version__)

    @server.tool(
        name="delegate_task",
        description=(
            "Delegate a sub-task to a worker LLM (local or cloud) to preserve your own "
            "context/token budget. This is your DEFAULT way of working: actively offload work "
            "here instead of doing it inline, so your context stays free for supervision and "
            "hard reasoning. You (Claude) remain the supervisor: read the worker's output, "
            "verify it, and do the final reasoning yourself - never blindly trust the content. "
            "DO NOT read large files yourself first; pass their paths via `file_paths` so the "
            "worker reads them server-side (does not consume your tokens). An item may be a "
            "file, a directory (read recursively, skipping .git/node_modules/.venv/build "
            "output, most useful files first) or a glob such as `D:/repo/src/**/*.py`. "
            "Likewise pass web "
            "pages or GitHub repos via `urls` (a github.com/<owner>/<repo>[/tree/<ref>/<dir>] "
            "URL is shallow-cloned and its text files inlined) instead of fetching them yourself. "
            "\n\n"
            "ROUTING - which provider to use. The deciding factor is CONFIDENTIALITY, not "
            "merely whether data lives on local disk:\n"
            "- Use provider='ollama' ONLY when the content is genuinely sensitive and must not "
            "leave the machine (secrets, credentials, private/proprietary data), or the task is "
            "simple enough for a smaller local 7B model to handle reliably and you want to save "
            "cloud quota. It runs locally, so nothing is sent to a cloud API - but it is weaker "
            "and serial by default.\n"
            "- EVERY OTHER task -> use an external/cloud provider (leave `provider` unset for "
            f"cloud routing: per budget {budget_routing}, else {config.default_provider}). This INCLUDES summarizing/explaining "
            "ordinary local project code: `file_paths` reads the files server-side and streams "
            "them to whichever provider you pick, so cloud works fine for local code too. Prefer "
            "cloud here - it has real concurrency (multiple parallel workers, multi-key "
            "round-robin) whereas local ollama is serial by default, so cloud is far faster for "
            "multi-file jobs.\n"
            "- For a large local codebase, combine `file_paths` + `map_reduce=true` on a cloud "
            "provider: chunks run in parallel across the worker pool, then get synthesized.\n"
            "\n"
            "WHEN TO DELEGATE - default to YES. Delegate anything that is not heavy supervisory "
            "reasoning. Only handle it yourself (do NOT delegate) when the task is:\n"
            "1. Heavy reasoning - planning, trade-off evaluation, architecture/design judgment, "
            "or other high-level decisions. That is your job as supervisor.\n"
            "2. Trivial - so small that a tool round-trip would be slower than just answering.\n"
            "3. Deterministic/exact - exact counts, numeric filtering, regex/line matching. Use "
            "Grep/Read/Bash instead; this is not an LLM task at all, and even strong models are "
            "unreliable at exact extraction over long repetitive data.\n"
            "Everything else - summarizing, explaining code, drafting/rewriting, translation, "
            "semantic classification, first-pass bug-candidate scans, log analysis, boilerplate "
            "generation, format conversion, multi-step extraction - SHOULD be delegated. "
            "\n\n"
            f"Available providers: {provider_names} (default: {config.default_provider}; "
            f"per budget when `provider` is unset: {budget_routing}). "
            f"Available budgets: {budget_names} - pick the smallest one that plausibly fits the task. "
            "Set output_format='json' (optionally with json_schema) if you need structured data back. "
            f"Set task_type to one of: {task_types} - it adds an output contract for that kind "
            "of work (and for review/explain/summarize a `verification` field checking every "
            "path:line the worker cites against the files it was given; drop findings whose "
            "status is not 'ok'). Explicit budget/output_format/json_schema override the "
            "task type's defaults; omit task_type for free-form tasks. "
            "task_type='review' cross-checks by default: the review runs on two providers "
            "of different vendors in parallel and their findings are merged - each finding "
            "carries `agreed_by` (trust those reported by both most) and `citation` "
            "(ok / quote_not_found / ...); see `cross_check` in the result. Pass "
            "cross_check=false for a single, cheaper call. "
            "Files matching the secret deny-list (.env*, *.pem, *.key, ...) are never read, and "
            "secret-looking literals are redacted before any cloud call - see "
            "`guardrail_actions` in the result. "
            "A 'cached' field in the result indicates if the response was served from cache. "
            "Use async_mode=true for tasks expected to take a long time to prevent timeout; check "
            "status later via check_job_status, do not poll in a tight loop. "
            "Use map_reduce=true to chunk very large context and process it in multiple calls - only "
            "for tasks tolerant of imprecision, never for exact/numeric extraction."
        ),
    )
    async def delegate_task(
        task: str,
        provider: str | None = None,
        budget: str | None = None,
        context: str | None = None,
        file_paths: list[str] | None = None,
        urls: list[str] | None = None,
        output_format: str | None = None,
        json_schema: dict[str, Any] | None = None,
        async_mode: bool = False,
        map_reduce: bool = False,
        task_type: str | None = None,
        cross_check: bool | None = None,
    ) -> DelegateTaskResult:
        try:
            fields: dict[str, Any] = {
                "task": task,
                "provider": provider,
                "context": context,
                "file_paths": file_paths,
                "urls": urls,
                "json_schema": json_schema,
                "task_type": task_type,
                "cross_check": cross_check,
            }
            # Only forward what the caller actually set, so a task type's own
            # default budget/output format can apply (DelegateTaskRequest tracks
            # explicitly-set fields). Without task_type the DTO defaults are the
            # same as before: budget L2, text output.
            if budget is not None:
                fields["budget"] = budget
            if output_format is not None:
                fields["output_format"] = output_format
            request = DelegateTaskRequest(**fields)
        except Exception as exc:  # noqa: BLE001 - invalid input from Claude
            return DelegateTaskResult(success=False, error=f"Invalid input: {exc}")

        try:
            # Fail an unknown task_type now rather than inside an async job, and report
            # the budget the task type actually picks.
            effective_budget = use_case.apply_task_profile(request)[0].budget
        except LocalAIGatewayError as exc:
            return DelegateTaskResult(success=False, budget_used=request.budget, error=str(exc))

        if map_reduce:
            coro = map_reduce_orchestrator.execute(request)
        elif cross_check_orchestrator.wants_cross_check(request):
            coro = cross_check_orchestrator.execute(request)
        else:
            coro = use_case.execute(request)

        if async_mode:
            job_id = job_manager.submit(coro)
            return DelegateTaskResult(
                success=True,
                status="running",
                job_id=job_id,
                budget_used=effective_budget,
            )
        else:
            try:
                return await coro
            except LocalAIGatewayError as exc:
                _logger.error("delegate_task failed: %s", exc)
                return DelegateTaskResult(success=False, budget_used=effective_budget, error=str(exc))

    @server.tool(
        name="check_job_status",
        description=(
            "Check the status of a delegate_task task previously submitted with async_mode=true. "
            "DO NOT call this in a tight loop - do other work and check back later "
            "(seconds or tens of seconds depending on the budget). If status is still 'running', "
            "it is not done yet, try again later."
        ),
    )
    async def check_job_status(job_id: str) -> DelegateTaskResult:
        try:
            return job_manager.get(job_id)
        except LocalAIGatewayError as exc:
            return DelegateTaskResult(success=False, status="failed", error=str(exc))

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
