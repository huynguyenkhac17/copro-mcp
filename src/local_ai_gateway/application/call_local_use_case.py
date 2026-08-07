"""CallLocalUseCase: coordinates the entire flow of a call_local invocation.

Does not contain detailed logic - delegates to domain services (PromptBuilder,
OutputParser) and provider (only via LLMProvider interface, agnostic of
specific Ollama/vLLM/OpenAI details).
"""

from __future__ import annotations

import asyncio
import time

from local_ai_gateway.application.dto import CallLocalRequest, CallLocalResult
from local_ai_gateway.config.settings import AppConfig
from local_ai_gateway.domain.exceptions import ConfigurationError, LocalAIGatewayError
from local_ai_gateway.domain.file_context import FileContextResolver
from local_ai_gateway.domain.interfaces import LLMProvider
from local_ai_gateway.domain.models import BudgetProfile
from local_ai_gateway.domain.output_parser import OutputParser
from local_ai_gateway.domain.prompt_builder import PromptBuilder
from local_ai_gateway.infra.logging import InvocationLogger
from local_ai_gateway.infra.retry import with_backoff


class CallLocalUseCase:
    def __init__(
        self,
        provider: LLMProvider,
        config: AppConfig,
        budget_profiles: dict[str, BudgetProfile],
        invocation_logger: InvocationLogger,
        prompt_builder: PromptBuilder | None = None,
        output_parser: OutputParser | None = None,
        file_context_resolver: FileContextResolver | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._budget_profiles = budget_profiles
        self._logger = invocation_logger
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._output_parser = output_parser or OutputParser()
        self._file_context_resolver = file_context_resolver or FileContextResolver(
            allowed_roots=[]
        )
        # Limited GPU VRAM might easily overflow if multiple requests with large 
        # contexts run in parallel. Thus, we limit concurrent provider calls.
        self._semaphore = asyncio.Semaphore(config.concurrency.max_parallel_requests)

    async def execute(self, request: CallLocalRequest) -> CallLocalResult:
        profile = self._resolve_budget(request.budget)

        file_warnings: list[str] = []
        combined_context = request.context
        if request.file_paths:
            file_result = self._file_context_resolver.resolve(request.file_paths)
            file_warnings = file_result.warnings
            combined_context = "\n\n".join(
                part for part in (request.context, file_result.text) if part
            )

        built = self._prompt_builder.build(
            task=request.task,
            context=combined_context,
            profile=profile,
            output_format=request.output_format,
            json_schema=request.json_schema,
        )

        options = profile.options.model_copy(
            update={
                "system_prompt": built.system_prompt,
                "format": self._resolve_format(profile, request),
            }
        )

        timeout = profile.timeout_seconds or self._config.timeout_seconds
        start = time.monotonic()

        try:
            async with self._semaphore:
                response = await asyncio.wait_for(
                    with_backoff(
                        lambda: self._provider.generate(built.user_prompt, options),
                        max_attempts=self._config.retry.max_attempts,
                        backoff_base_seconds=self._config.retry.backoff_base_seconds,
                    ),
                    timeout=timeout,
                )
        except TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            error = (
                f"Local model did not respond within {timeout}s (budget={request.budget}). "
                "Try a lower budget or truncate the context."
            )
            self._logger.log_failure(
                budget=request.budget,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=latency_ms,
                error=error,
                task=request.task,
            )
            return CallLocalResult(
                success=False,
                budget_used=request.budget,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=latency_ms,
                error=error,
            )
        except LocalAIGatewayError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._logger.log_failure(
                budget=request.budget,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=latency_ms,
                error=str(exc),
                task=request.task,
            )
            return CallLocalResult(
                success=False,
                budget_used=request.budget,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=latency_ms,
                error=str(exc),
            )

        warnings = [*file_warnings, *built.warnings]
        content: object = response.raw_text

        if request.output_format == "json":
            parsed = self._output_parser.parse(response.raw_text, request.json_schema)
            warnings.extend(parsed.warnings)
            content = parsed.content if parsed.content is not None else response.raw_text

        self._logger.log_success(
            budget=request.budget,
            provider=response.provider_name,
            model=response.model_name,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            warnings=warnings,
            task=request.task,
            raw_response=response.raw_text,
        )

        return CallLocalResult(
            success=True,
            content=content,
            raw_text=response.raw_text,
            provider=response.provider_name,
            model=response.model_name,
            budget_used=request.budget,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            cached=response.from_cache,
            warnings=warnings,
            error=None,
        )

    def available_budgets(self) -> list[str]:
        return list(self._budget_profiles)

    def _resolve_budget(self, budget: str) -> BudgetProfile:
        profile = self._budget_profiles.get(budget)
        if profile is None:
            available = ", ".join(sorted(self._budget_profiles))
            raise ConfigurationError(
                f"Budget '{budget}' does not exist. Available budgets: {available}"
            )
        return profile

    @staticmethod
    def _resolve_format(profile: BudgetProfile, request: CallLocalRequest) -> object:
        if request.output_format == "json" and profile.options.format is None:
            return "json"
        return profile.options.format
