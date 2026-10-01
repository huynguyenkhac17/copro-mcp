"""DelegateTaskUseCase: coordinates the entire flow of a delegate_task invocation.

Does not contain detailed logic - delegates to domain services (PromptBuilder,
OutputParser) and provider (only via LLMProvider interface, agnostic of
specific Ollama/vLLM/OpenAI details).
"""

from __future__ import annotations

import asyncio
import time

from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.config.settings import AppConfig
from llm_worker_pool.domain.exceptions import ConfigurationError, LocalAIGatewayError
from llm_worker_pool.domain.file_context import FileContextResolver, resolve_request_context
from llm_worker_pool.domain.interfaces import LLMProvider
from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions, ProviderResponse
from llm_worker_pool.domain.output_parser import OutputParser
from llm_worker_pool.domain.prompt_builder import PromptBuilder
from llm_worker_pool.domain.rag_indexer import RagIndexer
from llm_worker_pool.infra.logging import InvocationLogger
from llm_worker_pool.infra.retry import with_backoff


class DelegateTaskUseCase:
    _THINKING_WARN_RATIO = 0.5
    """Warn once internal reasoning eats more than this share of the output budget."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default_provider: str,
        config: AppConfig,
        budget_profiles: dict[str, BudgetProfile],
        invocation_logger: InvocationLogger,
        prompt_builder: PromptBuilder | None = None,
        output_parser: OutputParser | None = None,
        file_context_resolver: FileContextResolver | None = None,
        rag_indexer: RagIndexer | None = None,
    ) -> None:
        self._providers = providers
        self._default_provider = default_provider
        self._config = config
        self._budget_profiles = budget_profiles
        self._logger = invocation_logger
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._output_parser = output_parser or OutputParser()
        self._file_context_resolver = file_context_resolver or FileContextResolver(
            allowed_roots=[]
        )
        self._rag_indexer = rag_indexer

    async def execute(self, request: DelegateTaskRequest) -> DelegateTaskResult:
        profile = self.resolve_budget(request.budget)

        provider_name = request.provider or self._default_provider
        provider = self._providers.get(provider_name)
        if not provider:
            available = ", ".join(sorted(self._providers))
            raise ConfigurationError(
                f"Provider '{provider_name}' does not exist. Available providers: {available}"
            )

        combined_context, file_warnings = await resolve_request_context(
            self._file_context_resolver,
            context=request.context,
            file_paths=request.file_paths,
            query=request.task,
            rag_indexer=self._rag_indexer,
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
            response = await self._call_provider(provider, built.user_prompt, options, timeout)
        except TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            error = (
                f"Local model did not respond within {timeout}s (budget={request.budget}). "
                "Try a lower budget or truncate the context."
            )
            self._logger.log_failure(
                budget=request.budget,
                provider=provider.provider_name,
                model=provider.model_name,
                latency_ms=latency_ms,
                error=error,
                task=request.task,
            )
            return DelegateTaskResult(
                success=False,
                budget_used=request.budget,
                provider=provider.provider_name,
                model=provider.model_name,
                latency_ms=latency_ms,
                error=error,
            )
        except LocalAIGatewayError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._logger.log_failure(
                budget=request.budget,
                provider=provider.provider_name,
                model=provider.model_name,
                latency_ms=latency_ms,
                error=str(exc),
                task=request.task,
            )
            return DelegateTaskResult(
                success=False,
                budget_used=request.budget,
                provider=provider.provider_name,
                model=provider.model_name,
                latency_ms=latency_ms,
                error=str(exc),
            )

        retried_without_thinking = False
        thinking_knob = getattr(provider, "thinking_option", "think")
        if self._looks_thinking_exhausted(response, options) and self._can_curb_thinking(
            options, thinking_knob
        ):
            # Thinking-capable models can burn the entire num_predict budget on internal
            # reasoning, leaving nothing for the actual answer. Rather than surface a
            # useless result, retry once with reasoning curbed.
            #
            # Both knobs are set because they address different wire protocols and each
            # provider ignores the one that is not its own: `think` is Ollama's, while
            # `reasoning_effort` is the OpenAI-compatible field. Setting only `think` here
            # is what made this retry a silent no-op for cloud providers - it re-sent a
            # byte-identical payload and burned a second full call for nothing.
            fallback_options = options.model_copy(
                update={"think": False, "reasoning_effort": "low"}
            )
            try:
                response = await self._call_provider(
                    provider, built.user_prompt, fallback_options, timeout
                )
                retried_without_thinking = True
            except (TimeoutError, LocalAIGatewayError):
                pass  # keep the original (empty) response, fall through to the warning below

        warnings = [*file_warnings, *built.warnings]
        content: object = response.raw_text

        if retried_without_thinking:
            warnings.append("retried_with_thinking_disabled_after_empty_response")
        elif self._looks_thinking_exhausted(response, options):
            warnings.append("empty_response_possible_thinking_exhaustion")

        # The provider reported this answer as cut off mid-sentence. Without this the
        # result comes back success=True with no warnings at all, and the caller has no
        # way to tell a complete answer from a decapitated one.
        if response.truncated:
            warnings.append(
                f"output_truncated_at_max_tokens (num_predict={options.num_predict}); "
                "the answer is incomplete - raise the budget or split the task"
            )

        # Thinking is billed inside the same budget as the answer, so a reasoning model
        # can quietly leave only scraps for the visible reply. Worth flagging before the
        # user wonders why a large budget produced three sentences.
        thinking = response.thinking_tokens
        if thinking and thinking > options.num_predict * self._THINKING_WARN_RATIO:
            warnings.append(
                f"thinking_consumed_{thinking}_of_{options.num_predict}_output_tokens; "
                "set a lower reasoning_effort for this budget"
            )

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
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            thinking_tokens=response.thinking_tokens,
            truncated=response.truncated,
            cached=response.from_cache,
            warnings=warnings,
            task=request.task,
            raw_response=response.raw_text,
        )

        return DelegateTaskResult(
            success=True,
            content=content,
            raw_text=response.raw_text,
            provider=response.provider_name,
            model=response.model_name,
            budget_used=request.budget,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            thinking_tokens=response.thinking_tokens,
            truncated=response.truncated,
            cached=response.from_cache,
            warnings=warnings,
            error=None,
        )

    def available_budgets(self) -> list[str]:
        return list(self._budget_profiles)

    def available_providers(self) -> list[str]:
        return list(self._providers)

    def resolve_budget(self, budget: str) -> BudgetProfile:
        profile = self._budget_profiles.get(budget)
        if profile is None:
            available = ", ".join(sorted(self._budget_profiles))
            raise ConfigurationError(
                f"Budget '{budget}' does not exist. Available budgets: {available}"
            )
        return profile

    @staticmethod
    def _resolve_format(profile: BudgetProfile, request: DelegateTaskRequest) -> object:
        if request.output_format == "json":
            return request.json_schema or profile.options.format or "json"
        return profile.options.format

    @staticmethod
    def _looks_thinking_exhausted(response: ProviderResponse, options: GenerationOptions) -> bool:
        """Did the model spend its whole output budget thinking and say nothing useful?

        The comparison must be against *completion* tokens. Using the grand total here
        (its previous form) compared a prompt+thinking+completion sum against a cap that
        governs completion alone, so any request with a non-trivial prompt satisfied it
        automatically and the whole clause collapsed into "is the text empty".
        """
        if response.raw_text.strip():
            return False

        # An empty answer that also hit the ceiling is the signature of the failure:
        # something consumed the budget, and it was not the visible reply.
        spent = response.completion_tokens
        if spent is None:
            spent = response.tokens_used  # provider gave no breakdown; fall back
        return bool(response.truncated or (spent and spent >= options.num_predict))

    @staticmethod
    def _can_curb_thinking(options: GenerationOptions, thinking_knob: str | None) -> bool:
        """Is there room left to turn reasoning down on the knob this backend reads?

        Judged against the provider's own knob, because a retry that lowers a field
        the backend ignores costs a full extra call and changes nothing.
        """
        if thinking_knob == "think":
            return options.think is not False
        if thinking_knob == "reasoning_effort":
            return options.reasoning_effort not in ("low", "minimal")
        return False  # backend exposes no thinking control - retrying is pointless

    async def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        options: GenerationOptions,
        timeout: float,
    ) -> ProviderResponse:
        return await asyncio.wait_for(
            with_backoff(
                lambda: provider.generate(prompt, options),
                max_attempts=self._config.retry.max_attempts,
                backoff_base_seconds=self._config.retry.backoff_base_seconds,
            ),
            timeout=timeout,
        )
