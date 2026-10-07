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
from llm_worker_pool.infra.url_fetcher import UrlContextFetcher


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
        url_fetcher: UrlContextFetcher | None = None,
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
        self._url_fetcher = url_fetcher

    async def execute(self, request: DelegateTaskRequest) -> DelegateTaskResult:
        profile = self.resolve_budget(request.budget)

        provider_name = request.provider or self._budget_default_provider(request.budget)
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
            urls=request.urls,
            url_fetcher=self._url_fetcher,
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

        # Walk the fallback chain: a 503 "model overloaded" hits every key of a provider
        # alike, so retrying the same provider cannot help - only another model/vendor can.
        response: ProviderResponse | None = None
        used_options = options  # what the answering tier was actually sent (see _tier_options)
        used_timeout = timeout  # and the timeout it ran under (see _tier_timeout)
        failures: list[str] = []
        # Tiers skipped at startup (e.g. their API keys are unset) are dropped up front so
        # "is there a later tier?" below only counts tiers that can actually answer.
        tiers = [
            self._providers[name]
            for name in self._fallback_chain(provider_name, request)
            if name in self._providers
        ]
        for index, tier in enumerate(tiers):
            has_next = index < len(tiers) - 1
            tier_timeout = self._tier_timeout(timeout, has_next)
            tier_options = self._tier_options(options, is_fallback=index > 0)
            tier_start = time.monotonic()
            try:
                response = await self._call_provider(
                    tier, built.user_prompt, tier_options, tier_timeout
                )
            except TimeoutError:
                error = (
                    f"{tier.provider_name} did not respond within {tier_timeout}s "
                    f"(budget={request.budget}). Try a lower budget or truncate the context."
                )
            except LocalAIGatewayError as exc:
                error = str(exc)
            else:
                provider = tier
                used_options = tier_options
                used_timeout = tier_timeout
                break

            self._logger.log_failure(
                budget=request.budget,
                provider=tier.provider_name,
                model=tier.model_name,
                latency_ms=(time.monotonic() - tier_start) * 1000,
                error=error,
                task=request.task,
            )
            failures.append(f"{tier.provider_name}: {self._short_error(error)}")
            provider = tier

        if response is None:
            return DelegateTaskResult(
                success=False,
                budget_used=request.budget,
                provider=provider.provider_name,
                model=provider.model_name,
                latency_ms=(time.monotonic() - start) * 1000,
                # A single-provider chain keeps the original, unabridged error.
                error=error if len(failures) == 1 else "All providers failed: " + "; ".join(failures),
            )

        retried_without_thinking = False
        thinking_knob = getattr(provider, "thinking_option", "think")
        if self._looks_thinking_exhausted(response, used_options) and self._can_curb_thinking(
            used_options, thinking_knob
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
            fallback_options = used_options.model_copy(
                update={"think": False, "reasoning_effort": "low"}
            )
            try:
                # Same cap as the tier's first call: a non-last tier must not be able to
                # hold the request past `tier_timeout_seconds` through this retry either.
                response = await self._call_provider(
                    provider, built.user_prompt, fallback_options, used_timeout
                )
                retried_without_thinking = True
            except (TimeoutError, LocalAIGatewayError):
                pass  # keep the original (empty) response, fall through to the warning below

        warnings = [*file_warnings, *built.warnings]
        if failures:
            warnings.append(
                f"fallback_used:{provider.provider_name} after " + "; ".join(failures)
            )
        content: object = response.raw_text

        if retried_without_thinking:
            warnings.append("retried_with_thinking_disabled_after_empty_response")
        elif self._looks_thinking_exhausted(response, used_options):
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

    @property
    def url_fetcher(self) -> UrlContextFetcher | None:
        return self._url_fetcher

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

    def _tier_timeout(self, timeout: float, has_next: bool) -> float:
        """A tier that still has a fallback behind it is cut off at `tier_timeout_seconds`,
        so a model that accepts the call and then hangs cannot eat the whole budget; the
        last tier keeps the full budget timeout."""
        cap = self._config.tier_timeout_seconds
        if has_next and cap is not None:
            return min(timeout, cap)
        return timeout

    _HEAVY_REASONING = frozenset({"medium", "high"})

    def _tier_options(self, options: GenerationOptions, *, is_fallback: bool) -> GenerationOptions:
        """Curb reasoning on fallback tiers (see `AppConfig.fallback_reasoning_effort`).
        Never raises effort: a budget already at "low" stays as it is."""
        effort = self._config.fallback_reasoning_effort
        if is_fallback and effort and options.reasoning_effort in self._HEAVY_REASONING:
            return options.model_copy(update={"reasoning_effort": effort})
        return options

    def _budget_default_provider(self, budget: str) -> str:
        """Provider used when the request names none: the budget's own pick if that
        provider is up, otherwise the global default (a budget routed to a vendor whose
        keys are missing must degrade, not fail every call)."""
        preferred = self._config.budget_providers.get(budget)
        if preferred and preferred in self._providers:
            return preferred
        return self._default_provider

    def _fallback_chain(self, provider_name: str, request: DelegateTaskRequest) -> list[str]:
        chain = [provider_name]
        fallbacks = self._config.fallbacks.get(provider_name, [])
        # A budget-specific chain applies only when the budget itself chose the provider:
        # an explicitly requested provider keeps its own chain (or none, for ollama).
        if request.provider is None and self._config.budget_providers.get(request.budget) == provider_name:
            fallbacks = self._config.budget_fallbacks.get(request.budget, fallbacks)
        for name in fallbacks:
            if name not in chain:
                chain.append(name)
        return chain

    @staticmethod
    def _short_error(error: str, limit: int = 160) -> str:
        """Provider errors embed multi-line JSON bodies; flatten them for a one-line summary."""
        flat = " ".join(error.split())
        return flat if len(flat) <= limit else flat[: limit - 3] + "..."

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
