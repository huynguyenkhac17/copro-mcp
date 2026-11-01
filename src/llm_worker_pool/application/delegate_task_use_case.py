"""DelegateTaskUseCase: coordinates the entire flow of a delegate_task invocation.

Does not contain detailed logic - delegates to domain services (PromptBuilder,
OutputParser) and provider (only via LLMProvider interface, agnostic of
specific Ollama/vLLM/OpenAI details).
"""

from __future__ import annotations

import asyncio
import time

from llm_worker_pool.application import progress
from llm_worker_pool.application.conversation_store import (
    INHERITED_FIELDS,
    Conversation,
    ConversationStore,
    render_history,
)
from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.config.settings import AppConfig
from llm_worker_pool.domain.exceptions import ConfigurationError, LocalAIGatewayError
from llm_worker_pool.domain.file_context import FileContextResolver, resolve_request_context
from llm_worker_pool.domain.guardrails import SecretScanner, format_counts
from llm_worker_pool.domain.interfaces import LLMProvider
from llm_worker_pool.domain.models import (
    BudgetProfile,
    GenerationOptions,
    ProviderResponse,
    TaskProfile,
)
from llm_worker_pool.domain.output_parser import OutputParser
from llm_worker_pool.domain.prompt_builder import PromptBuilder
from llm_worker_pool.domain.rag_indexer import RagIndexer
from llm_worker_pool.domain.token_estimate import (
    CONTEXT_RESERVE_RATIO,
    estimate_prompt_tokens,
    estimate_tokens,
)
from llm_worker_pool.domain.validators import (
    CitationValidator,
    build_json_repair_prompt,
    citation_warnings,
    extract_citations,
    split_context_files,
)
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
        task_profiles: dict[str, TaskProfile] | None = None,
        secret_scanner: SecretScanner | None = None,
        local_providers: list[str] | None = None,
        conversation_store: ConversationStore | None = None,
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
        self._task_profiles = task_profiles or {}
        self._secret_scanner = secret_scanner
        """None = no redaction and no output scan (guardrails.redact_secrets: false)."""
        self._local_providers = frozenset(local_providers or ())
        self._conversations = conversation_store
        """None = no continuation_id support (results carry none)."""

    async def execute(
        self,
        request: DelegateTaskRequest,
        *,
        allow_fallback: bool = True,
        keep_conversation: bool = True,
    ) -> DelegateTaskResult:
        """`allow_fallback=False` pins the call to its provider (cross-check arms: an
        arm answered by another tier would no longer be an independent opinion).
        `keep_conversation=False` answers without recording a turn (internal calls
        whose answer the caller never sees as such, e.g. cross-check arms)."""
        request, conversation = self.resume_conversation(request)
        request, task_profile = self.apply_task_profile(request)
        profile = self.resolve_budget(request.budget)

        provider_name = request.provider or self._routed_provider(request.budget, task_profile)
        provider = self._providers.get(provider_name)
        if not provider:
            available = ", ".join(sorted(self._providers))
            raise ConfigurationError(
                f"Provider '{provider_name}' does not exist. Available providers: {available}"
            )
        chain = self._fallback_chain(provider_name, request) if allow_fallback else [provider_name]

        stored_context = conversation.context if conversation is not None else None
        combined_context, file_warnings = await self.resolve_context(
            request, reserved_tokens=estimate_tokens(stored_context or "")
        )
        task = request.task
        history_warnings: list[str] = []
        if conversation is not None:
            if request.cross_check:
                history_warnings.append("cross_check_skipped:follow_up_runs_on_one_provider")
            if combined_context and stored_context and combined_context in stored_context:
                combined_context = None  # the caller resent what the conversation already has
            combined_context = "\n\n".join(
                part for part in (stored_context, combined_context) if part
            ) or None
            room = self._context_window_tokens(request.budget) - estimate_tokens(combined_context or "")
            task, dropped = render_history(conversation.turns, request.task, max(room, 0))
            if dropped:
                history_warnings.append(
                    f"continuation_history_trimmed:{dropped} oldest turns left out to fit num_ctx"
                )
        unredacted_context = combined_context  # what the conversation keeps, see conversation_store
        task, combined_context, redaction_warnings = self._guard_input(
            task, combined_context, chain
        )
        guardrail_actions = [
            w for w in file_warnings if w.startswith("file_denied_sensitive:")
        ] + redaction_warnings

        built = self._prompt_builder.build(
            task=task,
            context=combined_context,
            profile=profile,
            output_format=request.output_format,
            json_schema=request.json_schema,
            system_addon=task_profile.system_addon if task_profile else None,
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
        quota_skips: list[str] = []
        # Tiers skipped at startup (e.g. their API keys are unset) are dropped up front so
        # "is there a later tier?" below only counts tiers that can actually answer.
        names = [name for name in chain if name in self._providers]
        configured = list(names)
        names, prompt_cap_skips = self._drop_over_prompt_cap(names, built.user_prompt, options)
        tiers = [self._providers[name] for name in names]
        for index, tier in enumerate(tiers):
            has_next = index < len(tiers) - 1
            tier_timeout = self._tier_timeout(timeout, has_next)
            # Reasoning follows the tier's place in the configured chain, so dropping an
            # over-cap tier changes no answer: on review-large-l3 (2026-11-01) lite at
            # full reasoning took +5.5s for precision 0.67 -> 0.57, recall 1.00 both.
            tier_options = self._tier_options(
                options, is_fallback=configured.index(names[index]) > 0
            )
            if has_next:
                # Skip a tier already known to be out of quota or hanging. The last tier is
                # always tried: failing fast there would only turn a slow answer into none.
                check = getattr(tier, "unready_reason", None)
                reason = check(built.user_prompt, tier_options) if check else None
                if reason:
                    quota_skips.append(f"quota_skip:{tier.provider_name}")
                    failures.append(f"{tier.provider_name}: {self._short_error(reason)}")
                    provider = tier
                    error = reason
                    continue
            await progress.note(
                f"calling {tier.provider_name}"
                + (f" after {failures[-1]}"[:200] if failures else "")
            )
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
                task_type=request.task_type,
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
                task_type=request.task_type,
                guardrail_actions=guardrail_actions,
                # Unchanged conversation: the caller can retry the same follow-up.
                continuation_id=request.continuation_id,
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
            await progress.note(
                f"retrying {provider.provider_name} with reasoning curbed after an empty answer"
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

        warnings = [
            *file_warnings, *history_warnings, *redaction_warnings, *built.warnings,
            *prompt_cap_skips, *quota_skips,
        ]
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

        validators = task_profile.validators if task_profile else []
        if request.output_format == "json":
            content, response = await self._parse_json(
                request, response, provider, used_options, used_timeout, built.system_prompt,
                warnings, repair="json_repair" in validators,
            )

        verification = None
        if "citations" in validators:
            verification = self.verify_citations(response.raw_text, content, combined_context)
            warnings.extend(citation_warnings(verification))

        leaked = self._guard_output(response.raw_text)
        warnings.extend(leaked)
        guardrail_actions.extend(leaked)

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
            task_type=request.task_type,
            # The chain's head even when its cap dropped it (stats count by it).
            requested_provider=self._providers[configured[0]].provider_name,
            fallback_chain=[t.provider_name for t in tiers[: len(failures) + 1]],
            guardrail_actions=guardrail_actions,
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
            task_type=request.task_type,
            guardrail_actions=guardrail_actions,
            verification=verification,
            continuation_id=(
                self.remember_turn(request, unredacted_context, response.raw_text)
                if keep_conversation else None
            ),
        )

    def resume_conversation(
        self, request: DelegateTaskRequest
    ) -> tuple[DelegateTaskRequest, Conversation | None]:
        """For a follow-up: the stored conversation, and the request with the
        conversation's settings filled in wherever the caller left them unset."""
        if request.continuation_id is None:
            return request, None
        conversation = self._conversations.get(request.continuation_id) if self._conversations else None
        if conversation is None:
            raise ConfigurationError(
                f"continuation_not_found: '{request.continuation_id}' is unknown or expired "
                "(conversations live in server memory for a limited time and are lost on "
                "restart) - resend the task with its context/file_paths."
            )
        explicit = {name: getattr(request, name) for name in request.model_fields_set}
        # An explicit None is "unset": the server passes provider/task_type/json_schema
        # through even when the caller left them out.
        inherited = {
            name: value for name, value in conversation.settings.items()
            if explicit.get(name) is None and value is not None
        }
        # Rebuilt (not model_copy) so inherited fields count as explicitly set, and the
        # task profile does not override them with its own defaults.
        return DelegateTaskRequest(**{**explicit, **inherited}), conversation

    def remember_turn(
        self, request: DelegateTaskRequest, context: str | None, answer: str
    ) -> str | None:
        """Record an answered turn; returns the id to continue from (None when
        conversations are off)."""
        if self._conversations is None:
            return None
        return self._conversations.append(
            request.continuation_id,
            context=context,
            settings={name: getattr(request, name) for name in INHERITED_FIELDS},
            task=request.task,
            answer=answer,
        )

    def apply_task_profile(
        self, request: DelegateTaskRequest
    ) -> tuple[DelegateTaskRequest, TaskProfile | None]:
        """Fill budget/output_format/json_schema from the task profile wherever the
        caller left them unset. task_type=None returns the request untouched."""
        if request.task_type is None:
            return request, None
        task_profile = self._task_profiles.get(request.task_type)
        if task_profile is None:
            available = ", ".join(sorted(self._task_profiles)) or "(none configured)"
            raise ConfigurationError(
                f"Task type '{request.task_type}' does not exist. Available task types: {available}"
            )
        explicit = request.model_fields_set
        update: dict[str, object] = {}
        if task_profile.default_budget and "budget" not in explicit:
            update["budget"] = task_profile.default_budget
        if task_profile.output_format and "output_format" not in explicit:
            update["output_format"] = task_profile.output_format
        output_format = update.get("output_format", request.output_format)
        if output_format == "json" and request.json_schema is None and task_profile.json_schema:
            update["json_schema"] = task_profile.json_schema
        return (request.model_copy(update=update) if update else request), task_profile

    def _is_local(self, name: str) -> bool:
        if name in self._local_providers:
            return True
        instance = self._config.providers.get(name)
        return instance is not None and instance.type == "ollama"

    def _guard_input(
        self, task: str, context: str | None, chain: list[str]
    ) -> tuple[str, str | None, list[str]]:
        """Redact secrets from task + context when ANY tier of the chain is a cloud
        provider - the fallback decides where the text ends up, not the first pick."""
        if self._secret_scanner is None or all(self._is_local(name) for name in chain):
            return task, context, []
        task, counts = self._secret_scanner.redact(task)
        if context:
            context, context_counts = self._secret_scanner.redact(context)
            counts.update(context_counts)
        return task, context, format_counts("secret_redacted", counts)

    def _guard_output(self, raw_text: str) -> list[str]:
        """The worker may echo a secret that slipped past input redaction (or was
        local-only). Flag it; the caller decides what to do with the answer."""
        if self._secret_scanner is None or not raw_text:
            return []
        return format_counts("secret_in_response", self._secret_scanner.scan(raw_text))

    async def _parse_json(
        self,
        request: DelegateTaskRequest,
        response: ProviderResponse,
        provider: LLMProvider,
        options: GenerationOptions,
        timeout: float,
        system_prompt: str | None,
        warnings: list[str],
        *,
        repair: bool,
    ) -> tuple[object, ProviderResponse]:
        """Parse the answer as JSON; with `repair`, a failed parse/schema check gets
        one retry on the answering tier, shown its own output and the errors.
        Returns (content, response) - the response is the repaired one on success."""
        parsed = self._output_parser.parse(response.raw_text, request.json_schema)
        failed = [
            w for w in parsed.warnings
            if w.startswith(("json_parse_failed", "json_schema_validation_failed"))
        ]
        if repair and failed:
            repaired = await self._repair_json(
                request, response, provider, options, timeout, system_prompt, failed
            )
            if repaired is not None:
                fixed = self._output_parser.parse(repaired.raw_text, request.json_schema)
                if not any(
                    w.startswith(("json_parse_failed", "json_schema_validation_failed"))
                    for w in fixed.warnings
                ):
                    warnings.extend(fixed.warnings)
                    warnings.append("json_repaired")
                    return fixed.content, repaired
        warnings.extend(parsed.warnings)
        content = parsed.content if parsed.content is not None else response.raw_text
        return content, response

    async def _repair_json(
        self,
        request: DelegateTaskRequest,
        response: ProviderResponse,
        provider: LLMProvider,
        options: GenerationOptions,
        timeout: float,
        system_prompt: str | None,
        errors: list[str],
    ) -> ProviderResponse | None:
        prompt = build_json_repair_prompt(response.raw_text, errors, request.json_schema)
        # The repair prompt quotes the previous answer, which may hold an echoed secret.
        if self._secret_scanner is not None and not self._is_local(provider.provider_name):
            prompt, _ = self._secret_scanner.redact(prompt)
        try:
            return await self._call_provider(
                provider, prompt, options.model_copy(update={"system_prompt": system_prompt}), timeout
            )
        except (TimeoutError, LocalAIGatewayError):
            return None

    @staticmethod
    def verify_citations(raw_text: str, content: object, context: str | None) -> dict:
        """Public for MapReduceOrchestrator, whose reduce step only sees the map
        extracts and must re-check citations against the original files."""
        files = split_context_files(context)
        return CitationValidator(files).validate(extract_citations(raw_text, content))

    async def resolve_context(
        self, request: DelegateTaskRequest, *, reserved_tokens: int = 0
    ) -> tuple[str | None, list[str]]:
        """The context text exactly as it will be sent (context + file_paths + urls),
        plus the warnings raised while gathering it. Public so `worker-eval` can hand
        the very same text to RAGAS as retrieved_contexts. `reserved_tokens`: window
        already taken by other context (a conversation's earlier context)."""
        return await resolve_request_context(
            self._file_context_resolver,
            context=request.context,
            file_paths=request.file_paths,
            query=request.task,
            rag_indexer=self._rag_indexer,
            urls=request.urls,
            url_fetcher=self._url_fetcher,
            # RAG replaces whole files with top_k chunks, so only use it when the files
            # would not fit the budget's context window anyway.
            rag_min_tokens=self._context_window_tokens(request.budget) - reserved_tokens,
        )

    def _context_window_tokens(self, budget: str) -> int:
        """Tokens of context the budget's num_ctx leaves after the prompt/answer reserve."""
        return int(self.resolve_budget(budget).options.num_ctx * (1 - CONTEXT_RESERVE_RATIO))

    @property
    def url_fetcher(self) -> UrlContextFetcher | None:
        return self._url_fetcher

    def _drop_over_prompt_cap(
        self, names: list[str], user_prompt: str, options: GenerationOptions
    ) -> tuple[list[str], list[str]]:
        """(tiers to try, prompt_cap_skip warnings). A tier whose max_prompt_tokens this
        prompt exceeds would refuse it locally anyway (groq: 6000, while most L3 prompts
        are larger). Calling it anyway logged a failure and made the next tier a
        "fallback": fallback_used and a logged failure for a call that never left the
        machine. Dropped up front, the chain starts at the next tier (its reasoning still
        follows its configured position, see execute).
        Same estimate as OpenAICompatibleProvider's own check. When every tier is over,
        the chain is kept so the refusal still reaches the caller."""
        limits = {name: self.prompt_limit(name) for name in names}
        if not any(limit is not None for limit in limits.values()):
            return names, []
        estimated = estimate_prompt_tokens((options.system_prompt or "") + user_prompt)
        over = [name for name in names if limits[name] is not None and estimated > limits[name]]
        if not over or len(over) == len(names):
            return names, []
        return [name for name in names if name not in over], [
            f"prompt_cap_skip:{name} (~{estimated} tokens > max_prompt_tokens={limits[name]})"
            for name in over
        ]

    def prompt_limit(self, provider_name: str) -> int | None:
        """The provider's `max_prompt_tokens` (None = no per-request cap)."""
        instance = self._config.providers.get(provider_name)
        return instance.max_prompt_tokens if instance else None

    def available_task_types(self) -> dict[str, TaskProfile]:
        return dict(self._task_profiles)

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

    def routed_provider(self, request: DelegateTaskRequest) -> str:
        """The provider `execute` starts this request on (its own, else routing)."""
        request, task_profile = self.apply_task_profile(request)
        return request.provider or self._routed_provider(request.budget, task_profile)

    def _routed_provider(self, budget: str, task_profile: TaskProfile | None) -> str:
        """Provider when the request names none: the task type's measured pick if that
        provider is up, else the budget's."""
        preferred = task_profile.preferred_provider if task_profile else None
        if preferred and preferred in self._providers:
            return preferred
        return self._budget_default_provider(budget)

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
                deadline=time.monotonic() + timeout,
            ),
            timeout=timeout,
        )
