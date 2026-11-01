"""Cross-check: run one review on several providers in parallel and merge the findings.

Measured on worker-eval (2026-10-30, harness-compare): with the review contract,
groq found 0.94 of the seeded bugs but cited the wrong line more often (citation
validity 0.93 -> 0.81), while gemini-lite cited precisely but found only 0.38. The
providers miss different things, so their union - with each finding marked by who
reported it and whether its citation checks out - beats either one alone.

Each arm is pinned to its provider (no fallback chain): an arm that silently fell back
to the other arm's model would count one opinion twice. An arm whose provider caps the
prompt size (groq: max_prompt_tokens) gets the files in batches that fit, instead of
being skipped - small files are where it does best.

Arms are awaited, never cancelled: a cancelled call is recorded as a hang by the
quota tracker (infra/quota_tracker.py) and could block a healthy model.

A review that runs on ONE provider (cross_check=false, fewer than two arms up, or a
provider named in the request) is batched the same way by `execute_single`: on
review-large-l3 (~15k tokens, 2026-11-01) groq skipped the whole prompt and the
fallback to gemini-lite found 0.33-0.50 of the bugs, while groq on batches of whole
files is what made the cross-check reach 1.00. Each batch keeps its fallback chain,
and the findings are concatenated - no LLM reduce step, which would lose the
file/line/quote evidence (map_reduce does that for free-form tasks).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_worker_pool.application import progress
from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.domain.finding_merge import findings_of, merge_findings
from llm_worker_pool.domain.token_estimate import estimate_prompt_tokens
from llm_worker_pool.domain.validators import citation_warnings, split_context_files

# System prompt (budget + grounding rules + profile addon + JSON schema) on top of
# task + context; measured ~500-700 tokens for review, rounded up.
_PROMPT_OVERHEAD_TOKENS = 1000


class CrossCheckOrchestrator:
    def __init__(self, delegate_task_use_case: DelegateTaskUseCase) -> None:
        self._use_case = delegate_task_use_case

    def wants_cross_check(self, request: DelegateTaskRequest) -> bool:
        if request.cross_check is False:
            return False
        _, profile = self._use_case.apply_task_profile(request)
        return bool(request.cross_check or (profile and profile.cross_check))

    def arms(self, request: DelegateTaskRequest) -> list[str]:
        _, profile = self._use_case.apply_task_profile(request)
        wanted = profile.cross_check_providers if profile else []
        available = set(self._use_case.available_providers())
        return [name for name in dict.fromkeys(wanted) if name in available]

    async def execute(self, request: DelegateTaskRequest) -> DelegateTaskResult:
        request, profile = self._use_case.apply_task_profile(request)
        arms = self.arms(request)
        if request.output_format != "json" or len(arms) < 2:
            reason = "needs_json_findings" if request.output_format != "json" else "fewer_than_two_providers"
            result = await self.execute_single(request)
            result.warnings.append(f"cross_check_skipped:{reason}")
            return result

        context, context_warnings = await self._use_case.resolve_context(request)
        base = request.model_copy(
            update={"context": context, "file_paths": None, "urls": None, "provider": None}
        )
        splittable = bool(request.file_paths) and not request.context and not request.urls
        files = split_context_files(context) if splittable else {}

        batches = {arm: self._batches(arm, base, files) for arm in arms}
        await progress.begin(
            sum(len(b) for b in batches.values()),
            "cross-check on " + ", ".join(f"{arm} ({len(b)} calls)" for arm, b in batches.items()),
        )
        arm_jobs = [self._run_arm(arm, base, batches[arm]) for arm in arms]
        outcomes = await asyncio.gather(*arm_jobs)
        return self._combine(request, arms, outcomes, context, context_warnings)

    # --- one provider, batched ---------------------------------------------------

    def wants_batching(self, request: DelegateTaskRequest) -> bool:
        """The request (after its task profile) asks for a JSON `findings` list, the
        contract whose answers can be concatenated batch by batch."""
        request, _ = self._use_case.apply_task_profile(request)
        return request.output_format == "json" and _asks_for_findings(request.json_schema)

    async def execute_single(self, request: DelegateTaskRequest) -> DelegateTaskResult:
        """A findings review on one provider (the requested one, else the routed one).
        Context over that provider's prompt cap that is made of several files goes out
        as batches of whole files that fit; anything else is a normal call."""
        request, _ = self._use_case.apply_task_profile(request)
        provider = self._use_case.routed_provider(request)
        splittable = bool(request.file_paths) and not request.context and not request.urls
        if (
            not splittable
            or self._use_case.prompt_limit(provider) is None
            or not _asks_for_findings(request.json_schema)
        ):
            return await self._use_case.execute(request)

        context, context_warnings = await self._use_case.resolve_context(request)
        # provider stays as requested (None = routed): each batch keeps its fallbacks.
        base = request.model_copy(update={"context": context, "file_paths": None, "urls": None})
        files = split_context_files(context)
        batches = self._batches(provider, base, files)
        if len(batches) < 2:
            # Fits, or one file too big to split: the call it would have been anyway,
            # minus a second read of the files.
            result = await self._use_case.execute(base)
            return self._with_context_warnings(result, context_warnings)

        await progress.begin(len(batches), f"review in {len(batches)} batches on {provider}")
        label = progress.counter("review batch {done}/{total} done", len(batches))
        results = await asyncio.gather(
            *(
                progress.step(
                    self._use_case.execute(base.model_copy(update={"context": ctx}), keep_conversation=False),
                    label,
                )
                for ctx in batches
            ),
            return_exceptions=True,
        )
        by_provider: dict[str, list[dict[str, Any]]] = {}
        answered: list[DelegateTaskResult] = []
        errors: list[str] = []
        for i, result in enumerate(results, start=1):
            if isinstance(result, BaseException):
                errors.append(f"review_batch_failed:{i}/{len(batches)}:{type(result).__name__}: {result}"[:300])
                continue
            batch_findings = findings_of(result.content) if result.success else None
            if batch_findings is None:
                reason = result.error or "answer has no findings list"
                errors.append(f"review_batch_failed:{i}/{len(batches)}:{reason}"[:300])
                continue
            answered.append(result)
            # Grouped by who answered: a batch that fell back is another model's opinion.
            by_provider.setdefault(result.provider or provider, []).extend(batch_findings)

        if not answered:
            return DelegateTaskResult(
                success=False,
                budget_used=request.budget,
                task_type=request.task_type,
                error=f"Every one of {len(batches)} review batches failed: " + "; ".join(errors),
                warnings=list(context_warnings),
            )
        warnings = [*context_warnings, f"review_batched:{len(batches)}", *errors]
        return self._merged_result(request, by_provider, answered, context, warnings, cross_check=None)

    @staticmethod
    def _with_context_warnings(result: DelegateTaskResult, context_warnings: list[str]) -> DelegateTaskResult:
        """Warnings (and denied-file guardrail actions) of a context the use case did
        not resolve itself, since it was handed the text."""
        result.warnings = [*context_warnings, *(w for w in result.warnings if w not in context_warnings)]
        denied = [w for w in context_warnings if w.startswith("file_denied_sensitive:")]
        result.guardrail_actions = [*denied, *(a for a in result.guardrail_actions if a not in denied)]
        return result

    # --- one arm ---------------------------------------------------------------

    async def _run_arm(
        self, arm: str, base: DelegateTaskRequest, batches: list[str | None]
    ) -> dict[str, Any]:
        requests = [
            base.model_copy(update={"provider": arm, "context": ctx}) for ctx in batches
        ]
        label = progress.counter(f"{arm}: {{done}}/{{total}} calls done", len(batches))
        results = await asyncio.gather(
            *(
                progress.step(
                    self._use_case.execute(r, allow_fallback=False, keep_conversation=False),
                    label,
                )
                for r in requests
            ),
            return_exceptions=True,
        )

        findings: list[dict[str, Any]] = []
        errors: list[str] = []
        answered: list[DelegateTaskResult] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(f"{type(result).__name__}: {result}")
                continue
            if not result.success:
                errors.append(result.error or "failed")
                continue
            batch_findings = findings_of(result.content)
            if batch_findings is None:
                errors.append("answer has no findings list")
                continue
            answered.append(result)
            findings.extend(batch_findings)
        return {
            "arm": arm,
            "batches": len(batches),
            "results": answered,
            "findings": findings,
            "errors": errors,
        }

    def _batches(
        self, arm: str, base: DelegateTaskRequest, files: dict[str, str]
    ) -> list[str | None]:
        """The whole context, or - when it is too big for this arm and is made of
        files - groups of whole files that each fit. A single file over the limit is
        sent alone and left to the provider to refuse."""
        limit = self._use_case.prompt_limit(arm)
        overhead = estimate_prompt_tokens(base.task) + _PROMPT_OVERHEAD_TOKENS
        if limit is None or not base.context:
            return [base.context]
        if overhead + estimate_prompt_tokens(base.context) <= limit or len(files) < 2:
            return [base.context]

        batches: list[str] = []
        current: list[str] = []
        used = overhead
        for path, body in files.items():
            block = f"--- file: {path} ---\n{body}"
            cost = estimate_prompt_tokens(block)
            if current and used + cost > limit:
                batches.append("\n\n".join(current))
                current, used = [], overhead
            current.append(block)
            used += cost
        if current:
            batches.append("\n\n".join(current))
        return batches

    # --- merge -----------------------------------------------------------------

    def _combine(
        self,
        request: DelegateTaskRequest,
        arms: list[str],
        outcomes: list[dict[str, Any]],
        context: str | None,
        context_warnings: list[str],
    ) -> DelegateTaskResult:
        answered = [o for o in outcomes if o["results"]]
        arm_summary = {
            o["arm"]: {
                "success": bool(o["results"]),
                "findings": len(o["findings"]),
                "batches": o["batches"],
                "errors": o["errors"],
            }
            for o in outcomes
        }
        if not answered:
            return DelegateTaskResult(
                success=False,
                budget_used=request.budget,
                task_type=request.task_type,
                error="Cross-check: every provider failed: " + "; ".join(
                    f"{o['arm']}: {' | '.join(o['errors']) or 'no answer'}" for o in outcomes
                ),
                warnings=list(context_warnings),
                cross_check={"arms": arm_summary},
            )

        warnings = list(context_warnings)
        for o in outcomes:
            if not o["results"]:
                warnings.append(f"cross_check_partial:{o['arm']} failed: {' | '.join(o['errors'])[:300]}")
            elif o["errors"]:
                warnings.append(f"cross_check_partial:{o['arm']} {len(o['errors'])} of {o['batches']} batches failed")
        return self._merged_result(
            request,
            {o["arm"]: o["findings"] for o in answered},
            [r for o in answered for r in o["results"]],
            context,
            warnings,
            cross_check={"arms": arm_summary},
        )

    def _merged_result(
        self,
        request: DelegateTaskRequest,
        by_provider: dict[str, list[dict[str, Any]]],
        results: list[DelegateTaskResult],
        context: str | None,
        warnings: list[str],
        *,
        cross_check: dict[str, Any] | None,
    ) -> DelegateTaskResult:
        """One answer out of several calls' findings: merged, re-verified against the
        whole context, usage summed, remembered as a single conversation turn."""
        merged = merge_findings(by_provider, files=split_context_files(context))
        content = {"findings": merged}
        raw_text = json.dumps(content, ensure_ascii=False, indent=2)

        warnings = list(warnings)
        guardrail_actions = [w for w in warnings if w.startswith("file_denied_sensitive:")]
        for result in results:
            for warning in result.warnings:
                if warning not in warnings:
                    warnings.append(warning)
            for action in result.guardrail_actions:
                if action not in guardrail_actions:
                    guardrail_actions.append(action)

        # Re-check citations on the merged list against the full context: per-arm
        # verification saw only that arm's answer (and, when batched, one batch).
        warnings = [w for w in warnings if not w.startswith("citation_unverified:")]
        verification = self._use_case.verify_citations(raw_text, content, context)
        warnings.extend(citation_warnings(verification))

        if cross_check is not None:
            cross_check = {
                **cross_check,
                "findings_total": len(merged),
                "findings_agreed": sum(1 for f in merged if len(f["agreed_by"]) > 1),
            }

        def total(field: str) -> int | None:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            return sum(values) if values else None

        return DelegateTaskResult(
            success=True,
            content=content,
            raw_text=raw_text,
            provider="+".join(by_provider),
            model="+".join(dict.fromkeys(r.model for r in results)),
            budget_used=request.budget,
            latency_ms=max(r.latency_ms for r in results),
            tokens_used=total("tokens_used"),
            prompt_tokens=total("prompt_tokens"),
            completion_tokens=total("completion_tokens"),
            thinking_tokens=total("thinking_tokens"),
            truncated=any(r.truncated for r in results),
            cached=all(r.cached for r in results),
            warnings=warnings,
            task_type=request.task_type,
            guardrail_actions=guardrail_actions,
            verification=verification,
            cross_check=cross_check,
            # A follow-up continues from the merged findings, on a single provider.
            continuation_id=self._use_case.remember_turn(request, context, raw_text),
        )


def _asks_for_findings(schema: dict[str, Any] | None) -> bool:
    """The JSON schema's top level has a `findings` array (the review contract)."""
    if not isinstance(schema, dict):
        return False
    findings = (schema.get("properties") or {}).get("findings")
    return isinstance(findings, dict) and findings.get("type") == "array"
