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
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_worker_pool.application.delegate_task_use_case import DelegateTaskUseCase
from llm_worker_pool.application.dto import DelegateTaskRequest, DelegateTaskResult
from llm_worker_pool.domain.finding_merge import findings_of, merge_findings
from llm_worker_pool.domain.token_estimate import estimate_tokens
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
            result = await self._use_case.execute(request)
            result.warnings.append(f"cross_check_skipped:{reason}")
            return result

        context, context_warnings = await self._use_case.resolve_context(request)
        base = request.model_copy(
            update={"context": context, "file_paths": None, "urls": None, "provider": None}
        )
        splittable = bool(request.file_paths) and not request.context and not request.urls
        files = split_context_files(context) if splittable else {}

        arm_jobs = [self._run_arm(arm, base, files) for arm in arms]
        outcomes = await asyncio.gather(*arm_jobs)
        return self._combine(request, arms, outcomes, context, context_warnings)

    # --- one arm ---------------------------------------------------------------

    async def _run_arm(
        self, arm: str, base: DelegateTaskRequest, files: dict[str, str]
    ) -> dict[str, Any]:
        batches = self._batches(arm, base, files)
        requests = [
            base.model_copy(update={"provider": arm, "context": ctx}) for ctx in batches
        ]
        results = await asyncio.gather(
            *(self._use_case.execute(r, allow_fallback=False) for r in requests),
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
        overhead = estimate_tokens(base.task) + _PROMPT_OVERHEAD_TOKENS
        if limit is None or not base.context:
            return [base.context]
        if overhead + estimate_tokens(base.context) <= limit or len(files) < 2:
            return [base.context]

        batches: list[str] = []
        current: list[str] = []
        used = overhead
        for path, body in files.items():
            block = f"--- file: {path} ---\n{body}"
            cost = estimate_tokens(block)
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

        merged = merge_findings(
            {o["arm"]: o["findings"] for o in answered}, files=split_context_files(context)
        )
        content = {"findings": merged}
        raw_text = json.dumps(content, ensure_ascii=False, indent=2)

        warnings = list(context_warnings)
        guardrail_actions = [w for w in context_warnings if w.startswith("file_denied_sensitive:")]
        results = [r for o in answered for r in o["results"]]
        for result in results:
            for warning in result.warnings:
                if warning not in warnings:
                    warnings.append(warning)
            for action in result.guardrail_actions:
                if action not in guardrail_actions:
                    guardrail_actions.append(action)
        for o in outcomes:
            if not o["results"]:
                warnings.append(f"cross_check_partial:{o['arm']} failed: {' | '.join(o['errors'])[:300]}")
            elif o["errors"]:
                warnings.append(f"cross_check_partial:{o['arm']} {len(o['errors'])} of {o['batches']} batches failed")

        # Re-check citations on the merged list against the full context: per-arm
        # verification saw only that arm's answer (and, when batched, one batch).
        warnings = [w for w in warnings if not w.startswith("citation_unverified:")]
        verification = self._use_case.verify_citations(raw_text, content, context)
        warnings.extend(citation_warnings(verification))

        agreed = sum(1 for f in merged if len(f["agreed_by"]) > 1)

        def total(field: str) -> int | None:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            return sum(values) if values else None

        return DelegateTaskResult(
            success=True,
            content=content,
            raw_text=raw_text,
            provider="+".join(o["arm"] for o in answered),
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
            cross_check={
                "arms": arm_summary,
                "findings_total": len(merged),
                "findings_agreed": agreed,
            },
        )
