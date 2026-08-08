"""Combines task + context into the final prompt, truncating context if it exceeds
the context capacity of the current budget.

The truncation strategy here is a simple character-based heuristic (see design flaw
#2 in architecture plan): truncate the bottom of the context, keep the top, and report
a clear warning so Claude knows the data has been truncated.
"""

from __future__ import annotations

from pydantic import BaseModel

from local_ai_gateway.domain.models import BudgetProfile, OutputFormat
from local_ai_gateway.domain.token_estimate import CONTEXT_RESERVE_RATIO, max_chars_for_tokens

_JSON_INSTRUCTION = (
    "\n\nYou MUST respond with exactly one valid JSON, without any explanations, "
    "and do not wrap it in markdown code fences."
)
_JSON_SCHEMA_INSTRUCTION_TMPL = (
    "\n\nThe returned JSON must follow this schema:\n{schema}"
)


class BuiltPrompt(BaseModel):
    system_prompt: str | None
    user_prompt: str
    warnings: list[str] = []


class PromptBuilder:
    def build(
        self,
        task: str,
        context: str | None,
        profile: BudgetProfile,
        output_format: OutputFormat = "text",
        json_schema: dict | None = None,
    ) -> BuiltPrompt:
        warnings: list[str] = []

        system_prompt = profile.options.system_prompt
        if output_format == "json":
            extra = _JSON_INSTRUCTION
            if json_schema is not None:
                extra += _JSON_SCHEMA_INSTRUCTION_TMPL.format(schema=json_schema)
            system_prompt = (system_prompt or "") + extra

        max_context_chars = max_chars_for_tokens(
            int(profile.options.num_ctx * (1 - CONTEXT_RESERVE_RATIO))
        )

        used_context = context or ""
        if len(used_context) > max_context_chars:
            used_context = used_context[:max_context_chars]
            warnings.append("context_truncated")

        if used_context:
            user_prompt = f"{task}\n\n---\nContext:\n{used_context}"
        else:
            user_prompt = task

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            warnings=warnings,
        )
