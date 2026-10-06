"""Combines task + context into the final prompt, truncating context if it exceeds
the context capacity of the current budget.

The truncation strategy here is a simple character-based heuristic (see design flaw
#2 in architecture plan): truncate the bottom of the context, keep the top, and report
a clear warning so Claude knows the data has been truncated.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_worker_pool.domain.models import BudgetProfile, OutputFormat
from llm_worker_pool.domain.token_estimate import CONTEXT_RESERVE_RATIO, max_chars_for_tokens

_JSON_INSTRUCTION = (
    "\n\nYou MUST respond with exactly one valid JSON, without any explanations, "
    "and do not wrap it in markdown code fences."
)
# Appended to every system prompt. Each rule answers a failure seen in real use:
# asked to fill in a lab notebook, a worker returned only a fragment of a cell as
# its "full new content" (replacing it would have deleted the rest of the cell),
# made up a classmate's results for a pair-work section, and answered a section
# marked "do this without AI".
_GROUNDING_RULES = (
    "\n\nGround rules:"
    "\n- Never invent facts you cannot observe from the input: other people's "
    "results, actions the user took, measurements you did not compute. Leave such "
    "fields unfilled and say they need the user."
    "\n- Respect restrictions written in the input itself (e.g. a section marked "
    "NO_AI or 'do by hand'): skip it and say so instead of answering it."
    "\n- When asked for the full/new content of something, return all of it, not "
    "an excerpt - keep every unchanged part verbatim."
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

        system_prompt = (profile.options.system_prompt or "").rstrip() + _GROUNDING_RULES
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
