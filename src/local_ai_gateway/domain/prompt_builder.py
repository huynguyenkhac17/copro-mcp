"""Ghep task + context thanh prompt cuoi cung, cat bot context neu vuot
kha nang context cua budget dang dung.

Chien luoc cat o day la heuristic don gian theo so ky tu (xem design flaw
#2 trong plan kien truc): cat phan duoi cua context, giu phan dau, va bao
warning ro rang de Claude biet du lieu da bi rut gon.
"""

from __future__ import annotations

from pydantic import BaseModel

from local_ai_gateway.domain.models import BudgetProfile, OutputFormat

# Heuristic tho: trung binh ~4 ky tu / token cho van ban Anh-Viet tron.
_CHARS_PER_TOKEN = 4
# Danh rieng mot phan context window cho system prompt + task + output.
_CONTEXT_RESERVE_RATIO = 0.25

_JSON_INSTRUCTION = (
    "\n\nBan PHAI tra loi bang duy nhat mot JSON hop le, khong kem giai "
    "thich, khong bao trong markdown code fence."
)
_JSON_SCHEMA_INSTRUCTION_TMPL = (
    "\n\nJSON tra ve phai tuan theo schema sau:\n{schema}"
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

        max_context_chars = int(
            profile.options.num_ctx * _CHARS_PER_TOKEN * (1 - _CONTEXT_RESERVE_RATIO)
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
