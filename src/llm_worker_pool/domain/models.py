"""Pure domain models, completely independent of specific providers/frameworks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OutputFormat = Literal["text", "json"]


class GenerationOptions(BaseModel):
    """Text generation parameters - 1-1 mapping to Ollama 'options', but
    independent of Ollama (other providers will map this to their own APIs).
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 512
    num_ctx: int = 8192
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    format: Literal["json"] | dict[str, Any] | None = None
    think: bool | None = None
    """Enable/disable 'thinking' for reasoning-capable models (e.g. qwen3). None = model default.
    Thinking models can consume all num_predict for internal reasoning
    leaving zero tokens for the final answer - budgets needing high speed (L1)
    should set think=False.

    Ollama-only. Cloud models reached over the OpenAI-compatible wire protocol have
    no such boolean - use `reasoning_effort` for those."""

    reasoning_effort: str | None = None
    """How much of the token budget a reasoning model may spend on internal thought.

    The OpenAI-compatible equivalent of `think`. Accepted by Gemini's shim as
    "low" / "medium" / "high"; "none" is rejected with HTTP 400, so the floor is
    "low". None = leave the field out entirely and take the model's default, which
    for gemini-flash means unrestricted thinking - measured at ~979 of 1024 tokens
    on a trivial prompt, leaving 41 for the answer."""


class BudgetProfile(BaseModel):
    """A 'budget' level (L1/L2/L3): name + generation options + purpose description."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    options: GenerationOptions
    timeout_seconds: float | None = None
    """None = use default timeout_seconds from AppConfig."""


class TaskProfile(BaseModel):
    """A kind of work (review, summarize, extract...) - see config/task_profiles.yaml.
    Adds an output contract on top of a budget; selected by `task_type` on a request."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    system_addon: str = ""
    """Appended to the budget's system prompt."""
    default_budget: str | None = None
    """Used when the request leaves budget at its default."""
    output_format: OutputFormat | None = None
    """Forced output format; None = whatever the request asks for."""
    json_schema: dict[str, Any] | None = None
    """Schema used when the request supplies none."""
    validators: list[str] = Field(default_factory=list)
    """Post-checks run on the answer: "json_repair", "citations"."""


class ProviderResponse(BaseModel):
    """Raw result returned from an LLMProvider, before JSON parse/validation."""

    raw_text: str
    provider_name: str
    model_name: str
    latency_ms: float
    tokens_used: int | None = None
    """Grand total: prompt + thinking + completion. Kept as-is for backward compatibility -
    do NOT compare it against num_predict, which caps completion alone."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    from_cache: bool = False

    @property
    def thinking_tokens(self) -> int | None:
        """Tokens spent on internal reasoning, which providers bill inside the total
        but report in neither prompt nor completion. Derived rather than reported."""
        if self.tokens_used is None or self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return max(0, self.tokens_used - self.prompt_tokens - self.completion_tokens)

    @property
    def truncated(self) -> bool:
        """The model stopped because it ran out of output budget, not because it
        finished its answer. "length" is the OpenAI-compatible spelling, "MAX_TOKENS"
        Gemini's native one, "limit" what Ollama reports."""
        return (self.finish_reason or "").lower() in ("length", "max_tokens", "limit")
