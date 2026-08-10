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
    should set think=False."""


class BudgetProfile(BaseModel):
    """A 'budget' level (L1/L2/L3): name + generation options + purpose description."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    options: GenerationOptions
    timeout_seconds: float | None = None
    """None = use default timeout_seconds from AppConfig."""


class ProviderResponse(BaseModel):
    """Raw result returned from an LLMProvider, before JSON parse/validation."""

    raw_text: str
    provider_name: str
    model_name: str
    latency_ms: float
    tokens_used: int | None = None
    finish_reason: str | None = None
    from_cache: bool = False
