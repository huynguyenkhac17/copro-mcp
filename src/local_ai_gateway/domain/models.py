"""Cac model thuan domain, khong phu thuoc provider/framework cu the."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OutputFormat = Literal["text", "json"]


class GenerationOptions(BaseModel):
    """Tham so sinh van ban - anh xa 1-1 toi 'options' cua Ollama, nhung
    khong phu thuoc Ollama (provider khac se tu map lai theo API rieng).
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
    """Bat/tat 'thinking' cho model ho tro reasoning (vd qwen3.5). None = mac
    dinh cua model. Cac model thinking co the tieu het num_predict cho phan
    suy nghi noi bo va khong con token nao cho cau tra loi cuoi - budget can
    toc do nhanh (L1) nen dat think=False."""


class BudgetProfile(BaseModel):
    """Mot muc do 'budget' (L1/L2/L3): ten + tham so sinh + mo ta muc dich."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    options: GenerationOptions
    timeout_seconds: float | None = None
    """None = dung timeout_seconds mac dinh cua AppConfig."""


class ProviderResponse(BaseModel):
    """Ket qua tho tra ve tu mot LLMProvider, chua qua parse/validate JSON."""

    raw_text: str
    provider_name: str
    model_name: str
    latency_ms: float
    tokens_used: int | None = None
    finish_reason: str | None = None
