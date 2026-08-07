"""Interface (Protocol) ma moi provider phai tuan theo.

Day la ranh gioi duy nhat giua application layer va cac backend LLM cu the
(Ollama, vLLM, OpenAI, Gemini, ...). Application layer chi biet den
LLMProvider, khong bao gio import truc tiep mot provider cu the nao.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from local_ai_gateway.domain.models import GenerationOptions, ProviderResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Mot backend sinh van ban tu prompt + tham so sinh.

    Cac loi transient (khong ket noi duoc, timeout) phai duoc raise thanh
    ProviderUnavailableError / ProviderTimeoutError (xem domain.exceptions),
    khong de lo exception rieng cua thu vien HTTP/SDK ra ngoai provider.
    """

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate(
        self, prompt: str, options: GenerationOptions
    ) -> ProviderResponse: ...
