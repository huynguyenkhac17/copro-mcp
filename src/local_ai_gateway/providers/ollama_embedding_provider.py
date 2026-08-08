"""Ollama-based embedding provider via httpx."""

from __future__ import annotations

import httpx

from local_ai_gateway.domain.exceptions import InvalidResponseError, ProviderUnavailableError


class OllamaEmbeddingProvider:
    def __init__(
        self,
        endpoint: str,
        model: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._client = http_client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = http_client is None

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        payload = {
            "model": self._model,
            "input": texts,
        }
        
        try:
            response = await self._client.post(
                f"{self._endpoint}/api/embed",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Ollama embedding failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"Ollama embedding returned non-JSON body: {exc}") from exc

        return data.get("embeddings", [])
        
    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
