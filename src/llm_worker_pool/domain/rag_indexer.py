"""Coordinates embedding and indexing of files."""

from __future__ import annotations

from llm_worker_pool.config.settings import RagConfig
from llm_worker_pool.domain.chunker import Chunker
from llm_worker_pool.domain.interfaces import EmbeddingProvider


class RagIndexer:
    def __init__(
        self,
        config: RagConfig,
        embedding_provider: EmbeddingProvider,
        chunker: Chunker,
    ) -> None:
        self._config = config
        self._provider = embedding_provider
        self._chunker = chunker

    async def index_and_search(self, texts_by_file: dict[str, str], query: str) -> str:
        if not self._config.enabled or not texts_by_file:
            # If disabled or no text, fallback to full text
            return "\n\n".join(
                f"--- file: {path} ---\n{text}" for path, text in texts_by_file.items()
            )
            
        chunks = []
        metadata = []
        
        for file_path, text in texts_by_file.items():
            file_chunks = self._chunker.split(
                text, 
                self._config.chunk_size_tokens, 
                self._config.overlap_tokens
            )
            for i, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                metadata.append({"file_path": file_path, "chunk_idx": i, "text": chunk})
                
        if not chunks:
            return ""

        # Batch embeddings to avoid OOM or timeout
        embeddings = []
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_emb = await self._provider.embed(batch)
            embeddings.extend(batch_emb)
            
        if not embeddings:
            return ""
            
        dimension = len(embeddings[0])
        # Imported here: faiss/numpy are the optional [rag] extra (server.py checks
        # they import before enabling RAG).
        from llm_worker_pool.infra.vector_store import FaissVectorStore

        vector_store = FaissVectorStore(dimension)
        vector_store.add(embeddings, metadata)
        
        # Embed query
        query_emb = (await self._provider.embed([query]))[0]
        results = vector_store.search(query_emb, self._config.top_k)
        
        context_parts = []
        for res in results:
            context_parts.append(
                f"--- From {res['file_path']} (score: {res['score']:.2f}) ---\n{res['text']}"
            )
            
        return "\n\n".join(context_parts)
