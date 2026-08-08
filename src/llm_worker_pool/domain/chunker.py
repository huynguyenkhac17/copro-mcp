"""Splits large text into smaller chunks based on token estimates."""

from __future__ import annotations

from llm_worker_pool.domain.token_estimate import max_chars_for_tokens


class Chunker:
    def split(self, text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
        if chunk_size_tokens <= overlap_tokens:
            raise ValueError("chunk_size_tokens must be strictly greater than overlap_tokens")
            
        chunk_size_chars = max_chars_for_tokens(chunk_size_tokens)
        overlap_chars = max_chars_for_tokens(overlap_tokens)
        
        if len(text) <= chunk_size_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size_chars
            
            # Simple chunking without complex sentence boundary logic for now
            chunks.append(text[start:end])
            
            if end >= len(text):
                break
                
            start += (chunk_size_chars - overlap_chars)
            
        return chunks
