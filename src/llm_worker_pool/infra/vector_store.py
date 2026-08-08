"""FAISS vector store."""

from __future__ import annotations

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        # Flat inner product for cosine similarity (assumes normalized vectors)
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []

    def add(self, embeddings: list[list[float]], metadata: list[dict]) -> None:
        if not embeddings:
            return
            
        arr = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(arr)
        
        self._index.add(arr)
        self._metadata.extend(metadata)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self._index.ntotal == 0:
            return []
            
        arr = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(arr)
        
        # Search returns distances and indices
        distances, indices = self._index.search(arr, min(top_k, self._index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                meta = dict(self._metadata[idx])
                meta["score"] = float(dist)
                results.append(meta)
                
        return results
