from __future__ import annotations

import math
from collections import Counter

from .embedding import EmbeddingModel, cosine
from .models import Chunk, SearchHit
from .text import term_overlap, tokens


def _bm25(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    documents = [tokens(chunk.text) for chunk in chunks]
    if not documents:
        return {}
    avg_len = sum(map(len, documents)) / len(documents) or 1.0
    query_terms = set(tokens(query))
    document_frequency = {
        term: sum(1 for document in documents if term in document) for term in query_terms
    }
    scores: dict[str, float] = {}
    for chunk, document in zip(chunks, documents):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            idf = math.log(1 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(document) / avg_len)
            score += idf * frequency * (k1 + 1) / denominator
        scores[chunk.id] = score
    return scores


class HybridRetriever:
    def __init__(self, embedder: EmbeddingModel, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        candidates: int = 20,
        query_vector: list[float] | None = None,
    ) -> list[SearchHit]:
        if not query.strip() or not chunks:
            return []
        query_vector = query_vector or self.embedder.embed(query)
        dense = {chunk.id: cosine(query_vector, chunk.embedding) for chunk in chunks}
        sparse = _bm25(query, chunks)
        dense_rank = sorted(chunks, key=lambda chunk: dense[chunk.id], reverse=True)[:candidates]
        sparse_rank = sorted(chunks, key=lambda chunk: sparse[chunk.id], reverse=True)[:candidates]
        fused: dict[str, float] = {}
        for ranking in (dense_rank, sparse_rank):
            for rank, chunk in enumerate(ranking, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (self.rrf_k + rank)
        by_id = {chunk.id: chunk for chunk in chunks}
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:candidates]
        hits = [
            SearchHit(
                chunk=by_id[chunk_id],
                score=fused[chunk_id],
                dense_score=dense[chunk_id],
                sparse_score=sparse[chunk_id],
                rerank_score=0.7 * term_overlap(query, by_id[chunk_id].text) + 0.3 * max(dense[chunk_id], 0),
            )
            for chunk_id in candidate_ids
        ]
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        return self._mmr(hits, top_k)

    def _mmr(self, hits: list[SearchHit], top_k: int, diversity: float = 0.2) -> list[SearchHit]:
        selected: list[SearchHit] = []
        remaining = hits.copy()
        while remaining and len(selected) < top_k:
            def utility(hit: SearchHit) -> float:
                redundancy = max(
                    (cosine(hit.chunk.embedding, item.chunk.embedding) for item in selected),
                    default=0.0,
                )
                return (1 - diversity) * hit.rerank_score - diversity * redundancy

            best = max(remaining, key=utility)
            selected.append(best)
            remaining.remove(best)
        return selected
