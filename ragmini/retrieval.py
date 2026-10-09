from __future__ import annotations

import math
import time
from collections import Counter

from .embedding import EmbeddingModel, cosine
from .models import Chunk, RetrievalResult, SearchHit
from .text import tokens


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
    def __init__(
        self,
        embedder: EmbeddingModel,
        rrf_k: int = 60,
        mmr_lambda: float = 0.8,
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        candidates: int = 20,
        query_vector: list[float] | None = None,
    ) -> RetrievalResult:
        if not query.strip() or not chunks:
            return RetrievalResult([], {}, {}, {"enabled": False, "degraded": False})
        started = time.perf_counter()
        query_vector = query_vector or self.embedder.embed(query)
        dense = {chunk.id: cosine(query_vector, chunk.embedding) for chunk in chunks}
        sparse = _bm25(query, chunks)
        dense_rank = sorted(chunks, key=lambda chunk: (-dense[chunk.id], chunk.id))
        sparse_rank = sorted(chunks, key=lambda chunk: (-sparse[chunk.id], chunk.id))
        dense_positions = {chunk.id: rank for rank, chunk in enumerate(dense_rank, start=1)}
        sparse_positions = {chunk.id: rank for rank, chunk in enumerate(sparse_rank, start=1)}
        initial_at = time.perf_counter()
        fused: dict[str, float] = {}
        for ranking in (dense_rank[:candidates], sparse_rank[:candidates]):
            for rank, chunk in enumerate(ranking, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (self.rrf_k + rank)
        by_id = {chunk.id: chunk for chunk in chunks}
        candidate_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:candidates]
        hits = [
            SearchHit(
                chunk=by_id[chunk_id],
                score=fused[chunk_id],
                dense_score=dense[chunk_id],
                dense_rank=dense_positions[chunk_id],
                sparse_score=sparse[chunk_id],
                sparse_rank=sparse_positions[chunk_id],
                rrf_score=fused[chunk_id],
                rrf_rank=rank,
            )
            for rank, chunk_id in enumerate(candidate_ids, start=1)
        ]
        rrf_at = time.perf_counter()
        self._set_rrf_relevance(hits)
        selected = self._mmr(hits, top_k)
        finished = time.perf_counter()
        return RetrievalResult(
            hits=selected,
            rankings={
                "dense": [chunk.id for chunk in dense_rank],
                "bm25": [chunk.id for chunk in sparse_rank],
                "rrf": candidate_ids,
                "rerank": candidate_ids,
                "mmr": [hit.chunk.id for hit in selected],
            },
            timings={
                "initial_retrieval_ms": round((initial_at - started) * 1000, 2),
                "rrf_ms": round((rrf_at - initial_at) * 1000, 2),
                "rerank_ms": 0.0,
                "mmr_ms": round((finished - rrf_at) * 1000, 2),
            },
            reranker={"enabled": False, "degraded": False, "strategy": "rrf_fallback"},
        )

    @staticmethod
    def _set_rrf_relevance(hits: list[SearchHit]) -> None:
        if not hits:
            return
        values = [hit.rrf_score for hit in hits]
        low, high = min(values), max(values)
        for hit in hits:
            hit.mmr_relevance = 1.0 if high == low else (hit.rrf_score - low) / (high - low)

    def _mmr(self, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        selected: list[SearchHit] = []
        remaining = hits.copy()
        while remaining and len(selected) < top_k:
            def utility(hit: SearchHit) -> tuple[float, float]:
                redundancy = max(
                    (cosine(hit.chunk.embedding, item.chunk.embedding) for item in selected),
                    default=0.0,
                )
                score = self.mmr_lambda * hit.mmr_relevance - (1 - self.mmr_lambda) * redundancy
                return score, redundancy

            best = max(remaining, key=lambda hit: (utility(hit)[0], -hit.rrf_rank))
            best.mmr_score, best.mmr_redundancy = utility(best)
            best.final_rank = len(selected) + 1
            best.score = best.mmr_score
            selected.append(best)
            remaining.remove(best)
        return selected
