from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    dense_rank: int = 0
    sparse_score: float = 0.0
    sparse_rank: int = 0
    rrf_score: float = 0.0
    rrf_rank: int = 0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    mmr_relevance: float = 0.0
    mmr_redundancy: float = 0.0
    mmr_score: float = 0.0
    final_rank: int = 0


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    rankings: dict[str, list[str]]
    timings: dict[str, float]
    reranker: dict[str, Any]


@dataclass
class RAGResponse:
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]
