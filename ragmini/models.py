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
    sparse_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]
