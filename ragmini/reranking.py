from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import SearchHit


@dataclass(frozen=True)
class RerankResult:
    scores: dict[str, float]
    ranking: list[str]
    usage: dict[str, int]
    model: str


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit]) -> RerankResult: ...


class QwenTextReranker:
    """Qwen text reranking through Alibaba Model Studio's DashScope API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        instruct: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("RERANK_BASE_URL", "")).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("RERANK_API_KEY")
            or os.getenv("EMBEDDING_API_KEY", "")
        )
        self.model = model or os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
        self.timeout = timeout or float(os.getenv("RERANK_TIMEOUT", "30"))
        self.instruct = instruct or os.getenv(
            "RERANK_INSTRUCT",
            "Given a user question, retrieve passages that directly answer the question.",
        )
        if not self.base_url or "{WorkspaceId}" in self.base_url:
            raise ValueError("RERANK_BASE_URL must contain a real Workspace ID")
        if not self.api_key:
            raise ValueError("RERANK_API_KEY or EMBEDDING_API_KEY is required")

    def rerank(self, query: str, hits: list[SearchHit]) -> RerankResult:
        if not hits:
            return RerankResult({}, [], {}, self.model)
        payload = json.dumps(
            {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": [hit.chunk.text for hit in hits],
                },
                "parameters": {
                    "top_n": len(hits),
                    "instruct": self.instruct,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").replace(self.api_key, "***")
            raise RuntimeError(f"rerank API failed ({exc.code}): {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"rerank API connection failed: {exc.reason}") from exc

        results = body.get("output", {}).get("results", [])
        scores: dict[str, float] = {}
        ranking: list[str] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(hits):
                raise RuntimeError("rerank API returned an invalid document index")
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise RuntimeError("rerank API returned an invalid relevance score")
            chunk_id = hits[index].chunk.id
            scores[chunk_id] = float(score)
            ranking.append(chunk_id)
        if len(scores) != len(hits):
            raise RuntimeError("rerank API did not score every candidate")
        return RerankResult(
            scores=scores,
            ranking=ranking,
            usage=body.get("usage", {}),
            model=body.get("model", self.model),
        )


def reranker_from_env() -> Reranker | None:
    mode = os.getenv("RAG_RERANKER", "rrf").lower()
    if mode == "rrf":
        return None
    if mode == "qwen":
        return QwenTextReranker()
    raise ValueError("RAG_RERANKER must be 'rrf' or 'qwen'")
