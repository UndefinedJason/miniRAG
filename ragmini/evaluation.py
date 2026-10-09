from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .pipeline import RAGPipeline


@dataclass
class EvalCase:
    question: str
    expected_source: str
    answer_keywords: list[str]


def evaluate(pipeline: RAGPipeline, cases: list[EvalCase], top_k: int = 5) -> dict:
    """Small retrieval/generation evaluation without LLM-as-a-judge."""
    rows = []
    for case in cases:
        response = pipeline.ask(case.question, top_k=top_k)
        sources = [citation["source"] for citation in response.citations]
        rank = sources.index(case.expected_source) + 1 if case.expected_source in sources else None
        keyword_score = mean(
            [keyword.lower() in response.answer.lower() for keyword in case.answer_keywords]
        ) if case.answer_keywords else 1.0
        rows.append(
            {
                "question": case.question,
                "hit": rank is not None,
                "reciprocal_rank": 1 / rank if rank else 0.0,
                "keyword_score": keyword_score,
            }
        )
    return {
        f"recall@{top_k}": mean(row["hit"] for row in rows) if rows else 0.0,
        "mrr": mean(row["reciprocal_rank"] for row in rows) if rows else 0.0,
        "answer_keyword_score": mean(row["keyword_score"] for row in rows) if rows else 0.0,
        "cases": rows,
    }
