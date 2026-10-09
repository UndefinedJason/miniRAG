from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean

from .models import Chunk
from .pipeline import RAGPipeline


@dataclass(frozen=True)
class RelevanceJudgment:
    source: str
    contains: str
    relevance: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.relevance <= 3:
            raise ValueError("relevance must be between 0 and 3")


@dataclass
class EvalCase:
    question: str
    judgments: list[RelevanceJudgment]
    answer_keywords: list[str] = field(default_factory=list)
    should_answer: bool = True


def _graded_chunk_ids(chunks: list[Chunk], judgments: list[RelevanceJudgment]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        matching = [
            judgment.relevance
            for judgment in judgments
            if judgment.source == source and judgment.contains in chunk.text
        ]
        if matching:
            grades[chunk.id] = max(matching)
    return grades


def ranking_metrics(
    ranking: list[str], grades: dict[str, int], top_k: int
) -> dict[str, float | None]:
    ranked = ranking[:top_k]
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade > 0}
    hits = [chunk_id for chunk_id in ranked if chunk_id in relevant]
    first_rank = next(
        (rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
        None,
    )
    dcg = sum(
        (2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked, start=1)
    )
    ideal_grades = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:top_k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return {
        "precision": len(hits) / len(ranked) if ranked else 0.0,
        "recall": len(set(hits)) / len(relevant) if relevant else None,
        "mrr": 1 / first_rank if first_rank else 0.0,
        "ndcg": dcg / idcg if idcg else None,
    }


def evaluate(pipeline: RAGPipeline, cases: list[EvalCase], top_k: int = 5) -> dict:
    """Evaluate every retrieval stage and deterministic answer-level signals."""
    chunks = pipeline.store.all()
    rows: list[dict] = []
    stage_names = ("dense", "bm25", "rrf", "rerank", "mmr")
    for case in cases:
        response = pipeline.ask(case.question, top_k=top_k)
        grades = _graded_chunk_ids(chunks, case.judgments)
        stage_metrics = {
            stage: ranking_metrics(response.trace.get("rankings", {}).get(stage, []), grades, top_k)
            for stage in stage_names
        }
        keyword_recall = (
            mean(keyword.lower() in response.answer.lower() for keyword in case.answer_keywords)
            if case.answer_keywords
            else 1.0
        )
        refused = "根据现有资料无法确定" in response.answer
        rows.append(
            {
                "question": case.question,
                "relevant_chunks": len(grades),
                "stages": stage_metrics,
                "answer_keyword_recall": keyword_recall,
                "refused": refused,
                "answerability_correct": refused != case.should_answer,
            }
        )

    report: dict = {"top_k": top_k, "case_count": len(rows), "stages": {}, "cases": rows}
    for stage in stage_names:
        def metric_mean(metric: str) -> float:
            values = [
                row["stages"][stage][metric]
                for row in rows
                if row["stages"][stage][metric] is not None
            ]
            return mean(values) if values else 0.0

        report["stages"][stage] = {
            f"precision@{top_k}": metric_mean("precision"),
            f"recall@{top_k}": metric_mean("recall"),
            "mrr": metric_mean("mrr"),
            f"ndcg@{top_k}": metric_mean("ndcg"),
        }
    report["answers"] = {
        "keyword_recall": mean(row["answer_keyword_recall"] for row in rows) if rows else 0.0,
        "answerability_accuracy": mean(row["answerability_correct"] for row in rows) if rows else 0.0,
        "refusal_accuracy": mean(
            row["answerability_correct"] for row, case in zip(rows, cases) if not case.should_answer
        ) if any(not case.should_answer for case in cases) else None,
    }
    return report
