from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_env
from .embedding import OpenAICompatibleEmbedding
from .generation import OpenAICompatibleGenerator
from .pipeline import RAGPipeline


load_env()


def _generator():
    return OpenAICompatibleGenerator() if os.getenv("RAG_GENERATOR") == "openai" else None


def _embedder():
    return OpenAICompatibleEmbedding() if os.getenv("RAG_EMBEDDING") == "openai" else None


app = FastAPI(title="RAG Mini", version="0.1.0")
pipeline = RAGPipeline(
    os.getenv("RAG_DB", "data/rag.db"), generator=_generator(), embedder=_embedder()
)


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": pipeline.store.count()}


@app.post("/documents")
def ingest(request: IngestRequest) -> dict:
    return pipeline.ingest(request.text, request.source, request.metadata)


@app.post("/query")
def query(request: AskRequest) -> dict:
    try:
        return asdict(pipeline.ask(request.question, request.top_k))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
