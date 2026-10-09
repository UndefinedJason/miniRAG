from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .chunking import TextChunker
from .embedding import EmbeddingModel, HashEmbedding
from .generation import ExtractiveGenerator, Generator
from .models import Chunk, RAGResponse
from .retrieval import HybridRetriever
from .store import SQLiteChunkStore
from .text import stable_id


class RAGPipeline:
    def __init__(
        self,
        db_path: str | Path = "data/rag.db",
        chunk_size: int = 180,
        overlap: int = 30,
        generator: Generator | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.store = SQLiteChunkStore(db_path)
        self.chunker = TextChunker(chunk_size, overlap)
        self.embedder = embedder or HashEmbedding()
        self.retriever = HybridRetriever(self.embedder)
        self.generator = generator or ExtractiveGenerator()

    def ingest(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        document_id = stable_id(source)
        base_metadata = {**(metadata or {}), "source": source}
        pieces = self.chunker.split_with_metadata(text)
        contents = [piece.text for piece in pieces]
        embeddings = self.embedder.embed_many(contents)
        if embeddings:
            self.store.ensure_embedding_config(self.embedder.model_id, len(embeddings[0]))
        chunks = [
            Chunk(
                id=stable_id(document_id, str(position), content),
                document_id=document_id,
                text=content,
                position=position,
                metadata={**base_metadata, "section": piece.section},
                embedding=embedding,
            )
            for position, (piece, content, embedding) in enumerate(
                zip(pieces, contents, embeddings)
            )
        ]
        count = self.store.replace_document(document_id, chunks)
        return {"document_id": document_id, "chunks": count, "source": source}

    def ingest_file(self, path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        if file_path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("this minimal version supports .txt and .md files")
        return self.ingest(file_path.read_text(encoding="utf-8"), str(file_path))

    def ask(self, question: str, top_k: int = 5, include_trace: bool = True) -> RAGResponse:
        started = time.perf_counter()
        chunks = self.store.all()
        query_vector = None
        if chunks:
            # Validate before retrieval so a query never compares incompatible vectors.
            query_vector = self.embedder.embed(question)
            self.store.ensure_embedding_config(self.embedder.model_id, len(query_vector))
        hits = self.retriever.search(
            question, chunks, top_k=top_k, query_vector=query_vector
        )
        retrieved_at = time.perf_counter()
        answer = self.generator.generate(question, hits)
        finished = time.perf_counter()
        citations = [
            {
                "index": index,
                "chunk_id": hit.chunk.id,
                "document_id": hit.chunk.document_id,
                "source": hit.chunk.metadata.get("source"),
                "position": hit.chunk.position,
                "text": hit.chunk.text,
                "score": round(hit.rerank_score, 6),
            }
            for index, hit in enumerate(hits, start=1)
        ]
        trace = {}
        if include_trace:
            trace = {
                "retrieved": len(hits),
                "corpus_chunks": self.store.count(),
                "retrieval_ms": round((retrieved_at - started) * 1000, 2),
                "generation_ms": round((finished - retrieved_at) * 1000, 2),
                "total_ms": round((finished - started) * 1000, 2),
                "scores": [
                    {
                        "chunk_id": hit.chunk.id,
                        "dense": round(hit.dense_score, 6),
                        "bm25": round(hit.sparse_score, 6),
                        "rerank": round(hit.rerank_score, 6),
                    }
                    for hit in hits
                ],
            }
        return RAGResponse(answer=answer, citations=citations, trace=trace)
