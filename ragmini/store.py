from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Chunk


class SQLiteChunkStore:
    """Tiny persistent document/vector store backed by SQLite."""

    def __init__(self, path: str | Path = "data/rag.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.commit()

    def ensure_embedding_config(self, model_id: str, dimensions: int) -> None:
        rows = dict(self.connection.execute("SELECT key, value FROM index_metadata").fetchall())
        expected = {"embedding_model": model_id, "embedding_dimensions": str(dimensions)}
        if not rows:
            existing = self.connection.execute(
                "SELECT embedding FROM chunks LIMIT 1"
            ).fetchone()
            if existing:
                legacy_dimensions = len(json.loads(existing[0]))
                legacy_model = f"hash-embedding-v1-{legacy_dimensions}"
                if expected != {
                    "embedding_model": legacy_model,
                    "embedding_dimensions": str(legacy_dimensions),
                }:
                    raise ValueError(
                        "this legacy index has no embedding model metadata; rebuild it "
                        "in a new database before using a real embedding model"
                    )
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO index_metadata(key, value) VALUES (?, ?)", expected.items()
                )
            return
        if any(rows.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "embedding configuration does not match this index; use a new database "
                "or rebuild all documents with the same embedding model"
            )

    def replace_document(self, document_id: str, chunks: Iterable[Chunk]) -> int:
        rows = list(chunks)
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.position,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        json.dumps(chunk.embedding),
                    )
                    for chunk in rows
                ],
            )
        return len(rows)

    def all(self) -> list[Chunk]:
        rows = self.connection.execute(
            "SELECT id, document_id, position, text, metadata, embedding FROM chunks"
        ).fetchall()
        return [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                position=row["position"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
                embedding=json.loads(row["embedding"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        self.connection.close()
