from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .text import tokens


class EmbeddingModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class HashEmbedding:
    """Deterministic feature-hashing embedding for a zero-dependency demo.

    It is not a replacement for a trained embedding model, but exercises the
    same vector-store and cosine-search path. Swap this class in production.
    """

    dimensions: int = 384

    @property
    def model_id(self) -> str:
        return f"hash-embedding-v1-{self.dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        items = tokens(text)
        # Word/character features plus local bigrams improve phrase matching.
        features = items + [f"{a}::{b}" for a, b in zip(items, items[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAICompatibleEmbedding:
    """Real embedding model through an OpenAI-compatible /embeddings API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        if self.batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be a positive integer")
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required for real embeddings")

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"embedding API failed ({exc.code}): {detail}") from exc
            ordered = sorted(body["data"], key=lambda item: item["index"])
            if len(ordered) != len(batch):
                raise RuntimeError("embedding API returned an unexpected number of vectors")
            vectors.extend([self._normalize(item["embedding"]) for item in ordered])
        return vectors

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
