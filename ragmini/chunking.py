from __future__ import annotations

from dataclasses import dataclass

from .text import split_sentences


@dataclass
class TextChunker:
    """Sentence-aware character chunking with overlap.

    Character limits keep this demo tokenizer-independent. Production systems
    normally use the target LLM tokenizer and split by token count.
    """

    chunk_size: int = 500
    overlap: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("require chunk_size > overlap >= 0")

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        sentences = split_sentences(text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            # Hard-split a single unusually long sentence.
            pieces = [sentence[i : i + self.chunk_size] for i in range(0, len(sentence), self.chunk_size)]
            for piece in pieces:
                candidate = f"{current}\n{piece}".strip() if current else piece
                if current and len(candidate) > self.chunk_size:
                    chunks.append(current)
                    prefix = current[-self.overlap :] if self.overlap else ""
                    current = f"{prefix}\n{piece}".strip()
                    if len(current) > self.chunk_size:
                        current = current[-self.chunk_size :]
                else:
                    current = candidate
        if current:
            chunks.append(current)
        return chunks
