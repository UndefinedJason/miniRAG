from __future__ import annotations

import re
from dataclasses import dataclass

from .text import split_sentences, token_count


MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")
CHINESE_HEADING_RE = re.compile(r"^(?:第[^\s]{1,12}章|附件[^\s]{1,12})(?:\s|$).*$")


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    section: str = ""


@dataclass
class TextChunker:
    """Section-aware, word-budgeted chunking with whole-sentence overlap."""

    chunk_size: int = 180
    overlap: int = 30

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("require chunk_size > overlap >= 0")

    def split(self, text: str) -> list[str]:
        return [piece.text for piece in self.split_with_metadata(text)]

    def split_with_metadata(self, text: str) -> list[ChunkPiece]:
        text = text.strip()
        if not text:
            return []
        chunks: list[ChunkPiece] = []
        for section, body in self._sections(text):
            chunks.extend(self._chunk_section(section, body))
        return chunks

    @staticmethod
    def _is_heading(line: str) -> bool:
        return bool(MARKDOWN_HEADING_RE.match(line) or CHINESE_HEADING_RE.match(line))

    def _sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        heading = ""
        body: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if self._is_heading(line):
                if heading or any(item for item in body):
                    sections.append((heading, "\n".join(body).strip()))
                heading, body = line, []
            else:
                body.append(raw_line)
        if heading or any(item for item in body):
            sections.append((heading, "\n".join(body).strip()))
        return sections

    def _chunk_section(self, section: str, body: str) -> list[ChunkPiece]:
        units: list[str] = []
        for paragraph in re.split(r"\n\s*\n", body):
            units.extend(split_sentences(paragraph))
        if not units and section:
            return [ChunkPiece(section, section)]

        available = max(1, self.chunk_size - token_count(section))
        expanded: list[str] = []
        for unit in units:
            expanded.extend(self._hard_split(unit, available))

        results: list[ChunkPiece] = []
        current: list[str] = []
        for unit in expanded:
            candidate = current + [unit]
            if current and self._units_count(candidate) > available:
                results.append(self._piece(section, current))
                current = self._overlap_units(current)
                while current and self._units_count(current + [unit]) > available:
                    current.pop(0)
            current.append(unit)
        if current:
            results.append(self._piece(section, current))
        return results

    def _hard_split(self, text: str, budget: int) -> list[str]:
        if token_count(text) <= budget:
            return [text]
        pieces: list[str] = []
        remaining = text
        while remaining:
            low, high, best = 1, len(remaining), 1
            while low <= high:
                middle = (low + high) // 2
                if token_count(remaining[:middle]) <= budget:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            pieces.append(remaining[:best].strip())
            remaining = remaining[best:].strip()
        return [piece for piece in pieces if piece]

    def _overlap_units(self, units: list[str]) -> list[str]:
        if not self.overlap:
            return []
        selected: list[str] = []
        for unit in reversed(units):
            if selected and self._units_count([unit] + selected) > self.overlap:
                break
            selected.insert(0, unit)
        return selected

    @staticmethod
    def _units_count(units: list[str]) -> int:
        return token_count("\n".join(units))

    @staticmethod
    def _piece(section: str, units: list[str]) -> ChunkPiece:
        body = "\n".join(units).strip()
        text = f"{section}\n{body}".strip() if section else body
        return ChunkPiece(text=text, section=section)
