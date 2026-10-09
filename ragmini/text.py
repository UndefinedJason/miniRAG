from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter

import jieba

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<!\d)\.(?!\d)\s*|\n+")
jieba.setLogLevel(logging.WARNING)


def tokens(text: str) -> list[str]:
    """Tokenize Chinese as words while preserving normalized Latin terms."""
    result: list[str] = []
    for item in jieba.lcut(text, cut_all=False):
        result.extend(token.lower() for token in TOKEN_RE.findall(item))
    return result


def token_count(text: str) -> int:
    return len(tokens(text))


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def term_overlap(query: str, text: str) -> float:
    q = Counter(tokens(query))
    d = Counter(tokens(text))
    if not q:
        return 0.0
    return sum(min(count, d[token]) for token, count in q.items()) / sum(q.values())
