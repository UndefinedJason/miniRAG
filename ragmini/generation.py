from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from .models import SearchHit
from .text import split_sentences, term_overlap


SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。只使用给定上下文回答。
无法从上下文确认时，明确回答“根据现有资料无法确定”。
每个事实后使用 [1]、[2] 格式标注来源，不要编造来源。"""


def build_context(hits: list[SearchHit], max_chars: int = 4000) -> str:
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("source", hit.chunk.document_id)
        block = f"[{index}] 来源: {source}\n{hit.chunk.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


class Generator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...


class ExtractiveGenerator:
    """Offline grounded baseline; selects the most query-relevant sentences."""

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        candidates: list[tuple[float, str, int]] = []
        for index, hit in enumerate(hits, start=1):
            for sentence in split_sentences(hit.chunk.text):
                score = term_overlap(question, sentence)
                if score > 0:
                    candidates.append((score, sentence, index))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return "根据现有资料无法确定。"
        chosen: list[str] = []
        seen: set[str] = set()
        for _, sentence, index in candidates:
            normalized = sentence.lower()
            if normalized not in seen:
                chosen.append(f"{sentence} [{index}]")
                seen.add(normalized)
            if len(chosen) == 3:
                break
        return " ".join(chosen)


class OpenAICompatibleGenerator:
    """Calls an OpenAI-compatible /chat/completions endpoint via stdlib."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("RAG_MODEL", "gpt-4.1-mini")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai generator")

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"上下文：\n{build_context(hits)}\n\n问题：{question}"},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API failed ({exc.code}): {detail}") from exc
        return body["choices"][0]["message"]["content"]
