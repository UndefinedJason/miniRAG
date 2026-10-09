from __future__ import annotations

import argparse
import json
import os

from .config import load_env
from .embedding import OpenAICompatibleEmbedding
from .generation import OpenAICompatibleGenerator
from .pipeline import RAGPipeline


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Small but complete RAG demo")
    parser.add_argument("--db", default=os.getenv("RAG_DB", "data/rag.db"))
    parser.add_argument(
        "--embedding",
        choices=["hash", "openai"],
        default=os.getenv("RAG_EMBEDDING", "hash"),
        help="use 'openai' for a real OpenAI-compatible embedding model",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="ingest .txt/.md files")
    ingest.add_argument("files", nargs="+")
    ask = subparsers.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument(
        "--generator",
        choices=["extractive", "openai"],
        default=os.getenv("RAG_GENERATOR", "extractive"),
    )
    args = parser.parse_args()

    generator = OpenAICompatibleGenerator() if getattr(args, "generator", None) == "openai" else None
    embedder = OpenAICompatibleEmbedding() if args.embedding == "openai" else None
    pipeline = RAGPipeline(args.db, generator=generator, embedder=embedder)
    if args.command == "ingest":
        results = [pipeline.ingest_file(path) for path in args.files]
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        result = pipeline.ask(args.question, top_k=args.top_k)
        print(json.dumps({"answer": result.answer, "citations": result.citations, "trace": result.trace}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
