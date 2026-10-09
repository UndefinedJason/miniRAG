import json

from ragmini.evaluation import EvalCase, evaluate
from ragmini.pipeline import RAGPipeline


pipeline = RAGPipeline("data/example.db")
pipeline.ingest_file("examples/knowledge.md")
cases = [
    EvalCase("RAG 的典型流程是什么？", "examples/knowledge.md", ["检索", "生成"]),
    EvalCase("Recall@K 和 MRR 属于哪一层指标？", "examples/knowledge.md", ["Recall", "MRR"]),
]
print(json.dumps(evaluate(pipeline, cases), ensure_ascii=False, indent=2))
