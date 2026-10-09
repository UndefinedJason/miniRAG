from __future__ import annotations

import argparse
import json

from ragmini.config import load_env
from ragmini.embedding import OpenAICompatibleEmbedding
from ragmini.evaluation import EvalCase, RelevanceJudgment, evaluate
from ragmini.generation import OpenAICompatibleGenerator
from ragmini.pipeline import RAGPipeline


POLICY = "examples/company_management_policy.txt"
KNOWLEDGE = "examples/knowledge.md"


CASES = [
    EvalCase("RAG 的典型流程是什么？", [RelevanceJudgment(KNOWLEDGE, "典型流程包括", 3)], ["检索", "生成"]),
    EvalCase("RAG 检索应该看哪些指标？", [RelevanceJudgment(KNOWLEDGE, "Recall@K、MRR、nDCG", 3)], ["Recall", "MRR"]),
    EvalCase("混合检索为什么结合稠密检索和 BM25？", [RelevanceJudgment(KNOWLEDGE, "混合检索可用 RRF", 3)], ["语义", "关键词"]),
    EvalCase("国内出差每天补贴多少钱？", [RelevanceJudgment(POLICY, "出差日补贴为每人每天150元", 3)], ["150"]),
    EvalCase("工作日餐费补贴和月度上限是多少？", [RelevanceJudgment(POLICY, "工作日餐费补贴为每天25元", 3)], ["25", "550"]),
    EvalCase("请假超过三天由谁审批？", [RelevanceJudgment(POLICY, "超过3天由部门负责人和人力资源部共同审批", 3)], ["部门负责人", "人力资源部"]),
    EvalCase("工作满十年但不满二十年有几天年假？", [RelevanceJudgment(POLICY, "满10年不满20年为10天", 3)], ["10天"]),
    EvalCase("晋升委员会需要多少票才能通过？", [RelevanceJudgment(POLICY, "获得不少于4票同意视为通过", 3)], ["4票"]),
    EvalCase("转正员工主动离职要提前多久通知？", [RelevanceJudgment(POLICY, "转正员工提前30日", 3)], ["30日"]),
    EvalCase("发生疑似数据泄露后多久报告？", [RelevanceJudgment(POLICY, "应在30分钟内通知信息安全团队", 3)], ["30分钟", "2小时"]),
    EvalCase("员工绩效申诉期限是多少？", [RelevanceJudgment(POLICY, "可在五个工作日内申诉", 3)], ["五个工作日"]),
    EvalCase("公司每月提供多少远程办公补贴？", [], [], should_answer=False),
]


def build_pipeline(mode: str) -> RAGPipeline:
    if mode == "configured":
        pipeline = RAGPipeline(
            "data/evaluation-configured.db",
            embedder=OpenAICompatibleEmbedding(),
            generator=OpenAICompatibleGenerator(),
        )
    else:
        pipeline = RAGPipeline("data/evaluation-offline.db")
    pipeline.ingest_file(KNOWLEDGE)
    pipeline.ingest_file(POLICY)
    return pipeline


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "configured"], default="offline")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(build_pipeline(args.mode), CASES, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
