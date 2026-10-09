# RAG Mini：可解释的中文混合检索

这是一个面向学习和面试的最小 RAG 项目，保留了完整、可复算的检索链路：

```text
文档 → 章节/段落/句子切分 → Qwen Embedding → SQLite
问题 → Dense + 中文BM25 → RRF → Qwen Rerank → MMR → DeepSeek → 引用/Trace
```

项目重点不是堆框架，而是明确区分每个阶段的分数、排名、耗时和失败降级行为。

## 1. 安装与快速开始

要求 Python 3.9+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

离线模式不需要 API Key：

```bash
python3 -m ragmini.cli --db data/demo.db ingest examples/knowledge.md
python3 -m ragmini.cli --db data/demo.db ask "混合检索有什么好处？" --top-k 3
```

### 使用 `.env` 持久保存配置

```bash
cp .env.example .env
```

程序启动时自动读取当前目录的 `.env`，但不会覆盖终端中已经存在的同名环境变量。`.env` 已加入 `.gitignore`，不要把真实 Key 提交或发送给他人。

DeepSeek 生成 + Qwen Embedding 示例：

```dotenv
OPENAI_API_KEY=你的DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com
RAG_MODEL=deepseek-flash
RAG_GENERATOR=openai

EMBEDDING_API_KEY=你的百炼Key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=qwen3.7-text-embedding-flash
EMBEDDING_BATCH_SIZE=20
RAG_EMBEDDING=openai

RAG_DB=data/qwen37-structured.db
```

配置完成后重新建立索引；切块方式或 embedding 模型改变时不要复用旧数据库：

```bash
python3 -m ragmini.cli ingest \
  examples/knowledge.md \
  examples/company_management_policy.txt

python3 -m ragmini.cli ask "国内出差每天补贴多少钱？" --top-k 3
```

## 2. 接入 Qwen3.7 Text Rerank

Embedding 将问题和文档分别编码后比较向量；cross-encoder reranker 会同时阅读问题与候选片段，更适合判断片段是否真正回答问题。

从百炼控制台复制 Workspace ID，把下面配置加入 `.env`：

```dotenv
RAG_RERANKER=qwen
RERANK_BASE_URL=https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_MODEL=qwen3.7-text-rerank
RERANK_CANDIDATES=20
RERANK_TIMEOUT=30
RERANK_INSTRUCT=Given a user question, retrieve passages that directly answer the question.
MMR_LAMBDA=0.8
```

默认复用 `EMBEDDING_API_KEY`；如需单独的重排 Key，可设置 `RERANK_API_KEY`。

当重排接口超时或报错时，系统会自动使用本次候选的归一化 RRF 分数继续 MMR，并在 trace 中标记：

```json
{
  "reranker": {
    "enabled": true,
    "degraded": true,
    "strategy": "rrf_fallback",
    "error": "..."
  }
}
```

`RERANK_MIN_SCORE` 是可选的请求内阈值，默认不设置。Qwen 返回的是本次请求内的相对相关性，不应在没有标注集校准时把它当成跨请求概率。

## 3. 如何阅读分数

每个最终命中项包含：

```json
{
  "dense": {"score": 0.51, "rank": 2},
  "bm25": {"score": 6.94, "rank": 1},
  "rrf": {"score": 0.0325, "rank": 1},
  "rerank": {"score": 0.93, "rank": 1},
  "mmr": {
    "relevance": 0.93,
    "redundancy": 0.18,
    "lambda": 0.8,
    "score": 0.708,
    "rank": 1
  }
}
```

- `dense.score`：Qwen 向量余弦相似度，只在同一模型、同一查询内解释。
- `bm25.score`：中文词级关键词相关性，与 dense 不同尺度，不能直接相加。
- `rrf.score`：`Σ 1/(60 + rank)`，融合 Dense 与 BM25 排名，不是概率。
- `rerank.score`：cross-encoder 对本批候选的相对相关性；未启用或降级时为 `null`。
- `mmr.score`：`0.8 × relevance - 0.2 × redundancy`，兼顾相关性与结果多样性。

Trace 会分别输出 query embedding、初召回、RRF、rerank、MMR 和生成耗时，并保存每个阶段的 chunk 排名。

## 4. 中文切块与检索

- `jieba` 精确模式将中文切成词语，而不是单汉字。
- Markdown 标题、`第X章`、`附件X` 会创建独立 section，chunk 不跨 section。
- section 内按段落、句子累计，默认目标 180 个词级 token、重叠 30 个 token。
- 重叠复制完整尾句；只有单句超长时才回退硬切。
- section 标题写入 chunk 文本和 metadata，帮助 embedding 保留主题。

## 5. 分阶段评估

离线评估不会调用付费 API：

```bash
python3 -m examples.evaluate --mode offline --top-k 3
```

使用 `.env` 中的真实 Embedding 和 LLM：

```bash
python3 -m examples.evaluate --mode configured --top-k 3
```

12 条 golden cases 覆盖精确数字、同义改写、出差补贴/餐费补贴硬负例和不可回答问题。相关性标注使用 `source + passage contains + 0~3级 relevance`，避免同一文件的任意 chunk 都被判为命中。

报告分别展示 Dense、BM25、RRF、rerank、MMR 的：

- Precision@K
- Recall@K
- MRR
- nDCG@K
- 答案关键词召回率
- 可回答性准确率与拒答准确率

## 6. HTTP API

```bash
pip install -e '.[api,dev]'
uvicorn ragmini.api:app --reload
```

访问 `http://127.0.0.1:8000/docs`：

- `POST /documents`：写入文本及元数据；相同 `source` 幂等替换。
- `POST /query`：检索并生成带引用、分阶段 trace 的回答。
- `GET /health`：健康状态和 chunk 数。

## 7. 项目结构

```text
ragmini/
  chunking.py     # 章节/段落/句子切块与完整句重叠
  text.py         # jieba中文分词、英文/数字规范化
  embedding.py    # Hash基线与OpenAI-compatible Embedding
  retrieval.py    # Dense、BM25、RRF、rerank、MMR
  reranking.py    # Qwen3.7 Text Rerank与协议校验
  store.py        # SQLite文档库/向量库
  generation.py   # Grounded prompt与生成模型
  pipeline.py     # ingestion、query、引用与trace编排
  evaluation.py   # 分阶段Precision/Recall/MRR/nDCG
  api.py          # FastAPI接口
  cli.py          # 命令行入口
tests/             # 分词、切块、协议、降级、指标与端到端测试
```

## 8. 测试

```bash
python3 -m unittest discover -s tests -v
# 安装dev依赖后也可运行
pytest -q
```

测试包含中文词级分词、章节隔离、完整句 overlap、索引模型一致性、RRF、Qwen 请求映射、失败降级、MMR、评估指标和端到端引用。

## 9. 边界与生产化方向

- SQLite 当前对向量全量扫描，适合学习和小数据；规模扩大后替换为 pgvector、Milvus 或 Qdrant。
- Qwen rerank 分数只能在当前请求内比较；固定阈值必须使用业务标注集校准。
- 生成答案仍需结合引用一致性检查、ACL、Prompt Injection 隔离、敏感内容过滤和人工抽检。
- 真实评估应持续扩充 hard negatives，并分别观察检索、重排和生成，不能只看最终答案。
