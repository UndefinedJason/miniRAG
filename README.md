# RAG Mini：麻雀虽小，五脏俱全

这是一个面向学习和面试的最小 RAG 项目。它只有少量 Python 代码，默认零第三方依赖即可运行，但包含一条完整链路：

```text
文档 → 句子感知切分 → 向量化 → SQLite 持久化
                              ↓
问题 → 稠密召回 + BM25 → RRF 融合 → 轻量重排 + MMR → 上下文 → 生成 → 引用/Trace
```

## 1. 快速开始

要求 Python 3.9+。

```bash
python -m ragmini.cli --db data/demo.db ingest examples/knowledge.md
python -m ragmini.cli --db data/demo.db ask "混合检索有什么好处？" --top-k 3
python -m examples.evaluate
```

默认使用离线抽取式生成器，方便无密钥测试。要运行真正的 RAG，推荐直接使用下面的“双模型模式”。输出包含回答、引用原文和检索各阶段分数。

### 使用 `.env` 持久保存配置

项目启动时会自动读取当前目录的 `.env`，且不会覆盖终端中已经存在的同名环境变量。先复制示例配置：

```bash
cp .env.example .env
```

然后只需在 `.env` 中填写 API Key。`.env` 已加入 `.gitignore`，不要把真实密钥提交或发送给他人。示例文件默认配置为 DeepSeek 生成加本地 Hash Embedding；之后可以直接运行：

```bash
python -m ragmini.cli ingest examples/knowledge.md
python -m ragmini.cli ask "什么是 RAG？" --top-k 3
```

### 使用真实 Embedding + 真实 LLM

实现遵循 OpenAI-compatible API：Embedding 请求发送到 `/embeddings`，生成请求发送到 `/chat/completions`。OpenAI 默认配置如下：

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export EMBEDDING_MODEL='text-embedding-3-small'
export RAG_MODEL='gpt-4.1-mini'

# 必须用真实 embedding 重新建一个索引；不要混用旧的 hash 向量库
python -m ragmini.cli \
  --db data/real-model.db \
  --embedding openai \
  ingest examples/knowledge.md

python -m ragmini.cli \
  --db data/real-model.db \
  --embedding openai \
  ask "混合检索有什么好处？" \
  --generator openai \
  --top-k 3
```

如果 embedding 和生成模型来自不同的兼容服务，可以分别设置：

```bash
export EMBEDDING_BASE_URL='https://embedding-provider.example/v1'
export EMBEDDING_API_KEY='embedding-key'
export EMBEDDING_MODEL='provider-embedding-model'

export OPENAI_BASE_URL='https://llm-provider.example/v1'
export OPENAI_API_KEY='llm-key'
export RAG_MODEL='provider-chat-model'
```

程序会在 SQLite 中记录 embedding 服务、模型名和向量维度。查询时配置不一致会立即报错，避免不同模型的向量被静默混用。

启动 HTTP API：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api,dev]'
uvicorn ragmini.api:app --reload
```

访问 `http://127.0.0.1:8000/docs` 查看 Swagger。主要接口：

- `POST /documents`：写入文本及元数据；同一 `source` 会幂等替换。
- `POST /query`：检索并生成带引用的回答。
- `GET /health`：健康状态和当前 chunk 数。

仅把生成阶段切换为 OpenAI-compatible 模型（仍使用演示用 Hash Embedding）：

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # 可替换为兼容服务
export RAG_MODEL='gpt-4.1-mini'
python -m ragmini.cli --db data/demo.db ask "什么是 RAG？" --generator openai
```

API 服务同时设置 `RAG_EMBEDDING=openai` 和 `RAG_GENERATOR=openai` 即可启用双模型模式。建议使用全新的 `RAG_DB` 路径：

```bash
export RAG_EMBEDDING=openai
export RAG_GENERATOR=openai
export RAG_DB=data/real-model.db
uvicorn ragmini.api:app --reload
```

## 2. 项目结构

```text
ragmini/
  chunking.py     # 句子边界、chunk size、overlap
  embedding.py    # Hash 基线 + 真实 OpenAI-compatible Embedding + cosine
  store.py        # SQLite 文档库/向量库
  retrieval.py    # Dense + BM25 + RRF + rerank + MMR
  generation.py   # Prompt、上下文预算、离线/OpenAI-compatible 生成
  pipeline.py     # ingestion 与 query 编排、引用、耗时 trace
  evaluation.py   # Recall@K、MRR、答案关键词分数
  api.py          # FastAPI 接口
  cli.py          # 命令行入口
tests/             # 切分、幂等、端到端、拒答测试
```

## 3. 设计取舍（面试重点）

### 为什么要切分和重叠？

整篇文档直接向量化会稀释主题，也浪费 LLM 上下文。小块召回更精确，但容易切断语义。本项目按句子边界优先切分，超长句硬切，并把上一块尾部带入下一块。生产环境应按目标模型 tokenizer 计数，并基于标题层级、段落或语义做结构化切分。

### 稠密检索与 BM25 有什么差异？

- 稠密检索适合语义相近但字面不同的表达。
- BM25 对产品名、错误码、人名等精确关键词更可靠。
- 本项目分别生成两个排名，用 Reciprocal Rank Fusion（RRF）融合。RRF 不要求两种分数同尺度，比直接加权分数稳健。

为了零依赖，本地 dense embedding 使用 feature hashing，它只能近似词面/短语相似，并不具备真正神经向量的语义能力。生产时只需替换 `HashEmbedding.embed()`，可接入 BGE、E5、OpenAI embeddings 等；存储层则可替换为 pgvector、Milvus、Qdrant、Elasticsearch/OpenSearch。

### 为什么还需要 rerank 和 MMR？

首阶段召回追求 Recall，候选较多；reranker 追求前几名 Precision。这里用 query-term overlap 和 dense score 实现可解释的轻量精排。生产常用 cross-encoder 或专用 rerank API。MMR 会惩罚与已选 chunk 过于相似的候选，减少相邻重叠块挤占上下文。

### 如何减少幻觉？

- system prompt 限定只能依据上下文；证据不足时拒答。
- 返回 chunk 级引用，答案可追溯。
- `temperature=0` 降低随机性，但不能保证事实正确。
- 真正上线还要做检索置信度阈值、引用一致性检查、敏感内容过滤、prompt injection 隔离和权限过滤（ACL 必须在召回前执行）。

### 如何评估？

不要只评最终答案，应分层定位问题：

| 层次 | 常用指标 | 本项目 |
|---|---|---|
| 索引/数据 | 覆盖率、重复率、解析失败率、新鲜度 | 幂等覆盖、chunk count |
| 检索 | Recall@K、MRR、nDCG | Recall@K、MRR |
| 生成 | 正确性、faithfulness、引用准确率、拒答率 | 关键词分数、引用、拒答测试 |
| 系统 | P50/P95 延迟、吞吐、token/费用、缓存命中 | retrieval/generation/total trace |

`python -m examples.evaluate` 会运行一个最小离线 golden set。真实项目需要人工标注问题、相关文档及参考答案；LLM-as-a-judge 适合扩展规模，但应抽样人工校准，避免位置偏差和自我偏好。

### RAG 常见故障如何定位？

1. 正确 chunk 没进候选：检查解析、切分、embedding、召回与 ACL。
2. 正确 chunk 召回但排名低：调 hybrid 权重、候选数或 reranker。
3. 证据正确但回答错：检查 prompt、上下文顺序、模型能力与引用约束。
4. 回答旧信息：做增量索引、版本字段、删除传播和 freshness 监控。
5. 延迟或成本高：缓存 query/embedding、批量 embedding、ANN 索引、缩小候选及上下文。

## 4. 生产化演进路线

这个实现刻意把接口分开，便于逐层替换：

1. `HashEmbedding` → BGE/E5/OpenAI embedding，并记录模型版本和向量维度。
2. SQLite 全量扫描 → 支持 HNSW/IVF 的向量数据库；数据量小于数十万时先测再选型。
3. 轻量 rerank → cross-encoder，并使用离线集调候选数与阈值。
4. txt/md → PDF、HTML、Office 解析，保留标题、页码、表格和 ACL 元数据。
5. 单轮问题 → query rewrite、多轮对话去指代；谨慎使用 HyDE，避免引入错误假设。
6. 单实例 → 异步 ingestion、队列、批处理、缓存、Tracing、限流和灰度评估。

## 5. 测试

```bash
python -m unittest discover -s tests -v
# 安装 dev 依赖后也可运行：pytest -q
```

测试覆盖 chunk overlap、同源幂等更新、端到端检索/引用和证据不足拒答。
