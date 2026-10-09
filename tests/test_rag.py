import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragmini.chunking import TextChunker
from ragmini.embedding import OpenAICompatibleEmbedding
from ragmini.generation import OpenAICompatibleGenerator
from ragmini.models import Chunk, SearchHit
from ragmini.pipeline import RAGPipeline
from ragmini.retrieval import HybridRetriever
from ragmini.reranking import QwenTextReranker, RerankResult
from ragmini.text import split_sentences, tokens


class FakeEmbedding:
    model_id = "fake-real-embedding-v1"

    def embed(self, text):
        return [1.0, 0.0, 0.0] if "RAG" in text else [0.0, 1.0, 0.0]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class FakeLLM:
    def generate(self, question, hits):
        return f"模型回答：{hits[0].chunk.text} [1]"


class FakeReranker:
    def rerank(self, query, hits):
        ranking = [hit.chunk.id for hit in reversed(hits)]
        return RerankResult(
            scores={chunk_id: 0.9 - index * 0.1 for index, chunk_id in enumerate(ranking)},
            ranking=ranking,
            usage={"total_tokens": 42},
            model="fake-reranker",
        )


class BrokenReranker:
    def rerank(self, query, hits):
        raise RuntimeError("temporary failure")


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "rag.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chunker_has_overlap(self):
        chunks = TextChunker(chunk_size=10, overlap=3).split(
            "第一句话比较长。第二句话也比较长。第三句话结束。"
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("第二句话也比较长。", chunks[1])

    def test_chinese_tokenization_uses_words(self):
        result = tokens("我司人员出差补贴每天多少钱？RAG 2026")
        self.assertIn("出差", result)
        self.assertIn("补贴", result)
        self.assertIn("rag", result)
        self.assertNotIn("差", result)

    def test_sentence_split_preserves_decimal_numbers(self):
        sentences = split_sentences("绩效系数为1.5。下一句。")
        self.assertEqual(sentences, ["绩效系数为1.5。", "下一句。"])

    def test_chunker_never_crosses_sections_and_preserves_heading(self):
        pieces = TextChunker(chunk_size=12, overlap=2).split_with_metadata(
            "第一章 请假管理\n员工应提前申请。直属主管负责审批。\n\n"
            "第二章 差旅管理\n出差补贴每天一百五十元。"
        )
        self.assertTrue(all(piece.section in piece.text for piece in pieces))
        self.assertFalse(any("请假" in piece.text and "差旅管理" in piece.text for piece in pieces))

    def test_ingest_replaces_same_source(self):
        pipeline = RAGPipeline(self.db_path, chunk_size=20, overlap=5)
        pipeline.ingest("苹果是一种水果。" * 10, "fruit.md")
        pipeline.ingest("苹果是红色的。", "fruit.md")
        self.assertEqual(pipeline.store.count(), 1)

    def test_end_to_end_answer_with_citation(self):
        pipeline = RAGPipeline(self.db_path)
        pipeline.ingest(
            "RAG 包含检索和生成两个主要阶段。混合检索可以结合关键词和语义匹配。",
            "guide.md",
        )
        pipeline.ingest("北京是中国的首都。", "city.md")
        response = pipeline.ask("RAG 有哪些主要阶段？", top_k=2)
        self.assertIn("检索", response.answer)
        self.assertIn("[1]", response.answer)
        self.assertEqual(response.citations[0]["source"], "guide.md")
        self.assertEqual(response.trace["corpus_chunks"], 2)

    def test_unknown_question_refuses(self):
        pipeline = RAGPipeline(self.db_path)
        pipeline.ingest("猫是一种动物。", "animals.md")
        response = pipeline.ask("量子计算机有多少个量子比特？")
        self.assertEqual(response.answer, "根据现有资料无法确定。")

    def test_real_model_interfaces_are_used_end_to_end(self):
        pipeline = RAGPipeline(
            self.db_path, embedder=FakeEmbedding(), generator=FakeLLM()
        )
        pipeline.ingest("RAG 先检索，再生成。", "rag.md")
        response = pipeline.ask("RAG 是什么？")
        self.assertIn("模型回答", response.answer)
        self.assertEqual(response.citations[0]["source"], "rag.md")

    def test_retrieval_exposes_rrf_and_mmr_stages(self):
        embedder = FakeEmbedding()
        chunks = [
            Chunk("a", "d", "RAG 检索", 0, {}, [1.0, 0.0, 0.0]),
            Chunk("b", "d", "RAG 生成", 1, {}, [0.9, 0.1, 0.0]),
            Chunk("c", "d", "天气预报", 2, {}, [0.0, 1.0, 0.0]),
        ]
        result = HybridRetriever(embedder).search("RAG", chunks, top_k=2)
        first = result.hits[0]
        self.assertEqual(first.rrf_rank, 1)
        self.assertEqual(first.final_rank, 1)
        self.assertEqual(first.mmr_redundancy, 0.0)
        self.assertIsNone(first.rerank_score)
        self.assertEqual(result.rankings["mmr"], [hit.chunk.id for hit in result.hits])

    def test_rrf_uses_ranks_not_raw_score_scales(self):
        chunks = [
            Chunk("dense", "d", "无关键词", 0, {}, [1.0, 0.0, 0.0]),
            Chunk("sparse", "d", "目标词", 1, {}, [0.0, 1.0, 0.0]),
        ]
        result = HybridRetriever(FakeEmbedding()).search(
            "目标词", chunks, top_k=2, query_vector=[1.0, 0.0, 0.0]
        )
        by_id = {hit.chunk.id: hit for hit in result.hits}
        self.assertAlmostEqual(by_id["dense"].rrf_score, by_id["sparse"].rrf_score)

    def test_remote_reranker_scores_feed_mmr(self):
        chunks = [
            Chunk("a", "d", "RAG 检索", 0, {}, [1.0, 0.0, 0.0]),
            Chunk("b", "d", "RAG 生成", 1, {}, [0.8, 0.2, 0.0]),
        ]
        result = HybridRetriever(FakeEmbedding(), reranker=FakeReranker()).search(
            "RAG", chunks, top_k=2
        )
        self.assertEqual(result.hits[0].chunk.id, result.rankings["rerank"][0])
        self.assertEqual(result.hits[0].rerank_score, 0.9)
        self.assertFalse(result.reranker["degraded"])

    def test_reranker_failure_degrades_to_rrf(self):
        chunks = [Chunk("a", "d", "RAG", 0, {}, [1.0, 0.0, 0.0])]
        result = HybridRetriever(FakeEmbedding(), reranker=BrokenReranker()).search(
            "RAG", chunks, top_k=1
        )
        self.assertTrue(result.reranker["degraded"])
        self.assertEqual(result.hits[0].mmr_relevance, 1.0)

    @patch("urllib.request.urlopen")
    def test_qwen_reranker_protocol_and_index_mapping(self, urlopen):
        urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "output": {
                        "results": [
                            {"index": 1, "relevance_score": 0.9},
                            {"index": 0, "relevance_score": 0.2},
                        ]
                    },
                    "usage": {"total_tokens": 12},
                }
            ).encode()
        )
        reranker = QwenTextReranker(
            base_url="https://example.test/rerank", api_key="test-key"
        )
        hits = [
            SearchHit(Chunk("a", "d", "A", 0, {}, [1.0]), 0.0),
            SearchHit(Chunk("b", "d", "B", 1, {}, [1.0]), 0.0),
        ]
        result = reranker.rerank("question", hits)
        self.assertEqual(result.ranking, ["b", "a"])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3.7-text-rerank")
        self.assertEqual(payload["input"]["documents"], ["A", "B"])

    def test_embedding_model_mismatch_is_rejected(self):
        pipeline = RAGPipeline(self.db_path)
        pipeline.ingest("测试内容", "test.md")
        incompatible = RAGPipeline(self.db_path, embedder=FakeEmbedding())
        with self.assertRaisesRegex(ValueError, "embedding configuration"):
            incompatible.ask("测试")

    @patch("urllib.request.urlopen")
    def test_openai_compatible_embedding_protocol(self, urlopen):
        # Deliberately reverse response indexes to verify stable input ordering.
        urlopen.return_value = io.BytesIO(
            json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 0.0]},
                    ]
                }
            ).encode()
        )
        embedder = OpenAICompatibleEmbedding(
            base_url="https://example.test/v1", api_key="test-key", model="embed-test"
        )
        vectors = embedder.embed_many(["first", "second"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/embeddings")
        self.assertEqual(json.loads(request.data)["input"], ["first", "second"])

    @patch("urllib.request.urlopen")
    def test_embedding_batch_size_can_come_from_environment(self, urlopen):
        urlopen.side_effect = [
            io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1.0]}]}).encode()),
            io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1.0]}]}).encode()),
        ]
        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "1"}):
            embedder = OpenAICompatibleEmbedding(
                base_url="https://example.test/v1", api_key="test-key", model="embed-test"
            )
        embedder.embed_many(["first", "second"])
        self.assertEqual(urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_openai_compatible_llm_protocol(self, urlopen):
        urlopen.return_value = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "真实模型回答 [1]"}}]}).encode()
        )
        generator = OpenAICompatibleGenerator(
            base_url="https://example.test/v1", api_key="test-key", model="chat-test"
        )
        hit = SearchHit(
            chunk=Chunk("c1", "d1", "证据文本", 0, {"source": "doc.md"}, [1.0]),
            score=1.0,
        )
        answer = generator.generate("问题", [hit])
        self.assertEqual(answer, "真实模型回答 [1]")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(json.loads(request.data)["model"], "chat-test")


if __name__ == "__main__":
    unittest.main()
