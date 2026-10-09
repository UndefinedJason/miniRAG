from __future__ import annotations

import math
import unittest

from ragmini.evaluation import ranking_metrics


class EvaluationTests(unittest.TestCase):
    def test_ranking_metrics_match_hand_calculation(self) -> None:
        grades = {"a": 3, "b": 1}
        metrics = ranking_metrics(["x", "a", "b"], grades, top_k=3)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        dcg = 7 / math.log2(3) + 1 / math.log2(4)
        idcg = 7 + 1 / math.log2(3)
        self.assertAlmostEqual(metrics["ndcg"], dcg / idcg)

    def test_empty_judgments_mark_recall_and_ndcg_undefined(self) -> None:
        metrics = ranking_metrics(["x"], {}, top_k=3)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["ndcg"])


if __name__ == "__main__":
    unittest.main()
