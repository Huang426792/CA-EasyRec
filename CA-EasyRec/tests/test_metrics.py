import math
import unittest

import numpy as np

from ca_easyrec.metrics import recall_ndcg_at_k


class RankingMetricTests(unittest.TestCase):
    """Catch failures to mask seen items or normalize ranking gains."""

    def test_matches_hand_computed_macro_recall_and_ndcg(self):
        scores = np.array(
            [
                [0.9, 0.8, 0.7, 0.6],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        truth = [{2, 3}, {0}]
        seen = [{0}, {1}]
        first_user_ndcg = (1.0 / math.log2(3.0)) / (1.0 + 1.0 / math.log2(3.0))

        metrics = recall_ndcg_at_k(scores, truth, seen, k=2)

        self.assertAlmostEqual(metrics["recall@2"], 0.75, places=12)
        self.assertAlmostEqual(
            metrics["ndcg@2"],
            (first_user_ndcg + 1.0) / 2.0,
            places=12,
        )
        self.assertEqual(metrics["users"], 2)

    def test_empty_truth_users_are_excluded(self):
        scores = np.array([[0.3, 0.2], [0.2, 0.3]])

        metrics = recall_ndcg_at_k(scores, [set(), {1}], [set(), set()], k=1)

        self.assertEqual(metrics["users"], 1)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["ndcg@1"], 1.0)

    def test_out_of_range_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "item ID"):
            recall_ndcg_at_k(
                np.zeros((1, 2)),
                truth=[{2}],
                seen=[set()],
                k=1,
            )


if __name__ == "__main__":
    unittest.main()
