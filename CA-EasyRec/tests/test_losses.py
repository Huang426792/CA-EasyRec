import importlib.util
import math
import unittest

import numpy as np

from ca_easyrec.losses import (
    weighted_info_nce_numpy,
    weighted_info_nce_torch,
)


class WeightedInfoNCETests(unittest.TestCase):
    """Catch incorrect denominator weighting and unstable loss arithmetic."""

    def test_matches_hand_computed_denominator(self):
        logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)
        weights = np.array([[1.0, 0.5, 1.0]], dtype=np.float64)
        expected = -math.log(
            math.exp(2.0) / (math.exp(2.0) + 0.5 * math.exp(1.0) + math.exp(0.0))
        )

        loss = weighted_info_nce_numpy(logits, labels, weights)

        self.assertAlmostEqual(loss, expected, places=12)

    def test_positive_weight_is_always_one(self):
        logits = np.array([[2.0, 1.0]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)
        incorrect_positive_weight = np.array([[0.01, 1.0]], dtype=np.float64)
        unit_weights = np.ones_like(incorrect_positive_weight)

        got = weighted_info_nce_numpy(
            logits,
            labels,
            incorrect_positive_weight,
        )
        expected = weighted_info_nce_numpy(logits, labels, unit_weights)

        self.assertAlmostEqual(got, expected, places=12)

    def test_large_logits_produce_finite_loss(self):
        logits = np.array(
            [[10_000.0, 9_999.0], [8_000.0, 8_001.0]],
            dtype=np.float64,
        )
        labels = np.array([0, 1], dtype=np.int64)

        loss = weighted_info_nce_numpy(logits, labels, np.ones_like(logits))

        self.assertTrue(math.isfinite(loss))

    def test_invalid_inputs_are_rejected(self):
        logits = np.zeros((2, 3), dtype=np.float64)
        labels = np.array([0, 3], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "label"):
            weighted_info_nce_numpy(logits, labels, np.ones_like(logits))
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_info_nce_numpy(
                logits,
                np.array([0, 1]),
                np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
            )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchWeightedInfoNCETests(unittest.TestCase):
    """Catch differences between reference and differentiable objectives."""

    def test_torch_matches_numpy_and_backpropagates_to_logits(self):
        import torch

        logits_np = np.array(
            [[2.0, 1.0, -1.0], [0.0, 3.0, 2.0]],
            dtype=np.float32,
        )
        labels_np = np.array([0, 1], dtype=np.int64)
        weights_np = np.array(
            [[1.0, 0.3, 1.0], [0.4, 1.0, 0.7]],
            dtype=np.float32,
        )
        logits = torch.tensor(logits_np, requires_grad=True)

        loss = weighted_info_nce_torch(
            logits,
            torch.tensor(labels_np),
            torch.tensor(weights_np),
        )
        loss.backward()

        expected = weighted_info_nce_numpy(logits_np, labels_np, weights_np)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
