import importlib.util
import unittest

import numpy as np

from ca_easyrec.weighting import (
    confidence_weights_numpy,
    confidence_weights_torch,
)


class ConfidenceWeightTests(unittest.TestCase):
    """Catch incorrect confidence direction, masking, and fallback behavior."""

    def test_high_affinity_receives_lower_weight(self):
        scores = np.array([[0.0, 1.0, 3.0]], dtype=np.float64)
        eligible = np.array([[True, True, True]])

        weights = confidence_weights_numpy(
            scores,
            eligible,
            epsilon=0.2,
            gamma=1.0,
        )

        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[0, 1], weights[0, 2])
        self.assertGreaterEqual(weights.min(), 0.2)
        self.assertLessEqual(weights.max(), 1.0)

    def test_ineligible_entries_keep_unit_weight(self):
        scores = np.array([[0.0, 2.0, 4.0]], dtype=np.float64)
        eligible = np.array([[True, True, False]])

        weights = confidence_weights_numpy(scores, eligible, 0.3, 2.0)

        self.assertEqual(weights[0, 2], 1.0)
        self.assertLess(weights[0, 1], weights[0, 0])

    def test_fewer_than_two_eligible_negatives_keep_unit_weight(self):
        scores = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float64)
        eligible = np.array([[True, False], [False, False]])

        weights = confidence_weights_numpy(scores, eligible, 0.2, 1.0)

        np.testing.assert_allclose(weights, np.ones_like(scores))

    def test_zero_variance_scores_remain_finite(self):
        scores = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
        eligible = np.array([[True, True, True]])

        weights = confidence_weights_numpy(scores, eligible, 0.2, 1.0)

        np.testing.assert_allclose(weights, np.full_like(scores, 0.6))
        self.assertTrue(np.isfinite(weights).all())

    def test_invalid_hyperparameters_are_rejected(self):
        scores = np.zeros((1, 2))
        eligible = np.ones((1, 2), dtype=bool)

        with self.assertRaisesRegex(ValueError, "epsilon"):
            confidence_weights_numpy(scores, eligible, epsilon=0.0, gamma=1.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            confidence_weights_numpy(scores, eligible, epsilon=0.2, gamma=0.0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            confidence_weights_numpy(
                scores,
                np.ones((2, 1), dtype=bool),
                epsilon=0.2,
                gamma=1.0,
            )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchConfidenceWeightTests(unittest.TestCase):
    """Catch drift between the auditable NumPy and training implementations."""

    def test_torch_matches_numpy_and_detaches_weights(self):
        import torch

        scores_np = np.array(
            [[0.1, 0.4, 0.9], [0.7, 0.2, 0.5]],
            dtype=np.float32,
        )
        mask_np = np.array(
            [[True, True, False], [True, True, True]],
            dtype=bool,
        )
        scores = torch.tensor(scores_np, requires_grad=True)
        mask = torch.tensor(mask_np)

        got = confidence_weights_torch(scores, mask, epsilon=0.3, gamma=2.0)
        expected = confidence_weights_numpy(
            scores_np,
            mask_np,
            epsilon=0.3,
            gamma=2.0,
        )

        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(got.requires_grad)


if __name__ == "__main__":
    unittest.main()
