import unittest

import torch

from ca_easyrec.lightgcn import LightGCN


class LightGCNTests(unittest.TestCase):
    """Catch broken graph normalization, BPR scoring, and shape handling."""

    def setUp(self):
        torch.manual_seed(7)
        self.edge_index = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 1, 1, 2],
            ],
            dtype=torch.long,
        )

    def test_propagation_returns_finite_user_and_item_embeddings(self):
        model = LightGCN(
            num_users=3,
            num_items=4,
            embedding_dim=5,
            num_layers=2,
        )

        users, items = model.propagate(self.edge_index)

        self.assertEqual(users.shape, (3, 5))
        self.assertEqual(items.shape, (4, 5))
        self.assertTrue(torch.isfinite(users).all())
        self.assertTrue(torch.isfinite(items).all())

    def test_bpr_loss_is_finite_and_backpropagates(self):
        model = LightGCN(
            num_users=3,
            num_items=4,
            embedding_dim=4,
            num_layers=1,
        )
        users = torch.tensor([0, 1], dtype=torch.long)
        positives = torch.tensor([0, 1], dtype=torch.long)
        negatives = torch.tensor([3, 2], dtype=torch.long)

        loss = model.bpr_loss(
            users,
            positives,
            negatives,
            self.edge_index,
            l2_weight=1e-4,
        )
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.user_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())

    def test_out_of_range_edges_are_rejected(self):
        model = LightGCN(3, 4, embedding_dim=4, num_layers=1)
        invalid_edges = torch.tensor([[0, 3], [0, 1]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "user ID"):
            model.propagate(invalid_edges)


if __name__ == "__main__":
    unittest.main()
