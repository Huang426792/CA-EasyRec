import unittest

import torch

from ca_easyrec.teacher_bank import TeacherEmbeddingBank
from ca_easyrec.training import ca_easyrec_batch_loss


class ConfidenceAwareBatchTests(unittest.TestCase):
    """Catch incorrect candidate ordering, labels, and teacher-weight alignment."""

    def test_batch_loss_downweights_high_affinity_false_negative(self):
        bank = TeacherEmbeddingBank()
        bank.add_domain(
            "books",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                    [0.1, 0.9],
                ]
            ),
        )
        user_embeddings = torch.tensor(
            [[1.0, 0.1], [0.1, 1.0]],
            requires_grad=True,
        )
        positive_embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        negative_embeddings = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]],
            requires_grad=True,
        )

        output = ca_easyrec_batch_loss(
            user_embeddings=user_embeddings,
            positive_item_embeddings=positive_embeddings,
            negative_item_embeddings=negative_embeddings,
            user_domains=["books", "books"],
            user_ids=torch.tensor([0, 1]),
            positive_item_domains=["books", "books"],
            positive_item_ids=torch.tensor([0, 1]),
            negative_item_domains=["books", "books"],
            negative_item_ids=torch.tensor([2, 3]),
            teacher_bank=bank,
            temperature=0.5,
            epsilon=0.2,
            gamma=1.0,
        )
        output.loss.backward()

        torch.testing.assert_close(output.labels, torch.tensor([0, 1]))
        self.assertEqual(output.logits.shape, (2, 4))
        self.assertEqual(output.weights[0, 0].item(), 1.0)
        self.assertEqual(output.weights[1, 1].item(), 1.0)
        self.assertLess(output.weights[0, 2], output.weights[0, 3])
        self.assertLess(output.weights[1, 3], output.weights[1, 2])
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(user_embeddings.grad)

    def test_cross_domain_candidates_keep_unit_weight(self):
        bank = TeacherEmbeddingBank()
        bank.add_domain(
            "books",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        )
        bank.add_domain(
            "games",
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0], [0.5, 0.5]]),
        )

        output = ca_easyrec_batch_loss(
            user_embeddings=torch.eye(2),
            positive_item_embeddings=torch.eye(2),
            negative_item_embeddings=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            user_domains=["books", "games"],
            user_ids=torch.tensor([0, 0]),
            positive_item_domains=["books", "games"],
            positive_item_ids=torch.tensor([0, 0]),
            negative_item_domains=["books", "games"],
            negative_item_ids=torch.tensor([1, 1]),
            teacher_bank=bank,
            temperature=0.5,
        )

        self.assertEqual(output.weights[0, 1].item(), 1.0)
        self.assertEqual(output.weights[0, 3].item(), 1.0)
        self.assertEqual(output.weights[1, 0].item(), 1.0)
        self.assertEqual(output.weights[1, 2].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
