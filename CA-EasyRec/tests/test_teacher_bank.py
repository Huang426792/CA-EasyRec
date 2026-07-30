import tempfile
import unittest
from pathlib import Path

import torch

from ca_easyrec.teacher_bank import TeacherEmbeddingBank


class TeacherEmbeddingBankTests(unittest.TestCase):
    """Catch cross-domain leakage and corrupted teacher serialization."""

    def _make_bank(self):
        bank = TeacherEmbeddingBank()
        bank.add_domain(
            "books",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        )
        bank.add_domain(
            "games",
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[0.5, 0.5]]),
        )
        return bank

    def test_score_matrix_only_scores_matching_domains(self):
        bank = self._make_bank()

        scores, valid = bank.score_matrix(
            user_domains=["books", "books", "games"],
            user_ids=torch.tensor([0, 1, 0]),
            item_domains=["books", "games", "books"],
            item_ids=torch.tensor([0, 0, 1]),
        )

        expected_valid = torch.tensor(
            [
                [True, False, True],
                [True, False, True],
                [False, True, False],
            ]
        )
        torch.testing.assert_close(valid, expected_valid)
        self.assertEqual(scores[0, 0].item(), 2.0)
        self.assertEqual(scores[0, 2].item(), 0.0)
        self.assertEqual(scores[1, 2].item(), 3.0)
        self.assertEqual(scores[2, 1].item(), 1.0)
        self.assertTrue(torch.equal(scores[~valid], torch.zeros_like(scores[~valid])))

    def test_saved_bank_round_trips_without_gradients(self):
        bank = self._make_bank()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "teachers.pt"
            bank.save(path)
            loaded = TeacherEmbeddingBank.load(path)
            scores, valid = loaded.score_matrix(
                ["books"],
                torch.tensor([1]),
                ["books"],
                torch.tensor([1]),
            )

        self.assertEqual(scores.item(), 3.0)
        self.assertTrue(valid.item())
        self.assertFalse(scores.requires_grad)
        self.assertEqual(loaded.domain_names, ("books", "games"))

    def test_invalid_embedding_dimensions_are_rejected(self):
        bank = TeacherEmbeddingBank()

        with self.assertRaisesRegex(ValueError, "embedding dimension"):
            bank.add_domain(
                "broken",
                torch.zeros((2, 3)),
                torch.zeros((4, 2)),
            )


if __name__ == "__main__":
    unittest.main()
