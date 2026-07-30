import unittest

import torch

from ca_easyrec.text_model import HashingProfileEncoder, hash_profile_tokens


class HashingProfileEncoderTests(unittest.TestCase):
    """Catch nondeterministic token IDs and untrainable profile embeddings."""

    def test_hashing_is_deterministic_and_reserves_padding(self):
        first = hash_profile_tokens("Likes SPACE games!", vocabulary_size=101)
        second = hash_profile_tokens("likes space games", vocabulary_size=101)

        self.assertEqual(first, second)
        self.assertTrue(all(0 < token_id < 101 for token_id in first))

    def test_embeddings_are_normalized_and_backpropagate(self):
        torch.manual_seed(3)
        encoder = HashingProfileEncoder(vocabulary_size=127, embedding_dim=8)

        embeddings = encoder(["space strategy", "history book", ""])
        loss = embeddings[0] @ embeddings[1]
        loss.backward()

        self.assertEqual(embeddings.shape, (3, 8))
        torch.testing.assert_close(
            torch.linalg.vector_norm(embeddings, dim=1),
            torch.ones(3),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertIsNotNone(encoder.embedding.weight.grad)
        self.assertTrue(torch.isfinite(encoder.embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
