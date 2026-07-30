import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from ca_easyrec.data import load_compact_dataset, load_easyrec_domain

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tiny"


class CompactDataTests(unittest.TestCase):
    """Catch unstable ID mapping and unprofiled interaction entities."""

    def test_compact_loader_builds_stable_contiguous_ids(self):
        dataset = load_compact_dataset(FIXTURE_ROOT)
        books = dataset.domains["books"]

        self.assertEqual(books.raw_user_ids, ("u10", "u2"))
        self.assertEqual(books.raw_item_ids, ("i10", "i2"))
        np.testing.assert_array_equal(
            books.train_edges,
            np.array([[0, 1], [1, 0]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            books.test_edges,
            np.array([[0], [0]], dtype=np.int64),
        )
        self.assertEqual(books.user_profiles[0], "likes science fiction")
        self.assertEqual(books.item_profiles[1], "a space opera novel")

    def test_interaction_with_missing_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "interactions.csv").write_text(
                "domain,user_id,item_id,split\nbooks,missing,i1,train\n",
                encoding="utf-8",
            )
            (root / "user_profiles.jsonl").write_text(
                json.dumps({"domain": "books", "user_id": "u1", "profile": "reader"})
                + "\n",
                encoding="utf-8",
            )
            (root / "item_profiles.jsonl").write_text(
                json.dumps({"domain": "books", "item_id": "i1", "profile": "book"})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing user profile"):
                load_compact_dataset(root)


class OfficialEasyRecDataTests(unittest.TestCase):
    """Catch incompatibility with the public EasyRec data layout."""

    def test_loads_sparse_matrices_and_profiles_in_matrix_id_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            domain_root = root / "arts"
            domain_root.mkdir()
            train = sparse.coo_matrix(
                (
                    np.ones(2),
                    (np.array([0, 1]), np.array([1, 0])),
                ),
                shape=(2, 3),
            )
            validation = sparse.coo_matrix(
                (
                    np.ones(1),
                    (np.array([0]), np.array([2])),
                ),
                shape=(2, 3),
            )
            test = sparse.coo_matrix((2, 3))
            for name, matrix in (
                ("trn_mat.pkl", train),
                ("val_mat.pkl", validation),
                ("tst_mat.pkl", test),
            ):
                with (domain_root / name).open("wb") as output:
                    pickle.dump(matrix, output)
            (domain_root / "user_profile.json").write_text(
                "\n".join(
                    [
                        json.dumps({"user_id": 1, "profile": "second user"}),
                        json.dumps({"user_id": 0, "profile": "first user"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (domain_root / "item_profile.json").write_text(
                "\n".join(
                    json.dumps({"item_id": item_id, "profile": f"item {item_id}"})
                    for item_id in (2, 0, 1)
                )
                + "\n",
                encoding="utf-8",
            )

            domain = load_easyrec_domain(root, "arts")

        self.assertEqual(domain.user_profiles, ("first user", "second user"))
        self.assertEqual(domain.item_profiles, ("item 0", "item 1", "item 2"))
        np.testing.assert_array_equal(
            domain.train_edges,
            np.array([[0, 1], [1, 0]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            domain.validation_edges,
            np.array([[0], [2]], dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()
