import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from ca_easyrec.official import train_official_teachers
from ca_easyrec.teacher_bank import TeacherEmbeddingBank


class OfficialTeacherCommandTests(unittest.TestCase):
    """Catch failures between official EasyRec files and teacher export."""

    def test_trains_and_exports_selected_domains(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            domain_root = root / "arts"
            domain_root.mkdir()
            train = sparse.coo_matrix(
                (
                    np.ones(3),
                    (np.array([0, 0, 1]), np.array([0, 1, 2])),
                ),
                shape=(2, 4),
            )
            with (domain_root / "trn_mat.pkl").open("wb") as output:
                pickle.dump(train, output)
            (domain_root / "user_profile.json").write_text(
                "\n".join(
                    json.dumps({"user_id": index, "profile": f"user {index}"})
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            (domain_root / "item_profile.json").write_text(
                "\n".join(
                    json.dumps({"item_id": index, "profile": f"item {index}"})
                    for index in range(4)
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "teachers.pt"

            histories = train_official_teachers(
                data_root=root,
                domains=["arts"],
                output_path=output_path,
                embedding_dim=4,
                num_layers=1,
                epochs=1,
                batch_size=2,
                seed=17,
            )
            bank = TeacherEmbeddingBank.load(output_path)

        self.assertEqual(tuple(histories), ("arts",))
        self.assertEqual(len(histories["arts"]), 1)
        self.assertEqual(bank.domain_names, ("arts",))


if __name__ == "__main__":
    unittest.main()
