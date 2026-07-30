import json
import math
import tempfile
import unittest
from pathlib import Path

from ca_easyrec.demo import run_demo


class DemoTests(unittest.TestCase):
    """Catch failures in the complete teacher-to-text training pipeline."""

    def test_demo_writes_valid_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "demo"
            metrics = run_demo(
                output_directory=output,
                seed=11,
                teacher_epochs=2,
                text_epochs=2,
                embedding_dim=8,
                k=3,
            )
            stored_metrics = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )

            self.assertTrue((output / "teacher.pt").is_file())
            self.assertTrue((output / "text_model.pt").is_file())
            self.assertEqual(metrics, stored_metrics)
            for key in ("recall@3", "ndcg@3"):
                self.assertIn(key, metrics)
                self.assertTrue(math.isfinite(metrics[key]))
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)
            self.assertEqual(metrics["metadata"]["result_type"], "toy_smoke_test")


if __name__ == "__main__":
    unittest.main()
