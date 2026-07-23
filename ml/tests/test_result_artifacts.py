from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.model import load_model  # noqa: E402


class ResultArtifactTests(unittest.TestCase):
    def test_million_row_reports_and_models_are_consistent(self) -> None:
        result_dir = ML_DIR / "results" / "million-1m"
        manifest = json.loads(
            (result_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (result_dir / "experiment_report.json").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["records"], 1_000_000)
        self.assertEqual(report["records"], manifest["records"])
        self.assertFalse(report["subject_leakage"])
        self.assertEqual(
            report["subjects"], {"train": 70, "validation": 15, "test": 15}
        )

        windows = np.zeros((2, 32, 6))
        for name in ("mlp", "hmm"):
            diagnosis = report["models"][name]["fit_diagnosis"]
            self.assertEqual(diagnosis["status"], "balanced")
            self.assertFalse(diagnosis["underfit"])
            self.assertFalse(diagnosis["overfit"])

            model = load_model(
                result_dir / "models" / f"gesture-{name}-sim-1m.npz"
            )
            probabilities = model.predict_proba(windows)
            self.assertEqual(probabilities.shape, (2, 10))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            self.assertEqual(model.metadata["dataset_records"], 1_000_000)

    def test_drift_report_records_yaw_limit(self) -> None:
        report = json.loads(
            (ML_DIR / "results" / "drift" / "drift_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("not absolute yaw", report["yaw_observability"])
        self.assertIn("vqf", report["orientation"])
        self.assertLess(
            report["position"]["fused_zupt"]["final_error_m"],
            report["position"]["fused_orientation"]["final_error_m"],
        )


if __name__ == "__main__":
    unittest.main()
