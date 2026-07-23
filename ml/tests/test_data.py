from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.data import (  # noqa: E402
    grouped_train_validation_split,
    load_sessions,
    window_sessions,
)


class DatasetTests(unittest.TestCase):
    def write_session(self, directory: Path, label: str, index: int) -> Path:
        path = directory / f"{label}-{index}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for sequence in range(60):
                row = {
                    "schema": "ring-imu/v1",
                    "session_id": f"{label}-{index}",
                    "label": label,
                    "sample_rate_hz": 25,
                    "sequence": sequence,
                    "accel_raw": [sequence, -sequence, 100],
                    "gyro_raw": [10, 20, 30],
                }
                stream.write(json.dumps(row) + "\n")
        return path

    def test_load_window_and_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [
                self.write_session(directory, label, index)
                for label in ("idle", "wave")
                for index in range(3)
            ]
            sessions = load_sessions(paths)
            dataset = window_sessions(
                sessions,
                window_seconds=1.0,
                stride_seconds=0.5,
                target_steps=20,
            )
            train, validation = grouped_train_validation_split(
                dataset, validation_ratio=0.34, seed=3
            )

            self.assertEqual(dataset.windows.shape[1:], (20, 6))
            self.assertFalse(set(train.groups) & set(validation.groups))
            self.assertEqual(set(train.labels), {"idle", "wave"})
            self.assertEqual(set(validation.labels), {"idle", "wave"})
            self.assertTrue(np.max(np.abs(dataset.windows)) < 1)


if __name__ == "__main__":
    unittest.main()
