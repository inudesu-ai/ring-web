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
    WindowDataset,
    grouped_train_validation_test_split,
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

    def test_predefined_subject_split_has_no_leakage(self) -> None:
        labels = np.asarray(
            [label for subject in range(6) for label in ("idle", "wave")]
        )
        subjects = np.asarray(
            [f"subject-{subject}" for subject in range(6) for _ in range(2)]
        )
        partitions = np.asarray(
            [
                partition
                for subject in range(6)
                for partition in (
                    ("train", "train")
                    if subject < 2
                    else ("validation", "validation")
                    if subject < 4
                    else ("test", "test")
                )
            ]
        )
        dataset = WindowDataset(
            windows=np.zeros((12, 20, 6)),
            labels=labels,
            groups=np.asarray([f"session-{index}" for index in range(12)]),
            subjects=subjects,
            partitions=partitions,
        )
        train, validation, test, method = grouped_train_validation_test_split(
            dataset, use_predefined=True
        )
        self.assertEqual(method, "predefined")
        self.assertFalse(set(train.subjects) & set(validation.subjects))
        self.assertFalse(set(train.subjects) & set(test.subjects))
        self.assertEqual(set(test.labels), {"idle", "wave"})

    def test_predefined_subject_leakage_is_rejected(self) -> None:
        dataset = WindowDataset(
            windows=np.zeros((6, 20, 6)),
            labels=np.asarray(["idle", "wave"] * 3),
            groups=np.asarray([f"session-{index}" for index in range(6)]),
            subjects=np.asarray(
                ["shared", "train-only", "shared", "validation-only",
                 "test-a", "test-b"]
            ),
            partitions=np.asarray(
                ["train", "train", "validation", "validation", "test", "test"]
            ),
        )
        with self.assertRaisesRegex(ValueError, "group leakage"):
            grouped_train_validation_test_split(
                dataset, use_predefined=True
            )


if __name__ == "__main__":
    unittest.main()
