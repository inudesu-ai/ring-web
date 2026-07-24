from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.hmm import GaussianHMMClassifier  # noqa: E402
from ringml.mlp import MLPClassifier  # noqa: E402
from ringml.model import load_model  # noqa: E402
from ringml.temporal_cnn import TemporalCNNClassifier  # noqa: E402


def synthetic_windows(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 2 * np.pi, 20)
    windows = []
    labels = []
    for label, sign in (("left", -1.0), ("right", 1.0)):
        for _ in range(18):
            value = np.zeros((20, 6))
            value[:, 0] = sign * (0.7 + 0.2 * np.sin(time))
            value[:, 4] = sign * 0.5 * np.cos(time)
            value += rng.normal(0, 0.03, value.shape)
            windows.append(value)
            labels.append(label)
    return np.asarray(windows), np.asarray(labels)


class ModelTests(unittest.TestCase):
    def test_temporal_cnn_save_load_and_probabilities(self) -> None:
        rng = np.random.default_rng(12)
        parameters = {
            "stem_weight": rng.normal(0, 0.02, (24, 8, 3)),
            "stem_bias": np.zeros(24),
            "block1_dw_weight": rng.normal(0, 0.02, (24, 1, 5)),
            "block1_dw_bias": np.zeros(24),
            "block1_pw_weight": rng.normal(0, 0.02, (32, 24, 1)),
            "block1_pw_bias": np.zeros(32),
            "block1_skip_weight": rng.normal(0, 0.02, (32, 24, 1)),
            "block1_skip_bias": np.zeros(32),
            "block2_dw_weight": rng.normal(0, 0.02, (32, 1, 5)),
            "block2_dw_bias": np.zeros(32),
            "block2_pw_weight": rng.normal(0, 0.02, (32, 32, 1)),
            "block2_pw_bias": np.zeros(32),
            "block3_dw_weight": rng.normal(0, 0.02, (32, 1, 5)),
            "block3_dw_bias": np.zeros(32),
            "block3_pw_weight": rng.normal(0, 0.02, (32, 32, 1)),
            "block3_pw_bias": np.zeros(32),
            "attention_weight": rng.normal(0, 0.02, (1, 32, 1)),
            "attention_bias": np.zeros(1),
            "fc1_weight": rng.normal(0, 0.02, (48, 96)),
            "fc1_bias": np.zeros(48),
            "fc2_weight": rng.normal(0, 0.02, (2, 48)),
            "fc2_bias": np.zeros(2),
        }
        model = TemporalCNNClassifier(
            classes=["idle", "wave"],
            target_steps=20,
            window_seconds=0.8,
            stride_seconds=0.2,
            feature_mean=np.zeros(8),
            feature_std=np.ones(8),
            parameters=parameters,
        )
        windows = rng.normal(0, 0.05, (3, 20, 6))
        probabilities = model.predict_proba(windows)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "temporal-cnn.npz"
            model.save(path)
            loaded = load_model(path)
            np.testing.assert_allclose(
                loaded.predict_proba(windows),
                probabilities,
            )

    def test_mlp_fit_save_and_load(self) -> None:
        windows, labels = synthetic_windows()
        train_indices = np.r_[0:14, 18:32]
        validation_indices = np.r_[14:18, 32:36]
        model = MLPClassifier(
            classes=["left", "right"],
            target_steps=20,
            window_seconds=0.8,
            stride_seconds=0.2,
            hidden_size=16,
            seed=2,
        )
        model.fit(
            windows[train_indices],
            labels[train_indices],
            windows[validation_indices],
            labels[validation_indices],
            epochs=80,
            batch_size=8,
            patience=15,
            dropout=0,
        )
        predicted = model.predict(windows[validation_indices])
        self.assertGreaterEqual(np.mean(predicted == labels[validation_indices]), 0.99)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mlp.npz"
            model.save(path)
            loaded = load_model(path)
            np.testing.assert_allclose(
                loaded.predict_proba(windows[:2]),
                model.predict_proba(windows[:2]),
            )

    def test_hmm_fit_save_and_load(self) -> None:
        windows, labels = synthetic_windows()
        model = GaussianHMMClassifier(
            classes=["left", "right"],
            target_steps=20,
            window_seconds=0.8,
            stride_seconds=0.2,
            state_count=3,
        )
        model.fit(windows, labels, max_iterations=8)
        predicted = model.predict(windows)
        self.assertGreaterEqual(np.mean(predicted == labels), 0.95)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hmm.npz"
            model.save(path)
            loaded = load_model(path)
            np.testing.assert_allclose(
                loaded.predict_proba(windows[:2]),
                model.predict_proba(windows[:2]),
            )


if __name__ == "__main__":
    unittest.main()
