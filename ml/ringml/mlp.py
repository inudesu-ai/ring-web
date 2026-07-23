"""Small NumPy MLP classifier with Adam and early stopping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


class MLPClassifier:
    model_type = "ring-mlp-v1"

    def __init__(
        self,
        *,
        classes: Sequence[str],
        target_steps: int,
        window_seconds: float,
        stride_seconds: float,
        hidden_size: int = 64,
        seed: int = 7,
    ) -> None:
        if hidden_size < 2:
            raise ValueError("hidden_size must be at least 2")
        self.classes = np.asarray(classes, dtype=str)
        self.target_steps = int(target_steps)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.hidden_size = int(hidden_size)
        self.seed = int(seed)
        self.feature_mean = np.zeros(self.target_steps * 6)
        self.feature_std = np.ones(self.target_steps * 6)
        self.weights_1: np.ndarray | None = None
        self.bias_1: np.ndarray | None = None
        self.weights_2: np.ndarray | None = None
        self.bias_2: np.ndarray | None = None
        self.metadata: dict = {}

    def _flatten(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.target_steps, 6):
            raise ValueError(
                f"Expected windows shaped [N, {self.target_steps}, 6], "
                f"got {values.shape}"
            )
        return values.reshape(len(values), -1)

    def fit(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        validation_windows: np.ndarray,
        validation_labels: np.ndarray,
        *,
        epochs: int = 250,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
        dropout: float = 0.1,
        patience: int = 30,
        noise_std: float = 0.005,
    ) -> list[dict[str, float]]:
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        rng = np.random.default_rng(self.seed)
        train_x = self._flatten(train_windows)
        validation_x = self._flatten(validation_windows)
        self.feature_mean = train_x.mean(axis=0)
        self.feature_std = np.maximum(train_x.std(axis=0), 1e-6)
        train_x = (train_x - self.feature_mean) / self.feature_std
        validation_x = (validation_x - self.feature_mean) / self.feature_std

        label_to_index = {label: index for index, label in enumerate(self.classes)}
        try:
            train_y = np.asarray(
                [label_to_index[label] for label in train_labels], dtype=np.int64
            )
            validation_y = np.asarray(
                [label_to_index[label] for label in validation_labels], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(f"Unknown training label: {exc.args[0]}") from exc

        input_size = train_x.shape[1]
        output_size = len(self.classes)
        self.weights_1 = rng.normal(
            0.0, np.sqrt(2.0 / input_size), (input_size, self.hidden_size)
        )
        self.bias_1 = np.zeros(self.hidden_size)
        self.weights_2 = rng.normal(
            0.0, np.sqrt(2.0 / self.hidden_size), (self.hidden_size, output_size)
        )
        self.bias_2 = np.zeros(output_size)

        parameters = [self.weights_1, self.bias_1, self.weights_2, self.bias_2]
        first_moments = [np.zeros_like(value) for value in parameters]
        second_moments = [np.zeros_like(value) for value in parameters]
        counts = np.bincount(train_y, minlength=output_size)
        class_weights = len(train_y) / np.maximum(1, output_size * counts)

        best_loss = np.inf
        best_parameters = [value.copy() for value in parameters]
        stale_epochs = 0
        step = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(train_x))
            for begin in range(0, len(order), batch_size):
                indices = order[begin : begin + batch_size]
                x = train_x[indices].copy()
                if noise_std:
                    x += rng.normal(0.0, noise_std, x.shape)
                y = train_y[indices]
                weights = class_weights[y]
                weights /= np.mean(weights)

                hidden_linear = x @ self.weights_1 + self.bias_1
                hidden = np.maximum(hidden_linear, 0.0)
                if dropout:
                    keep = 1.0 - dropout
                    dropout_mask = (rng.random(hidden.shape) < keep) / keep
                    hidden_output = hidden * dropout_mask
                else:
                    dropout_mask = 1.0
                    hidden_output = hidden
                probabilities = _softmax(
                    hidden_output @ self.weights_2 + self.bias_2
                )

                gradient_logits = probabilities
                gradient_logits[np.arange(len(y)), y] -= 1.0
                gradient_logits *= weights[:, None] / len(y)
                gradient_w2 = hidden_output.T @ gradient_logits + l2 * self.weights_2
                gradient_b2 = gradient_logits.sum(axis=0)
                gradient_hidden = (
                    gradient_logits @ self.weights_2.T
                    * dropout_mask
                    * (hidden_linear > 0)
                )
                gradient_w1 = x.T @ gradient_hidden + l2 * self.weights_1
                gradient_b1 = gradient_hidden.sum(axis=0)
                gradients = [gradient_w1, gradient_b1, gradient_w2, gradient_b2]

                step += 1
                for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients)
                ):
                    first_moments[index] = (
                        0.9 * first_moments[index] + 0.1 * gradient
                    )
                    second_moments[index] = (
                        0.999 * second_moments[index] + 0.001 * gradient * gradient
                    )
                    m_hat = first_moments[index] / (1.0 - 0.9**step)
                    v_hat = second_moments[index] / (1.0 - 0.999**step)
                    parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            validation_probabilities = self._predict_standardized(validation_x)
            validation_loss = float(
                -np.mean(
                    np.log(
                        np.maximum(
                            validation_probabilities[
                                np.arange(len(validation_y)), validation_y
                            ],
                            1e-12,
                        )
                    )
                )
            )
            validation_accuracy = float(
                np.mean(np.argmax(validation_probabilities, axis=1) == validation_y)
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )

            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_parameters = [value.copy() for value in parameters]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        for parameter, best in zip(parameters, best_parameters):
            parameter[...] = best
        self.metadata["best_validation_loss"] = best_loss
        self.metadata["epochs_completed"] = len(history)
        return history

    def _predict_standardized(self, standardized: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (self.weights_1, self.bias_1, self.weights_2, self.bias_2)
        ):
            raise ValueError("Model has not been trained")
        hidden = np.maximum(standardized @ self.weights_1 + self.bias_1, 0.0)
        return _softmax(hidden @ self.weights_2 + self.bias_2)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        features = self._flatten(windows)
        standardized = (features - self.feature_mean) / self.feature_std
        return self._predict_standardized(standardized)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(windows)
        return self.classes[np.argmax(probabilities, axis=1)]

    def save(self, path: Path) -> None:
        if self.weights_1 is None or self.weights_2 is None:
            raise ValueError("Model has not been trained")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_type=np.asarray(self.model_type),
            classes=self.classes,
            target_steps=np.asarray(self.target_steps),
            window_seconds=np.asarray(self.window_seconds),
            stride_seconds=np.asarray(self.stride_seconds),
            hidden_size=np.asarray(self.hidden_size),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            weights_1=self.weights_1,
            bias_1=self.bias_1,
            weights_2=self.weights_2,
            bias_2=self.bias_2,
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> "MLPClassifier":
        with np.load(path, allow_pickle=False) as values:
            model_type = str(values["model_type"])
            if model_type != cls.model_type:
                raise ValueError(f"Not an MLP model: {model_type}")
            model = cls(
                classes=values["classes"].astype(str),
                target_steps=int(values["target_steps"]),
                window_seconds=float(values["window_seconds"]),
                stride_seconds=float(values["stride_seconds"]),
                hidden_size=int(values["hidden_size"]),
            )
            model.feature_mean = values["feature_mean"].copy()
            model.feature_std = values["feature_std"].copy()
            model.weights_1 = values["weights_1"].copy()
            model.bias_1 = values["bias_1"].copy()
            model.weights_2 = values["weights_2"].copy()
            model.bias_2 = values["bias_2"].copy()
            model.metadata = json.loads(str(values["metadata"]))
        return model
