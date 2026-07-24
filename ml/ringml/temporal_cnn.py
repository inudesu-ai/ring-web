"""Portable NumPy inference for the lightweight temporal CNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def _conv1d(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dilation: int = 1,
) -> np.ndarray:
    """Same-padded Conv1d for [batch, time, channels] NumPy arrays."""

    kernel = weight.shape[2]
    padding = dilation * (kernel // 2)
    padded = np.pad(values, ((0, 0), (padding, padding), (0, 0)))
    output = np.zeros(
        (len(values), values.shape[1], weight.shape[0]), dtype=np.float64
    )
    for index in range(kernel):
        frame = padded[
            :, index * dilation : index * dilation + values.shape[1], :
        ]
        output += frame @ weight[:, :, index].T
    return output + bias


def _depthwise_conv1d(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dilation: int,
) -> np.ndarray:
    kernel = weight.shape[2]
    padding = dilation * (kernel // 2)
    padded = np.pad(values, ((0, 0), (padding, padding), (0, 0)))
    output = np.zeros_like(values, dtype=np.float64)
    for index in range(kernel):
        frame = padded[
            :, index * dilation : index * dilation + values.shape[1], :
        ]
        output += frame * weight[:, 0, index]
    return output + bias


class TemporalCNNClassifier:
    """Depthwise temporal CNN with attentive mean/max pooling."""

    model_type = "ring-temporal-cnn-v1"

    def __init__(
        self,
        *,
        classes: Sequence[str],
        target_steps: int,
        window_seconds: float,
        stride_seconds: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        parameters: dict[str, np.ndarray],
        metadata: dict | None = None,
    ) -> None:
        self.classes = np.asarray(classes, dtype=str)
        self.target_steps = int(target_steps)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.parameters = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }
        self.metadata = dict(metadata or {})
        if self.feature_mean.shape != (8,) or self.feature_std.shape != (8,):
            raise ValueError("Temporal CNN feature statistics must have 8 channels")

    def _features(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.target_steps, 6):
            raise ValueError(
                f"Expected windows shaped [N, {self.target_steps}, 6], "
                f"got {values.shape}"
            )
        magnitudes = np.stack(
            (
                np.linalg.norm(values[:, :, :3], axis=2),
                np.linalg.norm(values[:, :, 3:], axis=2),
            ),
            axis=2,
        )
        features = np.concatenate((values, magnitudes), axis=2)
        return (features - self.feature_mean) / self.feature_std

    def _block(self, values: np.ndarray, index: int, dilation: int) -> np.ndarray:
        prefix = f"block{index}"
        depthwise = np.maximum(
            _depthwise_conv1d(
                values,
                self.parameters[f"{prefix}_dw_weight"],
                self.parameters[f"{prefix}_dw_bias"],
                dilation=dilation,
            ),
            0.0,
        )
        projected = _conv1d(
            depthwise,
            self.parameters[f"{prefix}_pw_weight"],
            self.parameters[f"{prefix}_pw_bias"],
        )
        if f"{prefix}_skip_weight" in self.parameters:
            skip = _conv1d(
                values,
                self.parameters[f"{prefix}_skip_weight"],
                self.parameters[f"{prefix}_skip_bias"],
            )
        else:
            skip = values
        return np.maximum(skip + projected, 0.0)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        features = self._features(windows)
        values = np.maximum(
            _conv1d(
                features,
                self.parameters["stem_weight"],
                self.parameters["stem_bias"],
            ),
            0.0,
        )
        values = self._block(values, 1, 1)
        values = self._block(values, 2, 2)
        values = self._block(values, 3, 4)
        attention_logits = _conv1d(
            values,
            self.parameters["attention_weight"],
            self.parameters["attention_bias"],
        )
        attention = _softmax(attention_logits, axis=1)
        attentive = np.sum(values * attention, axis=1)
        pooled = np.concatenate(
            (attentive, np.mean(values, axis=1), np.max(values, axis=1)), axis=1
        )
        hidden = np.maximum(
            pooled @ self.parameters["fc1_weight"].T
            + self.parameters["fc1_bias"],
            0.0,
        )
        logits = (
            hidden @ self.parameters["fc2_weight"].T
            + self.parameters["fc2_bias"]
        )
        return _softmax(logits)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(windows)
        return self.classes[np.argmax(probabilities, axis=1)]

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": np.asarray(self.model_type),
            "classes": self.classes,
            "target_steps": np.asarray(self.target_steps),
            "window_seconds": np.asarray(self.window_seconds),
            "stride_seconds": np.asarray(self.stride_seconds),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "metadata": np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        }
        payload.update(self.parameters)
        np.savez_compressed(destination, **payload)

    @classmethod
    def load(cls, path: Path) -> "TemporalCNNClassifier":
        with np.load(path, allow_pickle=False) as values:
            model_type = str(values["model_type"])
            if model_type != cls.model_type:
                raise ValueError(f"Not a temporal CNN model: {model_type}")
            reserved = {
                "model_type",
                "classes",
                "target_steps",
                "window_seconds",
                "stride_seconds",
                "feature_mean",
                "feature_std",
                "metadata",
            }
            parameters = {
                name: values[name].copy()
                for name in values.files
                if name not in reserved
            }
            return cls(
                classes=values["classes"].astype(str),
                target_steps=int(values["target_steps"]),
                window_seconds=float(values["window_seconds"]),
                stride_seconds=float(values["stride_seconds"]),
                feature_mean=values["feature_mean"].copy(),
                feature_std=values["feature_std"].copy(),
                parameters=parameters,
                metadata=json.loads(str(values["metadata"])),
            )
