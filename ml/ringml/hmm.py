"""Per-class left-to-right diagonal Gaussian HMM trained with EM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = safe_maximum + np.log(
            np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True)
        )
    result = np.where(finite, result, -np.inf)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _log_probabilities(
    sequence: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    difference = sequence[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variances)[None, :, :]
        + difference * difference / variances[None, :, :],
        axis=2,
    )


def _forward_backward(
    sequence: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    emissions = _log_probabilities(sequence, means, variances)
    log_initial = np.full_like(initial, -np.inf)
    log_initial[initial > 0] = np.log(initial[initial > 0])
    log_transitions = np.full_like(transitions, -np.inf)
    log_transitions[transitions > 0] = np.log(transitions[transitions > 0])

    length, state_count = emissions.shape
    forward = np.empty((length, state_count))
    forward[0] = log_initial + emissions[0]
    for index in range(1, length):
        forward[index] = emissions[index] + _logsumexp(
            forward[index - 1][:, None] + log_transitions, axis=0
        )
    log_likelihood = float(_logsumexp(forward[-1], axis=0))

    backward = np.zeros((length, state_count))
    for index in range(length - 2, -1, -1):
        backward[index] = _logsumexp(
            log_transitions
            + emissions[index + 1][None, :]
            + backward[index + 1][None, :],
            axis=1,
        )
    return log_likelihood, forward, backward


class GaussianHMMClassifier:
    model_type = "ring-hmm-v1"

    def __init__(
        self,
        *,
        classes: Sequence[str],
        target_steps: int,
        window_seconds: float,
        stride_seconds: float,
        state_count: int = 5,
        variance_floor: float = 1e-3,
        topology: str = "left_right",
        feature_mode: str = "raw_delta_magnitude",
    ) -> None:
        if state_count < 2:
            raise ValueError("state_count must be at least 2")
        if topology not in {"left_right", "ergodic"}:
            raise ValueError("topology must be left_right or ergodic")
        if feature_mode not in {"raw", "raw_delta_magnitude"}:
            raise ValueError("Unsupported HMM feature_mode")
        self.classes = np.asarray(classes, dtype=str)
        self.target_steps = int(target_steps)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.state_count = int(state_count)
        self.variance_floor = float(variance_floor)
        self.topology = topology
        self.feature_mode = feature_mode
        feature_count = 6 if feature_mode == "raw" else 14
        self.feature_mean = np.zeros(feature_count)
        self.feature_std = np.ones(feature_count)
        self.initial: np.ndarray | None = None
        self.transitions: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.variances: np.ndarray | None = None
        self.metadata: dict = {}

    def _transform(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.target_steps, 6):
            raise ValueError(
                f"Expected windows shaped [N, {self.target_steps}, 6], "
                f"got {values.shape}"
            )
        if self.feature_mode == "raw":
            return values
        differences = np.diff(values, axis=1, prepend=values[:, :1, :])
        magnitudes = np.stack(
            [
                np.linalg.norm(values[:, :, :3], axis=2),
                np.linalg.norm(values[:, :, 3:], axis=2),
            ],
            axis=2,
        )
        return np.concatenate([values, differences, magnitudes], axis=2)

    def _initialize_class(
        self, sequences: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pooled = np.concatenate(sequences)
        global_variance = np.maximum(pooled.var(axis=0), self.variance_floor)
        if self.topology == "left_right":
            initial = np.zeros(self.state_count)
            initial[0] = 1.0
            transitions = np.zeros((self.state_count, self.state_count))
            for state in range(self.state_count - 1):
                transitions[state, state] = 0.7
                transitions[state, state + 1] = 0.3
            transitions[-1, -1] = 1.0
            state_values: list[list[np.ndarray]] = [
                [] for _ in range(self.state_count)
            ]
            for sequence in sequences:
                boundaries = np.linspace(
                    0, len(sequence), self.state_count + 1
                ).astype(int)
                for state in range(self.state_count):
                    values = sequence[boundaries[state] : boundaries[state + 1]]
                    if len(values):
                        state_values[state].append(values)
            means = np.vstack(
                [
                    np.concatenate(values).mean(axis=0)
                    if values
                    else pooled.mean(axis=0)
                    for values in state_values
                ]
            )
        else:
            initial = np.full(self.state_count, 1.0 / self.state_count)
            transitions = np.full(
                (self.state_count, self.state_count),
                0.35 / (self.state_count - 1),
            )
            np.fill_diagonal(transitions, 0.65)
            # Deterministic farthest-point initialisation followed by short k-means.
            means = [pooled[0]]
            minimum_distance = np.sum((pooled - means[0]) ** 2, axis=1)
            for _ in range(1, self.state_count):
                means.append(pooled[int(np.argmax(minimum_distance))])
                distance = np.sum((pooled - means[-1]) ** 2, axis=1)
                minimum_distance = np.minimum(minimum_distance, distance)
            means = np.asarray(means)
            for _ in range(8):
                distance = np.sum(
                    (pooled[:, None, :] - means[None, :, :]) ** 2, axis=2
                )
                assignment = np.argmin(distance, axis=1)
                for state in range(self.state_count):
                    if np.any(assignment == state):
                        means[state] = pooled[assignment == state].mean(axis=0)
        distance = np.sum(
            (pooled[:, None, :] - means[None, :, :]) ** 2, axis=2
        )
        assignment = np.argmin(distance, axis=1)
        variances = np.vstack(
            [
                np.maximum(
                    pooled[assignment == state].var(axis=0)
                    if np.any(assignment == state)
                    else global_variance,
                    self.variance_floor,
                )
                for state in range(self.state_count)
            ]
        )
        return initial, transitions, means, variances

    def _fit_class(
        self,
        sequences: Sequence[np.ndarray],
        *,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
        initial, transitions, means, variances = self._initialize_class(sequences)
        transition_mask = transitions > 0
        history: list[float] = []

        for _ in range(max_iterations):
            transition_counts = np.zeros_like(transitions)
            state_weights = np.zeros(self.state_count)
            weighted_sum = np.zeros_like(means)
            weighted_square_sum = np.zeros_like(means)
            total_log_likelihood = 0.0

            for sequence in sequences:
                log_likelihood, forward, backward = _forward_backward(
                    sequence, initial, transitions, means, variances
                )
                total_log_likelihood += log_likelihood
                gamma = np.exp(forward + backward - log_likelihood)
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
                state_weights += gamma.sum(axis=0)
                weighted_sum += gamma.T @ sequence
                weighted_square_sum += gamma.T @ (sequence * sequence)

                emissions = _log_probabilities(sequence, means, variances)
                log_transitions = np.full_like(transitions, -np.inf)
                log_transitions[transition_mask] = np.log(
                    transitions[transition_mask]
                )
                for index in range(len(sequence) - 1):
                    log_xi = (
                        forward[index][:, None]
                        + log_transitions
                        + emissions[index + 1][None, :]
                        + backward[index + 1][None, :]
                        - log_likelihood
                    )
                    transition_counts += np.exp(log_xi)

            valid_states = state_weights > 1e-8
            means[valid_states] = (
                weighted_sum[valid_states] / state_weights[valid_states, None]
            )
            second_moment = np.zeros_like(means)
            second_moment[valid_states] = (
                weighted_square_sum[valid_states]
                / state_weights[valid_states, None]
            )
            variances[valid_states] = np.maximum(
                second_moment[valid_states] - means[valid_states] ** 2,
                self.variance_floor,
            )
            for state in range(self.state_count):
                allowed = transition_mask[state]
                # Tiny Dirichlet smoothing prevents valid transitions from
                # collapsing to exact zero during Baum-Welch.
                smoothed = transition_counts[state, allowed] + 1e-10
                total = smoothed.sum()
                if total > 1e-12:
                    transitions[state, allowed] = smoothed / total

            average = total_log_likelihood / len(sequences)
            history.append(float(average))
            if len(history) > 1 and abs(history[-1] - history[-2]) < tolerance:
                break
        return initial, transitions, means, variances, history

    def fit(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        *,
        max_iterations: int = 40,
        tolerance: float = 1e-3,
    ) -> dict[str, list[float]]:
        values = self._transform(windows)
        feature_count = values.shape[2]
        self.feature_mean = values.reshape(-1, feature_count).mean(axis=0)
        self.feature_std = np.maximum(
            values.reshape(-1, feature_count).std(axis=0), 1e-6
        )
        standardized = (values - self.feature_mean) / self.feature_std

        initial_values = []
        transition_values = []
        mean_values = []
        variance_values = []
        histories: dict[str, list[float]] = {}
        for label in self.classes:
            sequences = list(standardized[labels == label])
            if not sequences:
                raise ValueError(f"No training windows for class {label!r}")
            initial, transitions, means, variances, history = self._fit_class(
                sequences,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            initial_values.append(initial)
            transition_values.append(transitions)
            mean_values.append(means)
            variance_values.append(variances)
            histories[str(label)] = history

        self.initial = np.asarray(initial_values)
        self.transitions = np.asarray(transition_values)
        self.means = np.asarray(mean_values)
        self.variances = np.asarray(variance_values)
        return histories

    def score(self, windows: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (
                self.initial,
                self.transitions,
                self.means,
                self.variances,
            )
        ):
            raise ValueError("Model has not been trained")
        values = self._transform(windows)
        standardized = (values - self.feature_mean) / self.feature_std
        scores = np.empty((len(values), len(self.classes)))
        for row, sequence in enumerate(standardized):
            for class_index in range(len(self.classes)):
                likelihood, _, _ = _forward_backward(
                    sequence,
                    self.initial[class_index],
                    self.transitions[class_index],
                    self.means[class_index],
                    self.variances[class_index],
                )
                scores[row, class_index] = likelihood / len(sequence)
        return scores

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        scores = self.score(windows)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.score(windows), axis=1)]

    def save(self, path: Path) -> None:
        if self.initial is None:
            raise ValueError("Model has not been trained")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_type=np.asarray(self.model_type),
            classes=self.classes,
            target_steps=np.asarray(self.target_steps),
            window_seconds=np.asarray(self.window_seconds),
            stride_seconds=np.asarray(self.stride_seconds),
            state_count=np.asarray(self.state_count),
            variance_floor=np.asarray(self.variance_floor),
            topology=np.asarray(self.topology),
            feature_mode=np.asarray(self.feature_mode),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            initial=self.initial,
            transitions=self.transitions,
            means=self.means,
            variances=self.variances,
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> "GaussianHMMClassifier":
        with np.load(path, allow_pickle=False) as values:
            model_type = str(values["model_type"])
            if model_type != cls.model_type:
                raise ValueError(f"Not an HMM model: {model_type}")
            model = cls(
                classes=values["classes"].astype(str),
                target_steps=int(values["target_steps"]),
                window_seconds=float(values["window_seconds"]),
                stride_seconds=float(values["stride_seconds"]),
                state_count=int(values["state_count"]),
                variance_floor=float(values["variance_floor"]),
                topology=(
                    str(values["topology"])
                    if "topology" in values.files
                    else "left_right"
                ),
                feature_mode=(
                    str(values["feature_mode"])
                    if "feature_mode" in values.files
                    else "raw"
                ),
            )
            model.feature_mean = values["feature_mean"].copy()
            model.feature_std = values["feature_std"].copy()
            model.initial = values["initial"].copy()
            model.transitions = values["transitions"].copy()
            model.means = values["means"].copy()
            model.variances = values["variances"].copy()
            model.metadata = json.loads(str(values["metadata"]))
        return model
