"""Dataset loading, validation, windowing, and group-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Session:
    session_id: str
    label: str
    sample_rate_hz: float
    sequences: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class WindowDataset:
    windows: np.ndarray
    labels: np.ndarray
    groups: np.ndarray

    def subset(self, indices: np.ndarray) -> "WindowDataset":
        return WindowDataset(
            windows=self.windows[indices],
            labels=self.labels[indices],
            groups=self.groups[indices],
        )


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in sorted(glob.glob(pattern))]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(matches)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("No JSONL dataset files matched --data")
    return unique


def _number(record: dict, name: str, *, line_ref: str) -> float:
    value = record.get(name)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{line_ref}: {name} must be a finite number")
    return float(value)


def _axis(record: dict, name: str, *, line_ref: str) -> list[float]:
    value = record.get(name)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{line_ref}: {name} must contain three numbers")
    result = [float(item) for item in value]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{line_ref}: {name} contains a non-finite number")
    return result


def load_sessions(paths: Iterable[Path]) -> list[Session]:
    """Load ring-imu/v1 rows and group them by session_id."""

    grouped: dict[str, dict] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                line_ref = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_ref}: invalid JSON: {exc}") from exc

                if record.get("schema") != "ring-imu/v1":
                    raise ValueError(
                        f"{line_ref}: expected schema ring-imu/v1, "
                        f"got {record.get('schema')!r}"
                    )
                session_id = str(record.get("session_id", "")).strip()
                label = str(record.get("label", "")).strip()
                if not session_id or not label:
                    raise ValueError(f"{line_ref}: session_id and label are required")

                sample_rate = _number(record, "sample_rate_hz", line_ref=line_ref)
                if sample_rate <= 0:
                    raise ValueError(f"{line_ref}: sample_rate_hz must be positive")
                sequence = int(_number(record, "sequence", line_ref=line_ref))
                accel = _axis(record, "accel_raw", line_ref=line_ref)
                gyro = _axis(record, "gyro_raw", line_ref=line_ref)

                current = grouped.setdefault(
                    session_id,
                    {
                        "label": label,
                        "sample_rate_hz": sample_rate,
                        "sequences": [],
                        "samples": [],
                    },
                )
                if current["label"] != label:
                    raise ValueError(
                        f"{line_ref}: session {session_id!r} contains multiple labels"
                    )
                if abs(current["sample_rate_hz"] - sample_rate) > 1e-6:
                    raise ValueError(
                        f"{line_ref}: session {session_id!r} changes sample rate"
                    )

                # Raw signed int16 values are mapped to a stable [-1, 1) range.
                current["sequences"].append(sequence)
                current["samples"].append(np.asarray(accel + gyro) / 32768.0)

    sessions = [
        Session(
            session_id=session_id,
            label=str(values["label"]),
            sample_rate_hz=float(values["sample_rate_hz"]),
            sequences=np.asarray(values["sequences"], dtype=np.int64),
            samples=np.asarray(values["samples"], dtype=np.float64),
        )
        for session_id, values in grouped.items()
    ]
    if not sessions:
        raise ValueError("Dataset contains no samples")
    return sessions


def resample_window(window: np.ndarray, target_steps: int) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU window")
    if target_steps < 2:
        raise ValueError("target_steps must be at least 2")
    if len(window) < 2:
        raise ValueError("A window needs at least two samples")
    if len(window) == target_steps:
        return window.copy()

    source_x = np.linspace(0.0, 1.0, len(window))
    target_x = np.linspace(0.0, 1.0, target_steps)
    return np.column_stack(
        [np.interp(target_x, source_x, window[:, axis]) for axis in range(6)]
    )


def _continuous_ranges(sequences: np.ndarray) -> list[tuple[int, int]]:
    if not len(sequences):
        return []
    breaks = np.flatnonzero(np.diff(sequences) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(sequences)]))
    return list(zip(starts.tolist(), ends.tolist()))


def window_sessions(
    sessions: Sequence[Session],
    *,
    window_seconds: float = 1.6,
    stride_seconds: float = 0.4,
    target_steps: int = 40,
) -> WindowDataset:
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Window and stride durations must be positive")

    windows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for session in sessions:
        window_size = max(2, round(window_seconds * session.sample_rate_hz))
        stride = max(1, round(stride_seconds * session.sample_rate_hz))
        for start, end in _continuous_ranges(session.sequences):
            segment = session.samples[start:end]
            if len(segment) < window_size:
                continue
            for offset in range(0, len(segment) - window_size + 1, stride):
                windows.append(
                    resample_window(segment[offset : offset + window_size], target_steps)
                )
                labels.append(session.label)
                groups.append(session.session_id)

    if not windows:
        raise ValueError(
            "No complete windows were created; collect longer continuous sessions"
        )
    return WindowDataset(
        windows=np.asarray(windows, dtype=np.float64),
        labels=np.asarray(labels, dtype=str),
        groups=np.asarray(groups, dtype=str),
    )


def grouped_train_validation_split(
    dataset: WindowDataset,
    *,
    validation_ratio: float = 0.2,
    seed: int = 7,
) -> tuple[WindowDataset, WindowDataset]:
    """Split whole sessions, retaining every class in both partitions."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()

    for label in sorted(set(dataset.labels.tolist())):
        label_groups = np.unique(dataset.groups[dataset.labels == label])
        if len(label_groups) < 2:
            raise ValueError(
                f"Class {label!r} has only {len(label_groups)} session; "
                "collect at least two independent sessions per class"
            )
        shuffled = label_groups.copy()
        rng.shuffle(shuffled)
        count = min(
            len(shuffled) - 1,
            max(1, round(len(shuffled) * validation_ratio)),
        )
        validation_groups.update(shuffled[:count].tolist())

    validation_mask = np.isin(dataset.groups, list(validation_groups))
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    return dataset.subset(train_indices), dataset.subset(validation_indices)


def format_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    classes: Sequence[str],
) -> str:
    lines = ["class                 precision    recall        f1   support"]
    for label in classes:
        truth = expected == label
        guess = predicted == label
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        support = int(np.sum(truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        lines.append(
            f"{label:<20} {precision:>9.3f} {recall:>9.3f} "
            f"{f1:>9.3f} {support:>9d}"
        )
    accuracy = float(np.mean(expected == predicted))
    lines.append(f"\naccuracy: {accuracy:.3f} ({len(expected)} windows)")
    return "\n".join(lines)
