"""Dataset loading, validation, windowing, and group-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import gzip
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Session:
    session_id: str
    subject_id: str
    partition: str
    label: str
    sample_rate_hz: float
    sequences: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class WindowDataset:
    windows: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    subjects: np.ndarray
    partitions: np.ndarray

    def subset(self, indices: np.ndarray) -> "WindowDataset":
        return WindowDataset(
            windows=self.windows[indices],
            labels=self.labels[indices],
            groups=self.groups[indices],
            subjects=self.subjects[indices],
            partitions=self.partitions[indices],
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
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else path.open
        with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
            "r", encoding="utf-8"
        ) as stream:
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
                subject_id = str(record.get("subject_id", "")).strip()
                partition = str(record.get("partition", "")).strip().lower()
                label = str(record.get("label", "")).strip()
                if not session_id or not label:
                    raise ValueError(f"{line_ref}: session_id and label are required")
                if partition and partition not in {"train", "validation", "test"}:
                    raise ValueError(
                        f"{line_ref}: partition must be train, validation, or test"
                    )

                sample_rate = _number(record, "sample_rate_hz", line_ref=line_ref)
                if sample_rate <= 0:
                    raise ValueError(f"{line_ref}: sample_rate_hz must be positive")
                sequence = int(_number(record, "sequence", line_ref=line_ref))
                accel = _axis(record, "accel_raw", line_ref=line_ref)
                gyro = _axis(record, "gyro_raw", line_ref=line_ref)

                current = grouped.setdefault(
                    session_id,
                    {
                        "subject_id": subject_id,
                        "partition": partition,
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
                if current["subject_id"] != subject_id:
                    raise ValueError(
                        f"{line_ref}: session {session_id!r} changes subject_id"
                    )
                if current["partition"] != partition:
                    raise ValueError(
                        f"{line_ref}: session {session_id!r} changes partition"
                    )

                # Raw signed int16 values are mapped to a stable [-1, 1) range.
                current["sequences"].append(sequence)
                current["samples"].append(
                    [float(value) / 32768.0 for value in accel + gyro]
                )

    sessions = [
        Session(
            session_id=session_id,
            subject_id=str(values["subject_id"]),
            partition=str(values["partition"]),
            label=str(values["label"]),
            sample_rate_hz=float(values["sample_rate_hz"]),
            sequences=np.asarray(values["sequences"], dtype=np.int64),
            samples=np.asarray(values["samples"], dtype=np.float32),
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
    subjects: list[str] = []
    partitions: list[str] = []
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
                subjects.append(session.subject_id)
                partitions.append(session.partition)

    if not windows:
        raise ValueError(
            "No complete windows were created; collect longer continuous sessions"
        )
    return WindowDataset(
        windows=np.asarray(windows, dtype=np.float64),
        labels=np.asarray(labels, dtype=str),
        groups=np.asarray(groups, dtype=str),
        subjects=np.asarray(subjects, dtype=str),
        partitions=np.asarray(partitions, dtype=str),
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


def grouped_train_validation_test_split(
    dataset: WindowDataset,
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
    group_by: str = "auto",
    use_predefined: bool = False,
) -> tuple[WindowDataset, WindowDataset, WindowDataset, str]:
    """Create leakage-safe train/validation/test partitions.

    Subject grouping is preferred when every window carries subject_id.
    Predefined partitions are useful for a deliberately shifted test domain.
    """

    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be below one")

    all_labels = set(dataset.labels.tolist())
    if use_predefined:
        if set(dataset.partitions.tolist()) != {"train", "validation", "test"}:
            raise ValueError(
                "Predefined split requested but train/validation/test are incomplete"
            )
        partitions = [
            dataset.subset(np.flatnonzero(dataset.partitions == name))
            for name in ("train", "validation", "test")
        ]
        if any(set(part.labels.tolist()) != all_labels for part in partitions):
            raise ValueError("Every predefined partition must contain every class")
        split_groups = (
            dataset.subjects
            if bool(np.all(dataset.subjects != ""))
            else dataset.groups
        )
        group_sets = [
            set(split_groups[dataset.partitions == name].tolist())
            for name in ("train", "validation", "test")
        ]
        if (
            group_sets[0] & group_sets[1]
            or group_sets[0] & group_sets[2]
            or group_sets[1] & group_sets[2]
        ):
            raise ValueError("Predefined partitions contain group leakage")
        return (*partitions, "predefined")

    if group_by not in {"auto", "subject", "session"}:
        raise ValueError("group_by must be auto, subject, or session")
    has_subjects = bool(np.all(dataset.subjects != ""))
    selected = (
        "subject"
        if group_by == "auto" and has_subjects
        else "session"
        if group_by == "auto"
        else group_by
    )
    if selected == "subject" and not has_subjects:
        raise ValueError("Subject split requested but subject_id is missing")
    group_values = dataset.subjects if selected == "subject" else dataset.groups
    unique_groups = np.unique(group_values)
    rng = np.random.default_rng(seed)

    group_label_counts = {
        group: len(set(dataset.labels[group_values == group].tolist()))
        for group in unique_groups
    }
    groups_span_classes = any(count > 1 for count in group_label_counts.values())

    if groups_span_classes:
        # Global subject split: a person must never cross partitions in any class.
        for _ in range(1000):
            shuffled = unique_groups.copy()
            rng.shuffle(shuffled)
            test_count = max(1, round(len(shuffled) * test_ratio))
            validation_count = max(1, round(len(shuffled) * validation_ratio))
            test_groups = set(shuffled[:test_count].tolist())
            validation_groups = set(
                shuffled[test_count : test_count + validation_count].tolist()
            )
            train_groups = set(
                shuffled[test_count + validation_count :].tolist()
            )
            masks = [
                np.isin(group_values, list(values))
                for values in (train_groups, validation_groups, test_groups)
            ]
            if all(set(dataset.labels[mask].tolist()) == all_labels for mask in masks):
                train_mask, validation_mask, test_mask = masks
                break
        else:
            raise ValueError("Could not create class-complete global group split")
    else:
        train_groups: set[str] = set()
        validation_groups: set[str] = set()
        test_groups: set[str] = set()
        for label in sorted(all_labels):
            label_groups = np.unique(group_values[dataset.labels == label])
            if len(label_groups) < 3:
                raise ValueError(
                    f"Class {label!r} needs at least three {selected} groups"
                )
            shuffled = label_groups.copy()
            rng.shuffle(shuffled)
            test_count = min(
                len(shuffled) - 2,
                max(1, round(len(shuffled) * test_ratio)),
            )
            validation_count = min(
                len(shuffled) - test_count - 1,
                max(1, round(len(shuffled) * validation_ratio)),
            )
            test_groups.update(shuffled[:test_count].tolist())
            validation_groups.update(
                shuffled[test_count : test_count + validation_count].tolist()
            )
            train_groups.update(
                shuffled[test_count + validation_count :].tolist()
            )
        train_mask = np.isin(group_values, list(train_groups))
        validation_mask = np.isin(group_values, list(validation_groups))
        test_mask = np.isin(group_values, list(test_groups))

    return (
        dataset.subset(np.flatnonzero(train_mask)),
        dataset.subset(np.flatnonzero(validation_mask)),
        dataset.subset(np.flatnonzero(test_mask)),
        selected,
    )


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


def classification_scores(
    expected: np.ndarray,
    predicted: np.ndarray,
    classes: Sequence[str],
) -> dict[str, float]:
    f1_values = []
    for label in classes:
        truth = expected == label
        guess = predicted == label
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "accuracy": float(np.mean(expected == predicted)),
        "macro_f1": float(np.mean(f1_values)),
    }


def diagnose_fit(
    train_scores: dict[str, float],
    validation_scores: dict[str, float],
    test_scores: dict[str, float],
) -> dict[str, float | str | bool]:
    train_f1 = train_scores["macro_f1"]
    validation_f1 = validation_scores["macro_f1"]
    test_f1 = test_scores["macro_f1"]
    generalization_gap = train_f1 - validation_f1
    holdout_gap = validation_f1 - test_f1
    underfit = train_f1 < 0.80
    overfit = generalization_gap > 0.10
    domain_shift = holdout_gap > 0.10
    status = (
        "underfit"
        if underfit
        else "overfit"
        if overfit
        else "domain_shift_sensitive"
        if domain_shift
        else "balanced"
    )
    return {
        "status": status,
        "underfit": underfit,
        "overfit": overfit,
        "domain_shift_sensitive": domain_shift,
        "generalization_gap": float(generalization_gap),
        "validation_test_gap": float(holdout_gap),
    }
