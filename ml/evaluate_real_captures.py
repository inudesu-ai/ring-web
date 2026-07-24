#!/usr/bin/env python3
"""Evaluate command false activations on unlabeled real ring captures."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from ringml.data import resample_window
from ringml.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=0.4)
    parser.add_argument("--stationary-fraction", type=float, default=0.8)
    return parser


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path}: empty capture")
    samples = np.asarray(
        [row["accel"] + row["gyro"] for row in rows], dtype=np.float64
    )
    # Match realtime_infer.py: physical units are divided by configured range.
    samples[:, :3] /= 16.0
    samples[:, 3:] /= 2000.0
    stationary = np.asarray(
        [bool(row.get("stationary", False)) for row in rows], dtype=bool
    )
    positive_dt = [
        float(row["dt"])
        for row in rows
        if isinstance(row.get("dt"), (int, float)) and 0 < row["dt"] < 0.2
    ]
    sample_rate = 1.0 / np.median(positive_dt) if positive_dt else 100.0
    return samples, stationary, float(sample_rate)


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.model)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(model.metadata.get("recommended_threshold", 0.85))
    )
    paths = sorted(
        {
            Path(match).resolve()
            for pattern in args.data
            for match in glob.glob(pattern)
        }
    )
    if not paths:
        raise SystemExit("No capture files matched --data")

    file_reports: list[dict] = []
    all_stationary_fraction: list[float] = []
    all_prediction: list[str] = []
    all_confidence: list[float] = []
    for path in paths:
        samples, stationary, sample_rate = load_capture(path)
        window_size = max(2, round(model.window_seconds * sample_rate))
        stride = max(1, round(args.stride_seconds * sample_rate))
        windows: list[np.ndarray] = []
        stationary_fraction: list[float] = []
        for start in range(0, len(samples) - window_size + 1, stride):
            windows.append(
                resample_window(
                    samples[start : start + window_size], model.target_steps
                )
            )
            stationary_fraction.append(
                float(np.mean(stationary[start : start + window_size]))
            )
        if not windows:
            continue
        probabilities = model.predict_proba(np.asarray(windows))
        prediction = model.classes[np.argmax(probabilities, axis=1)]
        confidence = np.max(probabilities, axis=1)
        stationary_fraction_array = np.asarray(stationary_fraction)
        high_stationary = (
            stationary_fraction_array >= args.stationary_fraction
        )
        command = (prediction != "idle") & (confidence >= threshold)
        file_reports.append(
            {
                "path": str(path),
                "sample_rate_hz": sample_rate,
                "windows": len(windows),
                "high_stationary_windows": int(np.sum(high_stationary)),
                "high_stationary_idle_recall": (
                    float(np.mean(prediction[high_stationary] == "idle"))
                    if np.any(high_stationary)
                    else None
                ),
                "high_stationary_false_command_rate": (
                    float(np.mean(command[high_stationary]))
                    if np.any(high_stationary)
                    else None
                ),
                "prediction_counts": {
                    str(label): int(np.sum(prediction == label))
                    for label in np.unique(prediction)
                },
            }
        )
        all_stationary_fraction.extend(stationary_fraction)
        all_prediction.extend(prediction.tolist())
        all_confidence.extend(confidence.tolist())

    stationary_fraction = np.asarray(all_stationary_fraction)
    prediction = np.asarray(all_prediction, dtype=str)
    confidence = np.asarray(all_confidence)
    high_stationary = stationary_fraction >= args.stationary_fraction
    command = (prediction != "idle") & (confidence >= threshold)
    report = {
        "schema": "ring-real-unlabeled-evaluation/v1",
        "model": str(args.model),
        "model_type": model.model_type,
        "threshold": threshold,
        "files": file_reports,
        "summary": {
            "windows": len(prediction),
            "high_stationary_windows": int(np.sum(high_stationary)),
            "high_stationary_idle_recall": (
                float(np.mean(prediction[high_stationary] == "idle"))
                if np.any(high_stationary)
                else None
            ),
            "high_stationary_false_command_rate": (
                float(np.mean(command[high_stationary]))
                if np.any(high_stationary)
                else None
            ),
            "accepted_non_idle_windows": int(np.sum(command)),
            "prediction_counts": {
                str(label): int(np.sum(prediction == label))
                for label in np.unique(prediction)
            },
        },
        "interpretation": (
            "The captures have no command labels. This report can measure idle "
            "false activations, but not real gesture-class accuracy."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
