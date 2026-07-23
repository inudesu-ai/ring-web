#!/usr/bin/env python3
"""Generate deterministic synthetic ring-imu/v1 sessions for a smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def signal(label: str, phase: np.ndarray) -> np.ndarray:
    values = np.zeros((len(phase), 6))
    if label == "idle":
        values[:, 2] = 0.25
    elif label == "wave":
        values[:, 0] = 0.55 * np.sin(phase * 6)
        values[:, 4] = 0.65 * np.cos(phase * 6)
    elif label == "rotate_front":
        values[:, 2] = 0.2 + 0.2 * np.cos(phase * 2)
        values[:, 5] = 0.75 * np.sin(np.minimum(phase, np.pi))
    else:
        raise ValueError(label)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic demo dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions-per-class", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sample_rate = 25
    sample_count = round(args.seconds * sample_rate)
    phase = np.linspace(0, 2 * np.pi, sample_count)
    args.output.mkdir(parents=True, exist_ok=True)

    for label in ("idle", "wave", "rotate_front"):
        for session_index in range(args.sessions_per_class):
            session_id = f"demo-{label}-{session_index:02d}"
            values = signal(label, phase)
            values += rng.normal(0, 0.025, values.shape)
            values *= rng.uniform(0.92, 1.08)
            raw = np.clip(np.rint(values * 32768), -32768, 32767).astype(int)
            path = args.output / f"{session_id}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for sequence, row in enumerate(raw):
                    record = {
                        "schema": "ring-imu/v1",
                        "session_id": session_id,
                        "subject_id": f"demo-subject-{session_index:02d}",
                        "label": label,
                        "sdk_version": "demo",
                        "sample_rate_hz": sample_rate,
                        "accel_range_g": 16,
                        "gyro_range_dps": 2000,
                        "sequence": sequence,
                        "timestamp_ms": sequence * 40,
                        "accel_raw": row[:3].tolist(),
                        "gyro_raw": row[3:].tolist(),
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Generated demo sessions in {args.output}")


if __name__ == "__main__":
    main()
