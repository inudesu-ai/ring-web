#!/usr/bin/env python3
"""Generate exactly one million pre-segmented command-gesture IMU rows."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

from episodic_imu_simulator import LABELS, simulate_gesture_episode


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_for_subject(subject_index: int, subject_count: int) -> str:
    train_end = round(subject_count * 0.70)
    validation_end = train_end + round(subject_count * 0.15)
    if subject_index < train_end:
        return "train"
    if subject_index < validation_end:
        return "validation"
    return "test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--sample-rate", type=float, default=25.0)
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples_per_episode = round(args.sample_rate * args.window_seconds)
    denominator = args.subjects * len(LABELS) * samples_per_episode
    if args.records <= 0 or args.records % denominator:
        raise SystemExit(
            f"--records must be divisible by subjects*classes*samples "
            f"({denominator:,})"
        )
    episodes_per_subject_class = args.records // denominator
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    started = time.monotonic()
    total = 0
    session_count = 0
    class_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    shards: list[dict] = []
    for label_index, label in enumerate(LABELS):
        path = output / f"{label_index:02d}-{label}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as stream:
            for subject_index in range(args.subjects):
                partition = partition_for_subject(subject_index, args.subjects)
                for episode_index in range(episodes_per_subject_class):
                    session_id = (
                        f"episodic-{subject_index:03d}-{label}-{episode_index:02d}"
                    )
                    session = simulate_gesture_episode(
                        label=label,
                        sample_count=samples_per_episode,
                        sample_rate_hz=args.sample_rate,
                        subject_index=subject_index,
                        episode_index=episode_index,
                        session_seed=(
                            args.seed
                            + label_index * 1_000_000
                            + subject_index * 1_000
                            + episode_index
                        ),
                        shifted_domain=partition == "test",
                    )
                    for sequence in range(samples_per_episode):
                        row = {
                            "schema": "ring-imu/v1",
                            "session_id": session_id,
                            "subject_id": f"sim-subject-{subject_index:03d}",
                            "partition": partition,
                            "label": label,
                            "sdk_version": "simulation-episodic-v2",
                            "sample_rate_hz": args.sample_rate,
                            "accel_range_g": 16,
                            "gyro_range_dps": 2000,
                            "sequence": sequence,
                            "timestamp_ms": int(session.timestamps_ms[sequence]),
                            "accel_raw": session.accel_raw[sequence].astype(int).tolist(),
                            "gyro_raw": session.gyro_raw[sequence].astype(int).tolist(),
                            "episode_index": episode_index,
                        }
                        stream.write(
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                    total += samples_per_episode
                    session_count += 1
                    class_counts[label] += samples_per_episode
                    partition_counts[partition] += samples_per_episode
        shards.append(
            {
                "path": path.name,
                "records": class_counts[label],
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        print(f"{label:>14}: {class_counts[label]:,} rows", flush=True)

    if total != args.records:
        raise RuntimeError(f"Expected {args.records:,}, generated {total:,}")
    manifest = {
        "schema": "ring-imu-simulation-manifest/v2",
        "generator": "physics-informed-episodic-six-axis-v2",
        "records": total,
        "subjects": args.subjects,
        "classes": list(LABELS),
        "class_counts": dict(class_counts),
        "partition_counts": dict(partition_counts),
        "sessions": session_count,
        "episodes_per_subject_class": episodes_per_subject_class,
        "samples_per_session": samples_per_episode,
        "sample_rate_hz": args.sample_rate,
        "window_seconds": args.window_seconds,
        "seed": args.seed,
        "split": "70 train / 15 validation / 15 shifted test subjects",
        "test_domain_shift": {
            "placement_degrees": 17,
            "noise_bias_severity_multiplier": 1.35,
        },
        "variability": [
            "gesture onset, duration, amplitude, and speed",
            "subject and episode sensor placement",
            "axis gain, static bias, and bias random walk",
            "sensor bandwidth and white measurement noise",
            "sampling clock scale and jitter",
            "int16 quantisation",
        ],
        "shards": shards,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Generated exactly {total:,} rows in "
        f"{manifest['elapsed_seconds']:.1f}s: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
