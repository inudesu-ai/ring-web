#!/usr/bin/env python3
"""Generate exactly one million physics-informed ring-imu/v1 JSONL rows."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

from imu_simulator import LABELS, simulate_session


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
    parser = argparse.ArgumentParser(
        description="Generate exactly N physics-informed six-axis IMU samples."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--sample-rate", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session_count = args.subjects * len(LABELS)
    if args.records <= 0 or args.records % session_count:
        raise SystemExit(
            f"--records must be divisible by subjects*classes ({session_count})"
        )
    samples_per_session = args.records // session_count
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    started = time.monotonic()
    total = 0
    partition_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    shards = []

    for label_index, label in enumerate(LABELS):
        path = output / f"{label_index:02d}-{label}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as stream:
            for subject_index in range(args.subjects):
                partition = partition_for_subject(subject_index, args.subjects)
                shifted_domain = partition == "test"
                session_id = f"sim-{subject_index:03d}-{label}"
                session = simulate_session(
                    label=label,
                    sample_count=samples_per_session,
                    sample_rate_hz=args.sample_rate,
                    subject_index=subject_index,
                    session_seed=args.seed + label_index * 100_000 + subject_index,
                    shifted_domain=shifted_domain,
                )
                for sequence in range(samples_per_session):
                    row = {
                        "schema": "ring-imu/v1",
                        "session_id": session_id,
                        "subject_id": f"sim-subject-{subject_index:03d}",
                        "partition": partition,
                        "label": label,
                        "sdk_version": "simulation-physics-v1",
                        "sample_rate_hz": args.sample_rate,
                        "accel_range_g": 16,
                        "gyro_range_dps": 2000,
                        "sequence": sequence,
                        "timestamp_ms": int(session.timestamps_ms[sequence]),
                        "accel_raw": session.accel_raw[sequence].astype(int).tolist(),
                        "gyro_raw": session.gyro_raw[sequence].astype(int).tolist(),
                    }
                    stream.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                total += samples_per_session
                partition_counts[partition] += samples_per_session
                class_counts[label] += samples_per_session
        shards.append(
            {
                "path": path.name,
                "records": args.subjects * samples_per_session,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        print(
            f"{label:>14}: {class_counts[label]:,} rows "
            f"({path.stat().st_size / 1024 / 1024:.1f} MiB)"
        )

    if total != args.records:
        raise RuntimeError(f"Expected {args.records:,} rows, generated {total:,}")
    manifest = {
        "schema": "ring-imu-simulation-manifest/v1",
        "generator": "physics-informed-six-axis-v1",
        "records": total,
        "subjects": args.subjects,
        "classes": list(LABELS),
        "class_counts": dict(class_counts),
        "partition_counts": dict(partition_counts),
        "sessions": session_count,
        "samples_per_session": samples_per_session,
        "sample_rate_hz": args.sample_rate,
        "seed": args.seed,
        "test_domain_shift": {
            "subjects": [
                f"sim-subject-{index:03d}"
                for index in range(round(args.subjects * 0.85), args.subjects)
            ],
            "placement_noise_bias_severity_multiplier": 1.6,
        },
        "variability": [
            "movement amplitude and speed",
            "sensor placement rotation",
            "axis gain and static bias",
            "bias random walk",
            "sensor bandwidth",
            "white measurement noise",
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
        f"Generated exactly {total:,} rows in {manifest['elapsed_seconds']:.1f}s: "
        f"{output}"
    )


if __name__ == "__main__":
    main()
