#!/usr/bin/env python3
"""Capture labeled Ring Sound IMU samples as training-ready JSONL."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

import ring_sound as sdk  # noqa: E402


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one labeled Ring Sound IMU session as JSONL."
    )
    parser.add_argument("--address", required=True, help="Ring BLE MAC/UUID address.")
    parser.add_argument("--label", required=True, help="Gesture class for this session.")
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=30.0,
        help="Capture duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .jsonl path.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional stable session ID; generated when omitted.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append instead of refusing to overwrite an existing file.",
    )
    return parser


def make_session_id(label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{label}-{timestamp}-{uuid.uuid4().hex[:8]}"


async def capture(args: argparse.Namespace) -> int:
    output: Path = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "x"
    session_id = args.session_id or make_session_id(args.label)
    sample_count = 0
    report_started = False

    print("Put the ring in gesture mode before IMU capture starts.", file=sys.stderr)
    print(f"Connecting to {args.address} ...", file=sys.stderr)

    async with sdk.RingSoundClient(address=args.address) as ring:
        start_info = await sdk.start_sensor_report(ring)
        report_started = True
        print(
            "IMU started: "
            f"{start_info.sample_rate_hz} Hz, "
            f"±{start_info.accel_range_g} g, "
            f"±{start_info.gyro_range_dps} dps",
            file=sys.stderr,
        )

        deadline = time.monotonic() + args.duration
        try:
            with output.open(mode, encoding="utf-8") as stream:
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    try:
                        batch = await sdk.wait_sensor_data(
                            ring,
                            timeout_s=max(0.1, min(5.0, remaining)),
                        )
                    except sdk.TimeoutError:
                        continue

                    for index, sample in enumerate(batch.samples):
                        row = {
                            "schema": "ring-imu/v1",
                            "session_id": session_id,
                            "label": args.label,
                            "sdk_version": sdk.__version__,
                            "sample_rate_hz": start_info.sample_rate_hz,
                            "accel_range_g": start_info.accel_range_g,
                            "gyro_range_dps": start_info.gyro_range_dps,
                            "sequence": batch.sequence_start + index,
                            "timestamp_ms": sample.timestamp_ms,
                            "accel_raw": [
                                sample.accel_x,
                                sample.accel_y,
                                sample.accel_z,
                            ],
                            "gyro_raw": [
                                sample.gyro_x,
                                sample.gyro_y,
                                sample.gyro_z,
                            ],
                        }
                        stream.write(
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        sample_count += 1
                    stream.flush()
        finally:
            if report_started:
                await sdk.stop_sensor_report(ring)

    print(
        f"Saved {sample_count} samples to {output} (session {session_id}).",
        file=sys.stderr,
    )
    return sample_count


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(capture(args))
    except FileExistsError:
        raise SystemExit(
            f"{args.output} already exists; choose another path or pass --append"
        )
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
