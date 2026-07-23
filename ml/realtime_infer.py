#!/usr/bin/env python3
"""Run a saved gesture model on live ring IMU data and optionally publish it."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "sdk"))

import ring_sound as sdk  # noqa: E402

from ringml.data import resample_window  # noqa: E402
from ringml.model import load_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Ring Sound gesture inference.")
    parser.add_argument("--address", required=True, help="Ring BLE MAC/UUID address.")
    parser.add_argument("--model", type=Path, required=True, help="Trained .npz model.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum confidence; defaults to the model recommendation.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="New-probability weight for exponential smoothing.",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help="Optional API endpoint, e.g. https://api.inudesu.xyz/v1/gesture.",
    )
    parser.add_argument(
        "--token-env",
        default="RING_BRIDGE_TOKEN",
        help="Environment variable containing the API producer token.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=0,
        help="Stop after N predictions; zero keeps running.",
    )
    return parser


def post_json(url: str, payload: dict, token: str | None) -> None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=4) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Publisher returned HTTP {response.status}")


async def run(args: argparse.Namespace) -> None:
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    if not 0 < args.smoothing <= 1:
        raise ValueError("--smoothing must be in (0, 1]")
    if args.max_predictions < 0:
        raise ValueError("--max-predictions cannot be negative")

    model = load_model(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(model.metadata.get("recommended_threshold", 0.7))
    )
    token = os.getenv(args.token_env)
    raw_window: deque[list[float]] = deque()
    smoothed: np.ndarray | None = None
    samples_since_prediction = 0
    prediction_count = 0
    report_started = False

    print(
        f"Loaded {model.model_type}: {', '.join(model.classes.tolist())}; "
        f"threshold={threshold:.2f}",
        file=sys.stderr,
    )
    print("Put the ring in gesture mode before connecting.", file=sys.stderr)

    async with sdk.RingSoundClient(address=args.address) as ring:
        start_info = await sdk.start_sensor_report(ring)
        report_started = True
        source_window_size = max(
            2, round(model.window_seconds * start_info.sample_rate_hz)
        )
        stride_size = max(1, round(model.stride_seconds * start_info.sample_rate_hz))
        raw_window = deque(maxlen=source_window_size)

        try:
            while args.max_predictions == 0 or prediction_count < args.max_predictions:
                try:
                    batch = await sdk.wait_sensor_data(ring, timeout_s=5.0)
                except sdk.TimeoutError:
                    continue
                for sample in batch.samples:
                    raw_window.append(
                        [
                            sample.accel_x / 32768.0,
                            sample.accel_y / 32768.0,
                            sample.accel_z / 32768.0,
                            sample.gyro_x / 32768.0,
                            sample.gyro_y / 32768.0,
                            sample.gyro_z / 32768.0,
                        ]
                    )
                    samples_since_prediction += 1
                    if (
                        len(raw_window) < source_window_size
                        or samples_since_prediction < stride_size
                    ):
                        continue

                    samples_since_prediction = 0
                    window = resample_window(
                        np.asarray(raw_window), model.target_steps
                    )[None, :, :]
                    probabilities = model.predict_proba(window)[0]
                    smoothed = (
                        probabilities
                        if smoothed is None
                        else args.smoothing * probabilities
                        + (1.0 - args.smoothing) * smoothed
                    )
                    best_index = int(np.argmax(smoothed))
                    confidence = float(smoothed[best_index])
                    raw_gesture = str(model.classes[best_index])
                    gesture = raw_gesture if confidence >= threshold else "uncertain"
                    payload = {
                        "gesture": gesture,
                        "raw_gesture": raw_gesture,
                        "confidence": confidence,
                        "probabilities": {
                            str(label): float(value)
                            for label, value in zip(model.classes, smoothed)
                        },
                        "model_type": model.model_type,
                        "model_file": args.model.name,
                        "source": socket.gethostname(),
                        "device_timestamp_ms": sample.timestamp_ms,
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    prediction_count += 1

                    if args.publish_url:
                        try:
                            await asyncio.to_thread(
                                post_json, args.publish_url, payload, token
                            )
                        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                            print(f"publish failed: {exc}", file=sys.stderr)

                    if (
                        args.max_predictions
                        and prediction_count >= args.max_predictions
                    ):
                        break
        finally:
            if report_started:
                await sdk.stop_sensor_report(ring)


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
