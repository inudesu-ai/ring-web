#!/usr/bin/env python3
"""Run a saved gesture model on live ring IMU data and optionally publish it."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import suppress
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
from ringml.direction import (  # noqa: E402
    DirectionDecision,
    DirectionalGestureRecognizer,
    blend_direction_probabilities,
)
from ringml.displacement import DisplacementTracker  # noqa: E402
from ringml.model import load_model  # noqa: E402
from ringml.orientation import SixAxisAhrs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Ring Sound gesture inference.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--address", help="Ring BLE MAC/CoreBluetooth UUID.")
    target.add_argument(
        "--name",
        help="Advertised BLE name, e.g. ring; recommended on macOS.",
    )
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
        "--telemetry-url",
        default=None,
        help=(
            "Telemetry API endpoint. When omitted, /v1/gesture in --publish-url "
            "is replaced by /v1/telemetry."
        ),
    )
    parser.add_argument(
        "--telemetry-hz",
        type=float,
        default=10.0,
        help="Maximum orientation/IMU publish rate; zero disables telemetry.",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Publish predictions without printing every JSON event.",
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


def resolve_telemetry_url(
    publish_url: str | None, telemetry_url: str | None
) -> str | None:
    if telemetry_url:
        return telemetry_url
    if publish_url and publish_url.rstrip("/").endswith("/v1/gesture"):
        return f"{publish_url.rstrip('/')[:-len('/v1/gesture')]}/v1/telemetry"
    return None


async def telemetry_worker(
    url: str,
    token: str | None,
    queue: asyncio.Queue[dict],
) -> None:
    last_error_at = 0.0
    while True:
        payload = await queue.get()
        while not queue.empty():
            payload = queue.get_nowait()
        try:
            await asyncio.to_thread(post_json, url, payload, token)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            now = time.monotonic()
            if now - last_error_at >= 5.0:
                print(f"telemetry publish failed: {exc}", file=sys.stderr)
                last_error_at = now


def put_latest(queue: asyncio.Queue[dict], payload: dict) -> None:
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(payload)


async def start_sensor_stream(ring: sdk.RingSoundClient) -> sdk.SensorStartInfo:
    key_press_count = 0
    last_key_press_at: float | None = None

    def on_key_single_press(_packet: object) -> None:
        nonlocal key_press_count, last_key_press_at
        key_press_count += 1
        last_key_press_at = time.monotonic()
        print(
            f"Ring single-click received ({key_press_count}); "
            "waiting for gesture mode...",
            file=sys.stderr,
            flush=True,
        )

    ring.add_packet_handler(sdk.SensorCommand.KEY_SINGLE_PRESS, on_key_single_press)
    try:
        try:
            return await sdk.start_sensor_report(ring)
        except sdk.DeviceError as exc:
            if exc.error_code != 2:
                raise

        print(
            "Ring is in recording mode. Single-click the ring once to enter "
            "gesture mode...",
            file=sys.stderr,
            flush=True,
        )
        # Poll the actual mode instead of depending solely on the unsolicited
        # 0x0704 notification, which can be dropped. On this firmware revision,
        # a START_REPORT request can also race mode initialization immediately
        # after 0x0704, so re-arm it once the 800 ms transition window settles.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            try:
                start = await sdk.start_sensor_report(ring)
            except sdk.DeviceError as exc:
                if exc.error_code != 2:
                    raise
                continue

            if last_key_press_at is not None:
                remaining = 0.8 - (time.monotonic() - last_key_press_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                start = await sdk.start_sensor_report(ring)
            return start
        raise sdk.TimeoutError("Ring stayed in recording mode for 60 seconds")
    finally:
        ring.remove_packet_handler(
            sdk.SensorCommand.KEY_SINGLE_PRESS,
            on_key_single_press,
        )


async def run(args: argparse.Namespace) -> None:
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    if not 0 < args.smoothing <= 1:
        raise ValueError("--smoothing must be in (0, 1]")
    if args.max_predictions < 0:
        raise ValueError("--max-predictions cannot be negative")
    if not 0 <= args.telemetry_hz <= 30:
        raise ValueError("--telemetry-hz must be between zero and 30")

    model = load_model(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(model.metadata.get("recommended_threshold", 0.7))
    )
    token = os.getenv(args.token_env)
    telemetry_url = resolve_telemetry_url(
        args.publish_url, args.telemetry_url
    )
    telemetry_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
    telemetry_task = (
        asyncio.create_task(
            telemetry_worker(telemetry_url, token, telemetry_queue)
        )
        if telemetry_url and args.telemetry_hz > 0
        else None
    )
    raw_window: deque[list[float]] = deque()
    smoothed: np.ndarray | None = None
    samples_since_prediction = 0
    prediction_count = 0
    report_started = False
    ahrs = SixAxisAhrs()
    source = socket.gethostname()
    last_timestamp_ms: int | None = None
    last_telemetry_at = 0.0
    expected_sequence: int | None = None
    displacement: DisplacementTracker | None = None
    direction = DirectionalGestureRecognizer()
    direction_decision: DirectionDecision | None = None

    print(
        f"Loaded {model.model_type}: {', '.join(model.classes.tolist())}; "
        f"threshold={threshold:.2f}",
        file=sys.stderr,
    )
    print("Put the ring in gesture mode before connecting.", file=sys.stderr)

    try:
        async with sdk.RingSoundClient(
            address=args.address, name=args.name
        ) as ring:
            start_info = await start_sensor_stream(ring)
            report_started = True
            displacement = DisplacementTracker(
                sample_rate_hz=start_info.sample_rate_hz
            )
            source_window_size = max(
                2, round(model.window_seconds * start_info.sample_rate_hz)
            )
            stride_size = max(
                1, round(model.stride_seconds * start_info.sample_rate_hz)
            )
            raw_window = deque(maxlen=source_window_size)
            print(
                "IMU stream: "
                f"{start_info.sample_rate_hz} Hz, "
                f"±{start_info.accel_range_g} g, "
                f"±{start_info.gyro_range_dps} dps",
                file=sys.stderr,
            )

            while args.max_predictions == 0 or prediction_count < args.max_predictions:
                try:
                    batch = await sdk.wait_sensor_data(ring, timeout_s=5.0)
                except sdk.TimeoutError:
                    continue
                if (
                    expected_sequence is not None
                    and batch.sequence_start != expected_sequence
                ):
                    displacement.handle_transport_gap()
                    direction.reset()
                    raw_window.clear()
                    samples_since_prediction = 0
                    last_timestamp_ms = None
                expected_sequence = batch.sequence_start + len(batch.samples)

                # Firmware V2.000.0001.0015 appends one overlapping/corrupted
                # tail item per batch. Keep its sequence accounted for, but do
                # not feed it to fusion, integration, or recognition.
                valid_samples = (
                    batch.samples[:-1]
                    if len(batch.samples) > 1
                    else batch.samples
                )
                for sample_index, sample in enumerate(valid_samples):
                    accel_g = np.asarray(
                        [sample.accel_x, sample.accel_y, sample.accel_z],
                        dtype=np.float64,
                    ) / 32768.0 * start_info.accel_range_g
                    gyro_dps = np.asarray(
                        [sample.gyro_x, sample.gyro_y, sample.gyro_z],
                        dtype=np.float64,
                    ) / 32768.0 * start_info.gyro_range_dps
                    default_dt = 1.0 / start_info.sample_rate_hz
                    if last_timestamp_ms is None:
                        dt = default_dt
                    else:
                        elapsed_ms = (
                            sample.timestamp_ms - last_timestamp_ms
                        ) & 0xFFFFFFFF
                        dt = (
                            elapsed_ms / 1000.0
                            if 0 < elapsed_ms < 200
                            else default_dt
                        )
                    last_timestamp_ms = sample.timestamp_ms
                    quaternion = ahrs.update(accel_g, gyro_dps, dt)
                    motion = displacement.update(
                        dt_s=dt,
                        accel_body_g=tuple(float(value) for value in accel_g),
                        gyro_body_dps=tuple(
                            float(value) for value in ahrs.corrected_gyro_dps
                        ),
                        quaternion=tuple(
                            float(value) for value in quaternion
                        ),
                        stationary=ahrs.stationary,
                        stationary_confidence=ahrs.stationary_confidence,
                    )
                    direction_decision = direction.update(
                        motion,
                        timestamp_ms=sample.timestamp_ms,
                    )

                    now = time.monotonic()
                    if (
                        telemetry_task is not None
                        and now - last_telemetry_at
                        >= 1.0 / args.telemetry_hz
                    ):
                        telemetry = ahrs.telemetry(accel_g, gyro_dps)
                        telemetry["linear_accel_g"] = {
                            axis: float(value)
                            for axis, value in zip(
                                "xyz", motion.linear_accel_world_g
                            )
                        }
                        telemetry["motion"] = motion.as_payload()
                        telemetry.update(
                            {
                                "sample_rate_hz": start_info.sample_rate_hz,
                                "sequence": batch.sequence_start + sample_index,
                                "device_timestamp_ms": sample.timestamp_ms,
                                "source": source,
                            }
                        )
                        put_latest(telemetry_queue, telemetry)
                        last_telemetry_at = now

                    raw_window.append(
                        [
                            float(accel_g[0] / start_info.accel_range_g),
                            float(accel_g[1] / start_info.accel_range_g),
                            float(accel_g[2] / start_info.accel_range_g),
                            float(
                                ahrs.corrected_gyro_dps[0]
                                / start_info.gyro_range_dps
                            ),
                            float(
                                ahrs.corrected_gyro_dps[1]
                                / start_info.gyro_range_dps
                            ),
                            float(
                                ahrs.corrected_gyro_dps[2]
                                / start_info.gyro_range_dps
                            ),
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
                    fused, recognition_source = blend_direction_probabilities(
                        model.classes,
                        smoothed,
                        direction_decision,
                    )
                    best_index = int(np.argmax(fused))
                    confidence = float(fused[best_index])
                    raw_gesture = str(model.classes[best_index])
                    gesture = raw_gesture if confidence >= threshold else "uncertain"
                    payload = {
                        "gesture": gesture,
                        "raw_gesture": raw_gesture,
                        "confidence": confidence,
                        "probabilities": {
                            str(label): float(value)
                            for label, value in zip(model.classes, fused)
                        },
                        "model_type": f"{model.model_type}+zupt-direction-v1",
                        "model_file": args.model.name,
                        "recognition_source": recognition_source,
                        "direction_displacement_m": (
                            {
                                axis: float(value)
                                for axis, value in zip(
                                    "xyz",
                                    direction_decision.displacement_m,
                                )
                            }
                            if recognition_source == "zupt-direction"
                            and direction_decision is not None
                            else None
                        ),
                        "source": source,
                        "device_timestamp_ms": sample.timestamp_ms,
                    }
                    if not args.quiet:
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
            if report_started:
                await sdk.stop_sensor_report(ring)
                report_started = False
    finally:
        if report_started:
            with suppress(Exception):
                await sdk.stop_sensor_report(ring)
        if telemetry_task is not None:
            telemetry_task.cancel()
            with suppress(asyncio.CancelledError):
                await telemetry_task


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except sdk.RingSoundError as exc:
        print(f"ring connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
