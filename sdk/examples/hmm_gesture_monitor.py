#!/usr/bin/env python3
"""Print the ring firmware's built-in HMM gesture events as JSON lines."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

import ring_sound as sdk  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor 0x0702 HMM gestures.")
    parser.add_argument("--address", required=True, help="Ring BLE MAC/UUID address.")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Stop after N events; zero keeps monitoring.",
    )
    return parser


async def monitor(address: str, count: int) -> None:
    if count < 0:
        raise ValueError("--count cannot be negative")

    print(
        "Use gesture mode, then long-press, move, and release the ring.",
        file=sys.stderr,
    )
    seen = 0
    async with sdk.RingSoundClient(address=address) as ring:
        while count == 0 or seen < count:
            event = await sdk.wait_sensor_gesture_event(ring)
            print(
                json.dumps(
                    {
                        "schema": "ring-hmm-event/v1",
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "timestamp_ms": event.timestamp_ms,
                        "gesture_id": event.gesture_id,
                        "gesture": sdk.sensor_gesture_name(event.gesture_id),
                        "sdk_version": sdk.__version__,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            seen += 1


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(monitor(args.address, args.count))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
