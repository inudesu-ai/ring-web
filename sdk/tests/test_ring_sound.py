from __future__ import annotations

import asyncio
from pathlib import Path
import struct
import sys
import unittest

SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

import ring_sound as sdk  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_imported_version_and_transport_defaults(self) -> None:
        self.assertEqual(sdk.__version__, "0.4.1")
        self.assertEqual(sdk.DEFAULT_SCAN_TIMEOUT_S, 25.0)

    def test_packet_round_trip_and_fragmented_stream(self) -> None:
        first = sdk.encode_packet(0x0605, b"sensor-data")
        second = sdk.encode_packet(0x0702, b"\x00\x00\x00\x7b\x03")
        stream = sdk.PacketStream()

        self.assertEqual(stream.feed(first[:7]), [])
        packets = stream.feed(first[7:] + second)

        self.assertEqual([packet.command for packet in packets], [0x0605, 0x0702])
        self.assertEqual(packets[0].body, b"sensor-data")
        self.assertEqual(packets[1].body, b"\x00\x00\x00\x7b\x03")

    def test_parse_sensor_batch(self) -> None:
        sample = struct.pack(">Ihhhhhh", 1234, 100, -200, 300, -400, 500, -600)
        body = struct.pack(">HIHH", 0, 42, 1, 16) + sample

        batch = sdk.parse_sensor_data_batch(body)

        self.assertEqual(batch.sequence_start, 42)
        self.assertEqual(batch.frame_count, 1)
        self.assertEqual(batch.samples[0].timestamp_ms, 1234)
        self.assertEqual(
            (
                batch.samples[0].accel_x,
                batch.samples[0].accel_y,
                batch.samples[0].accel_z,
            ),
            (100, -200, 300),
        )
        self.assertEqual(
            (
                batch.samples[0].gyro_x,
                batch.samples[0].gyro_y,
                batch.samples[0].gyro_z,
            ),
            (-400, 500, -600),
        )

    def test_parse_gesture_event_and_unknown_name(self) -> None:
        event = sdk.parse_sensor_gesture_event(struct.pack(">IB", 99, 3))
        self.assertEqual(event.timestamp_ms, 99)
        self.assertEqual(sdk.sensor_gesture_name(event.gesture_id), "wave")
        self.assertEqual(sdk.sensor_gesture_name(250), "unknown(250)")

    def test_rejects_bad_sensor_sample_size(self) -> None:
        body = struct.pack(">HIHH", 0, 0, 0, 12)
        with self.assertRaises(sdk.ProtocolError):
            sdk.parse_sensor_data_batch(body)


class FakeBleakClient:
    def __init__(self) -> None:
        self.is_connected = True
        self.writes: list[tuple[str, bytes, bool]] = []

    async def write_gatt_char(
        self, characteristic: str, data: bytes, *, response: bool
    ) -> None:
        self.writes.append((characteristic, bytes(data), response))


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_nus_writes_fixed_twenty_byte_chunks(self) -> None:
        transport = sdk.NusClient(address="test-address")
        fake = FakeBleakClient()
        transport._client = fake

        await transport.write(bytes(range(45)))

        self.assertEqual([len(write[1]) for write in fake.writes], [20, 20, 5])
        self.assertTrue(all(write[0] == sdk.NUS_RX_UUID for write in fake.writes))
        self.assertTrue(all(write[2] is False for write in fake.writes))


if __name__ == "__main__":
    unittest.main()
