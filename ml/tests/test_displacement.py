from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.displacement import (  # noqa: E402
    GRAVITY_MPS2,
    DisplacementTracker,
    rotate_body_to_world,
)


class DisplacementTrackerTests(unittest.TestCase):
    def make_tracker(self, **overrides: float) -> DisplacementTracker:
        options = {
            "sample_rate_hz": 100.0,
            "accel_threshold_floor_g": 0.001,
            "accel_threshold_ceiling_g": 0.05,
            "noise_multiplier": 3.0,
            "velocity_damping_per_s": 0.0,
            "arm_stationary_s": 0.20,
        }
        options.update(overrides)
        return DisplacementTracker(**options)

    @staticmethod
    def update(
        tracker: DisplacementTracker,
        *,
        accel=(0.0, 0.0, 1.0),
        gyro=(0.0, 0.0, 0.0),
        stationary=True,
        confidence=1.0,
    ):
        return tracker.update(
            dt_s=0.01,
            accel_body_g=accel,
            gyro_body_dps=gyro,
            quaternion=(1.0, 0.0, 0.0, 0.0),
            stationary=stationary,
            stationary_confidence=confidence,
        )

    def arm(self, tracker: DisplacementTracker) -> None:
        for _ in range(80):
            latest = self.update(tracker)
        self.assertTrue(latest.armed)

    def test_static_signal_arms_without_drift(self) -> None:
        tracker = self.make_tracker(accel_threshold_floor_g=0.01)
        self.arm(tracker)
        for index in range(500):
            noise = 0.0015 * math.sin(index * 0.31)
            latest = self.update(
                tracker,
                accel=(noise, -0.7 * noise, 1.0 + 0.4 * noise),
            )
        self.assertFalse(latest.moving)
        self.assertEqual(latest.velocity_mps, (0.0, 0.0, 0.0))
        self.assertLess(math.dist(latest.position_m, (0.0, 0.0, 0.0)), 1e-9)

    def test_rest_to_rest_pulse_uses_double_integration_and_zupt(self) -> None:
        tracker = self.make_tracker()
        self.arm(tracker)
        one_mps2_g = 1.0 / GRAVITY_MPS2
        for _ in range(50):
            latest = self.update(
                tracker,
                accel=(one_mps2_g, 0.0, 1.0),
                stationary=False,
            )
        for _ in range(50):
            latest = self.update(
                tracker,
                accel=(-one_mps2_g, 0.0, 1.0),
                stationary=False,
            )
        for _ in range(20):
            latest = self.update(tracker)

        self.assertFalse(latest.moving)
        self.assertEqual(latest.velocity_mps, (0.0, 0.0, 0.0))
        self.assertGreaterEqual(latest.zupt_count, 1)
        self.assertAlmostEqual(latest.position_m[0], 0.25, delta=0.035)

    def test_rotation_only_does_not_create_translation(self) -> None:
        tracker = self.make_tracker(accel_threshold_floor_g=0.01)
        self.arm(tracker)
        for _ in range(100):
            latest = self.update(
                tracker,
                accel=(0.025, 0.0, 1.0),
                gyro=(0.0, 0.0, 180.0),
                stationary=False,
            )
        self.assertTrue(latest.rotating_only)
        self.assertFalse(latest.moving)
        self.assertEqual(latest.position_m, (0.0, 0.0, 0.0))

    def test_gravity_removal_with_rotated_quaternion(self) -> None:
        quaternion = (
            math.sqrt(0.5),
            0.0,
            -math.sqrt(0.5),
            0.0,
        )
        world = rotate_body_to_world(quaternion, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(world[0], 0.0, places=6)
        self.assertAlmostEqual(world[2], 1.0, places=6)

    def test_low_confidence_does_not_apply_zupt(self) -> None:
        tracker = self.make_tracker()
        self.arm(tracker)
        for _ in range(30):
            latest = self.update(
                tracker,
                accel=(0.1, 0.0, 1.0),
                stationary=False,
            )
        self.assertTrue(latest.moving)
        latest = self.update(tracker, stationary=True, confidence=0.4)
        self.assertTrue(latest.moving)
        self.assertNotEqual(latest.velocity_mps, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
