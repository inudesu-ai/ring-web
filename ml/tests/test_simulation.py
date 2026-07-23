from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
SIM_DIR = ML_DIR / "simulation"
sys.path.insert(0, str(SIM_DIR))

from imu_simulator import simulate_session  # noqa: E402
from orientation_filters import (  # noqa: E402
    integrate_gyroscope,
    madgwick_filter,
    quaternion_to_euler,
)


class SimulationTests(unittest.TestCase):
    def test_session_is_deterministic_and_bounded(self) -> None:
        first = simulate_session(
            label="circle",
            sample_count=200,
            sample_rate_hz=25,
            subject_index=3,
            session_seed=55,
        )
        second = simulate_session(
            label="circle",
            sample_count=200,
            sample_rate_hz=25,
            subject_index=3,
            session_seed=55,
        )
        np.testing.assert_array_equal(first.accel_raw, second.accel_raw)
        np.testing.assert_array_equal(first.gyro_raw, second.gyro_raw)
        self.assertEqual(first.accel_raw.shape, (200, 3))
        self.assertTrue(np.all(np.diff(first.timestamps_ms) > 0))

    def test_accelerometer_fusion_corrects_tilt_drift(self) -> None:
        sample_rate = 100
        count = sample_rate * 30
        gyro = np.tile([0.5, -0.35, 0.7], (count, 1))
        accel = np.tile([0.0, 0.0, 1.0], (count, 1))
        raw = integrate_gyroscope(gyro, 1 / sample_rate)
        fused = madgwick_filter(gyro, accel, 1 / sample_rate)
        raw_tilt = np.linalg.norm(
            np.rad2deg(quaternion_to_euler(raw[-1])[:2])
        )
        fused_tilt = np.linalg.norm(
            np.rad2deg(quaternion_to_euler(fused[-1])[:2])
        )
        self.assertGreater(raw_tilt, 10)
        self.assertLess(fused_tilt, 2)


if __name__ == "__main__":
    unittest.main()
