from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.orientation import (  # noqa: E402
    SixAxisAhrs,
    quaternion_from_euler,
    quaternion_to_euler_deg,
    quaternion_to_matrix,
)


class LiveOrientationTests(unittest.TestCase):
    def test_initial_tilt_comes_from_gravity(self) -> None:
        truth = quaternion_from_euler(
            np.deg2rad(22), np.deg2rad(-13), 0
        )
        accel = (
            quaternion_to_matrix(truth).T @ np.asarray([0.0, 0.0, 1.0])
        )
        estimator = SixAxisAhrs()
        estimate = estimator.update(accel, np.zeros(3), 0.01)
        euler = quaternion_to_euler_deg(estimate)

        self.assertAlmostEqual(euler[0], 22, delta=0.2)
        self.assertAlmostEqual(euler[1], -13, delta=0.2)
        telemetry = estimator.telemetry(accel, np.zeros(3))
        self.assertTrue(telemetry["stationary"])
        self.assertLess(
            np.linalg.norm(list(telemetry["linear_accel_g"].values())),
            1e-6,
        )

    def test_gravity_fusion_limits_tilt_bias_drift(self) -> None:
        estimator = SixAxisAhrs()
        for _ in range(6000):
            estimator.update(
                np.asarray([0.0, 0.0, 1.0]),
                np.asarray([0.5, -0.35, 0.7]),
                0.01,
            )
        roll, pitch, _yaw = quaternion_to_euler_deg(estimator.quaternion)
        self.assertLess(np.hypot(roll, pitch), 1.0)

    def test_large_stable_hardware_bias_is_zeroed_at_startup(self) -> None:
        estimator = SixAxisAhrs()
        hardware_bias = np.asarray([1.5, -24.6, 2.0])
        for _ in range(100):
            estimator.update(
                np.asarray([1.0, 0.0, 0.0]),
                hardware_bias,
                0.01,
            )
        self.assertTrue(estimator.calibrated)
        self.assertTrue(estimator.stationary)
        self.assertGreater(estimator.stationary_confidence, 0.9)
        self.assertLess(np.linalg.norm(estimator.corrected_gyro_dps), 1e-6)
        np.testing.assert_allclose(
            estimator.gyro_bias_dps,
            hardware_bias,
            atol=1e-6,
        )

    def test_low_fluctuation_bias_drift_remains_stationary(self) -> None:
        estimator = SixAxisAhrs()
        initial_bias = np.asarray([0.5, -9.0, -0.3])
        for index in range(100):
            noise = 0.002 * np.sin(index * 0.37)
            estimator.update(
                np.asarray([1.0 + noise, -0.5 * noise, 0.03]),
                initial_bias + np.asarray([noise, -noise, 0.5 * noise]),
                0.01,
            )
        self.assertTrue(estimator.calibrated)

        final_bias = np.asarray([0.5, -8.1, -0.3])
        for index in range(2000):
            progress = index / 1999
            drifted_bias = initial_bias + progress * (final_bias - initial_bias)
            noise = 0.0025 * np.sin(index * 0.31)
            estimator.update(
                np.asarray([0.999 + noise, 0.4 * noise, 0.035]),
                drifted_bias + np.asarray([noise, -noise, 0.5 * noise]),
                0.01,
            )

        self.assertTrue(estimator.stationary)
        self.assertGreater(estimator.stationary_confidence, 0.7)
        self.assertLess(np.linalg.norm(estimator.corrected_gyro_dps), 0.35)

    def test_obvious_change_exits_stationary_immediately(self) -> None:
        estimator = SixAxisAhrs()
        for _ in range(100):
            estimator.update(
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.5, -9.0, -0.3]),
                0.01,
            )
        self.assertTrue(estimator.stationary)
        estimator.update(
            np.asarray([1.0, 0.2, 0.0]),
            np.asarray([45.0, -9.0, -0.3]),
            0.01,
        )
        self.assertFalse(estimator.stationary)
        self.assertEqual(estimator.stationary_confidence, 0.0)

    def test_quiet_bias_step_is_relearned_instead_of_becoming_motion(self) -> None:
        estimator = SixAxisAhrs()
        old_bias = np.asarray([-2.0, -15.0, 0.5])
        for _ in range(100):
            estimator.update(
                np.asarray([1.0, 0.0, 0.0]),
                old_bias,
                0.01,
            )
        self.assertTrue(estimator.stationary)

        new_bias = np.asarray([0.5, -0.7, -0.9])
        for _ in range(120):
            estimator.update(
                np.asarray([1.0, 0.0, 0.0]),
                new_bias,
                0.01,
            )

        self.assertTrue(estimator.stationary)
        self.assertGreater(estimator.stationary_confidence, 0.8)
        np.testing.assert_allclose(
            estimator.gyro_bias_dps,
            new_bias,
            atol=0.05,
        )
        self.assertLess(np.linalg.norm(estimator.corrected_gyro_dps), 0.05)


if __name__ == "__main__":
    unittest.main()
