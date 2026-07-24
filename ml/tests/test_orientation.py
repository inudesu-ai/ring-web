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


if __name__ == "__main__":
    unittest.main()
