"""Stateful six-axis orientation estimation for live Ring Sound telemetry."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return np.asarray(quaternion, dtype=np.float64) / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return normalize_quaternion(
        np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_to_euler_deg(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.rad2deg([roll, pitch, yaw])


class SixAxisAhrs:
    """Mahony-style gravity fusion with runtime gyro-bias estimation.

    The quaternion maps sensor-frame vectors into a local world frame. Gravity
    corrects roll and pitch; absolute yaw remains unobservable in six-axis mode.
    """

    def __init__(self, *, kp: float = 1.25, ki: float = 0.06) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.initialized = False

    def _initialize_from_accel(self, accel_g: np.ndarray) -> None:
        ax, ay, az = accel_g / max(float(np.linalg.norm(accel_g)), 1e-12)
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
        self.quaternion = quaternion_from_euler(roll, pitch, 0.0)
        self.initialized = True

    def update(
        self,
        accel_g: np.ndarray,
        gyro_dps: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        accel = np.asarray(accel_g, dtype=np.float64)
        gyro = np.asarray(gyro_dps, dtype=np.float64)
        dt = float(np.clip(dt, 0.001, 0.1))
        accel_norm = float(np.linalg.norm(accel))

        if not self.initialized and accel_norm > 1e-9:
            self._initialize_from_accel(accel)

        corrected = np.deg2rad(gyro)
        # Reject strong translational acceleration while retaining smooth
        # gravity correction during ordinary hand motion.
        if 0.78 <= accel_norm <= 1.22:
            measured_gravity = accel / accel_norm
            estimated_gravity = (
                quaternion_to_matrix(self.quaternion).T
                @ np.asarray([0.0, 0.0, 1.0])
            )
            error = np.cross(measured_gravity, estimated_gravity)
            self.integral += self.ki * error * dt
            corrected += self.kp * error + self.integral

        derivative = 0.5 * quaternion_multiply(
            self.quaternion, np.r_[0.0, corrected]
        )
        self.quaternion = normalize_quaternion(
            self.quaternion + derivative * dt
        )
        return self.quaternion.copy()

    def telemetry(
        self,
        accel_g: np.ndarray,
        gyro_dps: np.ndarray,
    ) -> dict[str, object]:
        accel = np.asarray(accel_g, dtype=np.float64)
        gyro = np.asarray(gyro_dps, dtype=np.float64)
        rotation = quaternion_to_matrix(self.quaternion)
        linear_world = rotation @ accel - np.asarray([0.0, 0.0, 1.0])
        euler = quaternion_to_euler_deg(self.quaternion)
        stationary = bool(
            np.linalg.norm(gyro) < 3.0
            and abs(float(np.linalg.norm(accel)) - 1.0) < 0.05
        )
        return {
            "quaternion": {
                axis: float(value)
                for axis, value in zip("wxyz", self.quaternion)
            },
            "euler_deg": {
                axis: float(value)
                for axis, value in zip(("roll", "pitch", "yaw"), euler)
            },
            "accel_g": {
                axis: float(value) for axis, value in zip("xyz", accel)
            },
            "gyro_dps": {
                axis: float(value) for axis, value in zip("xyz", gyro)
            },
            "linear_accel_g": {
                axis: float(value) for axis, value in zip("xyz", linear_world)
            },
            "stationary": stationary,
            "gyro_bias_dps": {
                axis: float(value)
                for axis, value in zip("xyz", -np.rad2deg(self.integral))
            },
        }
