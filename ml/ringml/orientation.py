"""Stateful six-axis orientation estimation for live Ring Sound telemetry."""

from __future__ import annotations

from collections import deque

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

    def __init__(
        self,
        *,
        kp: float = 1.25,
        ki: float = 0.04,
        calibration_window_s: float = 0.65,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.calibration_window_s = max(0.35, float(calibration_window_s))
        self.quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.gravity_integral = np.zeros(3)
        self.gyro_bias_dps = np.zeros(3)
        self.corrected_gyro_dps = np.zeros(3)
        self.stationary = False
        self.stationary_confidence = 0.0
        self.calibrated = False
        self.linear_accel_world_g = np.zeros(3)
        self._rest_window: deque[tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=240
        )
        self.initialized = False

    def _initialize_from_accel(self, accel_g: np.ndarray) -> None:
        ax, ay, az = accel_g / max(float(np.linalg.norm(accel_g)), 1e-12)
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
        self.quaternion = quaternion_from_euler(roll, pitch, 0.0)
        self.initialized = True

    def _update_rest_state(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> None:
        self._rest_window.append((accel.copy(), gyro.copy(), dt))
        duration = 0.0
        values: list[tuple[np.ndarray, np.ndarray, float]] = []
        for item in reversed(self._rest_window):
            values.append(item)
            duration += item[2]
            if duration >= self.calibration_window_s:
                break
        values.reverse()
        if duration < min(0.30, self.calibration_window_s):
            self.stationary = bool(
                np.linalg.norm(gyro) < 3.0
                and abs(float(np.linalg.norm(accel)) - 1.0) < 0.05
            )
            self.stationary_confidence = 1.0 if self.stationary else 0.0
            return

        accelerations = np.asarray([item[0] for item in values])
        gyroscopes = np.asarray([item[1] for item in values])
        accel_norms = np.linalg.norm(accelerations, axis=1)
        # The current firmware occasionally emits a short sensor spike even
        # when the ring is resting. Robust median/MAD statistics describe the
        # visible fluctuation instead of letting one outlier keep the state in
        # "moving" indefinitely.
        accel_center = np.median(accelerations, axis=0)
        gyro_center = np.median(gyroscopes, axis=0)
        accel_axis_noise = float(
            np.max(
                1.4826
                * np.median(
                    np.abs(accelerations - accel_center),
                    axis=0,
                )
            )
        )
        gyro_axis_noise = float(
            np.max(
                1.4826
                * np.median(
                    np.abs(gyroscopes - gyro_center),
                    axis=0,
                )
            )
        )
        accel_mean_norm = float(np.median(accel_norms))
        half = max(1, len(accelerations) // 2)
        accel_drift = float(
            np.linalg.norm(
                np.median(accelerations[half:], axis=0)
                - np.median(accelerations[:half], axis=0)
            )
        )
        stable_signal = bool(
            0.94 <= accel_mean_norm <= 1.06
            and accel_axis_noise <= 0.014
            and accel_drift <= 0.022
            and gyro_axis_noise <= 1.0
        )

        norm_score = np.clip(1.0 - abs(accel_mean_norm - 1.0) / 0.06, 0.0, 1.0)
        accel_score = np.clip(1.0 - accel_axis_noise / 0.014, 0.0, 1.0)
        drift_score = np.clip(1.0 - accel_drift / 0.022, 0.0, 1.0)
        gyro_score = np.clip(1.0 - gyro_axis_noise / 1.0, 0.0, 1.0)
        signal_confidence = float(
            0.20 * norm_score
            + 0.30 * accel_score
            + 0.25 * drift_score
            + 0.25 * gyro_score
        )

        window_bias = gyro_center
        if stable_signal and not self.calibrated and duration >= self.calibration_window_s:
            self.gyro_bias_dps = window_bias
            self.corrected_gyro_dps = gyro - self.gyro_bias_dps
            self.calibrated = True
            # Remove tilt and heading accumulated while the startup bias was
            # still unknown. Six-axis heading is relative, so yaw=0 is valid.
            self._initialize_from_accel(np.mean(accelerations, axis=0))
            self.gravity_integral.fill(0.0)

        corrected_mean = gyro_center - self.gyro_bias_dps
        current_accel_delta = float(np.linalg.norm(accel - accel_center))
        current_gyro_delta = float(np.linalg.norm(gyro - gyro_center))
        instant_quiet = bool(
            current_accel_delta <= 0.075
            and current_gyro_delta <= 7.0
        )
        rate_score = float(
            np.clip(1.0 - np.linalg.norm(corrected_mean) / 4.0, 0.0, 1.0)
        )
        # A slowly changing constant zero-bias is not motion. Weight the
        # window's actual fluctuation much more heavily than its DC offset.
        self.stationary_confidence = (
            signal_confidence * (0.82 + 0.18 * rate_score)
            if instant_quiet
            else 0.0
        )
        self.stationary = bool(
            stable_signal
            and instant_quiet
            and self.stationary_confidence >= 0.62
        )

        if self.stationary and self.calibrated:
            bias_step = float(np.linalg.norm(window_bias - self.gyro_bias_dps))
            if bias_step >= 3.0:
                # Firmware sensor sessions can change their gyro DC offset.
                # A full quiet window proves this is a new zero rather than
                # motion, so re-anchor the bias and gravity tilt immediately.
                self.gyro_bias_dps = window_bias.copy()
                self._initialize_from_accel(accel_center)
                self.gravity_integral.fill(0.0)
                return
            # Track warm-up drift slowly, but only after a fused rest decision.
            # Use one sample's dt here; using the whole window duration on
            # every sample would make the bias adaptation unintentionally fast.
            alpha = 1.0 - np.exp(-values[-1][2] / 3.0)
            self.gyro_bias_dps += alpha * (window_bias - self.gyro_bias_dps)

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

        self._update_rest_state(accel, gyro, dt)
        if not self.calibrated and self.stationary_confidence >= 0.70:
            # Provisional startup zero prevents a stable, large hardware bias
            # from rotating the attitude while the full window is collected.
            provisional_bias = np.median(
                np.asarray([item[1] for item in self._rest_window]), axis=0
            )
            self.corrected_gyro_dps = gyro - provisional_bias
        else:
            self.corrected_gyro_dps = gyro - self.gyro_bias_dps
        corrected = np.deg2rad(self.corrected_gyro_dps)
        # Reject strong translational acceleration while retaining smooth
        # gravity correction during ordinary hand motion.
        if 0.78 <= accel_norm <= 1.22:
            measured_gravity = accel / accel_norm
            estimated_gravity = (
                quaternion_to_matrix(self.quaternion).T
                @ np.asarray([0.0, 0.0, 1.0])
            )
            error = np.cross(measured_gravity, estimated_gravity)
            self.gravity_integral += self.ki * error * dt
            self.gravity_integral = np.clip(
                self.gravity_integral, -np.deg2rad(2.0), np.deg2rad(2.0)
            )
            corrected += self.kp * error + self.gravity_integral

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
        self.linear_accel_world_g = linear_world
        euler = quaternion_to_euler_deg(self.quaternion)
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
                axis: float(value)
                for axis, value in zip("xyz", self.corrected_gyro_dps)
            },
            "gyro_raw_dps": {
                axis: float(value) for axis, value in zip("xyz", gyro)
            },
            "linear_accel_g": {
                axis: float(value) for axis, value in zip("xyz", linear_world)
            },
            "stationary": self.stationary,
            "stationary_confidence": self.stationary_confidence,
            "calibrated": self.calibrated,
            "gyro_bias_dps": {
                axis: float(value)
                for axis, value in zip("xyz", self.gyro_bias_dps)
            },
        }
