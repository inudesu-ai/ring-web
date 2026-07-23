"""Physics-informed wrist/ring IMU signal simulator.

The simulator starts from smooth orientation and world-frame acceleration,
projects gravity and motion into a displaced sensor frame, and then applies
subject, placement, clock, gain, bias, random-walk, bandwidth, noise, and
quantisation variability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = (
    "idle",
    "wave",
    "rotate_back",
    "rotate_front",
    "left",
    "right",
    "up",
    "down",
    "circle",
    "double_tap",
)


@dataclass(frozen=True)
class SimulatedSession:
    accel_raw: np.ndarray
    gyro_raw: np.ndarray
    timestamps_ms: np.ndarray
    accel_g: np.ndarray
    gyro_dps: np.ndarray
    true_euler_rad: np.ndarray
    true_linear_accel_world_g: np.ndarray
    stationary: np.ndarray


def _rotation_matrix_xyz(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def euler_zyx_to_matrices(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler.T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrices = np.empty((len(euler), 3, 3))
    matrices[:, 0, 0] = cy * cp
    matrices[:, 0, 1] = cy * sp * sr - sy * cr
    matrices[:, 0, 2] = cy * sp * cr + sy * sr
    matrices[:, 1, 0] = sy * cp
    matrices[:, 1, 1] = sy * sp * sr + cy * cr
    matrices[:, 1, 2] = sy * sp * cr - cy * sr
    matrices[:, 2, 0] = -sp
    matrices[:, 2, 1] = cp * sr
    matrices[:, 2, 2] = cp * cr
    return matrices


def _smooth_sensor(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def _periodic_gaussian(phase: np.ndarray, centre: float, width: float) -> np.ndarray:
    distance = np.angle(np.exp(1j * (phase - centre)))
    return np.exp(-0.5 * (distance / width) ** 2)


def _motion_template(
    label: str,
    phase: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(phase)
    euler = np.zeros((count, 3))
    linear = np.zeros((count, 3))
    s1, c1 = np.sin(phase), np.cos(phase)
    s2, c2 = np.sin(2 * phase), np.cos(2 * phase)

    if label == "idle":
        euler[:, 0] = np.deg2rad(0.8) * np.sin(phase * 0.23)
        euler[:, 1] = np.deg2rad(0.6) * np.cos(phase * 0.19)
    elif label == "wave":
        euler[:, 0] = np.deg2rad(34 * amplitude) * s1
        euler[:, 2] = np.deg2rad(15 * amplitude) * np.sin(phase + 0.5)
        linear[:, 0] = 0.20 * amplitude * s1
        linear[:, 1] = 0.08 * amplitude * s2
    elif label in {"rotate_back", "rotate_front"}:
        sign = -1.0 if label == "rotate_back" else 1.0
        euler[:, 1] = sign * np.deg2rad(46 * amplitude) * s1
        euler[:, 0] = np.deg2rad(8 * amplitude) * s2
        linear[:, 2] = sign * 0.10 * amplitude * c1
    elif label in {"left", "right"}:
        sign = -1.0 if label == "left" else 1.0
        linear[:, 0] = sign * 0.34 * amplitude * s1
        linear[:, 1] = sign * 0.07 * amplitude * s2
        euler[:, 0] = sign * np.deg2rad(13 * amplitude) * s1
    elif label in {"up", "down"}:
        sign = 1.0 if label == "up" else -1.0
        linear[:, 2] = sign * 0.32 * amplitude * s1
        linear[:, 0] = 0.05 * amplitude * c2
        euler[:, 1] = -sign * np.deg2rad(11 * amplitude) * s1
    elif label == "circle":
        linear[:, 0] = 0.25 * amplitude * c1
        linear[:, 1] = 0.25 * amplitude * s1
        euler[:, 0] = np.deg2rad(17 * amplitude) * s1
        euler[:, 1] = np.deg2rad(17 * amplitude) * c1
        euler[:, 2] = np.deg2rad(9 * amplitude) * s2
    elif label == "double_tap":
        first = _periodic_gaussian(phase, 1.8, 0.11)
        second = _periodic_gaussian(phase, 2.45, 0.11)
        rebound = _periodic_gaussian(phase, 2.8, 0.18)
        linear[:, 2] = amplitude * (0.75 * first + 0.70 * second - 0.30 * rebound)
        linear[:, 0] = amplitude * (0.14 * first + 0.12 * second)
        euler[:, 1] = np.deg2rad(7 * amplitude) * (first + second)
    else:
        raise ValueError(f"Unknown gesture label: {label}")
    return euler, linear


def simulate_session(
    *,
    label: str,
    sample_count: int,
    sample_rate_hz: float,
    subject_index: int,
    session_seed: int,
    shifted_domain: bool = False,
) -> SimulatedSession:
    rng = np.random.default_rng(session_seed)
    dt = 1.0 / sample_rate_hz
    time = np.arange(sample_count) * dt

    subject_rng = np.random.default_rng(10_000 + subject_index)
    subject_amplitude = subject_rng.uniform(0.80, 1.20)
    subject_speed = subject_rng.uniform(0.82, 1.18)
    handedness = -1.0 if subject_rng.random() < 0.15 else 1.0
    session_amplitude = subject_amplitude * rng.uniform(0.90, 1.10)
    session_speed = subject_speed * rng.uniform(0.92, 1.08)
    base_frequency = {
        "idle": 0.12,
        "wave": 1.15,
        "rotate_back": 0.72,
        "rotate_front": 0.72,
        "left": 0.85,
        "right": 0.85,
        "up": 0.80,
        "down": 0.80,
        "circle": 0.65,
        "double_tap": 0.55,
    }[label]
    phase = (
        2.0 * np.pi * base_frequency * session_speed * time
        + rng.uniform(0, 2 * np.pi)
        + 0.025 * np.sin(2 * np.pi * 0.07 * time)
    )
    euler, linear_world = _motion_template(label, phase, session_amplitude)
    euler[:, 2] *= handedness

    matrices = euler_zyx_to_matrices(euler)
    gravity_plus_linear = linear_world + np.asarray([0.0, 0.0, 1.0])
    accel_body = np.einsum("nji,nj->ni", matrices, gravity_plus_linear)

    roll, pitch, _yaw = euler.T
    roll_rate = np.gradient(roll, dt)
    pitch_rate = np.gradient(pitch, dt)
    yaw_rate = np.gradient(euler[:, 2], dt)
    gyro_body = np.column_stack(
        [
            roll_rate - yaw_rate * np.sin(pitch),
            pitch_rate * np.cos(roll)
            + yaw_rate * np.sin(roll) * np.cos(pitch),
            -pitch_rate * np.sin(roll)
            + yaw_rate * np.cos(roll) * np.cos(pitch),
        ]
    )
    gyro_body = np.rad2deg(gyro_body)

    placement_limit = 32.0 if shifted_domain else 18.0
    placement_angles = np.deg2rad(
        subject_rng.uniform(-placement_limit, placement_limit, 3)
        + rng.normal(0, 2.0, 3)
    )
    placement = _rotation_matrix_xyz(placement_angles)
    accel_sensor = accel_body @ placement
    gyro_sensor = gyro_body @ placement

    severity = 1.6 if shifted_domain else 1.0
    accel_gain = subject_rng.normal(1.0, 0.008 * severity, 3)
    gyro_gain = subject_rng.normal(1.0, 0.010 * severity, 3)
    accel_bias = subject_rng.normal(0.0, 0.009 * severity, 3)
    gyro_bias = subject_rng.normal(0.0, 0.65 * severity, 3)
    accel_walk = np.cumsum(
        rng.normal(0, 0.000025 * severity, (sample_count, 3)), axis=0
    )
    gyro_walk = np.cumsum(
        rng.normal(0, 0.0018 * severity, (sample_count, 3)), axis=0
    )
    accel_noise = rng.uniform(0.003, 0.011) * severity
    gyro_noise = rng.uniform(0.06, 0.24) * severity

    accel_measured = (
        accel_sensor * accel_gain
        + accel_bias
        + accel_walk
        + rng.normal(0, accel_noise, (sample_count, 3))
    )
    gyro_measured = (
        gyro_sensor * gyro_gain
        + gyro_bias
        + gyro_walk
        + rng.normal(0, gyro_noise, (sample_count, 3))
    )
    bandwidth_alpha = rng.uniform(0.70, 0.92)
    accel_measured = _smooth_sensor(accel_measured, bandwidth_alpha)
    gyro_measured = _smooth_sensor(gyro_measured, bandwidth_alpha)

    accel_range_g = 16.0
    gyro_range_dps = 2000.0
    accel_raw = np.clip(
        np.rint(accel_measured / accel_range_g * 32768.0), -32768, 32767
    ).astype(np.int16)
    gyro_raw = np.clip(
        np.rint(gyro_measured / gyro_range_dps * 32768.0), -32768, 32767
    ).astype(np.int16)

    clock_scale = rng.normal(1.0, 0.004 * severity)
    jitter = rng.normal(0.0, 0.0007 * severity, sample_count)
    sample_intervals = np.maximum(dt * 0.7, dt * clock_scale + jitter)
    timestamps_ms = np.rint(np.cumsum(sample_intervals) * 1000).astype(np.int64)
    timestamps_ms -= timestamps_ms[0]
    stationary = (
        (np.linalg.norm(linear_world, axis=1) < 0.025)
        & (np.linalg.norm(gyro_body, axis=1) < 2.5)
    )
    return SimulatedSession(
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        timestamps_ms=timestamps_ms,
        accel_g=accel_measured,
        gyro_dps=gyro_measured,
        true_euler_rad=euler,
        true_linear_accel_world_g=linear_world,
        stationary=stationary,
    )
