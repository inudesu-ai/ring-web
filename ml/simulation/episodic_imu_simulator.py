"""Physics-informed, pre-segmented ring gesture episode simulator.

Unlike the legacy continuous random-phase simulator, every generated session
contains one command gesture with a short neutral prefix and suffix.  This is
the data shape required by a gesture spotter followed by a classifier: the
classifier learns the ordered acceleration/angular-rate signature of a
command, rather than an arbitrary phase of a periodic motion.
"""

from __future__ import annotations

import numpy as np

from imu_simulator import (
    LABELS,
    SimulatedSession,
    _rotation_matrix_xyz,
    _smooth_sensor,
    euler_zyx_to_matrices,
)

GRAVITY_MPS2 = 9.80665


def _minimum_jerk(progress: np.ndarray) -> np.ndarray:
    progress = np.clip(progress, 0.0, 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _episode_template(
    label: str,
    time_s: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Euler attitude and world-frame linear acceleration in g."""

    count = len(time_s)
    duration_s = time_s[-1] + np.median(np.diff(time_s))
    onset_s = rng.uniform(0.12, 0.25) * duration_s
    nominal_motion_s = {
        "idle": duration_s,
        "wave": 0.95,
        "rotate_back": 0.75,
        "rotate_front": 0.75,
        "left": 0.72,
        "right": 0.72,
        "up": 0.78,
        "down": 0.78,
        "circle": 1.05,
        "double_tap": 0.72,
    }[label]
    motion_s = np.clip(nominal_motion_s / speed, 0.48, duration_s - onset_s - 0.10)
    progress = np.clip((time_s - onset_s) / motion_s, 0.0, 1.0)
    active = (time_s >= onset_s) & (time_s <= onset_s + motion_s)
    smooth = _minimum_jerk(progress)

    euler = np.zeros((count, 3), dtype=np.float64)
    position_m = np.zeros((count, 3), dtype=np.float64)
    linear_g = np.zeros((count, 3), dtype=np.float64)

    if label == "idle":
        euler[:, 0] = np.deg2rad(0.35) * np.sin(2 * np.pi * 0.35 * time_s)
        euler[:, 1] = np.deg2rad(0.25) * np.cos(2 * np.pi * 0.28 * time_s)
    elif label in {"left", "right", "up", "down"}:
        direction = {
            "left": np.asarray([-1.0, 0.0, 0.0]),
            "right": np.asarray([1.0, 0.0, 0.0]),
            "up": np.asarray([0.0, 0.0, 1.0]),
            "down": np.asarray([0.0, 0.0, -1.0]),
        }[label]
        distance_m = rng.uniform(0.16, 0.24) * amplitude
        position_m = smooth[:, None] * distance_m * direction[None, :]
        if label in {"left", "right"}:
            euler[:, 0] = direction[0] * np.deg2rad(10.0 * amplitude) * smooth
            euler[:, 2] = direction[0] * np.deg2rad(5.0 * amplitude) * smooth
        else:
            euler[:, 1] = -direction[2] * np.deg2rad(12.0 * amplitude) * smooth
    elif label in {"rotate_back", "rotate_front"}:
        sign = -1.0 if label == "rotate_back" else 1.0
        euler[:, 1] = sign * np.deg2rad(rng.uniform(42.0, 62.0) * amplitude) * smooth
        euler[:, 0] = sign * np.deg2rad(5.0 * amplitude) * np.sin(np.pi * smooth)
        position_m[:, 2] = sign * 0.025 * amplitude * np.sin(np.pi * smooth)
    elif label == "wave":
        envelope = np.sin(np.pi * smooth) ** 2
        euler[:, 0] = (
            np.deg2rad(rng.uniform(30.0, 42.0) * amplitude)
            * envelope
            * np.sin(4.0 * np.pi * smooth)
        )
        euler[:, 2] = (
            np.deg2rad(10.0 * amplitude)
            * envelope
            * np.sin(4.0 * np.pi * smooth + 0.4)
        )
        position_m[:, 0] = 0.045 * amplitude * envelope * np.sin(4.0 * np.pi * smooth)
    elif label == "circle":
        theta = 2.0 * np.pi * smooth
        radius_m = rng.uniform(0.075, 0.115) * amplitude
        position_m[:, 0] = radius_m * (np.cos(theta) - 1.0)
        position_m[:, 1] = radius_m * np.sin(theta)
        position_m[:, 2] = 0.012 * amplitude * np.sin(2.0 * theta)
        envelope = np.sin(np.pi * smooth) ** 2
        euler[:, 0] = np.deg2rad(13.0 * amplitude) * envelope * np.sin(theta)
        euler[:, 1] = np.deg2rad(13.0 * amplitude) * envelope * np.cos(theta)
    elif label == "double_tap":
        local = (time_s - onset_s) / motion_s
        first = np.exp(-0.5 * ((local - 0.30) / 0.045) ** 2)
        second = np.exp(-0.5 * ((local - 0.68) / 0.045) ** 2)
        rebound = np.exp(-0.5 * ((local - 0.82) / 0.065) ** 2)
        linear_g[:, 2] = amplitude * (0.72 * first + 0.68 * second - 0.28 * rebound)
        linear_g[:, 0] = amplitude * (0.10 * first + 0.09 * second)
        euler[:, 1] = np.deg2rad(5.0 * amplitude) * (first + second)
    else:
        raise ValueError(f"Unknown gesture label: {label}")

    if label not in {"idle", "double_tap"}:
        dt = float(np.median(np.diff(time_s)))
        velocity = np.gradient(position_m, dt, axis=0, edge_order=2)
        linear_g = np.gradient(velocity, dt, axis=0, edge_order=2) / GRAVITY_MPS2
        linear_g[~active] = 0.0
    return euler, linear_g


def simulate_gesture_episode(
    *,
    label: str,
    sample_count: int,
    sample_rate_hz: float,
    subject_index: int,
    episode_index: int,
    session_seed: int,
    shifted_domain: bool = False,
) -> SimulatedSession:
    """Simulate one pre-segmented six-axis command gesture."""

    if label not in LABELS:
        raise ValueError(f"Unknown gesture label: {label}")
    if sample_count < 20:
        raise ValueError("An episode needs at least 20 samples")
    rng = np.random.default_rng(session_seed)
    subject_rng = np.random.default_rng(50_000 + subject_index)
    dt = 1.0 / sample_rate_hz
    time_s = np.arange(sample_count, dtype=np.float64) * dt

    subject_amplitude = subject_rng.uniform(0.84, 1.16)
    subject_speed = subject_rng.uniform(0.86, 1.14)
    amplitude = subject_amplitude * rng.uniform(0.90, 1.10)
    speed = subject_speed * rng.uniform(0.91, 1.09)
    euler, linear_world_g = _episode_template(
        label, time_s, rng, amplitude, speed
    )

    matrices = euler_zyx_to_matrices(euler)
    gravity_plus_linear = linear_world_g + np.asarray([0.0, 0.0, 1.0])
    accel_body = np.einsum("nji,nj->ni", matrices, gravity_plus_linear)

    roll, pitch, yaw = euler.T
    roll_rate = np.gradient(roll, dt)
    pitch_rate = np.gradient(pitch, dt)
    yaw_rate = np.gradient(yaw, dt)
    gyro_body = np.rad2deg(
        np.column_stack(
            [
                roll_rate - yaw_rate * np.sin(pitch),
                pitch_rate * np.cos(roll)
                + yaw_rate * np.sin(roll) * np.cos(pitch),
                -pitch_rate * np.sin(roll)
                + yaw_rate * np.cos(roll) * np.cos(pitch),
            ]
        )
    )

    severity = 1.35 if shifted_domain else 1.0
    placement_limit = 17.0 if shifted_domain else 11.0
    placement_angles = np.deg2rad(
        subject_rng.uniform(-placement_limit, placement_limit, 3)
        + rng.normal(0.0, 1.5 * severity, 3)
    )
    # The physical Ring Sound unit rests with sensor +X approximately aligned
    # with gravity (confirmed by the local real captures).  The simulator's
    # body frame uses +Z for gravity, so apply the fixed package mounting
    # rotation before the smaller subject/episode placement perturbation.
    package_mount = _rotation_matrix_xyz(np.asarray([0.0, -np.pi / 2.0, 0.0]))
    placement = package_mount @ _rotation_matrix_xyz(placement_angles)
    accel_sensor = accel_body @ placement
    gyro_sensor = gyro_body @ placement

    accel_gain = subject_rng.normal(1.0, 0.007 * severity, 3)
    gyro_gain = subject_rng.normal(1.0, 0.009 * severity, 3)
    accel_bias = subject_rng.normal(0.0, 0.007 * severity, 3)
    gyro_bias = subject_rng.normal(0.0, 0.45 * severity, 3)
    # Episode-to-episode mounting and electronics variation.
    accel_bias += rng.normal(0.0, 0.0015 * severity, 3)
    gyro_bias += rng.normal(0.0, 0.08 * severity, 3)
    accel_walk = np.cumsum(
        rng.normal(0.0, 0.000025 * severity, (sample_count, 3)), axis=0
    )
    gyro_walk = np.cumsum(
        rng.normal(0.0, 0.0015 * severity, (sample_count, 3)), axis=0
    )
    accel_noise = rng.uniform(0.0025, 0.0080) * severity
    gyro_noise = rng.uniform(0.05, 0.18) * severity
    accel_measured = (
        accel_sensor * accel_gain
        + accel_bias
        + accel_walk
        + rng.normal(0.0, accel_noise, (sample_count, 3))
    )
    gyro_measured = (
        gyro_sensor * gyro_gain
        + gyro_bias
        + gyro_walk
        + rng.normal(0.0, gyro_noise, (sample_count, 3))
    )
    bandwidth_alpha = rng.uniform(0.72, 0.92)
    accel_measured = _smooth_sensor(accel_measured, bandwidth_alpha)
    gyro_measured = _smooth_sensor(gyro_measured, bandwidth_alpha)

    accel_raw = np.clip(
        np.rint(accel_measured / 16.0 * 32768.0), -32768, 32767
    ).astype(np.int16)
    gyro_raw = np.clip(
        np.rint(gyro_measured / 2000.0 * 32768.0), -32768, 32767
    ).astype(np.int16)
    clock_scale = rng.normal(1.0, 0.003 * severity)
    jitter = rng.normal(0.0, 0.0005 * severity, sample_count)
    intervals = np.maximum(dt * 0.75, dt * clock_scale + jitter)
    timestamps_ms = np.rint(np.cumsum(intervals) * 1000).astype(np.int64)
    timestamps_ms -= timestamps_ms[0]
    stationary = (
        (np.linalg.norm(linear_world_g, axis=1) < 0.018)
        & (np.linalg.norm(gyro_body, axis=1) < 2.2)
    )
    return SimulatedSession(
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        timestamps_ms=timestamps_ms,
        accel_g=accel_measured,
        gyro_dps=gyro_measured,
        true_euler_rad=euler,
        true_linear_accel_world_g=linear_world_g,
        stationary=stationary,
    )
