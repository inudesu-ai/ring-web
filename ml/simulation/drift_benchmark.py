#!/usr/bin/env python3
"""Benchmark six-axis attitude drift and ZUPT-constrained displacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imu_simulator import euler_zyx_to_matrices
from orientation_filters import (
    integrate_gyroscope,
    madgwick_filter,
    mahony_filter,
    quaternion_from_euler,
    quaternion_to_euler,
    quaternion_to_matrix,
)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def generate_trajectory(
    *, duration: float, sample_rate: float, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_rate
    time = np.arange(round(duration * sample_rate)) * dt
    cycle = np.mod(time, 20.0)
    moving = (cycle >= 5.0) & (cycle < 15.0)
    local = np.clip((cycle - 5.0) / 10.0, 0.0, 1.0)
    envelope = np.sin(np.pi * local) ** 2 * moving

    euler = np.zeros((len(time), 3))
    euler[:, 0] = np.deg2rad(24) * envelope * np.sin(2 * np.pi * 0.40 * time)
    euler[:, 1] = np.deg2rad(18) * envelope * np.sin(
        2 * np.pi * 0.30 * time + 0.7
    )
    euler[:, 2] = np.deg2rad(35) * envelope * np.sin(
        2 * np.pi * 0.20 * time + 0.2
    )
    linear_world = np.zeros((len(time), 3))
    linear_world[:, 0] = 0.22 * envelope * np.sin(2 * np.pi * 0.30 * time)
    linear_world[:, 1] = 0.15 * envelope * np.sin(
        2 * np.pi * 0.40 * time + 0.4
    )
    linear_world[:, 2] = 0.10 * envelope * np.sin(
        2 * np.pi * 0.50 * time + 0.8
    )

    matrices = euler_zyx_to_matrices(euler)
    accel_true = np.einsum(
        "nji,nj->ni", matrices, linear_world + np.asarray([0.0, 0.0, 1.0])
    )
    roll, pitch, _ = euler.T
    roll_rate = np.gradient(roll, dt)
    pitch_rate = np.gradient(pitch, dt)
    yaw_rate = np.gradient(euler[:, 2], dt)
    gyro_true = np.rad2deg(
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

    gyro_bias = np.asarray([0.48, -0.32, 0.72])
    accel_bias = np.asarray([0.012, -0.010, 0.018])
    gyro_walk = np.cumsum(rng.normal(0, 0.0007, gyro_true.shape), axis=0)
    accel_walk = np.cumsum(rng.normal(0, 0.000008, accel_true.shape), axis=0)
    gyro_measured = (
        gyro_true + gyro_bias + gyro_walk + rng.normal(0, 0.10, gyro_true.shape)
    )
    accel_measured = (
        accel_true + accel_bias + accel_walk + rng.normal(0, 0.004, accel_true.shape)
    )
    true_velocity = np.zeros_like(linear_world)
    true_position = np.zeros_like(linear_world)
    for index in range(1, len(time)):
        true_velocity[index] = true_velocity[index - 1] + 9.80665 * (
            linear_world[index - 1] + linear_world[index]
        ) * 0.5 * dt
        if not moving[index]:
            true_velocity[index] = 0.0
        true_position[index] = true_position[index - 1] + (
            true_velocity[index - 1] + true_velocity[index]
        ) * 0.5 * dt
    return {
        "time": time,
        "dt": np.asarray(dt),
        "euler": euler,
        "true_quaternion": quaternion_from_euler(euler),
        "linear_world": linear_world,
        "true_velocity": true_velocity,
        "true_position": true_position,
        "gyro_true": gyro_true,
        "accel_true": accel_true,
        "gyro": gyro_measured,
        "accel": accel_measured,
        "stationary": ~moving,
        "gyro_bias": gyro_bias,
        "accel_bias": accel_bias,
    }


def run_vqf(gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray | None:
    try:
        from vqf import VQF
    except ImportError:
        return None
    filter_ = VQF(gyrTs=dt, accTs=dt)
    result = filter_.updateBatch(np.deg2rad(gyro), accel)
    return np.asarray(result["quat6D"])


def run_fusion(
    gyro: np.ndarray, accel: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import imufusion
    except ImportError:
        return None
    ahrs = imufusion.Ahrs()
    settings = imufusion.AhrsSettings()
    settings.convention = imufusion.CONVENTION_NWU
    settings.gain = 0.5
    settings.gyroscope_range = 2000
    settings.acceleration_rejection = 10
    settings.magnetic_rejection = 10
    settings.rejection_timeout = 5
    settings.sample_rate = round(1 / dt)
    ahrs.set_settings(settings)
    ahrs.set_sample_period(dt)
    bias = imufusion.Bias()
    bias_settings = imufusion.BiasSettings()
    bias_settings.sample_rate = round(1 / dt)
    bias_settings.stationary_threshold = 3.0
    bias_settings.stationary_period = 3.0
    bias.set_settings(bias_settings)
    quaternions = np.empty((len(gyro), 4))
    corrected = np.empty_like(gyro)
    for index in range(len(gyro)):
        corrected[index] = bias.update(gyro[index])
        ahrs.update_no_magnetometer(corrected[index], accel[index])
        quaternions[index] = ahrs.get_quaternion()
    return quaternions, corrected


def orientation_metrics(
    estimate: np.ndarray, truth_euler: np.ndarray
) -> dict[str, float]:
    estimated_euler = quaternion_to_euler(estimate)
    error = wrap_angle(estimated_euler - truth_euler)
    tilt = np.rad2deg(
        np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
    )
    yaw = np.rad2deg(error[:, 2] - error[0, 2])
    return {
        "tilt_rmse_deg": float(np.sqrt(np.mean(tilt * tilt))),
        "tilt_p95_deg": float(np.percentile(np.abs(tilt), 95)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw * yaw))),
        "yaw_final_error_deg": float(yaw[-1]),
    }


def stationary_detector(gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
    candidate = (np.linalg.norm(gyro, axis=1) < 2.5) & (
        np.abs(np.linalg.norm(accel, axis=1) - 1.0) < 0.035
    )
    kernel = np.ones(20, dtype=int)
    stable = np.convolve(candidate.astype(int), kernel, mode="same") >= 16
    return stable


def integrate_position(
    quaternion: np.ndarray,
    accel: np.ndarray,
    dt: float,
    *,
    stationary: np.ndarray | None = None,
    accel_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(accel_bias if accel_bias is not None else np.zeros(3)).copy()
    velocity = np.zeros((len(accel), 3))
    position = np.zeros((len(accel), 3))
    linear_world = np.empty_like(accel)
    for index in range(len(accel)):
        world = quaternion_to_matrix(quaternion[index]) @ (accel[index] - bias)
        linear_world[index] = world - np.asarray([0.0, 0.0, 1.0])
        if stationary is not None and stationary[index]:
            bias += 0.002 * (
                accel[index]
                - quaternion_to_matrix(quaternion[index]).T
                @ np.asarray([0.0, 0.0, 1.0])
                - bias
            )
            linear_world[index] = 0.0
        if index == 0:
            continue
        velocity[index] = velocity[index - 1] + 9.80665 * (
            linear_world[index - 1] + linear_world[index]
        ) * 0.5 * dt
        if stationary is not None and stationary[index]:
            velocity[index] = 0.0
        position[index] = position[index - 1] + (
            velocity[index - 1] + velocity[index]
        ) * 0.5 * dt
    return velocity, position


def main() -> None:
    parser = argparse.ArgumentParser(description="Six-axis IMU drift benchmark.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = generate_trajectory(
        duration=args.duration, sample_rate=args.sample_rate, seed=args.seed
    )
    dt = float(data["dt"])
    calibration_count = round(4.0 * args.sample_rate)
    estimated_gyro_bias = data["gyro"][:calibration_count].mean(axis=0)
    estimated_accel_bias = (
        data["accel"][:calibration_count].mean(axis=0)
        - np.asarray([0.0, 0.0, 1.0])
    )
    filters: dict[str, np.ndarray] = {
        "gyro_raw": integrate_gyroscope(data["gyro"], dt),
        "gyro_calibrated": integrate_gyroscope(
            data["gyro"] - estimated_gyro_bias, dt
        ),
        "mahony": mahony_filter(data["gyro"], data["accel"], dt)[0],
        "madgwick": madgwick_filter(data["gyro"], data["accel"], dt),
    }
    vqf = run_vqf(data["gyro"], data["accel"], dt)
    if vqf is not None:
        filters["vqf"] = vqf
    fusion = run_fusion(data["gyro"], data["accel"], dt)
    if fusion is not None:
        filters["fusion"] = fusion[0]

    metrics = {
        name: orientation_metrics(quaternion, data["euler"])
        for name, quaternion in filters.items()
    }
    position_quaternion = filters.get("vqf", filters["mahony"])
    detected_stationary = stationary_detector(
        data["gyro"] - estimated_gyro_bias, data["accel"]
    )
    _, position_raw = integrate_position(
        filters["gyro_calibrated"],
        data["accel"],
        dt,
        accel_bias=estimated_accel_bias,
    )
    _, position_fused = integrate_position(
        position_quaternion,
        data["accel"],
        dt,
        accel_bias=estimated_accel_bias,
    )
    _, position_zupt = integrate_position(
        position_quaternion,
        data["accel"],
        dt,
        stationary=detected_stationary,
        accel_bias=estimated_accel_bias,
    )
    positions = {
        "raw_double_integration": position_raw,
        "fused_orientation": position_fused,
        "fused_zupt": position_zupt,
    }
    position_metrics = {}
    for name, position in positions.items():
        error = np.linalg.norm(position - data["true_position"], axis=1)
        position_metrics[name] = {
            "final_error_m": float(error[-1]),
            "rmse_m": float(np.sqrt(np.mean(error * error))),
            "max_error_m": float(np.max(error)),
        }

    report = {
        "schema": "six-axis-drift-benchmark/v1",
        "duration_seconds": args.duration,
        "sample_rate_hz": args.sample_rate,
        "known_gyro_bias_dps": data["gyro_bias"].tolist(),
        "initial_estimated_gyro_bias_dps": estimated_gyro_bias.tolist(),
        "known_accel_bias_g": data["accel_bias"].tolist(),
        "initial_estimated_accel_bias_g": estimated_accel_bias.tolist(),
        "stationary_detection": {
            "precision": float(
                np.sum(detected_stationary & data["stationary"])
                / max(1, np.sum(detected_stationary))
            ),
            "recall": float(
                np.sum(detected_stationary & data["stationary"])
                / max(1, np.sum(data["stationary"]))
            ),
        },
        "orientation": metrics,
        "position": position_metrics,
        "yaw_observability": (
            "Six-axis gravity updates constrain roll/pitch but not absolute yaw; "
            "yaw bias remains unobservable without an external heading reference."
        ),
    }
    (args.output / "drift_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, quaternion in filters.items():
        euler = np.rad2deg(quaternion_to_euler(quaternion))
        axes[0].plot(data["time"], euler[:, 0], label=f"{name} roll", alpha=0.8)
    axes[0].plot(
        data["time"],
        np.rad2deg(data["euler"][:, 0]),
        "k--",
        linewidth=2,
        label="truth roll",
    )
    axes[0].set_ylabel("roll (deg)")
    axes[0].legend(ncol=3, fontsize=7)
    for name, position in positions.items():
        error = np.linalg.norm(position - data["true_position"], axis=1)
        axes[1].plot(data["time"], error, label=name)
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_ylabel("position error (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output / "drift_benchmark.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
