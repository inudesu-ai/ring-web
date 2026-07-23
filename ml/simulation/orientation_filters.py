"""Dependency-free six-axis orientation filters used by the benchmark."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


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


def quaternion_from_euler(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler).T
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.column_stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(quaternion)
    w, x, y, z = values.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    result = np.column_stack([roll, pitch, yaw])
    return result[0] if np.asarray(quaternion).ndim == 1 else result


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def gravity_in_sensor(quaternion: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(quaternion).T @ np.asarray([0.0, 0.0, 1.0])


def integrate_gyroscope(
    gyro_dps: np.ndarray, dt: float, initial: np.ndarray | None = None
) -> np.ndarray:
    quaternion = normalize_quaternion(
        np.asarray(initial if initial is not None else [1.0, 0.0, 0.0, 0.0])
    )
    output = np.empty((len(gyro_dps), 4))
    for index, measurement in enumerate(gyro_dps):
        omega = np.deg2rad(measurement)
        derivative = 0.5 * quaternion_multiply(
            quaternion, np.r_[0.0, omega]
        )
        quaternion = normalize_quaternion(quaternion + derivative * dt)
        output[index] = quaternion
    return output


def mahony_filter(
    gyro_dps: np.ndarray,
    accel_g: np.ndarray,
    dt: float,
    *,
    kp: float = 1.0,
    ki: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    integral = np.zeros(3)
    output = np.empty((len(gyro_dps), 4))
    bias = np.empty((len(gyro_dps), 3))
    for index, (gyro, accel) in enumerate(zip(gyro_dps, accel_g)):
        norm = np.linalg.norm(accel)
        corrected = np.deg2rad(gyro)
        if norm > 1e-9:
            measured = accel / norm
            estimated = gravity_in_sensor(quaternion)
            error = np.cross(measured, estimated)
            integral += ki * error * dt
            corrected += kp * error + integral
        derivative = 0.5 * quaternion_multiply(
            quaternion, np.r_[0.0, corrected]
        )
        quaternion = normalize_quaternion(quaternion + derivative * dt)
        output[index] = quaternion
        bias[index] = -np.rad2deg(integral)
    return output, bias


def madgwick_filter(
    gyro_dps: np.ndarray,
    accel_g: np.ndarray,
    dt: float,
    *,
    beta: float = 0.07,
) -> np.ndarray:
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    output = np.empty((len(gyro_dps), 4))
    for index, (gyro, accel) in enumerate(zip(gyro_dps, accel_g)):
        q0, q1, q2, q3 = quaternion
        gx, gy, gz = np.deg2rad(gyro)
        derivative = 0.5 * quaternion_multiply(
            quaternion, np.asarray([0.0, gx, gy, gz])
        )
        norm = np.linalg.norm(accel)
        if norm > 1e-9:
            ax, ay, az = accel / norm
            gradient = np.asarray(
                [
                    4 * q0 * q2 * q2 + 2 * q2 * ax
                    + 4 * q0 * q1 * q1 - 2 * q1 * ay,
                    4 * q1 * q3 * q3 - 2 * q3 * ax
                    + 4 * q0 * q0 * q1 - 2 * q0 * ay
                    - 4 * q1 + 8 * q1 * q1 * q1 + 8 * q1 * q2 * q2
                    + 4 * q1 * az,
                    4 * q0 * q0 * q2 + 2 * q0 * ax
                    + 4 * q2 * q3 * q3 - 2 * q3 * ay
                    - 4 * q2 + 8 * q2 * q1 * q1 + 8 * q2 * q2 * q2
                    + 4 * q2 * az,
                    4 * q1 * q1 * q3 - 2 * q1 * ax
                    + 4 * q2 * q2 * q3 - 2 * q2 * ay,
                ]
            )
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1e-12:
                derivative -= beta * gradient / gradient_norm
        quaternion = normalize_quaternion(quaternion + derivative * dt)
        output[index] = quaternion
    return output
