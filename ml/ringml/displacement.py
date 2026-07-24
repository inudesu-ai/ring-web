"""Short-horizon 3-D IMU displacement with gravity removal and ZUPT.

The tracker is intentionally constrained to rest-motion-rest hand gestures.
It learns the stationary acceleration residual, rejects rotation-only motion,
uses trapezoidal double integration, and forces velocity to zero at a detected
stop. It does not claim absolute or long-duration inertial positioning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
GRAVITY_MPS2 = 9.80665


def norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(value * factor for value in vector)  # type: ignore[return-value]


def dot(left: Vector3, right: Vector3) -> float:
    return sum(left[index] * right[index] for index in range(3))


def unit(vector: Vector3) -> Vector3:
    magnitude = norm(vector)
    return scale(vector, 1.0 / magnitude) if magnitude > 1e-12 else (0.0, 0.0, 0.0)


def rotate_body_to_world(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector with a body-to-world wxyz quaternion."""

    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class DisplacementEstimate:
    armed: bool
    moving: bool
    rotating_only: bool
    translation_candidate: bool
    position_m: Vector3
    velocity_mps: Vector3
    linear_accel_world_g: Vector3
    corrected_accel_world_g: Vector3
    accel_bias_world_g: Vector3
    accel_threshold_g: float
    noise_sigma_g: float
    speed_mps: float
    distance_m: float
    segment_id: int
    segment_elapsed_s: float
    zupt_count: int
    zupt_confidence: float
    confidence: float
    position_correction_m: Vector3

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


class DisplacementTracker:
    """Constrained strapdown integrator for short ring gestures."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        accel_threshold_floor_g: float = 0.020,
        accel_threshold_ceiling_g: float = 0.080,
        noise_multiplier: float = 5.0,
        stationary_bias_tau_s: float = 1.5,
        arm_stationary_s: float = 0.30,
        zupt_confidence_threshold: float = 0.62,
        translation_start_samples: int = 4,
        translation_start_delta_v_mps: float = 0.012,
        translation_quiet_s: float = 0.18,
        velocity_damping_per_s: float = 0.8,
        coast_damping_per_s: float = 4.0,
        max_speed_mps: float = 0.60,
        rotation_gate_dps: float = 12.0,
        rotation_lever_arm_m: float = 0.018,
    ) -> None:
        self.sample_rate_hz = max(1.0, float(sample_rate_hz))
        self.accel_threshold_floor_g = max(0.002, float(accel_threshold_floor_g))
        self.accel_threshold_ceiling_g = max(
            self.accel_threshold_floor_g, float(accel_threshold_ceiling_g)
        )
        self.noise_multiplier = max(1.0, float(noise_multiplier))
        self.stationary_bias_tau_s = max(0.1, float(stationary_bias_tau_s))
        self.arm_samples = max(5, round(arm_stationary_s * self.sample_rate_hz))
        self.zupt_confidence_threshold = min(
            1.0, max(0.0, float(zupt_confidence_threshold))
        )
        self.translation_start_samples = max(2, int(translation_start_samples))
        self.translation_start_delta_v_mps = max(
            0.002, float(translation_start_delta_v_mps)
        )
        self.translation_quiet_samples = max(
            4, round(translation_quiet_s * self.sample_rate_hz)
        )
        self.velocity_damping_per_s = max(0.0, float(velocity_damping_per_s))
        self.coast_damping_per_s = max(0.0, float(coast_damping_per_s))
        self.max_speed_mps = max(0.1, float(max_speed_mps))
        self.rotation_gate_dps = max(1.0, float(rotation_gate_dps))
        self.rotation_lever_arm_m = max(0.0, float(rotation_lever_arm_m))

        self.noise_window: deque[Vector3] = deque(
            maxlen=max(100, round(self.sample_rate_hz * 3.0))
        )
        self.accel_bias_world_g: Vector3 = (0.0, 0.0, 0.0)
        self.noise_sigma_g = 0.0
        self.accel_threshold_g = self.accel_threshold_floor_g
        self.effective_accel_threshold_g = self.accel_threshold_floor_g
        self.stationary_samples = 0
        self.armed = False
        self.zupt_confidence = 0.0
        self.reset_position()

    def reset_position(self) -> None:
        """Clear trajectory while retaining learned stationary calibration."""

        self.position_m: Vector3 = (0.0, 0.0, 0.0)
        self.velocity_mps: Vector3 = (0.0, 0.0, 0.0)
        self.previous_accel_mps2: Vector3 = (0.0, 0.0, 0.0)
        self.distance_m = 0.0
        self.moving = False
        self.rotating_only = False
        self.segment_id = 0
        self.segment_elapsed_s = 0.0
        self.segment_direction: Vector3 = (0.0, 0.0, 0.0)
        self.segment_constraint = "free"
        self.quiet_samples = 0
        self.zupt_count = 0
        self.previous_gyro_rad_s: Vector3 = (0.0, 0.0, 0.0)
        self._reset_candidate()

    def handle_transport_gap(self) -> None:
        self.velocity_mps = (0.0, 0.0, 0.0)
        self.previous_accel_mps2 = (0.0, 0.0, 0.0)
        self.moving = False
        self.rotating_only = False
        self.stationary_samples = 0
        self.armed = False
        self.quiet_samples = 0
        self._reset_candidate()

    def _reset_candidate(self) -> None:
        self.candidate_samples = 0
        self.candidate_elapsed_s = 0.0
        self.candidate_velocity_mps: Vector3 = (0.0, 0.0, 0.0)
        self.candidate_position_m: Vector3 = (0.0, 0.0, 0.0)
        self.candidate_previous_accel_mps2: Vector3 = (0.0, 0.0, 0.0)

    def _finish_segment(self) -> None:
        if self.moving:
            self.zupt_count += 1
        self.velocity_mps = (0.0, 0.0, 0.0)
        self.previous_accel_mps2 = (0.0, 0.0, 0.0)
        self.moving = False
        self.segment_elapsed_s = 0.0
        self.segment_direction = (0.0, 0.0, 0.0)
        self.segment_constraint = "free"
        self.quiet_samples = 0

    def _update_stationary_model(self, linear_world_g: Vector3, dt_s: float) -> None:
        alpha = 1.0 - math.exp(-dt_s / self.stationary_bias_tau_s)
        self.accel_bias_world_g = tuple(
            self.accel_bias_world_g[index]
            + alpha * (linear_world_g[index] - self.accel_bias_world_g[index])
            for index in range(3)
        )  # type: ignore[assignment]
        residual = subtract(linear_world_g, self.accel_bias_world_g)
        self.noise_window.append(residual)
        if len(self.noise_window) < 50:
            return
        sigmas = []
        for axis in range(3):
            values = [sample[axis] for sample in self.noise_window]
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            sigmas.append(max(1.4826 * mad, statistics.pstdev(values)))
        self.noise_sigma_g = max(sigmas)
        self.accel_threshold_g = min(
            self.accel_threshold_ceiling_g,
            max(
                self.accel_threshold_floor_g,
                self.noise_multiplier * self.noise_sigma_g,
            ),
        )

    @staticmethod
    def _soft_deadband(vector: Vector3, threshold: float) -> Vector3:
        magnitude = norm(vector)
        if magnitude <= threshold:
            return (0.0, 0.0, 0.0)
        full_scale = threshold * 1.25
        if magnitude >= full_scale:
            return vector
        phase = (magnitude - threshold) / max(1e-9, full_scale - threshold)
        gain = phase * phase * (3.0 - 2.0 * phase)
        return scale(vector, gain)

    def _constraint_for(self, velocity: Vector3) -> str:
        speed = norm(velocity)
        if speed <= 1e-9:
            return "free"
        vertical_fraction = abs(velocity[2]) / speed
        if vertical_fraction >= 0.82:
            return "vertical"
        if vertical_fraction <= 0.42:
            return "horizontal"
        return "free"

    def _constrain(self, vector: Vector3) -> Vector3:
        if self.segment_constraint == "vertical":
            return (0.0, 0.0, vector[2])
        if self.segment_constraint == "horizontal":
            return (vector[0], vector[1], 0.0)
        return vector

    def _estimate(
        self,
        linear_world_g: Vector3,
        corrected_g: Vector3,
    ) -> DisplacementEstimate:
        confidence = 1.0 if not self.moving else math.exp(-self.segment_elapsed_s / 4.0)
        return DisplacementEstimate(
            armed=self.armed,
            moving=self.moving,
            rotating_only=self.rotating_only,
            translation_candidate=self.candidate_samples > 0,
            position_m=self.position_m,
            velocity_mps=self.velocity_mps,
            linear_accel_world_g=linear_world_g,
            corrected_accel_world_g=corrected_g,
            accel_bias_world_g=self.accel_bias_world_g,
            accel_threshold_g=self.effective_accel_threshold_g,
            noise_sigma_g=self.noise_sigma_g,
            speed_mps=norm(self.velocity_mps),
            distance_m=self.distance_m,
            segment_id=self.segment_id,
            segment_elapsed_s=self.segment_elapsed_s,
            zupt_count=self.zupt_count,
            zupt_confidence=self.zupt_confidence,
            confidence=confidence,
            position_correction_m=(0.0, 0.0, 0.0),
        )

    def update(
        self,
        *,
        dt_s: float,
        accel_body_g: Vector3,
        gyro_body_dps: Vector3,
        quaternion: Quaternion,
        stationary: bool,
        stationary_confidence: float = 1.0,
    ) -> DisplacementEstimate:
        dt_s = min(max(float(dt_s), 0.001), 0.05)
        world_force = rotate_body_to_world(quaternion, accel_body_g)
        linear_world_g: Vector3 = (
            world_force[0],
            world_force[1],
            world_force[2] - 1.0,
        )
        residual_g = subtract(linear_world_g, self.accel_bias_world_g)
        gyro_magnitude = norm(gyro_body_dps)
        self.zupt_confidence = min(1.0, max(0.0, stationary_confidence))
        effective_stationary = bool(
            stationary
            and self.zupt_confidence >= self.zupt_confidence_threshold
            and gyro_magnitude <= 1.2
        )

        if effective_stationary:
            self.rotating_only = False
            self.stationary_samples += 1
            self._update_stationary_model(linear_world_g, dt_s)
            self.effective_accel_threshold_g = self.accel_threshold_g
            self._reset_candidate()
            if not self.armed and self.stationary_samples >= self.arm_samples:
                self.armed = True
                self.reset_position()
            if self.armed:
                self._finish_segment()
                return self._estimate(linear_world_g, (0.0, 0.0, 0.0))
        else:
            self.stationary_samples = 0

        if not self.armed:
            self._reset_candidate()
            return self._estimate(linear_world_g, (0.0, 0.0, 0.0))

        gyro_rad_s: Vector3 = tuple(
            math.radians(value) for value in gyro_body_dps
        )  # type: ignore[assignment]
        angular_accel = norm(
            tuple(
                (gyro_rad_s[index] - self.previous_gyro_rad_s[index]) / dt_s
                for index in range(3)
            )
        )
        self.previous_gyro_rad_s = gyro_rad_s
        rotation_margin = min(
            0.16,
            self.rotation_lever_arm_m
            * (norm(gyro_rad_s) ** 2 + angular_accel)
            / GRAVITY_MPS2,
        )
        self.effective_accel_threshold_g = min(
            0.20, self.accel_threshold_g + rotation_margin
        )
        corrected_g = self._soft_deadband(
            residual_g, self.effective_accel_threshold_g
        )
        rotation_like = bool(
            gyro_magnitude >= self.rotation_gate_dps
            and norm(residual_g) <= self.effective_accel_threshold_g + 0.05
        )
        if rotation_like and not self.moving:
            self.rotating_only = True
            self._reset_candidate()
            return self._estimate(linear_world_g, (0.0, 0.0, 0.0))
        self.rotating_only = False

        if self.moving:
            corrected_g = self._constrain(corrected_g)
        translation_evidence = norm(corrected_g) > 0.0

        if self.moving and not translation_evidence:
            self.quiet_samples += 1
            if self.quiet_samples >= self.translation_quiet_samples:
                self._finish_segment()
                return self._estimate(linear_world_g, (0.0, 0.0, 0.0))
        elif self.moving:
            self.quiet_samples = 0

        acceleration_mps2 = scale(corrected_g, GRAVITY_MPS2)
        if not self.moving:
            if not translation_evidence:
                self._reset_candidate()
                return self._estimate(linear_world_g, (0.0, 0.0, 0.0))
            old_velocity = self.candidate_velocity_mps
            mean_acceleration = scale(
                add(self.candidate_previous_accel_mps2, acceleration_mps2), 0.5
            )
            next_velocity = add(old_velocity, scale(mean_acceleration, dt_s))
            delta = scale(add(old_velocity, next_velocity), 0.5 * dt_s)
            self.candidate_velocity_mps = next_velocity
            self.candidate_position_m = add(self.candidate_position_m, delta)
            self.candidate_previous_accel_mps2 = acceleration_mps2
            self.candidate_elapsed_s += dt_s
            self.candidate_samples += 1
            if (
                self.candidate_samples < self.translation_start_samples
                or norm(next_velocity) < self.translation_start_delta_v_mps
            ):
                return self._estimate(linear_world_g, corrected_g)

            self.moving = True
            self.segment_id += 1
            self.segment_elapsed_s = self.candidate_elapsed_s
            self.segment_constraint = self._constraint_for(next_velocity)
            self.velocity_mps = self._constrain(next_velocity)
            self.segment_direction = unit(self.velocity_mps)
            candidate_position = self._constrain(self.candidate_position_m)
            self.position_m = add(self.position_m, candidate_position)
            self.distance_m += norm(candidate_position)
            self.previous_accel_mps2 = self._constrain(acceleration_mps2)
            self._reset_candidate()
            return self._estimate(linear_world_g, corrected_g)

        acceleration_mps2 = self._constrain(acceleration_mps2)
        old_velocity = self.velocity_mps
        mean_acceleration = scale(
            add(self.previous_accel_mps2, acceleration_mps2), 0.5
        )
        next_velocity = add(old_velocity, scale(mean_acceleration, dt_s))
        damping = self.velocity_damping_per_s
        if not translation_evidence:
            damping += self.coast_damping_per_s
        next_velocity = scale(next_velocity, math.exp(-damping * dt_s))
        speed = norm(next_velocity)
        if speed > self.max_speed_mps:
            next_velocity = scale(next_velocity, self.max_speed_mps / speed)

        forward_before = dot(old_velocity, self.segment_direction)
        forward_after = dot(next_velocity, self.segment_direction)
        braking = dot(acceleration_mps2, self.segment_direction) < 0.0
        if forward_before > 1e-6 and forward_after <= 0.0 and braking:
            denominator = forward_before - forward_after
            fraction = min(1.0, max(0.0, forward_before / max(1e-9, denominator)))
            delta = scale(old_velocity, 0.5 * dt_s * fraction)
            self.position_m = add(self.position_m, delta)
            self.distance_m += norm(delta)
            self._finish_segment()
            return self._estimate(linear_world_g, (0.0, 0.0, 0.0))

        delta = scale(add(old_velocity, next_velocity), 0.5 * dt_s)
        self.position_m = add(self.position_m, delta)
        self.velocity_mps = next_velocity
        self.previous_accel_mps2 = acceleration_mps2
        self.segment_elapsed_s += dt_s
        self.distance_m += norm(delta)
        return self._estimate(linear_world_g, corrected_g)
