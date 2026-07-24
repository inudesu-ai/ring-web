"""Geometry-based circle recognition for short 3-D ring trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .displacement import DisplacementEstimate


@dataclass(frozen=True)
class CircleDecision:
    label: str
    confidence: float
    path_length_m: float
    radius_m: float
    turn_radians: float
    closure_ratio: float
    plane_score: float
    roundness: float

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


class CircleGestureRecognizer:
    """Detect a closed, coherent loop after projection onto its best-fit plane.

    PCA makes the detector independent of whether the user draws the circle
    horizontally, vertically, or on a tilted plane. The angular sweep,
    closure, roundness, radial consistency, and planar energy jointly reject
    straight and backtracking movements.
    """

    def __init__(
        self,
        *,
        minimum_path_m: float = 0.08,
        minimum_radius_m: float = 0.010,
        quiet_finish_ms: int = 320,
        hold_ms: int = 1400,
    ) -> None:
        self.minimum_path_m = max(0.03, float(minimum_path_m))
        self.minimum_radius_m = max(0.005, float(minimum_radius_m))
        self.quiet_finish_ms = max(150, int(quiet_finish_ms))
        self.hold_ms = max(500, int(hold_ms))
        self.previous_position = np.zeros(3, dtype=np.float64)
        self.points: list[np.ndarray] = []
        self.active = False
        self.last_motion_ms: int | None = None
        self.last_point_ms: int | None = None
        self.last_decision: CircleDecision | None = None
        self.hold_until_ms = 0

    def reset(self) -> None:
        self.__init__(
            minimum_path_m=self.minimum_path_m,
            minimum_radius_m=self.minimum_radius_m,
            quiet_finish_ms=self.quiet_finish_ms,
            hold_ms=self.hold_ms,
        )

    @staticmethod
    def _elapsed_ms(current: int, previous: int | None) -> int:
        if previous is None:
            return 0
        return (int(current) - int(previous)) & 0xFFFFFFFF

    def _append_point(self, position: np.ndarray, timestamp_ms: int) -> None:
        if not self.points:
            self.points.append(position.copy())
            self.last_point_ms = timestamp_ms
            return
        spacing = float(np.linalg.norm(position - self.points[-1]))
        elapsed = self._elapsed_ms(timestamp_ms, self.last_point_ms)
        if spacing >= 0.002 or elapsed >= 80:
            self.points.append(position.copy())
            self.last_point_ms = timestamp_ms
            if len(self.points) > 500:
                self.points = self.points[-500:]

    def _classify(self) -> CircleDecision | None:
        if len(self.points) < 14:
            return None
        points = np.asarray(self.points, dtype=np.float64)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        path_length = float(np.sum(steps))
        if path_length < self.minimum_path_m:
            return None

        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        basis = eigenvectors[:, order[:2]]
        total_energy = float(np.sum(eigenvalues))
        if total_energy <= 1e-12 or eigenvalues[0] <= 1e-12:
            return None

        plane_score = float((eigenvalues[0] + eigenvalues[1]) / total_energy)
        roundness = float(eigenvalues[1] / eigenvalues[0])
        projected = centered @ basis
        radii = np.linalg.norm(projected, axis=1)
        radius = float(np.median(radii))
        if radius < self.minimum_radius_m:
            return None
        radial_mad = float(np.median(np.abs(radii - radius)))
        radial_variation = 1.4826 * radial_mad / max(radius, 1e-9)

        angles = np.arctan2(projected[:, 1], projected[:, 0])
        angle_steps = np.arctan2(
            np.sin(np.diff(angles)),
            np.cos(np.diff(angles)),
        )
        projected_steps = np.linalg.norm(np.diff(projected, axis=0), axis=1)
        useful = projected_steps >= max(0.001, radius * 0.025)
        angle_steps = angle_steps[useful]
        if len(angle_steps) < 10:
            return None
        signed_turn = float(abs(np.sum(angle_steps)))
        absolute_turn = float(np.sum(np.abs(angle_steps)))
        angular_coherence = signed_turn / max(absolute_turn, 1e-9)
        closure_ratio = float(
            np.linalg.norm(projected[-1] - projected[0])
            / max(2.0 * radius, 1e-9)
        )

        if (
            signed_turn < 4.35
            or angular_coherence < 0.58
            or closure_ratio > 1.15
            or plane_score < 0.80
            or roundness < 0.16
            or radial_variation > 0.80
        ):
            return None

        turn_score = float(np.clip((signed_turn - 4.0) / 2.2, 0.0, 1.0))
        coherence_score = float(
            np.clip((angular_coherence - 0.50) / 0.50, 0.0, 1.0)
        )
        closure_score = float(np.clip(1.0 - closure_ratio / 1.15, 0.0, 1.0))
        plane_quality = float(np.clip((plane_score - 0.75) / 0.25, 0.0, 1.0))
        roundness_score = float(np.clip(roundness / 0.65, 0.0, 1.0))
        radial_score = float(np.clip(1.0 - radial_variation / 0.80, 0.0, 1.0))
        confidence = (
            0.25 * turn_score
            + 0.20 * coherence_score
            + 0.18 * closure_score
            + 0.12 * plane_quality
            + 0.15 * roundness_score
            + 0.10 * radial_score
        )
        if confidence < 0.72:
            return None
        return CircleDecision(
            label="circle",
            confidence=min(0.97, confidence),
            path_length_m=path_length,
            radius_m=radius,
            turn_radians=signed_turn,
            closure_ratio=closure_ratio,
            plane_score=plane_score,
            roundness=roundness,
        )

    def update(
        self,
        estimate: DisplacementEstimate,
        *,
        timestamp_ms: int,
    ) -> CircleDecision | None:
        timestamp_ms = int(timestamp_ms)
        position = np.asarray(estimate.position_m, dtype=np.float64)
        active_sample = bool(
            estimate.moving or estimate.translation_candidate
        )

        if active_sample:
            if not self.active:
                self.active = True
                self.points = [self.previous_position.copy()]
                self.last_point_ms = timestamp_ms
                self.last_decision = None
                self.hold_until_ms = 0
            self.last_motion_ms = timestamp_ms
            self._append_point(position, timestamp_ms)
            decision = self._classify()
            if decision is not None and (
                self.last_decision is None
                or decision.confidence >= self.last_decision.confidence
            ):
                self.last_decision = decision
                self.hold_until_ms = timestamp_ms + self.hold_ms
        elif self.active:
            self._append_point(position, timestamp_ms)
            if self._elapsed_ms(timestamp_ms, self.last_motion_ms) >= self.quiet_finish_ms:
                if self.last_decision is None:
                    self.last_decision = self._classify()
                if self.last_decision is not None:
                    self.hold_until_ms = timestamp_ms + self.hold_ms
                self.active = False
                self.points = []

        self.previous_position = position
        if self.last_decision is not None and (
            self.active or timestamp_ms <= self.hold_until_ms
        ):
            return self.last_decision
        return None


def blend_circle_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    decision: CircleDecision | None,
) -> tuple[np.ndarray, str]:
    """Override model output only for a geometrically verified circle."""

    values = np.asarray(probabilities, dtype=np.float64).copy()
    if decision is None:
        return values, "mlp"
    matches = np.flatnonzero(np.asarray(classes, dtype=str) == "circle")
    if not len(matches):
        return values, "mlp"
    target = int(matches[0])
    target_probability = max(0.82, min(0.97, decision.confidence))
    other_sum = float(np.sum(values) - values[target])
    if other_sum <= 1e-12:
        values.fill((1.0 - target_probability) / max(1, len(values) - 1))
    else:
        scale = (1.0 - target_probability) / other_sum
        for index in range(len(values)):
            if index != target:
                values[index] *= scale
    values[target] = target_probability
    values /= np.sum(values)
    return values, "zupt-circle"
