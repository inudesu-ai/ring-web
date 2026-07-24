"""Trajectory-aided recognition for left/right/up/down ring gestures."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .displacement import DisplacementEstimate


CARDINAL_GESTURES = frozenset({"left", "right", "up", "down"})


def swap_vertical_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Map the simulated model's vertical convention to the physical ring."""

    labels = np.asarray(classes, dtype=str)
    values = np.asarray(probabilities, dtype=np.float64).copy()
    up = np.flatnonzero(labels == "up")
    down = np.flatnonzero(labels == "down")
    if len(up) and len(down):
        up_index = int(up[0])
        down_index = int(down[0])
        values[up_index], values[down_index] = (
            values[down_index],
            values[up_index],
        )
    return values


def blend_stationary_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    stationary_confidence: float,
) -> tuple[np.ndarray, str]:
    """Expose confirmed ZUPT rest as a first-class gesture state."""

    labels = np.asarray(classes, dtype=str)
    values = np.asarray(probabilities, dtype=np.float64).copy()
    idle = np.flatnonzero(labels == "idle")
    if not len(idle):
        return values, "mlp"

    target = int(idle[0])
    confidence = min(0.995, max(0.90, 0.86 + 0.13 * stationary_confidence))
    other_sum = float(np.sum(values) - values[target])
    if other_sum <= 1e-12:
        values.fill((1.0 - confidence) / max(1, len(values) - 1))
    else:
        scale = (1.0 - confidence) / other_sum
        for index in range(len(values)):
            if index != target:
                values[index] *= scale
    values[target] = confidence
    values /= np.sum(values)
    return values, "zupt-stationary"


@dataclass(frozen=True)
class DirectionDecision:
    label: str
    confidence: float
    displacement_m: tuple[float, float, float]
    segment_id: int


class DirectionalGestureRecognizer:
    """Classify one-way trajectory segments by their dominant world axis.

    Gravity makes up/down observable. Horizontal heading is relative in a
    six-axis system, so left/right use the dominant horizontal axis in the
    startup frame. This avoids the synthetic MLP's left/right phase symmetry
    while keeping circles and rotation-only motion out of the override path.
    """

    def __init__(
        self,
        *,
        minimum_displacement_m: float = 0.012,
        hold_ms: int = 900,
    ) -> None:
        self.minimum_displacement_m = max(0.005, minimum_displacement_m)
        self.hold_ms = max(100, int(hold_ms))
        self.previous_position = (0.0, 0.0, 0.0)
        self.segment_start = (0.0, 0.0, 0.0)
        self.segment_id = 0
        self.was_moving = False
        self.gesture_active = False
        self.quiet_started_ms: int | None = None
        self.last_decision: DirectionDecision | None = None
        self.hold_until_ms = 0

    def reset(self) -> None:
        self.__init__(
            minimum_displacement_m=self.minimum_displacement_m,
            hold_ms=self.hold_ms,
        )

    def _classify(
        self,
        position: tuple[float, float, float],
        segment_id: int,
    ) -> DirectionDecision | None:
        displacement = tuple(
            position[index] - self.segment_start[index] for index in range(3)
        )
        dx, dy, dz = displacement
        horizontal = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < self.minimum_displacement_m:
            return None

        label: str | None = None
        dominant = 0.0
        if abs(dz) >= horizontal * 1.12:
            label = "up" if dz > 0 else "down"
            dominant = abs(dz) / max(distance, 1e-9)
        elif horizontal >= abs(dz) * 1.22:
            horizontal_axis = dx if abs(dx) >= abs(dy) else dy
            plane_dominance = abs(horizontal_axis) / max(abs(dx) + abs(dy), 1e-9)
            if plane_dominance >= 0.60:
                label = "right" if horizontal_axis > 0 else "left"
                dominant = abs(horizontal_axis) / max(distance, 1e-9)

        if label is None or dominant < 0.62:
            return None
        distance_score = min(1.0, distance / 0.08)
        confidence = min(0.97, 0.56 + 0.25 * dominant + 0.18 * distance_score)
        if confidence < 0.72:
            return None
        return DirectionDecision(
            label=label,
            confidence=confidence,
            displacement_m=displacement,
            segment_id=segment_id,
        )

    def update(
        self,
        estimate: DisplacementEstimate,
        *,
        timestamp_ms: int,
    ) -> DirectionDecision | None:
        position = estimate.position_m
        active_sample = bool(
            estimate.moving or estimate.translation_candidate
        )
        confirmed_rest = bool(
            not active_sample
            and estimate.armed
            and estimate.zupt_confidence >= 0.62
        )
        if active_sample:
            self.quiet_started_ms = None
        elif self.gesture_active and self.quiet_started_ms is None:
            self.quiet_started_ms = timestamp_ms
        quiet_elapsed_ms = (
            (timestamp_ms - self.quiet_started_ms) & 0xFFFFFFFF
            if self.quiet_started_ms is not None
            else 0
        )
        pause_complete = bool(
            confirmed_rest
            or (
                self.gesture_active
                and not active_sample
                and quiet_elapsed_ms >= 300
            )
        )

        # One physical rest→move→rest gesture may be split into several
        # integration segments by velocity zero crossings. Keep one anchor and
        # latch the first clear cardinal direction so the braking/rebound tail
        # cannot reverse an otherwise correct up/down decision.
        if active_sample and not self.gesture_active:
            self.gesture_active = True
            self.segment_id = estimate.segment_id
            self.segment_start = self.previous_position
            self.last_decision = None
            self.hold_until_ms = 0

        current: DirectionDecision | None = None
        if self.gesture_active and self.last_decision is None:
            current = self._classify(position, estimate.segment_id)
            if current is not None:
                self.last_decision = current
                self.hold_until_ms = timestamp_ms + self.hold_ms

        if pause_complete:
            if self.gesture_active:
                if self.last_decision is None:
                    self.last_decision = self._classify(
                        position,
                        estimate.segment_id,
                    )
                if self.last_decision is not None:
                    self.hold_until_ms = timestamp_ms + self.hold_ms
                self.gesture_active = False
                self.quiet_started_ms = None
            self.segment_start = position

        self.was_moving = active_sample
        self.previous_position = position
        if self.last_decision is not None and (
            self.gesture_active or timestamp_ms <= self.hold_until_ms
        ):
            return self.last_decision
        return current


def blend_direction_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    decision: DirectionDecision | None,
) -> tuple[np.ndarray, str]:
    """Fuse a reliable trajectory direction into the MLP probabilities."""

    values = np.asarray(probabilities, dtype=np.float64).copy()
    if decision is None:
        return values, "mlp"
    matches = np.flatnonzero(np.asarray(classes, dtype=str) == decision.label)
    if not len(matches):
        return values, "mlp"

    target = int(matches[0])
    target_probability = max(0.76, min(0.97, decision.confidence))
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
    return values, "zupt-direction"
