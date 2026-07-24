"""Trajectory-aided forward/backward translation recognition."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .displacement import DisplacementEstimate


@dataclass(frozen=True)
class DepthDecision:
    label: str
    confidence: float
    displacement_m: tuple[float, float, float]
    segment_id: int


class DepthGestureRecognizer:
    """Recognize depth-dominant translation without changing cardinal logic."""

    def __init__(
        self,
        *,
        minimum_displacement_m: float = 0.015,
        hold_ms: int = 900,
    ) -> None:
        self.minimum_displacement_m = max(0.006, float(minimum_displacement_m))
        self.hold_ms = max(100, int(hold_ms))
        self.previous_position = (0.0, 0.0, 0.0)
        self.segment_start = (0.0, 0.0, 0.0)
        self.segment_id = 0
        self.was_moving = False
        self.last_decision: DepthDecision | None = None
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
    ) -> DepthDecision | None:
        displacement = tuple(
            position[index] - self.segment_start[index] for index in range(3)
        )
        dx, dy, dz = displacement
        horizontal = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < self.minimum_displacement_m:
            return None
        depth_dominance = abs(dy) / max(horizontal, 1e-9)
        if (
            depth_dominance < 0.78
            or horizontal < abs(dz) * 1.22
            or abs(dy) < abs(dx) * 1.25
        ):
            return None
        distance_score = min(1.0, distance / 0.10)
        confidence = min(
            0.97,
            0.58 + 0.25 * depth_dominance + 0.15 * distance_score,
        )
        if confidence < 0.76:
            return None
        return DepthDecision(
            label="forward" if dy > 0 else "backward",
            confidence=confidence,
            displacement_m=displacement,
            segment_id=segment_id,
        )

    def update(
        self,
        estimate: DisplacementEstimate,
        *,
        timestamp_ms: int,
    ) -> DepthDecision | None:
        position = estimate.position_m
        if estimate.segment_id != self.segment_id and estimate.segment_id > 0:
            self.segment_id = estimate.segment_id
            self.segment_start = self.previous_position
            self.last_decision = None
            self.hold_until_ms = 0
        current = (
            self._classify(position, estimate.segment_id)
            if estimate.moving and estimate.segment_id > 0
            else None
        )
        if current is not None:
            self.last_decision = current
            self.hold_until_ms = timestamp_ms + self.hold_ms
        if self.was_moving and not estimate.moving and estimate.segment_id > 0:
            final = self._classify(position, estimate.segment_id)
            if final is not None:
                self.last_decision = final
                self.hold_until_ms = timestamp_ms + self.hold_ms
        self.was_moving = estimate.moving
        self.previous_position = position
        if self.last_decision is not None and timestamp_ms <= self.hold_until_ms:
            return self.last_decision
        return current


def augment_depth_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    decision: DepthDecision,
) -> dict[str, float]:
    """Add a physical-only depth class to the model probability mapping."""

    target_probability = max(0.80, min(0.97, decision.confidence))
    scale = 1.0 - target_probability
    output = {
        str(label): float(value) * scale
        for label, value in zip(classes, probabilities)
    }
    output[decision.label] = target_probability
    return output
