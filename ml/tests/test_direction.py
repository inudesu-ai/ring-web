from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.direction import (  # noqa: E402
    DirectionalGestureRecognizer,
    blend_direction_probabilities,
)
from ringml.displacement import DisplacementEstimate  # noqa: E402


def estimate(
    position,
    *,
    segment_id=1,
    moving=True,
) -> DisplacementEstimate:
    return DisplacementEstimate(
        armed=True,
        moving=moving,
        rotating_only=False,
        translation_candidate=False,
        position_m=position,
        velocity_mps=(0.1, 0.0, 0.0) if moving else (0.0, 0.0, 0.0),
        linear_accel_world_g=(0.0, 0.0, 0.0),
        corrected_accel_world_g=(0.0, 0.0, 0.0),
        accel_bias_world_g=(0.0, 0.0, 0.0),
        accel_threshold_g=0.02,
        noise_sigma_g=0.003,
        speed_mps=0.1 if moving else 0.0,
        distance_m=sum(abs(value) for value in position),
        segment_id=segment_id,
        segment_elapsed_s=0.3 if moving else 0.0,
        zupt_count=0 if moving else 1,
        zupt_confidence=0.9,
        confidence=0.9,
        position_correction_m=(0.0, 0.0, 0.0),
    )


class DirectionRecognizerTests(unittest.TestCase):
    def test_cardinal_segments_are_classified(self) -> None:
        cases = [
            ((0.06, 0.005, 0.002), "right"),
            ((-0.06, 0.005, 0.002), "left"),
            ((0.002, 0.004, 0.07), "up"),
            ((0.002, 0.004, -0.07), "down"),
        ]
        for position, expected in cases:
            with self.subTest(expected=expected):
                recognizer = DirectionalGestureRecognizer()
                decision = recognizer.update(
                    estimate(position),
                    timestamp_ms=1000,
                )
                self.assertIsNotNone(decision)
                self.assertEqual(decision.label, expected)
                self.assertGreater(decision.confidence, 0.75)

    def test_diagonal_or_tiny_motion_does_not_override(self) -> None:
        recognizer = DirectionalGestureRecognizer()
        self.assertIsNone(
            recognizer.update(
                estimate((0.006, 0.004, 0.002)),
                timestamp_ms=1000,
            )
        )
        recognizer = DirectionalGestureRecognizer()
        self.assertIsNone(
            recognizer.update(
                estimate((0.04, 0.04, 0.04)),
                timestamp_ms=1000,
            )
        )

    def test_decision_is_held_after_zupt(self) -> None:
        recognizer = DirectionalGestureRecognizer(hold_ms=900)
        moving = recognizer.update(
            estimate((0.06, 0.0, 0.0)),
            timestamp_ms=1000,
        )
        stopped = recognizer.update(
            estimate((0.07, 0.0, 0.0), moving=False),
            timestamp_ms=1200,
        )
        self.assertEqual(moving.label, "right")
        self.assertEqual(stopped.label, "right")

    def test_trajectory_probability_overrides_ambiguous_mlp(self) -> None:
        classes = np.asarray(["left", "right", "up", "down", "wave"])
        probabilities = np.asarray([0.22, 0.24, 0.18, 0.17, 0.19])
        recognizer = DirectionalGestureRecognizer()
        decision = recognizer.update(
            estimate((-0.08, 0.0, 0.0)),
            timestamp_ms=1000,
        )
        fused, source = blend_direction_probabilities(
            classes, probabilities, decision
        )
        self.assertEqual(source, "zupt-direction")
        self.assertEqual(classes[int(np.argmax(fused))], "left")
        self.assertGreater(float(np.max(fused)), 0.75)
        self.assertAlmostEqual(float(np.sum(fused)), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
