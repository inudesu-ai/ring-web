from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.circle import (  # noqa: E402
    CircleGestureRecognizer,
    blend_circle_probabilities,
)
from ringml.displacement import DisplacementEstimate  # noqa: E402


def estimate(position, *, moving=True, candidate=False) -> DisplacementEstimate:
    return DisplacementEstimate(
        armed=True,
        moving=moving,
        rotating_only=False,
        translation_candidate=candidate,
        position_m=tuple(float(value) for value in position),
        velocity_mps=(0.1, 0.0, 0.0) if moving else (0.0, 0.0, 0.0),
        linear_accel_world_g=(0.0, 0.0, 0.0),
        corrected_accel_world_g=(0.0, 0.0, 0.0),
        accel_bias_world_g=(0.0, 0.0, 0.0),
        accel_threshold_g=0.02,
        noise_sigma_g=0.003,
        speed_mps=0.1 if moving else 0.0,
        distance_m=0.0,
        segment_id=1,
        segment_elapsed_s=0.5,
        zupt_count=0,
        zupt_confidence=0.0 if moving else 0.9,
        confidence=0.9,
        position_correction_m=(0.0, 0.0, 0.0),
    )


def tilted_circle(count=100, radius=0.06, clockwise=False):
    first = np.asarray([1.0, 1.0, 0.2])
    first /= np.linalg.norm(first)
    second = np.asarray([-0.4, 0.2, 1.0])
    second -= np.dot(second, first) * first
    second /= np.linalg.norm(second)
    center = np.asarray([0.2, -0.1, 0.04])
    direction = -1.0 if clockwise else 1.0
    return [
        center
        + radius
        * (
            math.cos(direction * angle) * first
            + math.sin(direction * angle) * second
        )
        for angle in np.linspace(0.0, 2.0 * math.pi, count)
    ]


class CircleRecognizerTests(unittest.TestCase):
    def test_tilted_circle_is_detected_in_both_directions(self) -> None:
        for clockwise in (False, True):
            with self.subTest(clockwise=clockwise):
                points = tilted_circle(clockwise=clockwise)
                recognizer = CircleGestureRecognizer()
                recognizer.update(
                    estimate(points[0], moving=False),
                    timestamp_ms=0,
                )
                decision = None
                for index, point in enumerate(points[1:], start=1):
                    decision = recognizer.update(
                        estimate(point),
                        timestamp_ms=index * 20,
                    ) or decision
                self.assertIsNotNone(decision)
                self.assertEqual(decision.label, "circle")
                self.assertGreater(decision.confidence, 0.8)
                self.assertGreater(decision.turn_radians, 5.5)
                self.assertLess(decision.closure_ratio, 0.2)

    def test_line_and_partial_arc_are_rejected(self) -> None:
        paths = [
            [np.asarray([index * 0.004, 0.0, 0.0]) for index in range(50)],
            tilted_circle(count=45)[:30],
        ]
        for path in paths:
            recognizer = CircleGestureRecognizer()
            recognizer.update(
                estimate(path[0], moving=False),
                timestamp_ms=0,
            )
            decisions = []
            for index, point in enumerate(path[1:], start=1):
                decisions.append(
                    recognizer.update(
                        estimate(point),
                        timestamp_ms=index * 20,
                    )
                )
            self.assertTrue(all(decision is None for decision in decisions))

    def test_circle_probability_overrides_model(self) -> None:
        points = tilted_circle()
        recognizer = CircleGestureRecognizer()
        recognizer.update(estimate(points[0], moving=False), timestamp_ms=0)
        decision = None
        for index, point in enumerate(points[1:], start=1):
            decision = recognizer.update(
                estimate(point),
                timestamp_ms=index * 20,
            ) or decision
        classes = np.asarray(["circle", "idle", "left", "right"])
        model = np.asarray([0.1, 0.1, 0.4, 0.4])
        fused, source = blend_circle_probabilities(classes, model, decision)
        self.assertEqual(source, "zupt-circle")
        self.assertEqual(classes[int(np.argmax(fused))], "circle")
        self.assertGreater(fused[0], 0.8)
        self.assertAlmostEqual(float(np.sum(fused)), 1.0)


if __name__ == "__main__":
    unittest.main()
