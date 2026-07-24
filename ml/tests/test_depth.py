from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.depth import (  # noqa: E402
    DepthGestureRecognizer,
    augment_depth_probabilities,
)
from ringml.displacement import DisplacementEstimate  # noqa: E402


def estimate(position, *, moving=True, segment_id=1) -> DisplacementEstimate:
    return DisplacementEstimate(
        armed=True,
        moving=moving,
        rotating_only=False,
        translation_candidate=False,
        position_m=position,
        velocity_mps=(0.0, 0.1, 0.0) if moving else (0.0, 0.0, 0.0),
        linear_accel_world_g=(0.0, 0.0, 0.0),
        corrected_accel_world_g=(0.0, 0.0, 0.0),
        accel_bias_world_g=(0.0, 0.0, 0.0),
        accel_threshold_g=0.02,
        noise_sigma_g=0.003,
        speed_mps=0.1 if moving else 0.0,
        distance_m=0.1,
        segment_id=segment_id,
        segment_elapsed_s=0.4,
        zupt_count=0,
        zupt_confidence=0.0,
        confidence=0.9,
        position_correction_m=(0.0, 0.0, 0.0),
    )


class DepthRecognizerTests(unittest.TestCase):
    def test_forward_and_backward_are_recognized(self) -> None:
        for position, expected in (
            ((0.005, 0.09, 0.004), "forward"),
            ((-0.004, -0.09, 0.003), "backward"),
        ):
            recognizer = DepthGestureRecognizer()
            decision = recognizer.update(
                estimate(position),
                timestamp_ms=1000,
            )
            self.assertIsNotNone(decision)
            self.assertEqual(decision.label, expected)
            self.assertGreater(decision.confidence, 0.8)

    def test_lateral_vertical_and_diagonal_motion_are_rejected(self) -> None:
        for position in (
            (0.09, 0.005, 0.0),
            (0.0, 0.01, 0.09),
            (0.07, 0.07, 0.0),
        ):
            recognizer = DepthGestureRecognizer()
            self.assertIsNone(
                recognizer.update(
                    estimate(position),
                    timestamp_ms=1000,
                )
            )

    def test_depth_probability_is_added_without_model_class(self) -> None:
        recognizer = DepthGestureRecognizer()
        decision = recognizer.update(
            estimate((0.0, 0.10, 0.0)),
            timestamp_ms=1000,
        )
        classes = np.asarray(["idle", "left", "right"])
        model = np.asarray([0.2, 0.4, 0.4])
        output = augment_depth_probabilities(classes, model, decision)
        self.assertEqual(max(output, key=output.get), "forward")
        self.assertGreater(output["forward"], 0.8)
        self.assertAlmostEqual(sum(output.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
