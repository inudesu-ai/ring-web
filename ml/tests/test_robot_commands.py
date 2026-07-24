from __future__ import annotations

from pathlib import Path
import sys
import unittest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ringml.robot_commands import RobotCommandGate  # noqa: E402


class RobotCommandGateTests(unittest.TestCase):
    def test_requires_idle_rearm_and_two_confirmations(self) -> None:
        gate = RobotCommandGate(threshold=0.85, rearm_idle_ms=200)
        self.assertFalse(
            gate.update("left", 0.99, timestamp_ms=0).emitted
        )
        gate.update("idle", 0.99, timestamp_ms=100)
        gate.update("idle", 0.99, timestamp_ms=350)
        first = gate.update("left", 0.91, timestamp_ms=600)
        second = gate.update("left", 0.94, timestamp_ms=1000)
        self.assertFalse(first.emitted)
        self.assertTrue(second.emitted)
        self.assertEqual(second.command, "turn_left")
        self.assertFalse(
            gate.update("left", 0.99, timestamp_ms=1500).emitted
        )

    def test_uncertain_prediction_breaks_confirmation(self) -> None:
        gate = RobotCommandGate(rearm_idle_ms=0)
        gate.update("idle", 1.0, timestamp_ms=0)
        gate.update("right", 0.95, timestamp_ms=100)
        gate.update("right", 0.60, timestamp_ms=200)
        decision = gate.update("right", 0.95, timestamp_ms=300)
        self.assertFalse(decision.emitted)
        self.assertEqual(decision.confirmations, 1)

    def test_stop_is_immediate_and_latched(self) -> None:
        gate = RobotCommandGate()
        first = gate.update("double_tap", 0.96, timestamp_ms=100)
        second = gate.update("double_tap", 0.97, timestamp_ms=200)
        self.assertTrue(first.emitted)
        self.assertEqual(first.command, "stop")
        self.assertFalse(second.emitted)


if __name__ == "__main__":
    unittest.main()
