"""Temporal command gate between gesture recognition and a mobile robot."""

from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_COMMAND_MAP = {
    "left": "turn_left",
    "right": "turn_right",
    "up": "stand",
    "down": "sit",
    "forward": "move_forward",
    "backward": "move_backward",
    "rotate_front": "speed_up",
    "rotate_back": "slow_down",
    "circle": "spin",
    "wave": "greet",
    "double_tap": "stop",
}


@dataclass(frozen=True)
class RobotCommandDecision:
    command: str | None
    emitted: bool
    reason: str
    armed: bool
    confirmations: int
    source_gesture: str
    confidence: float

    def as_payload(self) -> dict:
        return asdict(self)


class RobotCommandGate:
    """Reject uncertain/repeated gestures and emit one command per gesture.

    Normal commands require two consecutive confident windows and a preceding
    idle period.  STOP is allowed immediately and has priority over the latch.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        confirmations_required: int = 2,
        rearm_idle_ms: int = 300,
        cooldown_ms: int = 700,
        command_map: dict[str, str] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if confirmations_required < 1:
            raise ValueError("confirmations_required must be positive")
        self.threshold = float(threshold)
        self.confirmations_required = int(confirmations_required)
        self.rearm_idle_ms = int(rearm_idle_ms)
        self.cooldown_ms = int(cooldown_ms)
        self.command_map = dict(command_map or DEFAULT_COMMAND_MAP)
        self.reset()

    def reset(self) -> None:
        self.armed = False
        self.idle_since_ms: int | None = None
        self.candidate: str | None = None
        self.confirmations = 0
        self.last_emitted_ms: int | None = None
        self.stop_latched = False

    def update(
        self,
        gesture: str,
        confidence: float,
        *,
        timestamp_ms: int,
    ) -> RobotCommandDecision:
        gesture = str(gesture)
        confidence = float(confidence)
        if gesture == "idle" and confidence >= self.threshold:
            if self.idle_since_ms is None:
                self.idle_since_ms = timestamp_ms
            if timestamp_ms - self.idle_since_ms >= self.rearm_idle_ms:
                self.armed = True
                self.stop_latched = False
            self.candidate = None
            self.confirmations = 0
            return self._decision(None, False, "idle-rearm", gesture, confidence)

        self.idle_since_ms = None
        command = self.command_map.get(gesture)
        if confidence < self.threshold or command is None:
            self.candidate = None
            self.confirmations = 0
            return self._decision(None, False, "rejected", gesture, confidence)

        if command == "stop":
            if self.stop_latched:
                return self._decision(None, False, "stop-latched", gesture, confidence)
            self.stop_latched = True
            self.armed = False
            self.last_emitted_ms = timestamp_ms
            return self._decision(command, True, "priority-stop", gesture, confidence)

        if not self.armed:
            self.candidate = None
            self.confirmations = 0
            return self._decision(None, False, "not-armed", gesture, confidence)
        if (
            self.last_emitted_ms is not None
            and timestamp_ms - self.last_emitted_ms < self.cooldown_ms
        ):
            return self._decision(None, False, "cooldown", gesture, confidence)

        if gesture == self.candidate:
            self.confirmations += 1
        else:
            self.candidate = gesture
            self.confirmations = 1
        if self.confirmations < self.confirmations_required:
            return self._decision(None, False, "confirming", gesture, confidence)

        self.armed = False
        self.last_emitted_ms = timestamp_ms
        self.candidate = None
        confirmations = self.confirmations
        self.confirmations = 0
        return RobotCommandDecision(
            command=command,
            emitted=True,
            reason="confirmed",
            armed=self.armed,
            confirmations=confirmations,
            source_gesture=gesture,
            confidence=confidence,
        )

    def _decision(
        self,
        command: str | None,
        emitted: bool,
        reason: str,
        gesture: str,
        confidence: float,
    ) -> RobotCommandDecision:
        return RobotCommandDecision(
            command=command,
            emitted=emitted,
            reason=reason,
            armed=self.armed,
            confirmations=self.confirmations,
            source_gesture=gesture,
            confidence=confidence,
        )
