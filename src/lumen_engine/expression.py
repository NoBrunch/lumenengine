"""Interpretable musical state and gesture selection.

This is intentionally an authored baseline rather than a black-box model. Every
decision carries a short reason and can later be influenced by stored feedback.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumen_engine.models import (
    ExpressionState,
    Gesture,
    MusicalObservation,
    PerformanceDecision,
    Vec3,
    clamp,
)


def _smooth(previous: float, current: float, attack: float, release: float) -> float:
    amount = attack if current > previous else release
    return previous + amount * (current - previous)


@dataclass(frozen=True, slots=True)
class ExpressionPolicy:
    room_center: Vec3 = Vec3(0.0, 0.0, 1.2)
    room_high: Vec3 = Vec3(0.0, 0.0, 2.4)
    room_wide: Vec3 = Vec3(2.0, 0.0, 1.4)
    minimum_gesture_hold_s: float = 0.8
    release_onset_threshold: float = 0.68
    high_energy_threshold: float = 0.48
    quiet_threshold: float = 0.16


class ExpressionEngine:
    def __init__(self, policy: ExpressionPolicy | None = None) -> None:
        self.policy = policy or ExpressionPolicy()
        self.state = ExpressionState()
        self._last_gesture = Gesture.HOLD
        self._last_target = self.policy.room_center
        self._last_gesture_at = float("-inf")

    def update_state(self, observation: MusicalObservation) -> ExpressionState:
        raw_energy = clamp(
            0.62 * observation.loudness
            + 0.14 * observation.onset_strength
            + 0.08 * observation.low_energy
            + 0.16 * observation.beat_pulse,
            0.0,
            1.0,
        )
        section_tension = 0.22 if observation.section == "build" else 0.0
        raw_tension = clamp(
            0.34 * observation.high_energy
            + 0.28 * observation.novelty
            + 0.18 * observation.onset_strength
            + section_tension,
            0.0,
            1.0,
        )
        raw_motion = clamp(
            0.42 * observation.onset_strength
            + 0.24 * observation.beat_confidence
            + 0.10 * observation.mid_energy
            + 0.24 * observation.beat_pulse,
            0.0,
            1.0,
        )
        raw_intimacy = clamp(
            0.75 - 0.45 * raw_energy - 0.20 * raw_motion, 0.0, 1.0
        )
        raw_confidence = clamp(
            0.45 * observation.beat_confidence
            + 0.35 * observation.section_confidence
            + 0.20 * (1.0 if observation.loudness > 0.02 else 0.0),
            0.0,
            1.0,
        )

        self.state = ExpressionState(
            energy=_smooth(self.state.energy, raw_energy, 0.45, 0.12),
            tension=_smooth(self.state.tension, raw_tension, 0.30, 0.08),
            motion=_smooth(self.state.motion, raw_motion, 0.40, 0.14),
            intimacy=_smooth(self.state.intimacy, raw_intimacy, 0.18, 0.12),
            confidence=_smooth(
                self.state.confidence, raw_confidence, 0.25, 0.10
            ),
        )
        return self.state

    def decide(self, observation: MusicalObservation) -> PerformanceDecision:
        state = self.update_state(observation)
        elapsed = observation.timestamp_s - self._last_gesture_at
        can_change = elapsed >= self.policy.minimum_gesture_hold_s

        gesture = self._last_gesture
        target = self._last_target
        reason = "Maintaining the current visual idea to avoid restless changes."

        is_release = (
            (
                observation.onset_strength
                >= self.policy.release_onset_threshold
                or observation.beat_pulse >= 0.90
            )
            and state.energy >= self.policy.high_energy_threshold
            and (
                observation.beat_confidence >= 0.25
                or observation.beat_pulse >= 0.90
            )
        )
        is_build = observation.section == "build" and state.tension >= 0.42

        if is_release and (can_change or self._last_gesture != Gesture.RELEASE):
            gesture = Gesture.RELEASE
            target = self.policy.room_wide
            reason = (
                "A strong, beat-aligned onset arrived with high accumulated energy."
            )
        elif is_build and can_change:
            gesture = Gesture.CONVERGE
            target = self.policy.room_center
            reason = (
                "The music is building, so the composition narrows while tension rises."
            )
        elif state.energy <= self.policy.quiet_threshold and can_change:
            gesture = Gesture.BREATHE
            target = self.policy.room_high
            reason = "Low energy favors a slow, spacious gesture with visual restraint."
        elif state.motion >= 0.42 and state.energy >= 0.32 and can_change:
            gesture = Gesture.SWEEP
            target = self.policy.room_wide
            reason = "Rhythmic activity is sustained enough to support coordinated motion."
        elif (
            observation.beat_pulse >= 0.48
            or observation.onset_strength >= 0.45
        ) and can_change:
            gesture = Gesture.PULSE
            target = self.policy.room_center
            reason = "A clear onset supports a contained accent without changing the motif."
        elif can_change and self._last_gesture == Gesture.RELEASE:
            gesture = Gesture.EXPAND
            target = self.policy.room_high
            reason = "The release resolves into a wider, sustained composition."

        if gesture != self._last_gesture:
            self._last_gesture = gesture
            self._last_gesture_at = observation.timestamp_s
        self._last_target = target

        brightness = clamp(
            0.09 + 0.78 * state.energy + 0.24 * observation.beat_pulse,
            0.0,
            1.0,
        )
        confidence = clamp(
            0.55 * state.confidence
            + 0.25 * observation.beat_confidence
            + 0.20 * max(observation.section_confidence, 0.3),
            0.0,
            1.0,
        )
        return PerformanceDecision(
            timestamp_s=observation.timestamp_s,
            gesture=gesture,
            expression=state,
            target=target,
            brightness=brightness,
            reason=reason,
            confidence=confidence,
        )

    def request_fresh_gesture(self) -> None:
        """Allow the next observation to reconsider the current visual idea."""

        self._last_gesture_at = float("-inf")
