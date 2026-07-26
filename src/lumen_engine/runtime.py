"""Connect perception, expression, spatial targeting, and DMX realization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from lumen_engine.dmx import (
    DMXFrame,
    DMXOutput,
    apply_moving_head_solution,
)
from lumen_engine.expression import ExpressionEngine
from lumen_engine.fixture_output import (
    apply_auxiliary_fixture,
    apply_moving_head_profile,
)
from lumen_engine.models import (
    FixturePatch,
    Gesture,
    MusicalObservation,
    PerformanceDecision,
    ProfileFixturePatch,
    Vec3,
    clamp,
)
from lumen_engine.spatial import (
    SpatialTargetingEngine,
    TargetingSolution,
    UnreachableTargetError,
)


@dataclass(frozen=True, slots=True)
class RuntimeFrame:
    decision: PerformanceDecision
    solutions: tuple[TargetingSolution, ...]
    dmx: DMXFrame
    warnings: tuple[str, ...]


class PerformanceRuntime:
    def __init__(
        self,
        fixtures: tuple[FixturePatch, ...],
        output: DMXOutput,
        auxiliary_fixtures: tuple[ProfileFixturePatch, ...] = (),
        expression: ExpressionEngine | None = None,
        targeting: SpatialTargetingEngine | None = None,
        motion_extents: Vec3 = Vec3(1.0, 3.0, 2.5),
    ) -> None:
        self.fixtures = fixtures
        self.output = output
        self.auxiliary_fixtures = auxiliary_fixtures
        self.expression = expression or ExpressionEngine()
        self.targeting = targeting or SpatialTargetingEngine()
        self.motion_extents = motion_extents
        self._previous: dict[str, tuple[float, float]] = {}
        self._last_timestamp_s: float | None = None
        self._audio_quiet_since_s: float | None = None
        self._audio_idle_amount = 0.0
        self._feedback_motion: dict[str, float] = {}
        self._feedback_intensity: dict[str, float] = {}
        self._motion_phase = 0.0
        self._motion_clock_s: float | None = None
        self._calibration_overrides: dict[str, dict[str, float]] = {}

    def set_calibration_override(
        self, fixture_id: str, *, active: bool, pan_dmx: float = 0.0,
        tilt_dmx: float = 0.0, speed: float = 192.0,
    ) -> None:
        if active:
            self._calibration_overrides[fixture_id] = {
                "pan_dmx": clamp(pan_dmx, 0.0, 65535.0),
                "tilt_dmx": clamp(tilt_dmx, 0.0, 65535.0),
                "speed": clamp(speed, 0.0, 255.0),
            }
        else:
            self._calibration_overrides.pop(fixture_id, None)

    def apply_feedback(
        self,
        *,
        scope: str,
        fixture_id: str | None,
        motion_delta: float = 0.0,
        intensity_delta: float = 0.0,
    ) -> None:
        key = fixture_id if scope == "fixture" and fixture_id else "overall"
        self._feedback_motion[key] = clamp(
            self._feedback_motion.get(key, 0.0) + motion_delta,
            -1.0,
            1.0,
        )
        self._feedback_intensity[key] = clamp(
            self._feedback_intensity.get(key, 0.0) + intensity_delta,
            -1.0,
            1.0,
        )

    def _feedback_for(self, fixture_id: str) -> tuple[float, float]:
        return (
            self._feedback_motion.get("overall", 0.0)
            + self._feedback_motion.get(fixture_id, 0.0),
            self._feedback_intensity.get("overall", 0.0)
            + self._feedback_intensity.get(fixture_id, 0.0),
        )

    def step(self, observation: MusicalObservation) -> RuntimeFrame:
        decision = self.expression.decide(observation)
        elapsed = (
            None
            if self._last_timestamp_s is None
            else max(0.0, observation.timestamp_s - self._last_timestamp_s)
        )
        idle_amount = self._update_audio_idle(observation)
        output_decision = replace(
            decision,
            brightness=decision.brightness
            + (24.0 / 255.0 - decision.brightness) * idle_amount,
        )
        frame = DMXFrame()
        solutions: list[TargetingSolution] = []
        warnings: list[str] = []

        for index, fixture in enumerate(self.fixtures):
            target = self._target_for_fixture(
                decision,
                fixture,
                index,
                observation,
            )
            previous = self._previous.get(fixture.fixture_id)
            motion_feedback, intensity_feedback = self._feedback_for(
                fixture.fixture_id
            )
            fixture_decision = replace(
                output_decision,
                brightness=clamp(
                    output_decision.brightness + intensity_feedback * 0.30,
                    0.0,
                    1.0,
                ),
            )
            try:
                calibration_override = self._calibration_overrides.get(fixture.fixture_id)
                if calibration_override is not None:
                    calibration = fixture.calibration
                    pan_norm = calibration_override["pan_dmx"] / 65535.0
                    tilt_norm = calibration_override["tilt_dmx"] / 65535.0
                    pan = calibration.pan_min_deg + pan_norm * (calibration.pan_max_deg - calibration.pan_min_deg)
                    tilt = calibration.tilt_min_deg + tilt_norm * (calibration.tilt_max_deg - calibration.tilt_min_deg)
                    direction = self.targeting.direction_for_angles(fixture, pan, tilt)
                    solution = TargetingSolution(
                        fixture_id=fixture.fixture_id,
                        target=fixture.position_m + direction * 5.0,
                        pan_deg=pan, tilt_deg=tilt, distance_m=5.0,
                        movement_cost_deg=0.0, aim_error_deg=0.0,
                        branch="calibration-jog",
                    )
                elif observation.loudness < 0.02 and previous is not None:
                    direction = self.targeting.direction_for_angles(
                        fixture, previous[0], previous[1]
                    )
                    solution = TargetingSolution(
                        fixture_id=fixture.fixture_id,
                        target=fixture.position_m + direction * 5.0,
                        pan_deg=previous[0],
                        tilt_deg=previous[1],
                        distance_m=5.0,
                        movement_cost_deg=0.0,
                        aim_error_deg=0.0,
                        branch="quiet-hold",
                    )
                else:
                    solution = self.targeting.solve(
                        fixture,
                        target,
                        previous_pan_deg=None
                        if previous is None
                        else previous[0],
                        previous_tilt_deg=None
                        if previous is None
                        else previous[1],
                    )
                if observation.loudness >= 0.02:
                    solution = self._performance_solution(
                        fixture,
                        index,
                        observation,
                        decision,
                        solution,
                        motion_feedback,
                    )
                if previous is not None and elapsed is not None:
                    solution = self._rate_limit(fixture, solution, previous, elapsed)
                self._previous[fixture.fixture_id] = (
                    solution.pan_deg,
                    solution.tilt_deg,
                )
                solutions.append(solution)
                apply_moving_head_solution(
                    frame, fixture, solution, fixture_decision.brightness
                )
                apply_moving_head_profile(
                    frame,
                    fixture,
                    fixture_decision,
                    observation,
                    idle_amount=idle_amount,
                )
                if calibration_override is not None:
                    # The profile's speed channel is intentionally overridden
                    # only while the operator is in calibration mode.
                    speed_channel = fixture.address + 5 - 1
                    frame.set_channel(fixture.universe, speed_channel, round(calibration_override["speed"]))
            except UnreachableTargetError as error:
                warnings.append(str(error))

        for fixture in self.auxiliary_fixtures:
            motion_feedback, intensity_feedback = self._feedback_for(
                fixture.fixture_id
            )
            fixture_decision = replace(
                output_decision,
                brightness=clamp(
                    output_decision.brightness + intensity_feedback * 0.30,
                    0.0,
                    1.0,
                ),
            )
            apply_auxiliary_fixture(
                frame,
                fixture,
                fixture_decision,
                observation,
                idle_amount=idle_amount,
                motion_feedback=motion_feedback,
            )

        self.output.send(frame)
        self._last_timestamp_s = observation.timestamp_s
        return RuntimeFrame(
            decision=decision,
            solutions=tuple(solutions),
            dmx=frame,
            warnings=tuple(warnings),
        )

    def _update_audio_idle(self, observation: MusicalObservation) -> float:
        """Fade active effects into Party Parrot's quiet/rest state."""

        if observation.loudness >= 0.02:
            self._audio_quiet_since_s = None
            self._audio_idle_amount = 0.0
            return 0.0
        if self._audio_quiet_since_s is None:
            self._audio_quiet_since_s = observation.timestamp_s
        quiet_for = max(0.0, observation.timestamp_s - self._audio_quiet_since_s)
        # Movers hold immediately; the center effect reaches its parked state
        # on the same short musical silence window instead of continuing its
        # fast motor program for several seconds.
        self._audio_idle_amount = clamp((quiet_for - 0.8) / 1.0, 0.0, 1.0)
        return self._audio_idle_amount

    def close(self) -> None:
        self.output.close()

    def _target_for_fixture(
        self,
        decision: PerformanceDecision,
        fixture: FixturePatch,
        index: int,
        observation: MusicalObservation,
    ) -> Vec3:
        target = decision.target
        phase = decision.timestamp_s
        if observation.loudness < 0.02:
            return Vec3(
                target.x,
                target.y,
                target.z + 0.25 * math.sin(phase * 0.55 + index * 0.4),
            )

        state = decision.expression
        activity = clamp(
            0.28 + 0.48 * state.energy + 0.42 * state.motion,
            0.0,
            1.0,
        )
        tempo_locked = (
            observation.bpm is not None
            and observation.beat_confidence >= 0.18
        )
        phase = (
            observation.bar_phase * math.tau
            if tempo_locked
            else decision.timestamp_s * (0.42 + 0.95 * state.motion)
        )
        extents = self.motion_extents
        pair_phase = phase + index * math.pi
        x = (
            extents.x
            * (0.35 + 0.65 * activity)
            * math.sin(pair_phase)
        )
        # Each ceiling mover aims into the opposite half of the garage. This
        # uses the calibrated room rather than asking a mover to point almost
        # straight beneath itself, which is outside the imported envelope.
        away = -1.0 if fixture.position_m.y > 0.0 else 1.0
        y = away * extents.y * (
            0.55 + 0.35 * math.cos(phase + index * 0.45)
        )
        floor_z = 0.65
        z_span = max(0.5, extents.z - floor_z)
        z = floor_z + z_span * (
            0.50 + 0.42 * math.sin(phase * 2.0 + index * math.pi / 2.0)
        )

        if decision.gesture is Gesture.CONVERGE:
            x *= 0.38
            y *= 0.72
            z = 1.15 + (z - 1.15) * 0.42
        elif decision.gesture in {Gesture.EXPAND, Gesture.RELEASE}:
            side = -1.0 if index % 2 == 0 else 1.0
            x = side * extents.x * (0.76 + 0.24 * activity)
            y = away * extents.y * (0.72 + 0.20 * activity)
        elif decision.gesture is Gesture.BREATHE:
            # Breathe is spacious rather than nearly stationary once music is
            # actually present.
            x *= 0.72
            z = 1.35 + (z - 1.35) * 0.65

        pulse = observation.beat_pulse
        if pulse > 0.02:
            beat_index = int(observation.bar_phase * 4.0) % 4
            accent_side = -1.0 if (beat_index + index) % 2 else 1.0
            x += accent_side * extents.x * 0.32 * pulse
            y += away * extents.y * 0.10 * pulse
            z += 0.38 * pulse

        return Vec3(
            clamp(x, -extents.x, extents.x),
            clamp(y, -extents.y, extents.y),
            clamp(z, floor_z, extents.z),
        )

    def _performance_solution(
        self,
        fixture: FixturePatch,
        index: int,
        observation: MusicalObservation,
        decision: PerformanceDecision,
        spatial_solution: TargetingSolution,
        motion_feedback: float = 0.0,
    ) -> TargetingSolution:
        """Use the calibrated fixture envelope as a choreographic instrument.

        A room target remains appropriate for static cues and calibration, but
        ordinary targets occupy a small angular patch when a mover is mounted
        several metres away. Music performance therefore follows four
        beat-indexed anchor points spread across the known-good DMX envelope.
        The resulting direction is projected back into room space so the
        dashboard still shows where the beam is actually being sent.
        """

        if observation.bpm is not None and observation.beat_confidence >= 0.12:
            bar_phase = observation.bar_phase % 1.0
        else:
            assumed_bpm = observation.bpm or 120.0
            bar_phase = (
                observation.timestamp_s * assumed_bpm / 60.0 / 4.0
            ) % 1.0
        beat_position = bar_phase * 4.0
        beat_index = math.floor(beat_position) % 4
        local_phase = beat_position - math.floor(beat_position)
        # Reach the next visual station early in the beat, then hold it long
        # enough for the physical beam to register as a deliberate accent.
        travel = clamp(local_phase / 0.68, 0.0, 1.0)
        travel = travel * travel * (3.0 - 2.0 * travel)

        state = decision.expression
        if self._motion_clock_s is None:
            elapsed = 1.0 / 24.0
        else:
            elapsed = clamp(observation.timestamp_s - self._motion_clock_s, 0.0, 0.20)
        self._motion_clock_s = observation.timestamp_s
        bpm = observation.bpm or 120.0
        self._motion_phase = (
            self._motion_phase + elapsed * math.tau * (bpm / 60.0) / 4.0
        ) % math.tau
        phase = self._motion_phase + index * 1.73
        envelope = clamp(
            0.16 + 0.72 * state.energy + 0.22 * state.motion
            + 0.38 * motion_feedback,
            0.12,
            1.0,
        )
        # The second corner mover (DMX 43) has a more restrictive usable
        # mounting envelope in the imported calibration. Give it a little
        # extra choreographic travel so it reads as an active partner rather
        # than a mostly stationary accent, while still respecting its limits.
        if fixture.address == 43:
            envelope = clamp(envelope * 1.22, 0.12, 1.0)
        if decision.gesture is Gesture.CONVERGE:
            envelope *= 0.66
        elif decision.gesture is Gesture.BREATHE:
            envelope *= 0.58
        mode = int((observation.timestamp_s * (bpm or 120.0) / 60.0 / 4.0)) % 4
        if mode == 0:  # figure eight: a fast vertical harmonic over a pan loop
            pan_motion = math.sin(phase)
            tilt_motion = math.sin(phase * 2.0 + index * 0.6)
        elif mode == 1:  # broad circle with fixture-to-fixture phase opposition
            pan_motion = math.sin(phase + index * math.pi)
            tilt_motion = math.cos(phase + index * math.pi)
        elif mode == 2:  # smooth pan sweep with a smaller tilt breathe
            pan_motion = 2.0 * ((phase / math.tau + 0.5) % 1.0) - 1.0
            pan_motion = 2.0 * abs(pan_motion) - 1.0
            tilt_motion = math.sin(phase * 0.5 + index)
        else:  # beat nods and alternating sides
            pan_motion = math.sin(phase * 0.5 + index * math.pi)
            tilt_motion = math.sin(phase * 2.0 + observation.beat_phase * math.tau)
        pan_normalized = 0.5 + envelope * 0.47 * pan_motion
        tilt_normalized = 0.5 + envelope * 0.44 * tilt_motion
        if observation.beat_pulse > 0.02:
            pan_normalized += (
                1.0 if (beat_index + index) % 2 else -1.0
            ) * 0.08 * observation.beat_pulse * envelope
            tilt_normalized += 0.07 * observation.beat_pulse * envelope

        calibration = fixture.calibration
        pan = calibration.pan_min_deg + pan_normalized * (
            calibration.pan_max_deg - calibration.pan_min_deg
        )
        tilt = calibration.tilt_min_deg + tilt_normalized * (
            calibration.tilt_max_deg - calibration.tilt_min_deg
        )
        direction = self.targeting.direction_for_angles(fixture, pan, tilt)
        distance = max(4.0, spatial_solution.distance_m)
        target = fixture.position_m + direction * distance
        return TargetingSolution(
            fixture_id=fixture.fixture_id,
            target=target,
            pan_deg=pan,
            tilt_deg=tilt,
            distance_m=distance,
            movement_cost_deg=0.0,
            aim_error_deg=0.0,
            branch="performance-envelope",
        )

    def _rate_limit(
        self,
        fixture: FixturePatch,
        solution: TargetingSolution,
        previous: tuple[float, float],
        elapsed_s: float,
    ) -> TargetingSolution:
        calibration = fixture.calibration
        max_pan_delta = calibration.max_pan_speed_deg_s * elapsed_s
        max_tilt_delta = calibration.max_tilt_speed_deg_s * elapsed_s
        pan = previous[0] + clamp(
            solution.pan_deg - previous[0], -max_pan_delta, max_pan_delta
        )
        tilt = previous[1] + clamp(
            solution.tilt_deg - previous[1], -max_tilt_delta, max_tilt_delta
        )
        actual = self.targeting.direction_for_angles(fixture, pan, tilt)
        desired = (solution.target - fixture.position_m).normalized()
        error = math.degrees(
            math.acos(max(-1.0, min(1.0, actual.dot(desired))))
        )
        return TargetingSolution(
            fixture_id=solution.fixture_id,
            target=solution.target,
            pan_deg=pan,
            tilt_deg=tilt,
            distance_m=solution.distance_m,
            movement_cost_deg=abs(pan - previous[0]) + abs(tilt - previous[1]),
            aim_error_deg=error,
            branch=solution.branch + "/rate-limited",
        )
