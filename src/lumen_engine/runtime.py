"""Connect perception, expression, spatial targeting, and DMX realization."""

from __future__ import annotations

from dataclasses import dataclass
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

    def step(self, observation: MusicalObservation) -> RuntimeFrame:
        decision = self.expression.decide(observation)
        elapsed = (
            None
            if self._last_timestamp_s is None
            else max(0.0, observation.timestamp_s - self._last_timestamp_s)
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
            try:
                solution = self.targeting.solve(
                    fixture,
                    target,
                    previous_pan_deg=None if previous is None else previous[0],
                    previous_tilt_deg=None if previous is None else previous[1],
                )
                if previous is not None and elapsed is not None:
                    solution = self._rate_limit(fixture, solution, previous, elapsed)
                self._previous[fixture.fixture_id] = (
                    solution.pan_deg,
                    solution.tilt_deg,
                )
                solutions.append(solution)
                apply_moving_head_solution(
                    frame, fixture, solution, decision.brightness
                )
                apply_moving_head_profile(
                    frame,
                    fixture,
                    decision,
                    observation,
                )
            except UnreachableTargetError as error:
                warnings.append(str(error))

        for fixture in self.auxiliary_fixtures:
            apply_auxiliary_fixture(
                frame,
                fixture,
                decision,
                observation,
            )

        self.output.send(frame)
        self._last_timestamp_s = observation.timestamp_s
        return RuntimeFrame(
            decision=decision,
            solutions=tuple(solutions),
            dmx=frame,
            warnings=tuple(warnings),
        )

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
