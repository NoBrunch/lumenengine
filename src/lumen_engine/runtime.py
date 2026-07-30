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
        # The saved Party Parrot-style envelope is the software movement
        # definition. There are no additional hidden limits beyond it.
        self.targeting = targeting or SpatialTargetingEngine()
        self.motion_extents = motion_extents
        self._previous: dict[str, tuple[float, float]] = {}
        self._last_timestamp_s: float | None = None
        self._audio_quiet_since_s: float | None = None
        self._audio_idle_amount = 0.0
        self._feedback_motion: dict[str, float] = {}
        self._feedback_intensity: dict[str, float] = {}
        self._feedback_strobe: dict[str, float] = {}
        self._feedback_palette: dict[str, float] = {}
        self._gesture_preferences: dict[str, dict[str, float]] = {}
        self._motion_phase = 0.0
        self._motion_clock_s: float | None = None
        self._calibration_overrides: dict[str, dict[str, float]] = {}
        self._active_song_id: int | None = None
        self._active_section: str | None = None
        self._active_artist: str | None = None

    def set_media_context(self, song_id: int | None, section: str | None = None, artist: str | None = None) -> None:
        """Set the identity/section used when resolving learned preferences."""
        self._active_song_id = song_id
        self._active_section = section
        self._active_artist = artist.casefold().strip() if artist else None

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
        strobe_delta: float = 0.0,
        palette_delta: float = 0.0,
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
        self._feedback_strobe[key] = clamp(self._feedback_strobe.get(key, 0.0) + strobe_delta, -1.0, 1.0)
        self._feedback_palette[key] = clamp(self._feedback_palette.get(key, 0.0) + palette_delta, -1.0, 1.0)

    def replace_feedback(self, biases: dict[str, dict[str, float]]) -> None:
        self._feedback_motion.clear()
        self._feedback_intensity.clear()
        self._feedback_strobe.clear()
        self._feedback_palette.clear()
        self._gesture_preferences.clear()
        for key, bias in biases.items():
            self.apply_feedback(
                scope="fixture" if key != "overall" else "overall",
                fixture_id=None if key == "overall" else key,
                motion_delta=bias.get("motion", 0.0),
                intensity_delta=bias.get("intensity", 0.0),
                strobe_delta=bias.get("strobe", 0.0),
                palette_delta=bias.get("palette", 0.0),
            )
            gestures = bias.get("gestures")
            if isinstance(gestures, dict):
                self._gesture_preferences[key] = {
                    str(name): clamp(float(value), -1.0, 1.0)
                    for name, value in gestures.items()
                }

    def _gesture_for_context(self, current: PerformanceDecision) -> Gesture:
        keys = ["overall"]
        if self._active_song_id is not None:
            keys.append(f"song:{self._active_song_id}")
            if self._active_section:
                keys.append(f"song:{self._active_song_id}:section:{self._active_section}")
        if self._active_artist:
            keys.append(f"artist:{self._active_artist}")
        scores: dict[str, float] = {}
        for key in keys:
            for name, score in self._gesture_preferences.get(key, {}).items():
                scores[name] = scores.get(name, 0.0) + score
        current_score = scores.get(current.gesture.value, 0.0)
        if current_score >= -0.22:
            return current.gesture
        candidates = [
            (score, name) for name, score in scores.items()
            if name != current.gesture.value and score > 0.18
        ]
        if not candidates:
            return current.gesture
        return Gesture(max(candidates)[1])

    def _feedback_for(self, fixture_id: str) -> tuple[float, float, float, float]:
        keys = ["overall", fixture_id]
        if self._active_song_id is not None:
            song_key = f"song:{self._active_song_id}"
            keys.append(song_key)
            if self._active_section:
                keys.append(f"{song_key}:section:{self._active_section}")
            keys.append(f"{song_key}:fixture:{fixture_id}")
        if self._active_artist:
            keys.append(f"artist:{self._active_artist}")
        return tuple(
            clamp(sum(mapping.get(key, 0.0) for key in keys), -1.0, 1.0)
            for mapping in (
                self._feedback_motion,
                self._feedback_intensity,
                self._feedback_strobe,
                self._feedback_palette,
            )
        )  # type: ignore[return-value]

    def step(self, observation: MusicalObservation) -> RuntimeFrame:
        self._active_section = observation.section
        decision = self.expression.decide(observation)
        learned_gesture = self._gesture_for_context(decision)
        if learned_gesture is not decision.gesture:
            targets = {
                Gesture.BREATHE: self.expression.policy.room_high,
                Gesture.CONVERGE: self.expression.policy.room_center,
                Gesture.PULSE: self.expression.policy.room_center,
                Gesture.SWEEP: self.expression.policy.room_wide,
                Gesture.EXPAND: self.expression.policy.room_high,
                Gesture.RELEASE: self.expression.policy.room_wide,
                Gesture.HOLD: decision.target,
            }
            decision = replace(
                decision,
                gesture=learned_gesture,
                target=targets[learned_gesture],
                reason=f"Learned preference replaced {decision.gesture.value} with {learned_gesture.value} for this context.",
            )
            self.expression.accept_gesture_override(learned_gesture, observation.timestamp_s)
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
            motion_feedback, intensity_feedback, strobe_feedback, palette_feedback = self._feedback_for(
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
            if self._calibration_overrides.get(fixture.fixture_id) is not None:
                fixture_decision = replace(fixture_decision, brightness=0.35)
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
                if observation.loudness >= 0.02 and calibration_override is None:
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
                    frame, fixture, solution, fixture_decision.brightness,
                    unrestricted=False,
                )
                if calibration_override is not None:
                    # Calibration jog is direct-DMX, matching Party Parrot.
                    # Do not remap the jog value through the saved envelope.
                    for relative, raw in (
                        (fixture.pan_coarse_channel, int(calibration_override["pan_dmx"])),
                        (fixture.tilt_coarse_channel, int(calibration_override["tilt_dmx"])),
                    ):
                        frame.set_channel(fixture.universe, fixture.address + relative - 1, (raw >> 8) & 0xFF)
                        fine = fixture.pan_fine_channel if relative == fixture.pan_coarse_channel else fixture.tilt_fine_channel
                        if fine is not None:
                            frame.set_channel(fixture.universe, fixture.address + fine - 1, raw & 0xFF)
                apply_moving_head_profile(
                    frame,
                    fixture,
                    fixture_decision,
                    observation,
                    idle_amount=idle_amount,
                    strobe_feedback=strobe_feedback,
                    palette_bias=palette_feedback,
                )
                if calibration_override is not None:
                    # The profile's speed channel is intentionally overridden
                    # only while the operator is in calibration mode.
                    speed_channel = fixture.address + 5 - 1
                    frame.set_channel(fixture.universe, speed_channel, round(calibration_override["speed"]))
            except UnreachableTargetError as error:
                warnings.append(str(error))

        for fixture in self.auxiliary_fixtures:
            motion_feedback, intensity_feedback, strobe_feedback, palette_feedback = self._feedback_for(
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
                strobe_feedback=strobe_feedback,
                palette_bias=palette_feedback,
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
            pan_motion = math.sin(phase)
            tilt_motion = math.sin(phase * 0.5 + index)
        else:  # beat nods and alternating sides
            pan_motion = math.sin(phase * 0.5 + index * math.pi)
            tilt_motion = math.sin(phase * 2.0 + observation.beat_phase * math.tau)
        pan_normalized = 0.5 + envelope * 0.47 * pan_motion
        tilt_normalized = 0.5 + envelope * 0.44 * tilt_motion
        if observation.beat_pulse > 0.02:
            pan_normalized += (
                1.0 if (beat_index + index) % 2 else -1.0
            ) * 0.045 * observation.beat_pulse * envelope
            tilt_normalized += 0.035 * observation.beat_pulse * envelope

        calibration = fixture.calibration
        pan = calibration.pan_min_deg + pan_normalized * (calibration.pan_max_deg - calibration.pan_min_deg)
        tilt = calibration.tilt_min_deg + tilt_normalized * (calibration.tilt_max_deg - calibration.tilt_min_deg)
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
