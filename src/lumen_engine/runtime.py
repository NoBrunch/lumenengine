"""Connect perception, expression, spatial targeting, and DMX realization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import math
import threading
from typing import Any

from lumen_engine.choreography import (
    BoundarySequencePlanner,
    CapturedChoreographyExample,
    ChoreographySequence,
    ChoreographyStep,
    FeedbackSignal,
    MusicalContext,
    SequencePreferenceModel,
)
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
from lumen_engine.motion import (
    MotionTuning,
    merged_motion_tunings,
    normalized_position,
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
        choreography_model: SequencePreferenceModel | None = None,
        motion_tunings: dict[str, MotionTuning] | None = None,
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
        self._routine_preferences: dict[str, dict[str, float]] = {}
        self._pending_feedback_biases: dict[str, dict[str, float]] | None = None
        self._feedback_lock = threading.RLock()
        self._active_routine = "auto"
        self._active_routine_bar: int | None = None
        self._active_routine_section: str | None = None
        self._routine_bar_counter = 0
        self._last_bar_phase: float | None = None
        self._motion_phase = 0.0
        self._motion_clock_s: float | None = None
        self._calibration_overrides: dict[str, dict[str, float]] = {}
        self._active_song_id: int | None = None
        self._active_section: str | None = None
        self._active_artist: str | None = None
        self._choreography_model = choreography_model
        self._choreography_planner = (
            BoundarySequencePlanner(choreography_model)
            if choreography_model is not None
            else None
        )
        self._last_choreography_context: MusicalContext | None = None
        self._structure_functional = "unknown"
        self._structure_energy = "unknown"
        self._structure_content = "unknown"
        self._structure_confidence = 0.0
        self._structure_boundary_probability = 0.0
        self._active_choreography_step: ChoreographyStep | None = None
        self._rehearsal_step: ChoreographyStep | None = None
        self._rehearsal_size = 1.0
        self._rehearsal_isolate = True
        self._rehearsal_phase_origin_s: float | None = None
        self._motion_tunings = merged_motion_tunings(
            None if motion_tunings is None else {
                key: value.as_dict() for key, value in motion_tunings.items()
            }
        )

    def set_rehearsal(
        self,
        routine: str | None,
        *,
        scope: str = "overall",
        intensity: float = 0.68,
        size: float = 1.0,
        palette: str | None = None,
        strobe: float = 0.0,
        isolate: bool = True,
    ) -> None:
        """Force one auditable routine without invoking the song planner."""
        if routine is None:
            self._rehearsal_step = None
            return
        previous_routine = (
            self._rehearsal_step.routine
            if self._rehearsal_step is not None
            else None
        )
        self._rehearsal_step = ChoreographyStep(
            start_beat=0.0,
            duration_beats=8.0,
            fixture_scope=scope,
            routine=routine,
            intensity=clamp(intensity, 0.0, 1.0),
            palette=palette,
            strobe=clamp(strobe, 0.0, 1.0),
        )
        self._rehearsal_size = clamp(size, 0.0, 1.0)
        self._rehearsal_isolate = bool(isolate)
        if previous_routine != routine:
            self._rehearsal_phase_origin_s = None

    def set_motion_tunings(
        self, tunings: dict[str, MotionTuning]
    ) -> None:
        self._motion_tunings = dict(tunings)

    def set_media_context(self, song_id: int | None, section: str | None = None, artist: str | None = None) -> None:
        """Set the identity/section used when resolving learned preferences."""
        if song_id != self._active_song_id:
            self.expression.reset()
            self._active_routine = "auto"
            self._active_routine_bar = None
            self._active_routine_section = None
            self._routine_bar_counter = 0
            self._last_bar_phase = None
            if self._choreography_planner is not None:
                self._choreography_planner.release()
            self._active_choreography_step = None
            self._activate_pending_feedback()
        self._active_song_id = song_id
        self._active_section = section
        self._active_artist = artist.casefold().strip() if artist else None

    def set_structure_context(
        self,
        *,
        functional: str = "unknown",
        energy: str = "unknown",
        content: str = "unknown",
        confidence: float = 0.0,
        boundary_probability: float = 0.0,
    ) -> None:
        self._structure_functional = functional or "unknown"
        self._structure_energy = energy or "unknown"
        self._structure_content = content or "unknown"
        self._structure_confidence = clamp(confidence, 0.0, 1.0)
        self._structure_boundary_probability = clamp(
            boundary_probability, 0.0, 1.0
        )

    def choreography_snapshot(self) -> dict[str, Any]:
        """Expose auditable planner provenance without changing its lease."""
        planner = self._choreography_planner
        active = planner.active if planner is not None else None
        step = self._active_choreography_step
        return {
            "bar_counter": self._routine_bar_counter,
            "phrase_index": self._routine_bar_counter // 2,
            "active_boundary_id": (
                active.boundary_id if active is not None else None
            ),
            "active_sequence_id": (
                active.sequence.sequence_id if active is not None else None
            ),
            "active_sequence_source": (
                active.sequence.source if active is not None else None
            ),
            "active_step": (
                None
                if step is None
                else {
                    "start_beat": step.start_beat,
                    "duration_beats": step.duration_beats,
                    "fixture_scope": step.fixture_scope,
                    "routine": step.routine,
                }
            ),
            "model_revision": (
                self._choreography_model.revision
                if self._choreography_model is not None
                else None
            ),
            "rehearsal": (
                None
                if self._rehearsal_step is None
                else {
                    "routine": self._rehearsal_step.routine,
                    "scope": self._rehearsal_step.fixture_scope,
                    "intensity": self._rehearsal_step.intensity,
                    "size": self._rehearsal_size,
                    "palette": self._rehearsal_step.palette,
                    "strobe": self._rehearsal_step.strobe,
                    "isolate": self._rehearsal_isolate,
                }
            ),
        }

    def learn_choreography_feedback(
        self,
        *,
        label: str,
        value: float,
        urgency: float = 0.5,
        occurrences: int = 1,
        scope: str = "overall",
        fixture_id: str | None = None,
        preferred_routine: str | None = None,
        created_unix_ms: int | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Teach the next phrase without changing the one currently running."""
        planner = self._choreography_planner
        model = self._choreography_model
        context = self._last_choreography_context
        selection = planner.active if planner is not None else None
        if model is None or context is None or selection is None:
            return None
        preferred = None
        preferred_scope = (
            fixture_id
            if scope in {"fixture", "group"} and fixture_id
            else "overall"
        )
        if preferred_routine:
            preferred = ChoreographySequence(
                sequence_id=f"preferred:{preferred_routine}",
                source="operator_preferred_action",
                steps=(
                    ChoreographyStep(
                        start_beat=0.0,
                        duration_beats=max(1.0, selection.sequence.end_beat),
                        fixture_scope=preferred_scope,
                        routine=preferred_routine,
                    ),
                ),
            )
        elif label in {
            "brighter",
            "too_dim",
            "dimmer",
            "too_bright",
            "strobe",
            "faster_strobe",
            "slower_strobe",
            "no_strobe",
            "no_strobes",
            "cool_blue_purple",
            "warmer_color",
        }:
            intensity_delta = {
                "brighter": 0.22,
                "too_dim": 0.22,
                "dimmer": -0.22,
                "too_bright": -0.22,
            }.get(label, 0.0)
            strobe_value = {
                "strobe": 0.48,
                "faster_strobe": 0.78,
                "slower_strobe": 0.20,
                "no_strobe": 0.0,
                "no_strobes": 0.0,
            }.get(label)
            palette = {
                "cool_blue_purple": "cool",
                "warmer_color": "warm",
            }.get(label)
            preferred = ChoreographySequence(
                sequence_id=f"preferred:{label}",
                source="operator_preferred_characteristic",
                steps=tuple(
                    ChoreographyStep(
                        start_beat=step.start_beat,
                        duration_beats=step.duration_beats,
                        fixture_scope=preferred_scope,
                        routine=step.routine,
                        intensity=clamp(
                            step.intensity + intensity_delta, 0.0, 1.0
                        ),
                        palette=palette or step.palette,
                        strobe=(
                            step.strobe
                            if strobe_value is None
                            else strobe_value
                        ),
                        beat_sync=step.beat_sync,
                    )
                    for step in selection.sequence.steps
                ),
            )
        receipt = model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=selection.sequence,
                feedback=(
                    FeedbackSignal(
                        label=label,
                        value=clamp(value, -1.0, 1.0),
                        urgency=clamp(urgency, 0.0, 1.0),
                        occurrences=max(1, int(occurrences)),
                        scope=scope,
                        fixture_id=fixture_id,
                        created_unix_ms=created_unix_ms,
                    ),
                ),
                preferred=preferred,
            ),
            event_id=event_id,
        )
        return {
            "model_revision": receipt.model_revision,
            "effective_strength": receipt.effective_strength,
            "urgency": receipt.urgency,
            "feedback_occurrences": receipt.feedback_occurrences,
            "preferred_sequence_learned": (
                receipt.preferred_sequence_learned
            ),
            "output_action": receipt.output_action,
            "performed_sequence": selection.sequence.as_dict(),
            "preferred_sequence": (
                preferred.as_dict() if preferred is not None else None
            ),
        }

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
        with self._feedback_lock:
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
            self._feedback_strobe[key] = clamp(
                self._feedback_strobe.get(key, 0.0) + strobe_delta,
                -1.0,
                1.0,
            )
            self._feedback_palette[key] = clamp(
                self._feedback_palette.get(key, 0.0) + palette_delta,
                -1.0,
                1.0,
            )

    def replace_feedback(self, biases: dict[str, dict[str, float]]) -> None:
        snapshot = deepcopy(biases)
        planner = self._choreography_planner
        if planner is not None and planner.active is not None:
            # A live phrase is a lease. Collect feedback immediately, but do
            # not rewrite scalar output characteristics underneath that lease.
            # The new state becomes active at the next musical boundary.
            with self._feedback_lock:
                self._pending_feedback_biases = snapshot
            return
        self._replace_feedback_now(snapshot)

    def _replace_feedback_now(
        self, biases: dict[str, dict[str, float]]
    ) -> None:
        with self._feedback_lock:
            self._replace_feedback_now_locked(biases)

    def _replace_feedback_now_locked(
        self, biases: dict[str, dict[str, float]]
    ) -> None:
        self._feedback_motion.clear()
        self._feedback_intensity.clear()
        self._feedback_strobe.clear()
        self._feedback_palette.clear()
        self._gesture_preferences.clear()
        self._routine_preferences.clear()
        # Rebuilding learned weights must not interrupt the choreography that
        # is already leased to the current phrase.  The new preferences are
        # considered naturally at the next bar/section boundary.
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
            routines = bias.get("routines")
            if isinstance(routines, dict):
                self._routine_preferences[key] = {
                    str(name): clamp(float(value), -1.0, 1.0)
                    for name, value in routines.items()
                }

    def _activate_pending_feedback(self) -> None:
        with self._feedback_lock:
            pending = self._pending_feedback_biases
            if pending is None:
                return
            self._pending_feedback_biases = None
            self._replace_feedback_now_locked(pending)

    def _routine_for_context(self, decision: PerformanceDecision, observation: MusicalObservation) -> str:
        """Choose one phrase motif and hold it across a two-bar phrase.

        The old implementation selected a gesture, then let each fixture infer
        a pattern from process time.  This made the rig look busy but made the
        phrase impossible to learn. A routine is now selected once per phrase
        from the section/energy state and contextual operator preferences.
        """
        if observation.beat_confidence >= 0.20:
            phase = observation.bar_phase % 1.0
            if (
                self._last_bar_phase is not None
                and self._last_bar_phase >= 0.72
                and phase <= 0.28
            ):
                self._routine_bar_counter += 1
            self._last_bar_phase = phase
            bar = self._routine_bar_counter // 2
        else:
            bar = self._routine_bar_counter // 2
        section = observation.section or "groove"
        if self._choreography_planner is not None:
            return self._sequence_routine_for_context(
                decision, observation, bar=bar, section=section
            )
        if (
            self._active_routine_bar == bar
            and self._active_routine != "auto"
            and self._active_routine_section == section
        ):
            return self._active_routine
        contexts = [("overall", 0.20)]
        if self._active_song_id is not None:
            contexts.append((f"song:{self._active_song_id}", 0.75))
            if self._active_section:
                contexts.append((f"song:{self._active_song_id}:section:{self._active_section}", 1.0))
        if self._active_artist:
            contexts.append((f"artist:{self._active_artist}", 0.35))
        learned: dict[str, float] = {}
        with self._feedback_lock:
            for key, context_weight in contexts:
                for name, score in self._routine_preferences.get(key, {}).items():
                    learned[name] = (
                        learned.get(name, 0.0) + score * context_weight
                    )

        if observation.loudness < 0.20 or decision.expression.energy < 0.24:
            routine = "breathe"
        elif observation.section == "build":
            routine = "fan_sweep"
        elif observation.section in {"drop", "release"} or decision.expression.energy >= 0.70:
            routine = ("opposing_chase", "beat_nod", "counter_rotate", "figure_eight")[bar % 4]
        elif decision.expression.motion >= 0.42:
            routine = ("figure_eight", "opposing_chase", "fan_sweep")[bar % 3]
        else:
            routine = "breathe"
        if learned:
            best_name, best_score = max(learned.items(), key=lambda item: item[1])
            base_score = learned.get(routine, 0.0)
            # Learning may replace the authored choice only when the contextual
            # preference is both meaningful and unambiguous. Conflicting old
            # feedback must not win a dictionary-order tie and latch "breathe".
            if best_score >= 0.18 and best_score >= base_score + 0.08:
                routine = best_name
        self._active_routine = routine
        self._active_routine_bar = bar
        self._active_routine_section = section
        return routine

    def _sequence_routine_for_context(
        self,
        decision: PerformanceDecision,
        observation: MusicalObservation,
        *,
        bar: int,
        section: str,
    ) -> str:
        energy_label = {
            "drop": "release",
            "release": "release",
            "build": "build",
            "breakdown": "breakdown",
            "silence": "silence",
        }.get(section, "groove")
        context = MusicalContext(
            functional_label=self._structure_functional,
            energy_label=(
                self._structure_energy
                if self._structure_energy != "unknown"
                else energy_label
            ),
            content_label=self._structure_content,
            energy=decision.expression.energy,
            motion=decision.expression.motion,
            tension=decision.expression.tension,
            bpm=observation.bpm,
            song_key=(
                str(self._active_song_id)
                if self._active_song_id is not None
                else None
            ),
            artist=self._active_artist,
        )
        self._last_choreography_context = context
        boundary_id = (
            f"{self._active_song_id or 'line-in'}:{section}:{bar}"
        )
        active = self._choreography_planner.active
        if active is None or active.boundary_id != boundary_id:
            self._activate_pending_feedback()
        candidates = list(_choreography_candidates(context))
        if self._choreography_model is not None:
            known = {
                sequence.semantic_signature for sequence in candidates
            }
            candidates.extend(
                sequence
                for sequence in self._choreography_model.learned_candidates()
                if sequence.semantic_signature not in known
            )
        selection = self._choreography_planner.choose(
            boundary_id=boundary_id,
            context=context,
            candidates=candidates,
        )
        beat_in_phrase = (
            (self._routine_bar_counter % 2) * 4.0
            + (observation.bar_phase % 1.0) * 4.0
        )
        matching = [
            step
            for step in selection.sequence.steps
            if step.start_beat <= beat_in_phrase
            < step.start_beat + step.duration_beats
        ]
        if matching:
            step = matching[-1]
        else:
            prior = [
                candidate
                for candidate in selection.sequence.steps
                if candidate.start_beat <= beat_in_phrase
            ]
            step = prior[-1] if prior else selection.sequence.steps[0]
        self._active_choreography_step = step
        self._active_routine = step.routine
        self._active_routine_bar = bar
        self._active_routine_section = section
        return step.routine

    def _gesture_for_context(self, current: PerformanceDecision) -> Gesture:
        keys = ["overall"]
        if self._active_song_id is not None:
            keys.append(f"song:{self._active_song_id}")
            if self._active_section:
                keys.append(f"song:{self._active_song_id}:section:{self._active_section}")
        if self._active_artist:
            keys.append(f"artist:{self._active_artist}")
        scores: dict[str, float] = {}
        with self._feedback_lock:
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
                keys.append(
                    f"{song_key}:section:{self._active_section}:fixture:{fixture_id}"
                )
            keys.append(f"{song_key}:fixture:{fixture_id}")
        if self._active_artist:
            keys.append(f"artist:{self._active_artist}")
            keys.append(f"artist:{self._active_artist}:fixture:{fixture_id}")
        with self._feedback_lock:
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
        rehearsal_step = self._rehearsal_step
        learned_gesture = (
            decision.gesture
            if rehearsal_step is not None
            else self._gesture_for_context(decision)
        )
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
        if rehearsal_step is not None:
            routine = rehearsal_step.routine
            choreography_step = rehearsal_step
            rehearsal_gesture = {
                "breathe": Gesture.BREATHE,
                "beat_nod": Gesture.PULSE,
                "hold_position": Gesture.HOLD,
            }.get(routine, Gesture.SWEEP)
            decision = replace(
                decision,
                gesture=rehearsal_gesture,
                routine=routine,
                expression=replace(
                    decision.expression,
                    energy=max(0.24, rehearsal_step.intensity),
                    motion=self._rehearsal_size,
                ),
                brightness=rehearsal_step.intensity,
                palette_hint=rehearsal_step.palette or decision.palette_hint,
                reason=(
                    f"Rehearsal isolates {routine.replace('_', ' ')} on "
                    f"{rehearsal_step.fixture_scope.replace('_', ' ')}."
                ),
            )
        else:
            routine = self._routine_for_context(decision, observation)
            choreography_step = self._active_choreography_step
            decision = replace(
                decision,
                routine=routine,
                reason=f"{decision.reason} Phrase routine: {routine.replace('_', ' ')}.",
            )
            if (
                choreography_step is not None
                and _step_is_overall(choreography_step)
            ):
                decision = _apply_choreography_step(
                    decision, choreography_step
                )
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
            applies_to_fixture = _step_applies_to_fixture(
                choreography_step,
                fixture.fixture_id,
                is_mover=True,
            )
            target_decision = decision
            fixture_output_decision = output_decision
            if (
                choreography_step is not None
                and not _step_is_overall(choreography_step)
                and applies_to_fixture
            ):
                target_decision = _apply_choreography_step(
                    target_decision, choreography_step
                )
                fixture_output_decision = _apply_choreography_step(
                    fixture_output_decision, choreography_step
                )
            target = self._target_for_fixture(
                target_decision,
                fixture,
                index,
                observation,
            )
            previous = self._previous.get(fixture.fixture_id)
            motion_feedback, intensity_feedback, strobe_feedback, palette_feedback = self._feedback_for(
                fixture.fixture_id
            )
            fixture_decision = replace(
                fixture_output_decision,
                brightness=clamp(
                    fixture_output_decision.brightness
                    + intensity_feedback * 0.30,
                    0.0,
                    1.0,
                ),
                routine=(
                    fixture_output_decision.routine
                    if applies_to_fixture
                    else "breathe"
                ),
            )
            if (
                rehearsal_step is not None
                and self._rehearsal_isolate
                and not applies_to_fixture
            ):
                fixture_decision = replace(
                    fixture_decision, brightness=0.0
                )
            if self._calibration_overrides.get(fixture.fixture_id) is not None:
                fixture_decision = replace(fixture_decision, brightness=0.35)
            try:
                calibration_override = self._calibration_overrides.get(fixture.fixture_id)
                rehearsal_inactive = (
                    rehearsal_step is not None
                    and self._rehearsal_isolate
                    and not applies_to_fixture
                )
                if rehearsal_inactive and previous is not None:
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
                        branch="rehearsal-hold",
                    )
                elif calibration_override is not None:
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
                if (
                    observation.loudness >= 0.02
                    and calibration_override is None
                    and not rehearsal_inactive
                ):
                    solution = self._performance_solution(
                        fixture,
                        index,
                        observation,
                        target_decision,
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
                    choreography_strobe=(
                        choreography_step.strobe
                        if (
                            choreography_step is not None
                            and applies_to_fixture
                        )
                        else 0.0
                    ),
                    palette_bias=palette_feedback,
                    enabled=not rehearsal_inactive,
                )
                if calibration_override is not None:
                    # The profile's speed channel is intentionally overridden
                    # only while the operator is in calibration mode.
                    speed_channel = fixture.address + 5 - 1
                    frame.set_channel(fixture.universe, speed_channel, round(calibration_override["speed"]))
            except UnreachableTargetError as error:
                warnings.append(str(error))

        for fixture in self.auxiliary_fixtures:
            applies_to_fixture = _step_applies_to_fixture(
                choreography_step,
                fixture.fixture_id,
                is_mover=False,
            )
            fixture_output_decision = output_decision
            if (
                choreography_step is not None
                and not _step_is_overall(choreography_step)
                and applies_to_fixture
            ):
                fixture_output_decision = _apply_choreography_step(
                    fixture_output_decision, choreography_step
                )
            motion_feedback, intensity_feedback, strobe_feedback, palette_feedback = self._feedback_for(
                fixture.fixture_id
            )
            fixture_decision = replace(
                fixture_output_decision,
                brightness=clamp(
                    fixture_output_decision.brightness
                    + intensity_feedback * 0.30,
                    0.0,
                    1.0,
                ),
                routine=(
                    fixture_output_decision.routine
                    if applies_to_fixture
                    else "breathe"
                ),
            )
            if (
                rehearsal_step is not None
                and self._rehearsal_isolate
                and not applies_to_fixture
            ):
                fixture_decision = replace(
                    fixture_decision, brightness=0.0
                )
            apply_auxiliary_fixture(
                frame,
                fixture,
                fixture_decision,
                observation,
                idle_amount=idle_amount,
                motion_feedback=motion_feedback,
                strobe_feedback=strobe_feedback,
                choreography_strobe=(
                    choreography_step.strobe
                    if (
                        choreography_step is not None
                        and applies_to_fixture
                    )
                    else 0.0
                ),
                palette_bias=palette_feedback,
                enabled=not (
                    rehearsal_step is not None
                    and self._rehearsal_isolate
                    and not applies_to_fixture
                ),
                motion_tuning=self._motion_tunings.get(
                    fixture_decision.routine
                ),
                motion_timestamp_s=(
                    None
                    if rehearsal_step is None
                    or self._rehearsal_phase_origin_s is None
                    else max(
                        0.0,
                        observation.timestamp_s
                        - self._rehearsal_phase_origin_s,
                    )
                ),
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

        state = decision.expression
        tuning = self._motion_tunings.get(
            decision.routine,
            self._motion_tunings["breathe"],
        )
        bpm = observation.bpm or 120.0
        absolute_beat = observation.timestamp_s * bpm / 60.0
        if self._rehearsal_step is not None:
            if self._rehearsal_phase_origin_s is None:
                self._rehearsal_phase_origin_s = observation.timestamp_s
            beat_position = (
                observation.timestamp_s - self._rehearsal_phase_origin_s
            ) * bpm / 60.0
            size = self._rehearsal_size
        else:
            beat_position = absolute_beat
            structural_motion = {
                "release": 0.24,
                "drop": 0.24,
                "build": 0.13,
                "groove": 0.06,
                "breakdown": -0.16,
                "low": -0.12,
                "outro": -0.08,
            }.get(self._structure_energy, 0.0)
            transition_expansion = (
                0.10 * self._structure_boundary_probability
                if structural_motion > 0.0
                else 0.0
            )
            size = clamp(
                0.16 + 0.72 * state.energy + 0.22 * state.motion
                + 0.38 * motion_feedback
                + structural_motion * self._structure_confidence
                + transition_expansion,
                0.12,
                1.0,
            )
        pan_normalized, tilt_normalized = normalized_position(
            decision.routine,
            beat_position,
            index,
            len(self.fixtures),
            tuning,
            size=size,
        )

        calibration = fixture.calibration
        # Motion paths use room semantics: 0→1 means left→right and low→high.
        # A fixture whose captured DMX order runs in the other direction must
        # invert the normalized path before it is mapped into the numerically
        # sorted angle/DMX envelope. Without this, the channel-31 mover's
        # correctly characterized reversed tilt performs every vertical path
        # upside down and spends "high" portions of a routine facing low.
        pan_fixture_normalized = (
            pan_normalized
            if calibration.pan_direction > 0
            else 1.0 - pan_normalized
        )
        tilt_fixture_normalized = (
            tilt_normalized
            if calibration.tilt_direction > 0
            else 1.0 - tilt_normalized
        )
        pan = calibration.pan_min_deg + pan_fixture_normalized * (calibration.pan_max_deg - calibration.pan_min_deg)
        tilt = calibration.tilt_min_deg + tilt_fixture_normalized * (calibration.tilt_max_deg - calibration.tilt_min_deg)
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


def _choreography_candidates(
    context: MusicalContext,
) -> tuple[ChoreographySequence, ...]:
    """Return phrase-level candidates ordered by the current musical role."""

    def sequence(
        sequence_id: str, *steps: tuple[float, float, str]
    ) -> ChoreographySequence:
        return ChoreographySequence(
            sequence_id=sequence_id,
            source="lumen_authored_v1",
            steps=tuple(
                ChoreographyStep(
                    start_beat=start,
                    duration_beats=duration,
                    fixture_scope="overall",
                    routine=routine,
                    intensity=context.energy,
                    strobe=0.0,
                )
                for start, duration, routine in steps
            ),
        )

    calm = sequence("calm-arc", (0.0, 8.0, "breathe"))
    groove = sequence(
        "groove-exchange",
        (0.0, 4.0, "figure_eight"),
        (4.0, 4.0, "opposing_chase"),
    )
    build = sequence(
        "build-and-answer",
        (0.0, 4.0, "fan_sweep"),
        (4.0, 2.0, "beat_nod"),
        (6.0, 2.0, "opposing_chase"),
    )
    release = sequence(
        "release-counterplay",
        (0.0, 2.0, "opposing_chase"),
        (2.0, 2.0, "beat_nod"),
        (4.0, 4.0, "counter_rotate"),
    )
    energy_label = context.energy_label
    if (
        energy_label in {"silence", "low", "breakdown"}
        or context.energy < 0.24
    ):
        return calm, groove, build, release
    if energy_label == "build":
        return build, groove, release, calm
    if energy_label == "release" or context.energy >= 0.70:
        return release, groove, build, calm
    return groove, build, release, calm


def _step_applies_to_fixture(
    step: ChoreographyStep | None,
    fixture_id: str,
    *,
    is_mover: bool,
) -> bool:
    if step is None:
        return True
    scope = step.fixture_scope.casefold().strip()
    if scope in {"overall", "all", "rig"}:
        return True
    if scope in {"movers", "group:movers"}:
        return is_mover
    if scope in {"center", "multi_effect", "group:center"}:
        return not is_mover
    return scope in {
        fixture_id.casefold(),
        f"fixture:{fixture_id.casefold()}",
    }


def _step_is_overall(step: ChoreographyStep) -> bool:
    return step.fixture_scope.casefold().strip() in {
        "overall",
        "all",
        "rig",
    }


def _apply_choreography_step(
    decision: PerformanceDecision,
    step: ChoreographyStep,
) -> PerformanceDecision:
    scale = 0.62 + 0.76 * step.intensity
    return replace(
        decision,
        expression=replace(
            decision.expression,
            motion=clamp(decision.expression.motion * scale, 0.0, 1.0),
        ),
        brightness=clamp(decision.brightness * scale, 0.0, 1.0),
        palette_hint=step.palette or decision.palette_hint,
    )
