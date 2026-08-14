"""Connect perception, expression, spatial targeting, and DMX realization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from copy import deepcopy
import math
import threading
from typing import Any, Iterable

from lumen_engine.choreography import (
    CapturedChoreographyExample,
    CHOREOGRAPHY_LANES,
    ChoreographySequence,
    ChoreographyStep,
    DmxHistorySample,
    FeedbackSignal,
    MusicalContext,
    ParallelBoundarySequencePlanner,
    SequencePreferenceModel,
    choreography_lanes_for_scope,
    sequence_for_lane,
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
    expression_rgb,
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
    CenterMotionTuning,
    DEFAULT_CENTER_MOTION_TUNINGS,
    MotionTuning,
    merged_motion_tunings,
    normalized_position,
)
from lumen_engine.spatial import (
    SpatialTargetingEngine,
    TargetingSolution,
    UnreachableTargetError,
)
from lumen_engine.profiles import party_parrot_profile


def _blend(current: float, target: float | str, amount: float) -> float:
    return current + (float(target) - current) * clamp(amount, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class FeedbackCharacteristics:
    """Literal feedback axes resolved for one fixture/context."""

    motion_speed: float = 0.0
    side_arm_speed: float = 0.0
    travel_size: float = 0.0
    activity_density: float = 0.0
    brightness: float = 0.0
    palette: float = 0.0
    strobe_enabled: float = 0.0
    strobe_rate: float = 0.0
    beat_sync: float = 0.0
    cue_timing: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "motion_speed": self.motion_speed,
            "side_arm_speed": self.side_arm_speed,
            "travel_size": self.travel_size,
            "activity_density": self.activity_density,
            "brightness": self.brightness,
            "palette": self.palette,
            "strobe_enabled": self.strobe_enabled,
            "strobe_rate": self.strobe_rate,
            "beat_sync": self.beat_sync,
            "cue_timing": self.cue_timing,
        }


@dataclass(frozen=True, slots=True)
class EffectiveCueOutput:
    """Auditable characteristics actually offered to a fixture lane."""

    lane: str
    fixture_id: str
    routine: str
    motion_speed: float
    side_arm_speed: float
    travel_size: float
    activity_density: float
    brightness: float
    palette: str
    color_activity: float
    strobe_enabled: bool
    strobe_rate: float
    beat_sync: float
    cue_timing: float
    cue_start_beat: float | None
    cue_end_beat: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "fixture_id": self.fixture_id,
            "routine": self.routine,
            "motion_speed": self.motion_speed,
            "side_arm_speed": self.side_arm_speed,
            "travel_size": self.travel_size,
            "activity_density": self.activity_density,
            "brightness": self.brightness,
            "palette": self.palette,
            "color_activity": self.color_activity,
            "strobe": {
                "enabled": self.strobe_enabled,
                "rate": self.strobe_rate,
            },
            "beat_sync": self.beat_sync,
            "cue_timing": self.cue_timing,
            "cue_start_beat": self.cue_start_beat,
            "cue_end_beat": self.cue_end_beat,
        }


@dataclass(frozen=True, slots=True)
class RuntimeFrame:
    decision: PerformanceDecision
    solutions: tuple[TargetingSolution, ...]
    dmx: DMXFrame
    warnings: tuple[str, ...]
    effective_outputs: tuple[EffectiveCueOutput, ...] = ()


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
        center_motion_tunings: dict[str, CenterMotionTuning] | None = None,
        gesture_movements: dict[str, tuple[str, ...]] | None = None,
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
        self._mover_brightness: dict[str, float] = {}
        self._last_timestamp_s: float | None = None
        self._audio_quiet_since_s: float | None = None
        self._audio_idle_amount = 0.0
        self._feedback_motion_speed: dict[str, float] = {}
        self._feedback_side_arm_speed: dict[str, float] = {}
        self._feedback_travel_size: dict[str, float] = {}
        self._feedback_activity_density: dict[str, float] = {}
        self._feedback_brightness: dict[str, float] = {}
        self._feedback_strobe_enabled: dict[str, float] = {}
        self._feedback_strobe_rate: dict[str, float] = {}
        self._feedback_beat_sync: dict[str, float] = {}
        self._feedback_cue_timing: dict[str, float] = {}
        # Stable aliases for callers that still load the v1 scalar snapshot.
        self._feedback_motion = self._feedback_travel_size
        self._feedback_intensity = self._feedback_brightness
        self._feedback_strobe = self._feedback_strobe_enabled
        self._feedback_palette: dict[str, float] = {}
        self._gesture_preferences: dict[str, dict[str, float]] = {}
        self._routine_preferences: dict[str, dict[str, float]] = {}
        self._gesture_movements = {
            str(gesture): tuple(str(routine) for routine in routines)
            for gesture, routines in (gesture_movements or {}).items()
        }
        self._pending_feedback_biases: dict[str, dict[str, float]] | None = None
        self._pending_replan_lanes: set[str] = set()
        self._feedback_lock = threading.RLock()
        self._active_routine = "auto"
        self._active_routine_bar: int | None = None
        self._active_routine_section: str | None = None
        self._routine_bar_counter = 0
        self._last_bar_phase: float | None = None
        self._motion_phase = 0.0
        self._motion_clock_s: float | None = None
        self._motion_clock_bpm = 120.0
        self._motion_path_phase = 0.0
        self._motion_path_source_phase: float | None = None
        self._lane_motion_path_phase = {lane: 0.0 for lane in CHOREOGRAPHY_LANES}
        self._lane_motion_path_source_phase: dict[str, float | None] = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._calibration_overrides: dict[str, dict[str, float]] = {}
        self._active_song_id: int | None = None
        self._active_section: str | None = None
        self._active_artist: str | None = None
        self._choreography_model = choreography_model
        self._choreography_planner = (
            ParallelBoundarySequencePlanner(choreography_model)
            if choreography_model is not None
            else None
        )
        self._last_choreography_context: MusicalContext | None = None
        self._structure_functional = "unknown"
        self._structure_energy = "unknown"
        self._structure_content = "unknown"
        self._structure_confidence = 0.0
        self._structure_boundary_probability = 0.0
        self._structure_resolution: dict[str, Any] = {}
        self._active_choreography_step: ChoreographyStep | None = None
        self._active_choreography_steps: dict[
            str, ChoreographyStep | None
        ] = {lane: None for lane in CHOREOGRAPHY_LANES}
        self._lane_sequence_origin_beats: dict[str, float | None] = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._lane_active_section: dict[str, str | None] = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._lane_boundary_serial: dict[str, int] = {
            lane: 0 for lane in CHOREOGRAPHY_LANES
        }
        self._lane_last_phrase: dict[str, int | None] = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._active_step_elapsed_beats: dict[str, float] = {
            lane: 0.0 for lane in CHOREOGRAPHY_LANES
        }
        self._recalled_choreography: tuple[ChoreographySequence, ...] = ()
        self._consumed_recalled_sequence_ids: dict[str, set[str]] = {
            lane: set() for lane in CHOREOGRAPHY_LANES
        }
        self._rehearsal_step: ChoreographyStep | None = None
        self._rehearsal_size = 1.0
        self._rehearsal_isolate = True
        self._rehearsal_phase_origin_s: float | None = None
        self._motion_tunings = merged_motion_tunings(
            None if motion_tunings is None else {
                key: value.as_dict() for key, value in motion_tunings.items()
            }
        )
        self._center_motion_tunings = (
            dict(DEFAULT_CENTER_MOTION_TUNINGS)
            if center_motion_tunings is None
            else dict(center_motion_tunings)
        )
        self._effective_outputs: dict[str, EffectiveCueOutput] = {}
        # One actual color is held per fixture lane. Palette selection is a
        # cue-level decision; brightness and movement may change every frame
        # without making the beam change hue.
        self._latched_colors: dict[str, tuple[float, float, float]] = {}
        self._boundary_changed_lanes: set[str] = set()
        self._lane_handoff_started_s: dict[str, float] = {}
        # Semantic samples are decoded from the final fixture frame, after
        # choreography, feedback, smoothing, latching and profile encoding.
        # They let feedback describe what the rig actually emitted rather
        # than merely the sequence Lumen intended to perform.
        self._dmx_history: deque[DmxHistorySample] = deque(maxlen=1536)
        # Explicit operator channel values are a direct, temporary boundary
        # override. They remain separate from learned semantic strobe
        # preferences so an exact DMX audition never becomes geometry.
        self._operator_strobe_dmx: dict[str, int | None] = {
            "movers": None,
            "center": None,
        }

    def set_operator_strobe_dmx(
        self, lane: str, value: int | None
    ) -> None:
        normalized = str(lane).strip().casefold()
        if normalized not in CHOREOGRAPHY_LANES:
            raise ValueError("strobe lane must be movers or center")
        self._operator_strobe_dmx[normalized] = (
            None
            if value is None
            else round(clamp(float(value), 0.0, 255.0))
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
            self._latched_colors.clear()
            return
        previous_routine = (
            self._rehearsal_step.routine
            if self._rehearsal_step is not None
            else None
        )
        previous_palette = (
            self._rehearsal_step.palette
            if self._rehearsal_step is not None else None
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
        if previous_routine != routine or previous_palette != palette:
            self._rehearsal_phase_origin_s = None
            self._latched_colors.clear()

    def set_motion_tunings(
        self,
        tunings: dict[str, MotionTuning],
        center_tunings: dict[str, CenterMotionTuning] | None = None,
    ) -> None:
        self._motion_tunings = dict(tunings)
        if center_tunings is not None:
            self._center_motion_tunings = dict(center_tunings)

    def set_gesture_movements(
        self, associations: dict[str, tuple[str, ...]]
    ) -> None:
        """Replace operator-authored gesture-to-movement associations.

        Associations constrain only Lumen's generated candidate pool. Exact
        song choreography remains authoritative and is never discarded by a
        general gesture configuration.
        """

        self._gesture_movements = {
            str(gesture): tuple(str(routine) for routine in routines)
            for gesture, routines in associations.items()
        }

    def invalidate_color_latches(self) -> None:
        """Resolve edited Color Studio values on the next fixture frame."""

        self._latched_colors.clear()

    def set_media_context(self, song_id: int | None, section: str | None = None, artist: str | None = None) -> None:
        """Set the identity/section used when resolving learned preferences."""
        if song_id != self._active_song_id:
            self._reset_timeline_state()
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
        resolution: dict[str, Any] | None = None,
    ) -> None:
        self._structure_functional = functional or "unknown"
        self._structure_energy = energy or "unknown"
        self._structure_content = content or "unknown"
        self._structure_confidence = clamp(confidence, 0.0, 1.0)
        self._structure_boundary_probability = clamp(
            boundary_probability, 0.0, 1.0
        )
        self._structure_resolution = dict(resolution or {})

    def notify_audio_discontinuity(self) -> None:
        """Prevent a capture gap from masquerading as a completed bar."""

        self._last_bar_phase = None

    def notify_timeline_discontinuity(self) -> None:
        """Reset causal choreography after a track change or playback seek."""

        self._reset_timeline_state()

    def _reset_timeline_state(self) -> None:
        self.expression.reset()
        self._active_routine = "auto"
        self._active_routine_bar = None
        self._active_routine_section = None
        self._routine_bar_counter = 0
        self._last_bar_phase = None
        self._motion_phase = 0.0
        self._motion_clock_s = None
        self._motion_clock_bpm = 120.0
        self._motion_path_phase = 0.0
        self._motion_path_source_phase = None
        self._lane_motion_path_phase = {lane: 0.0 for lane in CHOREOGRAPHY_LANES}
        self._lane_motion_path_source_phase = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        if self._choreography_planner is not None:
            self._choreography_planner.release()
        self._active_choreography_step = None
        self._active_choreography_steps = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._lane_sequence_origin_beats = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._lane_active_section = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._lane_boundary_serial = {
            lane: 0 for lane in CHOREOGRAPHY_LANES
        }
        self._lane_last_phrase = {
            lane: None for lane in CHOREOGRAPHY_LANES
        }
        self._active_step_elapsed_beats = {
            lane: 0.0 for lane in CHOREOGRAPHY_LANES
        }
        self._pending_replan_lanes.clear()
        self._latched_colors.clear()
        self._boundary_changed_lanes.clear()
        self._lane_handoff_started_s.clear()
        self._dmx_history.clear()
        for lane in CHOREOGRAPHY_LANES:
            self._consumed_recalled_sequence_ids[lane].clear()
        self._activate_pending_feedback()

    def set_recalled_choreography(
        self, sequences: tuple[ChoreographySequence, ...]
    ) -> None:
        """Stage song-specific semantic sequences for the next boundary.

        Updating this candidate pool never changes the active lane leases, so
        UI teaching and background timeline recall cannot jerk fixtures in the
        middle of a phrase.
        """

        incoming = tuple(sequences)
        previous_ids = {
            sequence.sequence_id for sequence in self._recalled_choreography
        }
        # A placement can enter the recall window while a phrase is already
        # playing.  Mark only newly arrived lanes for a boundary replan: this
        # lets the authored call begin at the next musical boundary without
        # interrupting the current phrase.  Removing an expired placement does
        # not mark a replan, so a selected multi-step sequence can finish.
        added = (
            sequence
            for sequence in incoming
            if sequence.sequence_id not in previous_ids
        )
        with self._feedback_lock:
            for lane in CHOREOGRAPHY_LANES:
                present_ids = {
                    projected.sequence_id
                    for sequence in incoming
                    if (projected := sequence_for_lane(sequence, lane))
                    is not None
                }
                # Once the placement leaves its time/section window it may be
                # recalled again on a later play of the song.
                self._consumed_recalled_sequence_ids[lane].intersection_update(
                    present_ids
                )
            for sequence in added:
                for step in sequence.steps:
                    self._pending_replan_lanes.update(
                        choreography_lanes_for_scope(step.fixture_scope)
                    )
        self._recalled_choreography = incoming

    def choreography_snapshot(self) -> dict[str, Any]:
        """Expose auditable planner provenance without changing its lease."""
        planner = self._choreography_planner
        active = planner.active if planner is not None else None
        step = self._active_choreography_step
        lanes: dict[str, dict[str, Any]] = {}
        for lane in CHOREOGRAPHY_LANES:
            lane_active = (
                planner.active_for(lane) if planner is not None else None
            )
            lane_step = self._active_choreography_steps.get(lane)
            lanes[lane] = {
                "active_boundary_id": (
                    lane_active.boundary_id
                    if lane_active is not None else None
                ),
                "active_sequence_id": (
                    lane_active.sequence.sequence_id
                    if lane_active is not None else None
                ),
                "active_sequence_source": (
                    lane_active.sequence.source
                    if lane_active is not None else None
                ),
                "active_step": _step_snapshot(lane_step),
                "confidence": (
                    lane_active.confidence
                    if lane_active is not None else 0.0
                ),
                "reason": (
                    lane_active.reason if lane_active is not None else None
                ),
                "effective_outputs": [
                    output.as_dict()
                    for output in self._effective_outputs.values()
                    if output.lane == lane
                ],
            }
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
            "lanes": lanes,
            "effective_outputs": {
                fixture_id: output.as_dict()
                for fixture_id, output in self._effective_outputs.items()
            },
            "model_revision": (
                self._choreography_model.revision
                if self._choreography_model is not None
                else None
            ),
            "structure_context": {
                "functional": self._structure_functional,
                "energy": self._structure_energy,
                "content": self._structure_content,
                "confidence": self._structure_confidence,
                "boundary_probability": (
                    self._structure_boundary_probability
                ),
                "resolution": self._structure_resolution,
            },
            "feedback_pending": self._pending_feedback_biases is not None,
            "replan_pending_lanes": sorted(self._pending_replan_lanes),
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
        event_ids_by_lane: dict[str, str] | None = None,
        occurrences_by_lane: dict[str, int] | None = None,
        urgency_by_lane: dict[str, float] | None = None,
        lifetime: str = "global",
        target_strobe_rate: float | None = None,
    ) -> dict[str, Any] | None:
        """Teach the next phrase without changing the one currently running."""
        planner = self._choreography_planner
        model = self._choreography_model
        context = self._last_choreography_context
        if model is None or context is None or planner is None:
            return None
        lanes = self._feedback_lanes(scope, fixture_id)
        selections = {
            lane: planner.active_for(lane) for lane in lanes
        }
        selections = {
            lane: selection for lane, selection in selections.items()
            if selection is not None
        }
        if not selections:
            return None
        receipts = []
        receipts_by_lane: dict[str, Any] = {}
        preferred_by_lane: dict[str, ChoreographySequence | None] = {}
        for lane, selection in selections.items():
            lane_occurrences = max(
                1,
                int((occurrences_by_lane or {}).get(lane, occurrences)),
            )
            lane_urgency = clamp(
                float((urgency_by_lane or {}).get(lane, urgency)),
                0.0,
                1.0,
            )
            preferred = None
            if preferred_routine:
                performed_steps = selection.sequence.steps
                active = self._active_choreography_steps.get(lane)
                replaced = False
                revised_steps: list[ChoreographyStep] = []
                for item in performed_steps:
                    is_active = (
                        active is not None
                        and item.start_beat == active.start_beat
                        and item.duration_beats == active.duration_beats
                        and item.routine == active.routine
                    )
                    if (is_active or (active is None and not replaced)) and not replaced:
                        revised_steps.append(replace(item, routine=preferred_routine))
                        replaced = True
                    else:
                        revised_steps.append(item)
                preferred = ChoreographySequence(
                    sequence_id=f"preferred:{preferred_routine}:{lane}",
                    source=(
                        "operator_preferred_revision"
                        if len(revised_steps) > 1
                        else "operator_preferred_action"
                    ),
                    steps=tuple(revised_steps),
                )
            # Directional characteristics (strobe, brightness, palette, and
            # movement) are learned as metric preferences below and applied
            # through feedback biases. Only the explicit Preferred Action
            # interface is allowed to create a named choreography candidate.
            preferred_by_lane[lane] = preferred
            lane_receipt = model.learn(
                CapturedChoreographyExample(
                    context=context,
                    performed=selection.sequence,
                    feedback=(FeedbackSignal(
                        label=label,
                        value=clamp(value, -1.0, 1.0),
                        urgency=lane_urgency,
                        occurrences=lane_occurrences,
                        scope=lane,
                        fixture_id=fixture_id,
                        created_unix_ms=created_unix_ms,
                        target_strobe_rate=target_strobe_rate,
                    ),),
                    preferred=preferred,
                    dmx_history=self._recent_dmx_history(lane),
                ),
                event_id=(
                    (event_ids_by_lane or {}).get(lane)
                    if event_ids_by_lane is not None
                    else None if event_id is None else f"{event_id}:{lane}"
                ),
                lane=lane,
                lifetime=lifetime,
            )
            receipts.append(lane_receipt)
            receipts_by_lane[lane] = lane_receipt
        receipt = receipts[-1]
        # Metric feedback teaches how the completed choice was received; it
        # does not call a replacement routine. Only the explicit Preferred
        # Action flow is an instruction to revise the next phrase early.
        if preferred_routine is not None:
            with self._feedback_lock:
                self._pending_replan_lanes.update(selections)
        return {
            "model_revision": receipt.model_revision,
            "effective_strength": max(
                item.effective_strength for item in receipts
            ),
            "urgency": max(item.urgency for item in receipts),
            "feedback_occurrences": sum(
                item.feedback_occurrences for item in receipts
            ),
            "preferred_sequence_learned": (
                any(item.preferred_sequence_learned for item in receipts)
            ),
            "output_action": receipt.output_action,
            "performed_sequence": next(iter(selections.values())).sequence.as_dict(),
            "preferred_sequence": next(
                (value.as_dict() for value in preferred_by_lane.values()
                 if value is not None), None
            ),
            "lanes": {
                lane: {
                    "model_event_id": (event_ids_by_lane or {}).get(lane),
                    "performed_sequence": selection.sequence.as_dict(),
                    "preferred_sequence": (
                        preferred_by_lane[lane].as_dict()
                        if preferred_by_lane[lane] is not None else None
                    ),
                    "effective_dmx": {
                        "sample_count": receipts_by_lane[lane].dmx_summary.sample_count,
                        "intensity": receipts_by_lane[lane].dmx_summary.intensity,
                        "movement": receipts_by_lane[lane].dmx_summary.movement,
                        "strobe": receipts_by_lane[lane].dmx_summary.strobe,
                        "color_change": receipts_by_lane[lane].dmx_summary.color_change,
                        "blackout_ratio": receipts_by_lane[lane].dmx_summary.blackout_ratio,
                    },
                }
                for lane, selection in selections.items()
            },
        }

    def _feedback_lanes(
        self, scope: str, fixture_id: str | None
    ) -> tuple[str, ...]:
        if scope == "overall":
            return CHOREOGRAPHY_LANES
        target = fixture_id or scope
        semantic = choreography_lanes_for_scope(target)
        if semantic:
            return semantic
        normalized = str(target).casefold().removeprefix("fixture:")
        if any(
            fixture.fixture_id.casefold() == normalized
            for fixture in self.fixtures
        ):
            return ("movers",)
        if any(
            fixture.fixture_id.casefold() == normalized
            for fixture in self.auxiliary_fixtures
        ):
            return ("center",)
        return CHOREOGRAPHY_LANES

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
        motion_speed_delta: float = 0.0,
        side_arm_speed_delta: float = 0.0,
        travel_size_delta: float | None = None,
        activity_density_delta: float = 0.0,
        brightness_delta: float | None = None,
        strobe_enabled_delta: float | None = None,
        strobe_rate_delta: float | None = None,
        beat_sync_delta: float = 0.0,
        cue_timing_delta: float = 0.0,
    ) -> None:
        with self._feedback_lock:
            self._apply_feedback_locked(
                scope=scope,
                fixture_id=fixture_id,
                motion_delta=motion_delta,
                intensity_delta=intensity_delta,
                strobe_delta=strobe_delta,
                palette_delta=palette_delta,
                motion_speed_delta=motion_speed_delta,
                side_arm_speed_delta=side_arm_speed_delta,
                travel_size_delta=(
                    motion_delta
                    if travel_size_delta is None else travel_size_delta
                ),
                activity_density_delta=activity_density_delta,
                brightness_delta=(
                    intensity_delta
                    if brightness_delta is None else brightness_delta
                ),
                strobe_enabled_delta=(
                    strobe_delta
                    if strobe_enabled_delta is None else strobe_enabled_delta
                ),
                strobe_rate_delta=(
                    strobe_delta
                    if strobe_rate_delta is None else strobe_rate_delta
                ),
                beat_sync_delta=beat_sync_delta,
                cue_timing_delta=cue_timing_delta,
            )

    def _apply_feedback_locked(
        self,
        *,
        scope: str,
        fixture_id: str | None,
        motion_delta: float,
        intensity_delta: float,
        strobe_delta: float,
        palette_delta: float,
        motion_speed_delta: float = 0.0,
        side_arm_speed_delta: float = 0.0,
        travel_size_delta: float | None = None,
        activity_density_delta: float = 0.0,
        brightness_delta: float | None = None,
        strobe_enabled_delta: float | None = None,
        strobe_rate_delta: float | None = None,
        beat_sync_delta: float = 0.0,
        cue_timing_delta: float = 0.0,
    ) -> None:
        """Apply one bias while the caller already owns `_feedback_lock`."""

        key = fixture_id if scope == "fixture" and fixture_id else "overall"
        deltas = (
            (self._feedback_motion_speed, motion_speed_delta),
            (self._feedback_side_arm_speed, side_arm_speed_delta),
            (
                self._feedback_travel_size,
                motion_delta if travel_size_delta is None else travel_size_delta,
            ),
            (self._feedback_activity_density, activity_density_delta),
            (
                self._feedback_brightness,
                intensity_delta if brightness_delta is None else brightness_delta,
            ),
            (
                self._feedback_strobe_enabled,
                strobe_delta
                if strobe_enabled_delta is None else strobe_enabled_delta,
            ),
            (
                self._feedback_strobe_rate,
                strobe_delta if strobe_rate_delta is None else strobe_rate_delta,
            ),
            (self._feedback_beat_sync, beat_sync_delta),
            (self._feedback_cue_timing, cue_timing_delta),
        )
        for mapping, delta in deltas:
            mapping[key] = clamp(mapping.get(key, 0.0) + delta, -1.0, 1.0)
        self._feedback_palette[key] = clamp(
            self._feedback_palette.get(key, 0.0) + palette_delta,
            -1.0,
            1.0,
        )

    def replace_feedback(
        self,
        biases: dict[str, dict[str, float]],
        *,
        replan_lanes: Iterable[str] | None = None,
    ) -> None:
        snapshot = deepcopy(biases)
        planner = self._choreography_planner
        requested = tuple(() if replan_lanes is None else replan_lanes)
        if planner is not None and planner.active is not None and requested:
            # A live phrase is a lease. Collect feedback immediately, but do
            # not rewrite scalar output characteristics underneath that lease.
            # The new state becomes active at the next musical boundary.
            with self._feedback_lock:
                self._pending_feedback_biases = snapshot
                for lane in requested:
                    normalized = str(lane).casefold().strip()
                    if normalized not in CHOREOGRAPHY_LANES:
                        raise ValueError(
                            f"unsupported choreography lane: {lane}"
                        )
                    self._pending_replan_lanes.add(normalized)
            return
        # Scalar feedback changes speed, travel, density, brightness, and
        # effect characteristics. It must not terminate or replace the
        # phrase that the operator is currently watching. The deterministic
        # output rate limiter absorbs the resulting target adjustment while
        # routine preferences wait for the sequence's natural boundary.
        self._replace_feedback_now(snapshot)

    def _replace_feedback_now(
        self, biases: dict[str, dict[str, float]]
    ) -> None:
        with self._feedback_lock:
            self._replace_feedback_now_locked(biases)

    def _replace_feedback_now_locked(
        self, biases: dict[str, dict[str, float]]
    ) -> None:
        self._feedback_motion_speed.clear()
        self._feedback_side_arm_speed.clear()
        self._feedback_travel_size.clear()
        self._feedback_activity_density.clear()
        self._feedback_brightness.clear()
        self._feedback_strobe_enabled.clear()
        self._feedback_strobe_rate.clear()
        self._feedback_beat_sync.clear()
        self._feedback_cue_timing.clear()
        self._feedback_palette.clear()
        self._gesture_preferences.clear()
        self._routine_preferences.clear()
        # Rebuilding learned weights must not interrupt the choreography that
        # is already leased to the current phrase.  The new preferences are
        # considered naturally at the next bar/section boundary.
        for key, bias in biases.items():
            self._apply_feedback_locked(
                scope="fixture" if key != "overall" else "overall",
                fixture_id=None if key == "overall" else key,
                motion_delta=bias.get("motion", 0.0),
                intensity_delta=bias.get("intensity", 0.0),
                strobe_delta=bias.get("strobe", 0.0),
                palette_delta=bias.get("palette", 0.0),
                motion_speed_delta=bias.get("motion_speed", bias.get("speed", 0.0)),
                side_arm_speed_delta=bias.get("side_arm_speed", 0.0),
                travel_size_delta=bias.get("travel_size", bias.get("motion", 0.0)),
                activity_density_delta=bias.get("activity_density", bias.get("density", 0.0)),
                brightness_delta=bias.get("brightness", bias.get("intensity", 0.0)),
                strobe_enabled_delta=bias.get("strobe_enabled", bias.get("strobe", 0.0)),
                strobe_rate_delta=bias.get("strobe_rate", bias.get("strobe", 0.0)),
                beat_sync_delta=bias.get("beat_sync", 0.0),
                cue_timing_delta=bias.get("cue_timing", 0.0),
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
            # Once the published tempo clock is unavailable, its eventual
            # replacement may have an unrelated bar origin. Forget the old
            # comparison point so reacquisition cannot masquerade as a true
            # high-to-low bar wrap and advance the choreography phrase.
            self._last_bar_phase = None
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
        song_key = (
            str(self._active_song_id)
            if self._active_song_id is not None
            else None
        )
        resolved_energy = (
            (self._structure_resolution.get("axes") or {}).get("energy") or {}
        )
        structural_cue = str(
            resolved_energy.get("cue_key")
            or f"section:{section}"
        )
        cue_key = (
            f"song:{song_key}|{structural_cue}"
            if song_key is not None else None
        )
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
            song_key=song_key,
            artist=self._active_artist,
            cue_key=cue_key,
        )
        self._last_choreography_context = context
        absolute_beat = (
            self._routine_bar_counter * 4.0
            + (observation.bar_phase % 1.0) * 4.0
        )
        feedback_activated = False
        for lane in CHOREOGRAPHY_LANES:
            active = self._choreography_planner.active_for(lane)
            origin = self._lane_sequence_origin_beats[lane]
            elapsed = (
                0.0 if origin is None
                else max(0.0, absolute_beat - origin)
            )
            should_reselect = (
                active is None
                or elapsed >= max(1.0, active.sequence.end_beat)
                or (
                    lane in self._pending_replan_lanes
                    and self._lane_last_phrase[lane] != bar
                )
            )
            if should_reselect:
                self._boundary_changed_lanes.add(lane)
                if not feedback_activated:
                    self._activate_pending_feedback()
                    feedback_activated = True
                self._lane_boundary_serial[lane] += 1
                boundary_id = (
                    f"{self._active_song_id or 'line-in'}:{lane}:"
                    f"{section}:{self._lane_boundary_serial[lane]}"
                )
            else:
                assert active is not None
                boundary_id = active.boundary_id
            recalled = (
                sequence_for_lane(sequence, lane)
                for sequence in self._recalled_choreography
            )
            recalled_candidates = [
                sequence
                for sequence in recalled
                if sequence is not None
                and sequence.sequence_id
                not in self._consumed_recalled_sequence_ids[lane]
            ]
            # A time/section placement is an explicit owner-authored call,
            # not a weak preference hint. When one is active for this lane it
            # is the complete candidate pool for the next boundary. Feedback
            # continues to rank multiple placed choices and future defaults.
            candidates = recalled_candidates or list(
                _choreography_candidates(
                    context,
                    lane=lane,
                    development_index=(
                        0
                        if active is None
                        or self._lane_active_section[lane] != section
                        else max(
                            0, self._lane_boundary_serial[lane] - 1
                        )
                    ),
                )
            )
            associated = tuple(dict.fromkeys(
                self._gesture_movements.get(decision.gesture.value, ())
            ))
            known_routines = frozenset(self._motion_tunings)
            gesture_constrained = bool(
                not recalled_candidates
                and associated
                and frozenset(associated) != known_routines
            )
            if gesture_constrained:
                # A deliberately narrowed mapping is literal: every step in
                # the generated phrase comes from the movements associated
                # with this gesture. Exact-song recall remains above this
                # branch and learned candidates cannot append an unassociated
                # routine after the filter.
                candidates = [
                    ChoreographySequence(
                        sequence_id=(
                            f"gesture-{decision.gesture.value}-{lane}-{routine}"
                        ),
                        source="operator_gesture_mapping_v1",
                        steps=(ChoreographyStep(
                            start_beat=0.0,
                            duration_beats=16.0,
                            fixture_scope=lane,
                            routine=routine,
                            intensity=context.energy,
                            strobe=0.0,
                        ),),
                    )
                    for routine in associated
                ]
            if (
                self._choreography_model is not None
                and not recalled_candidates
                and not gesture_constrained
            ):
                known = {
                    sequence.semantic_signature for sequence in candidates
                }
                learned = (
                    sequence_for_lane(sequence, lane)
                    for sequence in self._choreography_model.learned_candidates(
                        context
                    )
                )
                candidates.extend(
                    sequence for sequence in learned
                    if sequence is not None
                    and sequence.semantic_signature not in known
                )
            selection = self._choreography_planner.choose_lane(
                lane,
                boundary_id=boundary_id,
                context=context,
                candidates=candidates,
                recent_dmx=self._recent_dmx_history(lane),
            )
            if not selection.held_for_boundary:
                if selection.sequence in recalled_candidates:
                    self._consumed_recalled_sequence_ids[lane].add(
                        selection.sequence.sequence_id
                    )
                self._lane_sequence_origin_beats[lane] = absolute_beat
                self._lane_active_section[lane] = section
                self._lane_last_phrase[lane] = bar
                with self._feedback_lock:
                    self._pending_replan_lanes.discard(lane)
                elapsed = 0.0
            else:
                origin = self._lane_sequence_origin_beats[lane]
                elapsed = (
                    0.0 if origin is None
                    else max(0.0, absolute_beat - origin)
                )
            previous_step = self._active_choreography_steps[lane]
            next_step = _sequence_step_at(
                selection.sequence,
                min(
                    elapsed,
                    max(0.0, selection.sequence.end_beat - 1e-6),
                ),
            )
            if (
                previous_step is not None
                and previous_step.routine != next_step.routine
            ):
                self._boundary_changed_lanes.add(lane)
            self._active_choreography_steps[lane] = next_step
            active_step = self._active_choreography_steps[lane]
            self._active_step_elapsed_beats[lane] = max(
                0.0,
                elapsed - (active_step.start_beat if active_step else 0.0),
            )
        step = self._active_choreography_steps["movers"]
        assert step is not None
        # Compatibility: the public PerformanceDecision names the movers lane;
        # the snapshot and fixture output expose both simultaneous decisions.
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
        """Return the legacy four-axis view for compatibility."""

        resolved = self._characteristics_feedback_for(fixture_id)
        return (
            resolved.travel_size,
            resolved.brightness,
            resolved.strobe_enabled,
            resolved.palette,
        )

    def _characteristics_feedback_for(
        self, fixture_id: str
    ) -> FeedbackCharacteristics:
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
            def total(mapping: dict[str, float]) -> float:
                return clamp(
                    sum(mapping.get(key, 0.0) for key in keys), -1.0, 1.0
                )

            return FeedbackCharacteristics(
                motion_speed=total(self._feedback_motion_speed),
                side_arm_speed=total(self._feedback_side_arm_speed),
                travel_size=total(self._feedback_travel_size),
                activity_density=total(self._feedback_activity_density),
                brightness=total(self._feedback_brightness),
                palette=total(self._feedback_palette),
                strobe_enabled=total(self._feedback_strobe_enabled),
                strobe_rate=total(self._feedback_strobe_rate),
                beat_sync=total(self._feedback_beat_sync),
                cue_timing=total(self._feedback_cue_timing),
            )

    def _effective_cue_output(
        self,
        *,
        lane: str,
        fixture_id: str,
        decision: PerformanceDecision,
        step: ChoreographyStep | None,
        feedback: FeedbackCharacteristics,
        applies: bool,
    ) -> EffectiveCueOutput:
        active_step = step if applies else None
        speed = clamp(
            (0.5 if active_step is None else active_step.motion_speed),
            0.0,
            1.0,
        )
        travel = clamp(
            (1.0 if active_step is None else active_step.travel_size),
            0.05,
            1.0,
        )
        density = clamp(
            (1.0 if active_step is None else active_step.activity_density),
            0.05,
            1.0,
        )
        base_brightness = decision.brightness
        if active_step is not None and active_step.brightness is not None:
            base_brightness = active_step.brightness
        brightness = clamp(base_brightness, 0.0, 1.0)
        structure_profile, structure_strength = self._structure_output_profile()
        if structure_profile is not None and structure_strength > 0.0:
            speed = _blend(speed, structure_profile["motion_speed"], structure_strength)
            travel = _blend(travel, structure_profile["travel_size"], structure_strength)
            density = _blend(density, structure_profile["activity_density"], structure_strength)
            brightness = clamp(
                brightness
                * _blend(1.0, structure_profile["brightness_scale"], structure_strength),
                0.0,
                1.0,
            )
        # Structure supplies the automatic baseline. The operator's literal
        # instruction is applied afterward so a trusted section label cannot
        # quietly erase "more movement", "faster", or "calm down".
        speed = clamp(speed + 0.45 * feedback.motion_speed, 0.0, 1.0)
        travel = clamp(travel + 0.55 * feedback.travel_size, 0.05, 1.0)
        density = clamp(
            density + 0.55 * feedback.activity_density, 0.05, 1.0
        )
        brightness = clamp(
            brightness + 0.30 * feedback.brightness, 0.0, 1.0
        )
        authored_strobe = bool(
            active_step is not None
            and (
                active_step.strobe > 0.05
                if active_step.strobe_enabled is None
                else active_step.strobe_enabled
            )
        )
        # Positive preference cannot manufacture a strobe cue. It can rank
        # strobe-bearing choreography and tune its rate; only the currently
        # leased, duration-bounded step can explicitly enable the effect.
        strobe_enabled = (
            authored_strobe
            and feedback.strobe_enabled > -0.60
            and self._structure_strobe_eligible()
        )
        authored_rate = (
            0.0
            if active_step is None
            else active_step.strobe
            if active_step.strobe_rate is None
            else active_step.strobe_rate
        )
        strobe_rate = (
            clamp(authored_rate + 0.5 * feedback.strobe_rate, 0.06, 1.0)
            if strobe_enabled else 0.0
        )
        palette = (
            active_step.palette
            if active_step is not None and active_step.palette
            else decision.palette_hint
        )
        if (
            structure_profile is not None
            and palette == "auto"
            and structure_profile.get("palette")
        ):
            palette = str(structure_profile["palette"])
        color_activity = (
            1.0
            if structure_profile is None
            else _blend(
                1.0,
                structure_profile["color_activity"],
                structure_strength,
            )
        )
        return EffectiveCueOutput(
            lane=lane,
            fixture_id=fixture_id,
            routine=decision.routine if applies else "breathe",
            motion_speed=speed,
            side_arm_speed=feedback.side_arm_speed,
            travel_size=travel,
            activity_density=density,
            brightness=brightness,
            palette=palette,
            color_activity=color_activity,
            strobe_enabled=strobe_enabled,
            strobe_rate=strobe_rate,
            beat_sync=clamp(
                (1.0 if active_step is None else active_step.beat_sync)
                + 0.3 * feedback.beat_sync,
                0.0,
                1.0,
            ),
            cue_timing=clamp(
                (1.0 if active_step is None else active_step.cue_timing)
                + 0.3 * feedback.cue_timing,
                0.0,
                1.0,
            ),
            cue_start_beat=(
                None if active_step is None else active_step.start_beat
            ),
            cue_end_beat=(
                None if active_step is None
                else active_step.start_beat + active_step.duration_beats
            ),
        )

    def _structure_output_profile(
        self,
    ) -> tuple[dict[str, float | str] | None, float]:
        """Resolve how strongly the selected structure may shape a cue."""

        profiles: dict[str, dict[str, float | str]] = {
            "intro": {
                "motion_speed": 0.30, "travel_size": 0.55,
                "activity_density": 0.45, "brightness_scale": 0.72,
                "color_activity": 0.32, "palette": "midnight_teal",
            },
            "breakdown": {
                "motion_speed": 0.20, "travel_size": 0.42,
                "activity_density": 0.30, "brightness_scale": 0.58,
                "color_activity": 0.20, "palette": "cool",
            },
            "build": {
                "motion_speed": 0.48, "travel_size": 0.82,
                "activity_density": 0.76, "brightness_scale": 0.90,
                "color_activity": 0.62, "palette": "cyan_violet",
            },
            "groove": {
                "motion_speed": 0.42, "travel_size": 0.78,
                "activity_density": 0.70, "brightness_scale": 0.86,
                "color_activity": 0.50, "palette": "magenta_blue",
            },
            "drop": {
                "motion_speed": 0.68, "travel_size": 1.0,
                "activity_density": 0.96, "brightness_scale": 1.06,
                "color_activity": 0.88, "palette": "party_vivid",
            },
            "outro": {
                "motion_speed": 0.25, "travel_size": 0.50,
                "activity_density": 0.38, "brightness_scale": 0.64,
                "color_activity": 0.28, "palette": "midnight_teal",
            },
        }
        profile = profiles.get(self._structure_energy)
        if profile is None:
            return None, 0.0
        energy_axis = (self._structure_resolution.get("axes") or {}).get(
            "energy"
        ) or {}
        source = str(energy_axis.get("source") or "").casefold()
        provenance = energy_axis.get("provenance") or {}
        provenance_source = (
            str(provenance.get("source") or "").casefold()
            if isinstance(provenance, dict) else ""
        )
        if "operator" in provenance_source or "operator" in source:
            authority = 0.98
        elif "cached" in source or "teacher" in source:
            authority = 0.88
        elif "student" in source:
            authority = 0.74
        elif "live_analyzer" in source:
            authority = 0.35
        else:
            # Generated rehearsal must preserve the exact Motion Studio
            # parameters being auditioned.
            authority = 0.0
        strength = clamp(
            authority * max(0.35, self._structure_confidence), 0.0, 1.0
        )
        if self._structure_energy in {"build", "drop"}:
            strength = clamp(
                strength + 0.12 * self._structure_boundary_probability,
                0.0,
                1.0,
            )
        return profile, strength

    def _structure_strobe_eligible(self) -> bool:
        if self._rehearsal_step is not None:
            return True
        return bool(
            self._structure_energy in {"build", "drop"}
            and (
                self._structure_confidence >= 0.45
                or self._structure_boundary_probability >= 0.65
            )
        )

    def _latched_color_for(
        self,
        lane: str,
        fixture_id: str,
        decision: PerformanceDecision,
        palette_bias: float,
        color_activity: float,
        *,
        role: str = "base",
    ) -> tuple[float, float, float]:
        """Return the fixture's held color for the current cue lease."""

        key = f"{lane}:{fixture_id}:{role}"
        if key not in self._latched_colors or lane in self._boundary_changed_lanes:
            self._latched_colors[key] = expression_rgb(
                decision, palette_bias, color_activity
            )
        return self._latched_colors[key]

    def step(self, observation: MusicalObservation) -> RuntimeFrame:
        self._boundary_changed_lanes = set()
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
            mover_step = rehearsal_step
            center_step = rehearsal_step
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
            mover_step = (
                self._active_choreography_steps["movers"]
                if self._choreography_planner is not None
                else choreography_step
            )
            center_step = (
                self._active_choreography_steps["center"]
                if self._choreography_planner is not None
                else choreography_step
            )
            decision = replace(
                decision,
                routine=routine,
                reason=f"{decision.reason} Phrase routine: {routine.replace('_', ' ')}.",
            )
            if (
                choreography_step is not None
                and _step_is_overall(choreography_step)
            ):
                reference_fixture_id = (
                    self.fixtures[0].fixture_id
                    if self.fixtures
                    else self.auxiliary_fixtures[0].fixture_id
                    if self.auxiliary_fixtures
                    else "overall"
                )
                overall_feedback = self._characteristics_feedback_for(
                    reference_fixture_id
                )
                resolved_overall_step = replace(
                    choreography_step,
                    beat_sync=clamp(
                        choreography_step.beat_sync
                        + 0.3 * overall_feedback.beat_sync,
                        0.0,
                        1.0,
                    ),
                    cue_timing=clamp(
                        choreography_step.cue_timing
                        + 0.3 * overall_feedback.cue_timing,
                        0.0,
                        1.0,
                    ),
                )
                decision = _apply_choreography_step(
                    decision,
                    resolved_overall_step,
                    beat_pulse=observation.beat_pulse,
                    step_elapsed_beats=self._active_step_elapsed_beats["movers"],
                )
            for lane in self._boundary_changed_lanes:
                self._lane_handoff_started_s[lane] = observation.timestamp_s
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
        effective_outputs: list[EffectiveCueOutput] = []

        for index, fixture in enumerate(self.fixtures):
            choreography_step = mover_step
            applies_to_fixture = _step_applies_to_fixture(
                choreography_step,
                fixture.fixture_id,
                is_mover=True,
            )
            feedback = self._characteristics_feedback_for(fixture.fixture_id)
            resolved_step = (
                replace(
                    choreography_step,
                    beat_sync=clamp(
                        choreography_step.beat_sync
                        + 0.3 * feedback.beat_sync,
                        0.0,
                        1.0,
                    ),
                    cue_timing=clamp(
                        choreography_step.cue_timing
                        + 0.3 * feedback.cue_timing,
                        0.0,
                        1.0,
                    ),
                )
                if choreography_step is not None and applies_to_fixture
                else choreography_step
            )
            target_decision = decision
            fixture_output_decision = output_decision
            if (
                choreography_step is not None
                and not _step_is_overall(choreography_step)
                and applies_to_fixture
            ):
                target_decision = _apply_choreography_step(
                    target_decision,
                    resolved_step,
                    beat_pulse=observation.beat_pulse,
                    step_elapsed_beats=self._active_step_elapsed_beats["movers"],
                )
                fixture_output_decision = _apply_choreography_step(
                    fixture_output_decision,
                    resolved_step,
                    beat_pulse=observation.beat_pulse,
                    step_elapsed_beats=self._active_step_elapsed_beats["movers"],
                )
            target = self._target_for_fixture(
                target_decision,
                fixture,
                index,
                observation,
            )
            previous = self._previous.get(fixture.fixture_id)
            effective = self._effective_cue_output(
                lane="movers",
                fixture_id=fixture.fixture_id,
                decision=fixture_output_decision,
                step=choreography_step,
                feedback=feedback,
                applies=applies_to_fixture,
            )
            if idle_amount >= 1.0:
                effective = replace(
                    effective,
                    routine="hold",
                    motion_speed=0.0,
                    travel_size=0.0,
                    activity_density=0.0,
                    strobe_enabled=False,
                    strobe_rate=0.0,
                )
            effective_outputs.append(effective)
            fixture_decision = replace(
                fixture_output_decision,
                brightness=effective.brightness,
                palette_hint=effective.palette,
                routine=(
                    fixture_output_decision.routine
                    if applies_to_fixture
                    else "breathe"
                ),
            )
            # Beat/onset energy already drives color, gesture, and explicitly
            # authored chase/strobe routines. Smooth the ordinary mover dimmer
            # so a per-frame beat component cannot look like an unintended
            # strobe during a continuous figure eight, sweep, or nod.
            deliberate_beam_gate = fixture_decision.routine in {
                "opposing_chase",
                "blackout_accent",
            }
            fixture_decision = replace(
                fixture_decision,
                brightness=self._smoothed_mover_brightness(
                    fixture.fixture_id,
                    fixture_decision.brightness,
                    elapsed,
                    immediate=(
                        deliberate_beam_gate
                        or rehearsal_step is not None
                    ),
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
            latched_rgb = self._latched_color_for(
                "movers",
                fixture.fixture_id,
                fixture_decision,
                feedback.palette + (
                    0.70 * (1 if index % 2 == 0 else -1)
                    if fixture_decision.routine == "opposing_chase"
                    else 0.0
                ),
                effective.color_activity,
            )
            try:
                calibration_override = self._calibration_overrides.get(fixture.fixture_id)
                rehearsal_inactive = (
                    rehearsal_step is not None
                    and self._rehearsal_isolate
                    and not applies_to_fixture
                )
                if rehearsal_inactive:
                    calibration = fixture.calibration
                    hold_pan = (
                        previous[0]
                        if previous is not None
                        else (calibration.pan_min_deg + calibration.pan_max_deg)
                        * 0.5
                    )
                    hold_tilt = (
                        previous[1]
                        if previous is not None
                        else (calibration.tilt_min_deg + calibration.tilt_max_deg)
                        * 0.5
                    )
                    direction = self.targeting.direction_for_angles(
                        fixture, hold_pan, hold_tilt
                    )
                    solution = TargetingSolution(
                        fixture_id=fixture.fixture_id,
                        target=fixture.position_m + direction * 5.0,
                        pan_deg=hold_pan,
                        tilt_deg=hold_tilt,
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
                elif observation.loudness >= 0.02:
                    # Authored mover routines already live inside the captured
                    # pan/tilt envelope. An unrelated room target must not be
                    # solved first: if that target is unreachable, the former
                    # implementation omitted this fixture's entire DMX frame
                    # and resumed it later, visibly interrupting every path.
                    calibration = fixture.calibration
                    seed_pan = (
                        previous[0]
                        if previous is not None
                        else (calibration.pan_min_deg + calibration.pan_max_deg)
                        * 0.5
                    )
                    seed_tilt = (
                        previous[1]
                        if previous is not None
                        else (calibration.tilt_min_deg + calibration.tilt_max_deg)
                        * 0.5
                    )
                    direction = self.targeting.direction_for_angles(
                        fixture, seed_pan, seed_tilt
                    )
                    solution = TargetingSolution(
                        fixture_id=fixture.fixture_id,
                        target=fixture.position_m + direction * 5.0,
                        pan_deg=seed_pan,
                        tilt_deg=seed_tilt,
                        distance_m=5.0,
                        movement_cost_deg=0.0,
                        aim_error_deg=0.0,
                        branch="performance-seed",
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
                        effective,
                    )
                    handoff_start = self._lane_handoff_started_s.get("movers")
                    if previous is not None and handoff_start is not None:
                        handoff_seconds = max(
                            0.25, 240.0 / (observation.bpm or 120.0)
                        )
                        handoff_alpha = clamp(
                            (observation.timestamp_s - handoff_start)
                            / handoff_seconds,
                            0.0,
                            1.0,
                        )
                        if handoff_alpha < 1.0:
                            solution = replace(
                                solution,
                                pan_deg=previous[0]
                                + (solution.pan_deg - previous[0]) * handoff_alpha,
                                tilt_deg=previous[1]
                                + (solution.tilt_deg - previous[1]) * handoff_alpha,
                            )
                        else:
                            self._lane_handoff_started_s.pop("movers", None)
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
                    strobe_feedback=feedback.strobe_enabled,
                    strobe_rate_feedback=feedback.strobe_rate,
                    choreography_strobe=effective.strobe_rate,
                    choreography_strobe_enabled=effective.strobe_enabled,
                    palette_bias=feedback.palette,
                    color_activity=effective.color_activity,
                    enabled=not rehearsal_inactive,
                    fixture_index=index,
                    fixture_count=len(self.fixtures),
                    chase_beat_position=(
                        self._continuous_motion_path_beat(
                            observation,
                            (0.5 + effective.motion_speed)
                            * (
                                0.82
                                + 0.18 * effective.activity_density
                            ),
                        )
                    ),
                    latched_rgb=latched_rgb,
                    strobe_dmx_override=self._operator_strobe_dmx["movers"],
                )
                if calibration_override is not None:
                    # The profile's speed channel is intentionally overridden
                    # only while the operator is in calibration mode.
                    speed_channel = fixture.address + 5 - 1
                    frame.set_channel(fixture.universe, speed_channel, round(calibration_override["speed"]))
            except UnreachableTargetError as error:
                warnings.append(str(error))

        for fixture in self.auxiliary_fixtures:
            choreography_step = center_step
            applies_to_fixture = _step_applies_to_fixture(
                choreography_step,
                fixture.fixture_id,
                is_mover=False,
            )
            feedback = self._characteristics_feedback_for(fixture.fixture_id)
            resolved_step = (
                replace(
                    choreography_step,
                    beat_sync=clamp(
                        choreography_step.beat_sync
                        + 0.3 * feedback.beat_sync,
                        0.0,
                        1.0,
                    ),
                    cue_timing=clamp(
                        choreography_step.cue_timing
                        + 0.3 * feedback.cue_timing,
                        0.0,
                        1.0,
                    ),
                )
                if choreography_step is not None and applies_to_fixture
                else choreography_step
            )
            fixture_output_decision = output_decision
            if (
                choreography_step is not None
                and not _step_is_overall(choreography_step)
                and applies_to_fixture
            ):
                fixture_output_decision = _apply_choreography_step(
                    fixture_output_decision,
                    resolved_step,
                    beat_pulse=observation.beat_pulse,
                    step_elapsed_beats=self._active_step_elapsed_beats["center"],
                )
            effective = self._effective_cue_output(
                lane="center",
                fixture_id=fixture.fixture_id,
                decision=fixture_output_decision,
                step=choreography_step,
                feedback=feedback,
                applies=applies_to_fixture,
            )
            center_tuning = self._center_motion_tunings.get(
                fixture_output_decision.routine
            )
            if (
                applies_to_fixture
                and center_tuning is not None
                and center_tuning.strobe_level > 0.0
                and feedback.strobe_enabled > -0.60
                and self._structure_strobe_eligible()
            ):
                effective = replace(
                    effective,
                    strobe_enabled=True,
                    strobe_rate=clamp(
                        center_tuning.strobe_level
                        + 0.5 * feedback.strobe_rate,
                        0.06,
                        1.0,
                    ),
                )
            if idle_amount >= 1.0:
                effective = replace(
                    effective,
                    routine="parked",
                    motion_speed=0.0,
                    travel_size=0.0,
                    activity_density=0.0,
                    brightness=24.0 / 255.0,
                    strobe_enabled=False,
                    strobe_rate=0.0,
                )
            effective_outputs.append(effective)
            fixture_decision = replace(
                fixture_output_decision,
                brightness=effective.brightness,
                palette_hint=effective.palette,
                routine=(
                    fixture_output_decision.routine
                    if applies_to_fixture
                    else "breathe"
                ),
            )
            latched_rgb = self._latched_color_for(
                "center",
                fixture.fixture_id,
                fixture_decision,
                feedback.palette,
                effective.color_activity,
            )
            latched_secondary_rgb = self._latched_color_for(
                "center",
                fixture.fixture_id,
                fixture_decision,
                feedback.palette + 0.70,
                effective.color_activity,
                role="secondary",
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
                motion_feedback=feedback.activity_density,
                motion_speed=effective.motion_speed,
                side_arm_speed_feedback=effective.side_arm_speed,
                travel_size=effective.travel_size,
                activity_density=effective.activity_density,
                strobe_feedback=feedback.strobe_enabled,
                strobe_rate_feedback=feedback.strobe_rate,
                choreography_strobe=effective.strobe_rate,
                choreography_strobe_enabled=effective.strobe_enabled,
                palette_bias=feedback.palette,
                color_activity=effective.color_activity,
                enabled=not (
                    rehearsal_step is not None
                    and self._rehearsal_isolate
                    and not applies_to_fixture
                ),
                motion_tuning=self._center_motion_tunings.get(
                    fixture_decision.routine
                ),
                motion_beat_position=self._continuous_motion_path_beat(
                    observation,
                    (0.5 + effective.motion_speed)
                    * (0.82 + 0.18 * effective.activity_density),
                    lane="center",
                ),
                latched_rgb=latched_rgb,
                latched_secondary_rgb=latched_secondary_rgb,
                strobe_dmx_override=self._operator_strobe_dmx["center"],
            )

        self.output.send(frame)
        transmitted_frame = getattr(
            self.output, "last_transmitted_frame", frame
        )
        if not isinstance(transmitted_frame, DMXFrame):
            transmitted_frame = frame
        self._record_dmx_history(
            transmitted_frame,
            solutions,
            beat=self._motion_phase,
        )
        self._effective_outputs = {
            item.fixture_id: item for item in effective_outputs
        }
        self._last_timestamp_s = observation.timestamp_s
        return RuntimeFrame(
            decision=decision,
            solutions=tuple(solutions),
            dmx=frame,
            warnings=tuple(warnings),
            effective_outputs=tuple(effective_outputs),
        )

    def _recent_dmx_history(
        self,
        lane: str,
        *,
        limit: int = 240,
        maximum_samples: int = 32,
    ) -> tuple[DmxHistorySample, ...]:
        """Return a time-spanning, compact physical-output window.

        Raw fixture state is sampled at the show rate. Persisting every sample
        in every reversible feedback event would make simultaneous listener
        bursts grow and serialize a multi-megabyte model under the Python GIL.
        Even spacing preserves the complete four-to-eight-second trajectory
        while bounding one lane's durable evidence to 32 semantic points.
        """

        prefix = f"{lane}:"
        rows = [
            sample
            for sample in reversed(self._dmx_history)
            if sample.fixture_scope.startswith(prefix)
        ][:limit]
        rows.reverse()
        if len(rows) <= maximum_samples:
            return tuple(rows)
        indexes = {
            round(index * (len(rows) - 1) / (maximum_samples - 1))
            for index in range(maximum_samples)
        }
        return tuple(rows[index] for index in sorted(indexes))

    def _record_dmx_history(
        self,
        frame: DMXFrame,
        solutions: Iterable[TargetingSolution],
        *,
        beat: float,
    ) -> None:
        """Decode the effective fixture frame into learner-safe semantics."""

        by_fixture = {item.fixture_id: item for item in solutions}
        for fixture in self.fixtures:
            profile = party_parrot_profile(fixture.profile_key)
            channels = {} if profile is None else profile.channels

            def value(name: str, fallback: int | None = None) -> int:
                relative = channels.get(name, fallback)
                if relative is None:
                    return 0
                return frame.get_channel(
                    fixture.universe, fixture.address + relative - 1
                )

            solution = by_fixture.get(fixture.fixture_id)
            calibration = fixture.calibration
            pan = tilt = None
            if solution is not None:
                pan = clamp(
                    (solution.pan_deg - calibration.pan_min_deg)
                    / (calibration.pan_max_deg - calibration.pan_min_deg),
                    0.0,
                    1.0,
                )
                tilt = clamp(
                    (solution.tilt_deg - calibration.tilt_min_deg)
                    / (calibration.tilt_max_deg - calibration.tilt_min_deg),
                    0.0,
                    1.0,
                )
            red, green, blue, white = (
                value("red"), value("green"), value("blue"), value("white")
            )
            self._dmx_history.append(DmxHistorySample(
                beat=max(0.0, beat),
                fixture_scope=f"movers:{fixture.fixture_id}",
                dimmer=value("dimmer", fixture.dimmer_channel) / 255.0,
                pan=pan,
                tilt=tilt,
                strobe=value("strobe") / 255.0,
                color=(
                    min(1.0, (red + white) / 255.0),
                    min(1.0, (green + white) / 255.0),
                    min(1.0, (blue + white) / 255.0),
                ),
            ))

        for fixture in self.auxiliary_fixtures:
            profile = party_parrot_profile(fixture.profile_key)
            if profile is None:
                continue
            channels = profile.channels

            def value(name: str) -> int:
                relative = channels.get(name)
                if relative is None:
                    return 0
                return frame.get_channel(
                    fixture.universe, fixture.address + relative - 1
                )

            ball = tuple(value(f"magic_ball_{name}") for name in (
                "red", "green", "blue", "white"
            ))
            arms = tuple(value(f"arm_beams_{name}") for name in (
                "red", "green", "blue", "white"
            ))
            combined = tuple(max(ball[index], arms[index]) for index in range(4))
            self._dmx_history.append(DmxHistorySample(
                beat=max(0.0, beat),
                fixture_scope=f"center:{fixture.fixture_id}",
                dimmer=value("master_dimmer") / 255.0,
                strobe=value("strobe") / 255.0,
                color=(
                    min(1.0, (combined[0] + combined[3]) / 255.0),
                    min(1.0, (combined[1] + combined[3]) / 255.0),
                    min(1.0, (combined[2] + combined[3]) / 255.0),
                ),
                auxiliary_motion=(
                    value("body_rotation") / 255.0,
                    value("arm_1_motor") / 255.0,
                    value("arm_2_motor") / 255.0,
                ),
            ))

    def _smoothed_mover_brightness(
        self,
        fixture_id: str,
        requested: float,
        elapsed_s: float | None,
        *,
        immediate: bool,
    ) -> float:
        requested = clamp(requested, 0.0, 1.0)
        previous = self._mover_brightness.get(fixture_id)
        if previous is None or elapsed_s is None or immediate:
            resolved = requested
        else:
            # A real energy lift reads promptly, while a single analysis-frame
            # beat spike is reduced to a small continuous intensity accent.
            time_constant = 0.75 if requested > previous else 1.00
            alpha = 1.0 - math.exp(
                -clamp(elapsed_s, 0.0, 0.5) / time_constant
            )
            resolved = previous + (requested - previous) * alpha
        resolved = clamp(resolved, 0.0, 1.0)
        self._mover_brightness[fixture_id] = resolved
        return resolved

    def _update_audio_idle(self, observation: MusicalObservation) -> float:
        """Fade active effects into Party Parrot's quiet/rest state."""

        # The control resolver declares physical silence only after its own
        # 550 ms confirmation window. Once that authoritative state arrives,
        # do not add a second 1.8-second motor tail: park the compound fixture
        # immediately and make the effective-output trace agree with reality.
        if observation.section == "silence":
            self._audio_quiet_since_s = observation.timestamp_s
            self._audio_idle_amount = 1.0
            return 1.0
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
        effective: EffectiveCueOutput,
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
        # Speed and density change future velocity. They must never multiply
        # the accumulated song clock, which would rewrite the path's history
        # and teleport a fixture whenever an output characteristic changed.
        speed_multiplier = (
            (0.5 + effective.motion_speed)
            * (0.82 + 0.18 * effective.activity_density)
        )
        beat_position = self._continuous_motion_path_beat(
            observation, speed_multiplier
        )
        if self._rehearsal_step is not None:
            if self._rehearsal_phase_origin_s is None:
                self._rehearsal_phase_origin_s = observation.timestamp_s
            size = self._rehearsal_size * effective.travel_size
        else:
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
            automatic_size = clamp(
                0.16 + 0.72 * state.energy + 0.22 * state.motion
                + structural_motion * self._structure_confidence
                + transition_expansion,
                0.12,
                1.0,
            )
            # Energy decides how much motion the music calls for; the travel
            # axis decides how much of the calibrated room envelope to use.
            # Blending the two keeps soft sections restrained while giving an
            # explicit movement command visible authority. The former formula
            # attenuated travel twice and made positive input hard to see.
            size = clamp(
                0.55 * automatic_size
                + 0.45 * effective.travel_size,
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

    def _continuous_motion_beat(
        self, observation: MusicalObservation
    ) -> float:
        """Advance mover phase without multiplying time by a changing BPM.

        ``timestamp * current_bpm`` changes the entire history whenever the
        tempo estimate moves, so even a small BPM correction teleports every
        routine to another point on its curve. This clock integrates tempo,
        limits tempo slew, and gently follows the audio beat phase instead.
        Calling it more than once for one observation is idempotent, which is
        required because both movers share this clock.
        """

        timestamp = float(observation.timestamp_s)
        if self._motion_clock_s is None or timestamp < self._motion_clock_s:
            self._motion_clock_s = timestamp
            self._motion_clock_bpm = clamp(
                float(observation.bpm or 120.0), 40.0, 240.0
            )
            self._motion_phase = (
                (observation.bar_phase % 1.0) * 4.0
                if observation.beat_confidence >= 0.20
                else 0.0
            )
            return self._motion_phase
        if timestamp == self._motion_clock_s:
            return self._motion_phase

        elapsed = max(0.0, timestamp - self._motion_clock_s)
        self._motion_clock_s = timestamp
        requested_bpm = clamp(
            float(observation.bpm or self._motion_clock_bpm), 40.0, 240.0
        )
        # Tempo may legitimately change, but an analyzer correction must alter
        # velocity rather than position. Twelve BPM per second is responsive
        # without transmitting frame-to-frame estimator wobble to the motors.
        maximum_change = 12.0 * elapsed
        self._motion_clock_bpm += clamp(
            requested_bpm - self._motion_clock_bpm,
            -maximum_change,
            maximum_change,
        )
        self._motion_phase += elapsed * self._motion_clock_bpm / 60.0

        if observation.beat_confidence >= 0.35:
            measured_bar_beat = (observation.bar_phase % 1.0) * 4.0
            current_bar_beat = self._motion_phase % 4.0
            phase_error = (
                (measured_bar_beat - current_bar_beat + 2.0) % 4.0
            ) - 2.0
            # Correct at no more than 0.12 beat/second. This keeps long-term
            # beat alignment without turning a phase reacquisition into a
            # visible direction change.
            correction_limit = 0.12 * elapsed
            correction = clamp(
                phase_error * 0.10 * observation.beat_confidence,
                -correction_limit,
                correction_limit,
            )
            self._motion_phase += correction
        return self._motion_phase

    def _continuous_motion_path_beat(
        self,
        observation: MusicalObservation,
        speed_multiplier: float,
        *,
        lane: str = "movers",
    ) -> float:
        """Integrate routine velocity on top of the continuous audio clock."""

        if lane not in CHOREOGRAPHY_LANES:
            raise ValueError(f"unknown motion-clock lane {lane!r}")
        source_phase = self._continuous_motion_beat(observation)
        previous_source = self._lane_motion_path_source_phase[lane]
        if previous_source is None:
            self._lane_motion_path_source_phase[lane] = source_phase
            self._lane_motion_path_phase[lane] = source_phase
            if lane == "movers":
                self._motion_path_source_phase = source_phase
                self._motion_path_phase = source_phase
            return self._lane_motion_path_phase[lane]
        source_delta = max(
            0.0, source_phase - previous_source
        )
        self._lane_motion_path_source_phase[lane] = source_phase
        self._lane_motion_path_phase[lane] += source_delta * clamp(
            speed_multiplier, 0.25, 2.0
        )
        if lane == "movers":
            self._motion_path_source_phase = source_phase
            self._motion_path_phase = self._lane_motion_path_phase[lane]
        return self._lane_motion_path_phase[lane]

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
    *,
    lane: str = "movers",
    development_index: int = 0,
) -> tuple[ChoreographySequence, ...]:
    """Return phrase-level candidates ordered by the current musical role."""

    if lane not in CHOREOGRAPHY_LANES:
        raise ValueError(f"unsupported choreography lane: {lane}")

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
                    fixture_scope=lane,
                    routine=routine,
                    intensity=context.energy,
                    strobe=0.0,
                )
                for start, duration, routine in steps
            ),
        )

    if lane == "movers":
        calm = sequence("movers-calm-arc", (0.0, 16.0, "breathe"))
        calm_development = sequence(
            "movers-calm-open",
            (0.0, 8.0, "fan_sweep"),
            (8.0, 8.0, "breathe"),
        )
        groove = sequence(
            "movers-groove-exchange",
            (0.0, 8.0, "figure_eight"),
            (8.0, 8.0, "opposing_chase"),
        )
        groove_development = sequence(
            "movers-groove-wide-answer",
            (0.0, 8.0, "fan_sweep"),
            (8.0, 4.0, "beat_nod"),
            (12.0, 4.0, "figure_eight"),
        )
        build = sequence(
            "movers-build-and-answer",
            (0.0, 8.0, "fan_sweep"),
            (8.0, 4.0, "beat_nod"),
            (12.0, 4.0, "opposing_chase"),
        )
        build_development = sequence(
            "movers-build-figure-rise",
            (0.0, 8.0, "figure_eight"),
            (8.0, 4.0, "fan_sweep"),
            (12.0, 4.0, "beat_nod"),
        )
        release = sequence(
            "movers-release-counterplay",
            (0.0, 4.0, "opposing_chase"),
            (4.0, 4.0, "beat_nod"),
            (8.0, 8.0, "counter_rotate"),
        )
        release_development = sequence(
            "movers-release-wide-trade",
            (0.0, 8.0, "counter_rotate"),
            (8.0, 4.0, "opposing_chase"),
            (12.0, 4.0, "beat_nod"),
        )
    else:
        calm = sequence("center-calm-arc", (0.0, 16.0, "breathe"))
        calm_development = sequence(
            "center-calm-open",
            (0.0, 8.0, "fan_sweep"),
            (8.0, 8.0, "breathe"),
        )
        groove = sequence(
            "center-groove-counterplay",
            (0.0, 8.0, "counter_rotate"),
            (8.0, 8.0, "fan_sweep"),
        )
        groove_development = sequence(
            "center-groove-answer",
            (0.0, 8.0, "opposing_chase"),
            (8.0, 8.0, "counter_rotate"),
        )
        build = sequence(
            "center-build-chase",
            (0.0, 8.0, "opposing_chase"),
            (8.0, 8.0, "counter_rotate"),
        )
        build_development = sequence(
            "center-build-fan-answer",
            (0.0, 8.0, "fan_sweep"),
            (8.0, 4.0, "opposing_chase"),
            (12.0, 4.0, "counter_rotate"),
        )
        release = sequence(
            "center-release-exchange",
            (0.0, 4.0, "beat_nod"),
            (4.0, 4.0, "opposing_chase"),
            (8.0, 8.0, "counter_rotate"),
        )
        release_development = sequence(
            "center-release-counter-chase",
            (0.0, 8.0, "counter_rotate"),
            (8.0, 4.0, "opposing_chase"),
            (12.0, 4.0, "beat_nod"),
        )

    def developed(
        primary: ChoreographySequence,
        secondary: ChoreographySequence,
    ) -> tuple[ChoreographySequence, ...]:
        # Development advances only when the caller opens a new boundary. The
        # active planner lease is never touched merely because time or feedback
        # changed between boundaries.
        return (
            (primary, secondary)
            if development_index % 2 == 0
            else (secondary, primary)
        )

    energy_label = context.energy_label
    if (
        energy_label in {"silence", "low", "breakdown"}
        or context.energy < 0.24
    ):
        return developed(calm, calm_development)
    if energy_label == "build":
        return developed(build, build_development)
    if energy_label in {"drop", "release"} or context.energy >= 0.70:
        return developed(release, release_development)
    return developed(groove, groove_development)


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


def _sequence_step_at(
    sequence: ChoreographySequence, beat_in_phrase: float
) -> ChoreographyStep:
    matching = [
        step for step in sequence.steps
        if step.start_beat <= beat_in_phrase
        < step.start_beat + step.duration_beats
    ]
    if matching:
        return matching[-1]
    prior = [
        step for step in sequence.steps
        if step.start_beat <= beat_in_phrase
    ]
    return prior[-1] if prior else sequence.steps[0]


def _step_snapshot(step: ChoreographyStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "start_beat": step.start_beat,
        "duration_beats": step.duration_beats,
        "fixture_scope": step.fixture_scope,
        "routine": step.routine,
        "intensity": step.intensity,
        "palette": step.palette,
        "strobe": step.strobe,
        "beat_sync": step.beat_sync,
        "motion_speed": step.motion_speed,
        "travel_size": step.travel_size,
        "activity_density": step.activity_density,
        "brightness": step.brightness,
        "strobe_enabled": step.strobe_enabled,
        "strobe_rate": step.strobe_rate,
        "cue_timing": step.cue_timing,
        "entry_behavior": step.entry_behavior,
        "exit_behavior": step.exit_behavior,
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
    *,
    beat_pulse: float = 0.0,
    step_elapsed_beats: float = 0.0,
) -> PerformanceDecision:
    scale = 0.62 + 0.76 * step.intensity
    synchronized_accent = 1.0 + (
        0.18 * clamp(beat_pulse, 0.0, 1.0) * step.beat_sync
    )
    entry_scale = 1.0
    transition_beats = 0.25 + 0.75 * step.cue_timing
    if step.entry_behavior == "soft" and step_elapsed_beats < transition_beats:
        entry_phase = clamp(step_elapsed_beats / transition_beats, 0.0, 1.0)
        entry_scale = 0.45 + 0.55 * entry_phase
    elif step.entry_behavior == "accent" and step_elapsed_beats < transition_beats:
        synchronized_accent += (
            0.22
            * (1.0 - clamp(step_elapsed_beats / transition_beats, 0.0, 1.0))
        )
    remaining = step.duration_beats - step_elapsed_beats
    exit_scale = 1.0
    if remaining < transition_beats:
        phase = clamp(remaining / transition_beats, 0.0, 1.0)
        if step.exit_behavior == "blackout":
            exit_scale = phase
        elif step.exit_behavior == "crossfade":
            exit_scale = 0.62 + 0.38 * phase
        elif step.exit_behavior == "resolve":
            exit_scale = 0.82 + 0.18 * phase
        # "hold" deliberately keeps the final state unchanged until the
        # next boundary; "resolve" gently settles into the following step.
    final_scale = scale * entry_scale * exit_scale
    brightness_scale = (
        step.intensity if step.brightness is None else step.brightness
    )
    return replace(
        decision,
        routine=step.routine,
        expression=replace(
            decision.expression,
            motion=clamp(
                decision.expression.motion
                * final_scale
                * synchronized_accent,
                0.0,
                1.0,
            ),
        ),
        brightness=clamp(
            decision.brightness
            * (0.62 + 0.76 * brightness_scale)
            * entry_scale
            * exit_scale
            * synchronized_accent,
            0.0,
            1.0,
        ),
        palette_hint=step.palette or decision.palette_hint,
    )
