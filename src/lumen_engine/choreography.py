"""Dependency-light learning for complete semantic choreography sequences.

This module deliberately does not write DMX or control the live runtime.
``SequencePreferenceModel`` learns ranking weights from captured performances,
while ``BoundarySequencePlanner`` keeps the chosen sequence leased until the
caller announces a new musical boundary. Feedback can therefore update the
next choice without interrupting motion already in progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
import threading
import time
from typing import Any, Iterable


def _unit(value: float, name: str) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return resolved


def _signed_unit(value: float, name: str) -> float:
    resolved = float(value)
    if not -1.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return resolved


@dataclass(frozen=True, slots=True)
class ChoreographyStep:
    """One beat-addressed semantic action in a complete routine."""

    start_beat: float
    duration_beats: float
    fixture_scope: str
    routine: str
    intensity: float = 1.0
    palette: str | None = None
    strobe: float = 0.0
    beat_sync: float = 1.0
    # Independent choreography characteristics.  The neutral defaults retain
    # the pre-v6 motion while giving feedback one literal axis to adjust.
    motion_speed: float = 0.5
    travel_size: float = 1.0
    activity_density: float = 1.0
    brightness: float | None = None
    strobe_enabled: bool | None = None
    strobe_rate: float | None = None
    cue_timing: float = 1.0
    entry_behavior: str = "phrase_boundary"
    exit_behavior: str = "resolve"

    def __post_init__(self) -> None:
        if self.start_beat < 0:
            raise ValueError("start_beat must be non-negative")
        if self.duration_beats <= 0:
            raise ValueError("duration_beats must be positive")
        if not self.fixture_scope.strip():
            raise ValueError("fixture_scope must not be empty")
        if not self.routine.strip():
            raise ValueError("routine must not be empty")
        _unit(self.intensity, "intensity")
        _unit(self.strobe, "strobe")
        _unit(self.beat_sync, "beat_sync")
        _unit(self.motion_speed, "motion_speed")
        _unit(self.travel_size, "travel_size")
        _unit(self.activity_density, "activity_density")
        if self.brightness is not None:
            _unit(self.brightness, "brightness")
        if self.strobe_rate is not None:
            _unit(self.strobe_rate, "strobe_rate")
        _unit(self.cue_timing, "cue_timing")
        if self.entry_behavior not in {"phrase_boundary", "soft", "accent"}:
            raise ValueError("unknown choreography entry behavior")
        if self.exit_behavior not in {"resolve", "hold", "blackout", "crossfade"}:
            raise ValueError("unknown choreography exit behavior")

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_beat": self.start_beat,
            "duration_beats": self.duration_beats,
            "fixture_scope": self.fixture_scope,
            "routine": self.routine,
            "intensity": self.intensity,
            "palette": self.palette,
            "strobe": self.strobe,
            "beat_sync": self.beat_sync,
            "motion_speed": self.motion_speed,
            "travel_size": self.travel_size,
            "activity_density": self.activity_density,
            "brightness": self.brightness,
            "strobe_enabled": self.strobe_enabled,
            "strobe_rate": self.strobe_rate,
            "cue_timing": self.cue_timing,
            "entry_behavior": self.entry_behavior,
            "exit_behavior": self.exit_behavior,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChoreographyStep:
        raw_strobe = value.get("strobe", 0.0)
        if isinstance(raw_strobe, dict):
            legacy_strobe = float(raw_strobe.get("rate", 0.0))
            nested_strobe_enabled: bool | None = bool(
                raw_strobe.get("enabled", legacy_strobe > 0.0)
            )
        else:
            legacy_strobe = float(raw_strobe)
            nested_strobe_enabled = None
        parameters = value.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        return cls(
            start_beat=float(value["start_beat"]),
            duration_beats=float(value["duration_beats"]),
            fixture_scope=str(value["fixture_scope"]),
            routine=str(value["routine"]),
            intensity=float(value.get("intensity", 1.0)),
            palette=(
                str(value["palette"])
                if value.get("palette") is not None
                else None
            ),
            strobe=legacy_strobe,
            beat_sync=float(value.get("beat_sync", parameters.get("beat_sync", 1.0))),
            motion_speed=float(value.get("motion_speed", parameters.get("motion_speed", 0.5))),
            travel_size=float(value.get("travel_size", parameters.get("travel_size", 1.0))),
            activity_density=float(value.get("activity_density", parameters.get("activity_density", 1.0))),
            brightness=(
                None if value.get("brightness") is None
                else float(value["brightness"])
            ),
            strobe_enabled=(
                nested_strobe_enabled
                if value.get("strobe_enabled") is None
                else bool(value["strobe_enabled"])
            ),
            strobe_rate=(
                legacy_strobe
                if isinstance(raw_strobe, dict)
                else None if value.get("strobe_rate") is None
                else float(value["strobe_rate"])
            ),
            cue_timing=float(value.get("cue_timing", parameters.get("cue_timing", 1.0))),
            entry_behavior=str(value.get("entry_behavior", "phrase_boundary")),
            exit_behavior=str(value.get("exit_behavior", "resolve")),
        )


@dataclass(frozen=True, slots=True)
class ChoreographySequence:
    """A complete, reusable phrase-level choreography candidate."""

    sequence_id: str
    steps: tuple[ChoreographyStep, ...]
    source: str = "authored"
    base_priority: float = 0.0

    def __post_init__(self) -> None:
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must not be empty")
        if not self.steps:
            raise ValueError("a choreography sequence requires at least one step")
        previous_start = -1.0
        for step in self.steps:
            if step.start_beat < previous_start:
                raise ValueError("choreography steps must be ordered by start beat")
            previous_start = step.start_beat
        if not -1.0 <= self.base_priority <= 1.0:
            raise ValueError("base_priority must be in [-1, 1]")

    @property
    def end_beat(self) -> float:
        return max(
            step.start_beat + step.duration_beats for step in self.steps
        )

    @property
    def semantic_signature(self) -> str:
        material = "\x1e".join(
            "\x1f".join(
                (
                    f"{step.start_beat:.3f}",
                    f"{step.duration_beats:.3f}",
                    step.fixture_scope.casefold(),
                    step.routine.casefold(),
                    f"{step.intensity:.3f}",
                    (step.palette or "").casefold(),
                    f"{step.strobe:.3f}",
                    f"{step.beat_sync:.3f}",
                    f"{step.motion_speed:.3f}",
                    f"{step.travel_size:.3f}",
                    f"{step.activity_density:.3f}",
                    "" if step.brightness is None else f"{step.brightness:.3f}",
                    "" if step.strobe_enabled is None else str(step.strobe_enabled),
                    "" if step.strobe_rate is None else f"{step.strobe_rate:.3f}",
                    f"{step.cue_timing:.3f}",
                    step.entry_behavior,
                    step.exit_behavior,
                )
            )
            for step in self.steps
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "steps": [step.as_dict() for step in self.steps],
            "source": self.source,
            "base_priority": self.base_priority,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChoreographySequence:
        return cls(
            sequence_id=str(value["sequence_id"]),
            steps=tuple(
                ChoreographyStep.from_dict(item) for item in value["steps"]
            ),
            source=str(value.get("source", "authored")),
            base_priority=float(value.get("base_priority", 0.0)),
        )


CHOREOGRAPHY_LANES = ("movers", "center")


def _normalized_lane(lane: str | None) -> str | None:
    if lane is None:
        return None
    normalized = str(lane).casefold().strip()
    if normalized not in CHOREOGRAPHY_LANES:
        raise ValueError(f"unsupported choreography lane: {lane}")
    return normalized


def _normalized_learning_lifetime(lifetime: str | None) -> str:
    normalized = str(lifetime or "global").casefold().strip()
    if normalized not in {"cue", "song", "artist", "global"}:
        raise ValueError(f"unsupported feedback lifetime: {lifetime}")
    return normalized


def _features_for_lifetime(
    features: dict[str, float], lifetime: str
) -> dict[str, float]:
    """Keep only the namespace authorized by one feedback lifetime."""

    normalized = _normalized_learning_lifetime(lifetime)
    if normalized == "global":
        return {
            name: value
            for name, value in features.items()
            if "context:" not in name
        }
    marker = f"context:{normalized}:"
    return {
        name: value for name, value in features.items() if marker in name
    }


def choreography_lanes_for_scope(scope: str) -> tuple[str, ...]:
    """Resolve a persisted semantic scope to the live lanes it can address."""

    normalized = str(scope).casefold().strip()
    if normalized in {"overall", "all", "rig", "whole_rig"}:
        return CHOREOGRAPHY_LANES
    if normalized in {"movers", "group:movers"}:
        return ("movers",)
    if normalized in {
        "center", "multi_effect", "multi-effect", "group:center"
    }:
        return ("center",)
    return ()


def sequence_for_lane(
    sequence: ChoreographySequence, lane: str
) -> ChoreographySequence | None:
    """Project a mixed or whole-rig sequence onto one independent lane.

    Whole-rig steps are copied into each lane. Steps scoped to another lane are
    omitted, so learned center actions can never enter the movers candidate
    pool (and vice versa).
    """

    normalized_lane = str(lane).casefold().strip()
    if normalized_lane not in CHOREOGRAPHY_LANES:
        raise ValueError(f"unsupported choreography lane: {lane}")
    steps = tuple(
        ChoreographyStep(
            start_beat=step.start_beat,
            duration_beats=step.duration_beats,
            fixture_scope=normalized_lane,
            routine=step.routine,
            intensity=step.intensity,
            palette=step.palette,
            strobe=step.strobe,
            beat_sync=step.beat_sync,
            motion_speed=step.motion_speed,
            travel_size=step.travel_size,
            activity_density=step.activity_density,
            brightness=step.brightness,
            strobe_enabled=step.strobe_enabled,
            strobe_rate=step.strobe_rate,
            cue_timing=step.cue_timing,
            entry_behavior=step.entry_behavior,
            exit_behavior=step.exit_behavior,
        )
        for step in sequence.steps
        if normalized_lane in choreography_lanes_for_scope(
            step.fixture_scope
        )
    )
    if not steps:
        return None
    return ChoreographySequence(
        sequence_id=f"{sequence.sequence_id}@{normalized_lane}",
        steps=steps,
        source=sequence.source,
        base_priority=sequence.base_priority,
    )


@dataclass(frozen=True, slots=True)
class MusicalContext:
    """Normalized music state used to retrieve choreography preferences."""

    functional_label: str = "unknown"
    energy_label: str = "unknown"
    content_label: str = "unknown"
    energy: float = 0.0
    motion: float = 0.0
    tension: float = 0.0
    bpm: float | None = None
    song_key: str | None = None
    artist: str | None = None
    cue_key: str | None = None

    def __post_init__(self) -> None:
        _unit(self.energy, "energy")
        _unit(self.motion, "motion")
        _unit(self.tension, "tension")
        if self.bpm is not None and self.bpm <= 0:
            raise ValueError("bpm must be positive")


@dataclass(frozen=True, slots=True)
class DmxHistorySample:
    """Normalized semantic channels sampled from actual fixture output.

    Pan, tilt, auxiliary motion, color, dimmer, and strobe are all normalized
    to ``[0, 1]`` by the capture adapter. This keeps fixture channel layouts out
    of the learner while retaining evidence of what the rig physically did.
    """

    beat: float
    fixture_scope: str
    dimmer: float
    pan: float | None = None
    tilt: float | None = None
    strobe: float = 0.0
    color: tuple[float, float, float] | None = None
    auxiliary_motion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.fixture_scope.strip():
            raise ValueError("fixture_scope must not be empty")
        _unit(self.dimmer, "dimmer")
        _unit(self.strobe, "strobe")
        if self.pan is not None:
            _unit(self.pan, "pan")
        if self.tilt is not None:
            _unit(self.tilt, "tilt")
        if self.color is not None:
            for component in self.color:
                _unit(component, "color component")
        for position in self.auxiliary_motion:
            _unit(position, "auxiliary motion")


@dataclass(frozen=True, slots=True)
class DmxHistorySummary:
    sample_count: int = 0
    intensity: float = 0.0
    movement: float = 0.0
    strobe: float = 0.0
    color_change: float = 0.0
    blackout_ratio: float = 0.0


def summarize_dmx_history(
    samples: Iterable[DmxHistorySample],
) -> DmxHistorySummary:
    """Describe physical output without assuming a fixture profile."""

    rows = tuple(samples)
    if not rows:
        return DmxHistorySummary()
    last_by_scope: dict[str, DmxHistorySample] = {}
    movement_changes: list[float] = []
    color_changes: list[float] = []
    for row in rows:
        previous = last_by_scope.get(row.fixture_scope)
        if previous is not None:
            components: list[float] = []
            if row.pan is not None and previous.pan is not None:
                components.append(abs(row.pan - previous.pan))
            if row.tilt is not None and previous.tilt is not None:
                components.append(abs(row.tilt - previous.tilt))
            for current, old in zip(
                row.auxiliary_motion, previous.auxiliary_motion
            ):
                components.append(abs(current - old))
            if components:
                beat_delta = max(0.25, abs(row.beat - previous.beat))
                movement_changes.append(
                    min(1.0, sum(components) / len(components) / beat_delta)
                )
            if row.color is not None and previous.color is not None:
                squared = sum(
                    (current - old) ** 2
                    for current, old in zip(row.color, previous.color)
                )
                color_changes.append(min(1.0, math.sqrt(squared / 3.0)))
        last_by_scope[row.fixture_scope] = row
    return DmxHistorySummary(
        sample_count=len(rows),
        intensity=sum(row.dimmer for row in rows) / len(rows),
        movement=(
            sum(movement_changes) / len(movement_changes)
            if movement_changes
            else 0.0
        ),
        strobe=sum(row.strobe for row in rows) / len(rows),
        color_change=(
            sum(color_changes) / len(color_changes)
            if color_changes
            else 0.0
        ),
        blackout_ratio=(
            sum(1 for row in rows if row.dimmer <= 0.02) / len(rows)
        ),
    )


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    """One feedback label, optionally representing repeated simultaneous hits."""

    label: str
    value: float = 1.0
    urgency: float = 0.5
    occurrences: int = 1
    scope: str = "overall"
    fixture_id: str | None = None
    created_unix_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("feedback label must not be empty")
        _signed_unit(self.value, "feedback value")
        _unit(self.urgency, "feedback urgency")
        if self.occurrences < 1:
            raise ValueError("feedback occurrences must be positive")


@dataclass(frozen=True, slots=True)
class CapturedChoreographyExample:
    context: MusicalContext
    performed: ChoreographySequence
    feedback: tuple[FeedbackSignal, ...] = ()
    preferred: ChoreographySequence | None = None
    preferred_strength: float = 1.0
    dmx_history: tuple[DmxHistorySample, ...] = ()

    def __post_init__(self) -> None:
        _unit(self.preferred_strength, "preferred_strength")
        if not self.feedback and self.preferred is None:
            raise ValueError(
                "a learning example needs feedback or a preferred sequence"
            )


@dataclass(frozen=True, slots=True)
class RankedSequence:
    sequence: ChoreographySequence
    score: float
    confidence: float
    learned_support: float
    model_revision: int


@dataclass(frozen=True, slots=True)
class LearningReceipt:
    model_revision: int
    effective_strength: float
    urgency: float
    feedback_occurrences: int
    preferred_sequence_learned: bool
    dmx_summary: DmxHistorySummary
    output_action: str = "none"


@dataclass(frozen=True, slots=True)
class PlanSelection:
    boundary_id: str
    sequence: ChoreographySequence
    changed: bool
    held_for_boundary: bool
    model_revision: int
    confidence: float
    reason: str


_MOVEMENT_ROUTINES = {
    "hold_position": 0.02,
    "hold": 0.03,
    "blackout_accent": 0.08,
    "breathe": 0.15,
    "fan_sweep": 0.58,
    "beat_nod": 0.62,
    "figure_eight": 0.78,
    "counter_rotate": 0.82,
    "opposing_chase": 0.88,
}

# These profiles describe different properties of a routine, rather than one
# ambiguous "movement" score.  They are intentionally semantic: fixture
# geometry and channel encoding remain deterministic elsewhere.
_ROUTINE_SPEED = {
    "hold_position": 0.02, "hold": 0.02, "blackout_accent": 0.18,
    "breathe": 0.18, "fan_sweep": 0.42, "beat_nod": 0.70,
    "figure_eight": 0.55, "counter_rotate": 0.60,
    "opposing_chase": 0.76,
}
_ROUTINE_TRAVEL = {
    "hold_position": 0.02, "hold": 0.02, "blackout_accent": 0.12,
    "breathe": 0.30, "beat_nod": 0.38, "opposing_chase": 0.62,
    "figure_eight": 0.78, "counter_rotate": 0.84, "fan_sweep": 0.92,
}
_ROUTINE_ACTIVITY = {
    "hold_position": 0.02, "hold": 0.02, "blackout_accent": 0.22,
    "breathe": 0.28, "fan_sweep": 0.58, "figure_eight": 0.72,
    "beat_nod": 0.80, "counter_rotate": 0.82,
    "opposing_chase": 0.94,
}
_PALETTE_WARMTH = {
    "auto": 0.50,
    "party_vivid": 0.50,
    "saturated_jewel": 0.50,
    "magenta_blue": 0.35,
    "blue_violet": 0.15,
    "midnight_teal": 0.10,
    "cool": 0.10,
    "cyan_violet": 0.12,
    "warm": 0.90,
    "red_amber": 0.95,
}

_DIRECTIONAL_FEEDBACK: dict[str, tuple[str, float]] = {
    "increase_movement": ("travel_size", 1.0),
    "more_movement": ("travel_size", 1.0),
    "decrease_movement": ("travel_size", -1.0),
    "less_movement": ("travel_size", -1.0),
    "pick_it_up": ("activity_density", 1.0),
    "not_busy_enough": ("activity_density", 1.0),
    "calm_down": ("activity_density", -1.0),
    "too_busy": ("activity_density", -1.0),
    "faster": ("motion_speed", 1.0),
    "faster_side_arms": ("motion_speed", 1.0),
    "slower": ("motion_speed", -1.0),
    "slower_side_arms": ("motion_speed", -1.0),
    "brighter": ("brightness", 1.0),
    "more_intensity": ("brightness", 1.0),
    "too_dim": ("brightness", 1.0),
    "dimmer": ("brightness", -1.0),
    "too_bright": ("brightness", -1.0),
    "strobe": ("strobe_enabled", 1.0),
    "more_strobe": ("strobe_enabled", 1.0),
    "no_strobe": ("strobe_enabled", -1.0),
    "no_strobes": ("strobe_enabled", -1.0),
    "less_strobe": ("strobe_enabled", -1.0),
    "less_flashing": ("strobe_enabled", -1.0),
    "faster_strobe": ("strobe_rate", 1.0),
    "slower_strobe": ("strobe_rate", -1.0),
    "more_variety": ("variety", 1.0),
    "too_repetitive": ("variety", 1.0),
    "less_variety": ("variety", -1.0),
    "too_varied": ("variety", -1.0),
    "more_blackout": ("blackout", 1.0),
    "less_blackout": ("blackout", -1.0),
    "better_beat_sync": ("beat_sync", 1.0),
    "great_timing": ("cue_timing", 1.0),
    "good_timing": ("cue_timing", 1.0),
    "timing_on_point": ("cue_timing", 1.0),
    "bad_timing": ("cue_timing", -1.0),
    "poor_timing": ("cue_timing", -1.0),
    "cool_blue_purple": ("palette_warmth", -1.0),
    "warmer_color": ("palette_warmth", 1.0),
}

_POSITIVE_FEEDBACK = {
    "good",
    "good_motion",
    "good_timing",
    "timing_on_point",
    "on_point",
    "keep_current",
    "more_like_this",
    "liked_this",
    "hold_this",
    "great_transition",
    "great_timing",
    "perfect_motion",
    "movement_good",
    "color_good",
}

_NEGATIVE_FEEDBACK = {
    "bad",
    "not_good",
    "wrong",
    "wrong_color",
    "poor_timing",
    "bad_timing",
    "wrong_look",
}


def _is_developed_sequence(sequence: ChoreographySequence) -> bool:
    """Return whether a learned item is complete enough to enter Live.

    A Preferred Action button is a useful routine-level correction, but the
    generated one-step placeholder is not an authored choreography.  Its
    features still train the ranker, allowing complete sequences containing
    that routine to gain support, while the placeholder itself cannot replace
    a developed multi-step phrase.
    """

    return not (
        sequence.source == "operator_preferred_action"
        and len(sequence.steps) < 2
    )


class SequencePreferenceModel:
    """Online linear ranker over whole semantic choreography sequences."""

    STATE_VERSION = 7

    def __init__(
        self,
        *,
        learning_rate: float = 0.35,
        feedback_half_life_days: float = 120.0,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if feedback_half_life_days <= 0:
            raise ValueError("feedback_half_life_days must be positive")
        self.learning_rate = float(learning_rate)
        self.feedback_half_life_days = float(feedback_half_life_days)
        self._weights: dict[str, float] = {}
        self._evidence: dict[str, float] = {}
        self._learned_sequences: dict[str, ChoreographySequence] = {}
        self._learned_sequence_scopes: dict[str, set[str]] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._revision = 0
        self._lock = threading.RLock()
        # The live planner reads an immutable published generation. Feedback
        # updates may rebuild a large reversible event under `_lock`, but a
        # phrase boundary must never wait for that work to finish.
        self._published_weights: dict[str, float] = {}
        self._published_evidence: dict[str, float] = {}
        self._published_revision = 0
        self._published_learned_sequences: tuple[
            ChoreographySequence, ...
        ] = ()
        self._published_learned_sequence_scopes: dict[
            str, frozenset[str]
        ] = {}

    @property
    def revision(self) -> int:
        return self._published_revision

    def learn(
        self,
        example: CapturedChoreographyExample,
        *,
        now_unix_ms: int | None = None,
        event_id: str | None = None,
        lane: str | None = None,
        lifetime: str = "global",
    ) -> LearningReceipt:
        """Update future rankings only; never return or change a live plan."""

        normalized_event_id = (
            str(event_id).strip() if event_id is not None else None
        )
        if event_id is not None and not normalized_event_id:
            raise ValueError("event_id must not be empty")
        now_ms = (
            int(time.time() * 1000)
            if now_unix_ms is None
            else int(now_unix_ms)
        )
        normalized_lane = _normalized_lane(lane)
        normalized_lifetime = _normalized_learning_lifetime(lifetime)
        dmx_summary = summarize_dmx_history(example.dmx_history)
        context_tokens = _context_tokens(example.context, dmx_summary)
        masses = [
            _feedback_mass(
                signal,
                now_unix_ms=now_ms,
                half_life_days=self.feedback_half_life_days,
            )
            for signal in example.feedback
        ]
        total_mass = sum(masses)
        urgency = 1.0 - math.exp(-total_mass)
        effective_strength = min(3.0, total_mass)
        if example.preferred is not None and effective_strength <= 0:
            effective_strength = example.preferred_strength
        performed_features = _joint_features(
            context_tokens, example.performed, lane=normalized_lane
        )
        performed_features = _features_for_lifetime(
            performed_features, normalized_lifetime
        )
        with self._lock:
            if normalized_event_id is not None:
                # Keep event replacement atomic for the real-time ranker.  It
                # must never observe a temporary model with this event absent.
                self.forget(normalized_event_id)
            if example.preferred is not None:
                if _is_developed_sequence(example.preferred):
                    self._remember_learned_candidate(
                        example.preferred,
                        example.context,
                        normalized_lifetime,
                    )
                preferred_features = _joint_features(
                    context_tokens, example.preferred, lane=normalized_lane
                )
                preferred_features = _features_for_lifetime(
                    preferred_features, normalized_lifetime
                )
                pair_strength = (
                    self.learning_rate
                    * max(0.1, effective_strength)
                    * example.preferred_strength
                )
                self._update_vector(preferred_features, pair_strength)
                self._update_vector(performed_features, -pair_strength)
            for signal, mass in zip(example.feedback, masses):
                if mass <= 0:
                    continue
                label = signal.label.casefold().strip().replace(" ", "_")
                update_strength = self.learning_rate * mass
                directional = _DIRECTIONAL_FEEDBACK.get(label)
                if directional is not None:
                    metric, direction = directional
                    self._update_metric(
                        context_tokens,
                        metric,
                        direction * update_strength,
                        lane=normalized_lane,
                        lifetime=normalized_lifetime,
                    )
                    # A correction means the exact performed sequence did not
                    # satisfy the operator, even when no alternative was named.
                    if label in _POSITIVE_FEEDBACK:
                        self._update_vector(
                            performed_features, 0.55 * update_strength
                        )
                    else:
                        self._update_vector(
                            performed_features, -0.20 * update_strength
                        )
                elif label in _POSITIVE_FEEDBACK:
                    self._update_vector(
                        performed_features, 0.55 * update_strength
                    )
                elif label in _NEGATIVE_FEEDBACK:
                    self._update_vector(
                        performed_features, -0.55 * update_strength
                    )
                elif signal.value > 0:
                    self._update_vector(
                        performed_features, 0.20 * update_strength
                    )
                elif signal.value < 0:
                    self._update_vector(
                        performed_features, -0.20 * update_strength
                    )
            self._revision += 1
            revision = self._revision
            if normalized_event_id is not None:
                self._events[normalized_event_id] = {
                    "example": _example_as_dict(example),
                    "now_unix_ms": now_ms,
                    "lane": normalized_lane,
                    "lifetime": normalized_lifetime,
                }
            self._publish_locked()
        return LearningReceipt(
            model_revision=revision,
            effective_strength=effective_strength,
            urgency=urgency,
            feedback_occurrences=sum(
                signal.occurrences for signal in example.feedback
            ),
            preferred_sequence_learned=example.preferred is not None,
            dmx_summary=dmx_summary,
        )

    def forget(self, event_id: str) -> bool:
        """Remove one identified update without replaying the event history."""

        normalized = str(event_id).strip()
        if not normalized:
            raise ValueError("event_id must not be empty")
        with self._lock:
            matching_event_ids = self._matching_event_ids_locked(normalized)
            if not matching_event_ids:
                return False
            previous_revision = self._revision
            for matching_event_id in matching_event_ids:
                self._subtract_event_locked(
                    self._events[matching_event_id]
                )
                del self._events[matching_event_id]
            self._rebuild_learned_sequences_locked()
            self._revision = previous_revision + 1
            self._publish_locked()
        return True

    def revise_feedback_event(
        self,
        event_id: str,
        *,
        occurrences: int,
        urgency: float,
    ) -> bool:
        """Resize one consensus window and replay it without duplicate rows."""

        if occurrences < 1:
            raise ValueError("occurrences must be positive")
        _unit(urgency, "urgency")
        normalized = str(event_id).strip()
        if not normalized:
            raise ValueError("event_id must not be empty")
        with self._lock:
            matching = self._matching_event_ids_locked(normalized)
            if not matching:
                return False
            previous_revision = self._revision
            revised: list[tuple[str, dict[str, Any]]] = []
            for key in matching:
                stored = self._events[key]
                self._subtract_event_locked(stored)
                example = _example_from_dict(stored["example"])
                revised.append((key, {
                    "example": _example_as_dict(replace(
                        example,
                        feedback=tuple(
                            replace(
                                signal,
                                occurrences=occurrences,
                                urgency=urgency,
                            )
                            for signal in example.feedback
                        ),
                    )),
                    "now_unix_ms": int(stored["now_unix_ms"]),
                    "lane": stored.get("lane"),
                    "lifetime": stored.get("lifetime", "global"),
                }))
                del self._events[key]
            self._rebuild_learned_sequences_locked()
            for key, value in revised:
                self.learn(
                    _example_from_dict(value["example"]),
                    now_unix_ms=int(value["now_unix_ms"]),
                    event_id=key,
                    lane=value.get("lane"),
                    lifetime=value.get("lifetime", "global"),
                )
            self._revision = previous_revision + 1
            self._publish_locked()
            return True

    def _matching_event_ids_locked(self, normalized: str) -> list[str]:
        if normalized in self._events:
            return [normalized]
        return [
            key for key in self._events
            if key.startswith(f"{normalized}:")
            and key.rsplit(":", 1)[-1] in CHOREOGRAPHY_LANES
        ]

    def _subtract_event_locked(self, stored: dict[str, Any]) -> None:
        """Subtract one additive event using its original timestamp/context."""

        contribution = SequencePreferenceModel(
            learning_rate=self.learning_rate,
            feedback_half_life_days=self.feedback_half_life_days,
        )
        contribution.learn(
            _example_from_dict(stored["example"]),
            now_unix_ms=int(stored["now_unix_ms"]),
            lane=stored.get("lane"),
            lifetime=stored.get("lifetime", "global"),
        )
        for name, value in contribution._weights.items():
            updated = self._weights.get(name, 0.0) - value
            if abs(updated) <= 1e-12:
                self._weights.pop(name, None)
            else:
                self._weights[name] = updated
        for name, value in contribution._evidence.items():
            updated = self._evidence.get(name, 0.0) - value
            if updated <= 1e-12:
                self._evidence.pop(name, None)
            else:
                self._evidence[name] = updated

    def _rebuild_learned_sequences_locked(self) -> None:
        self._learned_sequences.clear()
        self._learned_sequence_scopes.clear()
        for stored in self._events.values():
            example = _example_from_dict(stored["example"])
            preferred = example.preferred
            if preferred is not None and _is_developed_sequence(preferred):
                self._remember_learned_candidate(
                    preferred,
                    example.context,
                    stored.get("lifetime", "global"),
                )

    @staticmethod
    def _candidate_scope(
        context: MusicalContext, lifetime: str
    ) -> str | None:
        normalized = _normalized_learning_lifetime(lifetime)
        if normalized == "global":
            return "global"
        value = {
            "cue": context.cue_key,
            "song": context.song_key,
            "artist": context.artist,
        }[normalized]
        if value is None or not str(value).strip():
            return None
        return f"{normalized}:{str(value).casefold().strip()}"

    def _remember_learned_candidate(
        self,
        sequence: ChoreographySequence,
        context: MusicalContext,
        lifetime: str,
    ) -> None:
        scope = self._candidate_scope(context, lifetime)
        if scope is None:
            return
        signature = sequence.semantic_signature
        self._learned_sequences[signature] = sequence
        self._learned_sequence_scopes.setdefault(signature, set()).add(scope)

    def learned_candidates(
        self, context: MusicalContext | None = None
    ) -> tuple[ChoreographySequence, ...]:
        """Return operator-authored sequences for future boundary choices."""

        if context is None:
            return self._published_learned_sequences
        permitted = {
            value
            for lifetime in ("cue", "song", "artist", "global")
            if (value := self._candidate_scope(context, lifetime)) is not None
        }
        return tuple(
            sequence
            for sequence in self._published_learned_sequences
            if self._published_learned_sequence_scopes.get(
                sequence.semantic_signature, frozenset({"global"})
            ).intersection(permitted)
        )

    def rank(
        self,
        context: MusicalContext,
        candidates: Iterable[ChoreographySequence],
        *,
        recent_dmx: Iterable[DmxHistorySample] = (),
        lane: str | None = None,
    ) -> tuple[RankedSequence, ...]:
        """Rank candidates without retaining or activating a selection."""

        choices = tuple(candidates)
        if not choices:
            raise ValueError("at least one choreography candidate is required")
        normalized_lane = _normalized_lane(lane)
        dmx_summary = summarize_dmx_history(recent_dmx)
        context_tokens = _context_tokens(context, dmx_summary)
        weights = self._published_weights
        evidence = self._published_evidence
        revision = self._published_revision
        ranked: list[tuple[int, RankedSequence]] = []
        for index, sequence in enumerate(choices):
            features = _joint_features(
                context_tokens, sequence, lane=normalized_lane
            )
            if normalized_lane is not None:
                # Models written before choreography lanes existed contain
                # unscoped weights.  Consult those weights as a read-only
                # fallback while keeping every new lane-scoped update inside
                # its own namespace.
                features.update(_joint_features(
                    context_tokens, sequence, lane=None
                ))
            learned_score = sum(
                weights.get(name, 0.0) * value
                for name, value in features.items()
            )
            support = sum(
                evidence.get(name, 0.0) * abs(value)
                for name, value in features.items()
            )
            confidence = 1.0 - math.exp(-support / 6.0)
            ranked.append(
                (
                    index,
                    RankedSequence(
                        sequence=sequence,
                        score=learned_score + sequence.base_priority * 0.15,
                        confidence=confidence,
                        learned_support=support,
                        model_revision=revision,
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[1].score, item[0]))
        return tuple(item[1] for item in ranked)

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": "lumen_sequence_preference_model",
                "version": self.STATE_VERSION,
                "learning_rate": self.learning_rate,
                "feedback_half_life_days": self.feedback_half_life_days,
                "revision": self._revision,
                "weights": dict(sorted(self._weights.items())),
                "evidence": dict(sorted(self._evidence.items())),
                "learned_sequences": [
                    sequence.as_dict()
                    for sequence in self.learned_candidates()
                ],
                "learned_sequence_scopes": {
                    key: sorted(value)
                    for key, value in sorted(
                        self._learned_sequence_scopes.items()
                    )
                },
                "events": {
                    key: value
                    for key, value in sorted(self._events.items())
                },
            }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any]
    ) -> SequencePreferenceModel:
        if state.get("format") != "lumen_sequence_preference_model":
            raise ValueError("not a Lumen sequence preference model")
        version = int(state.get("version", 0))
        if version not in {1, 2, 3, 4, 5, 6, cls.STATE_VERSION}:
            raise ValueError("unsupported sequence preference model version")
        model = cls(
            learning_rate=float(state["learning_rate"]),
            feedback_half_life_days=float(
                state["feedback_half_life_days"]
            ),
        )
        weights = state.get("weights")
        evidence = state.get("evidence")
        if not isinstance(weights, dict) or not isinstance(evidence, dict):
            raise ValueError("model weights and evidence must be objects")
        model._weights = {
            str(name): float(value) for name, value in weights.items()
        }
        model._evidence = {
            str(name): max(0.0, float(value))
            for name, value in evidence.items()
        }
        learned_sequences = state.get("learned_sequences", ())
        if not isinstance(learned_sequences, (list, tuple)):
            raise ValueError("learned_sequences must be an array")
        for value in learned_sequences:
            if not isinstance(value, dict):
                raise ValueError("learned sequence entries must be objects")
            sequence = ChoreographySequence.from_dict(value)
            if _is_developed_sequence(sequence):
                model._learned_sequences[
                    sequence.semantic_signature
                ] = sequence
        raw_scopes = state.get("learned_sequence_scopes", {})
        if version >= 7 and not isinstance(raw_scopes, dict):
            raise ValueError("learned sequence scopes must be an object")
        for signature in model._learned_sequences:
            values = raw_scopes.get(signature, ()) if version >= 7 else ()
            if not isinstance(values, (list, tuple)):
                raise ValueError("learned sequence scope must be an array")
            normalized_scopes = {
                str(value).casefold().strip()
                for value in values if str(value).strip()
            }
            # Models through v6 had no lifetime vocabulary. Preserve their
            # behavior explicitly as legacy-global instead of silently
            # pretending those old weights were cue-local.
            model._learned_sequence_scopes[signature] = (
                normalized_scopes or {"global"}
            )
        events = state.get("events", {})
        if not isinstance(events, dict):
            raise ValueError("model events must be an object")
        parsed_events: list[tuple[str, dict[str, Any]]] = []
        rebuild_events = False
        for key, value in events.items():
            if not isinstance(value, dict):
                raise ValueError("model event entries must be objects")
            example = value.get("example")
            if not isinstance(example, dict):
                raise ValueError("model event example must be an object")
            parsed = _example_from_dict(example)
            # Version 3 allowed broad directional feedback such as
            # ``pick_it_up`` to smuggle in a specifically named routine. Only
            # explicit annotation events may carry a preferred sequence.
            if (
                version == 3
                and str(key).startswith("feedback:")
                and parsed.preferred is not None
                and parsed.preferred.source == "operator_preferred_action"
            ):
                parsed = replace(parsed, preferred=None)
                example = _example_as_dict(parsed)
                rebuild_events = True
            # Versions through 4 materialized brightness, palette, and strobe
            # corrections as complete candidate sequences. A characteristic
            # must score/modify choreography; it must never replace the
            # choreography lease or pin its underlying routine for minutes.
            if (
                parsed.preferred is not None
                and parsed.preferred.source
                == "operator_preferred_characteristic"
            ):
                parsed = replace(parsed, preferred=None)
                example = _example_as_dict(parsed)
                rebuild_events = True
            parsed_events.append((str(key), {
                "example": example,
                "now_unix_ms": int(value["now_unix_ms"]),
                "lane": _normalized_lane(value.get("lane")),
                "lifetime": (
                    _normalized_learning_lifetime(value.get("lifetime"))
                    if version >= 7 else "global"
                ),
            }))
        if rebuild_events and parsed_events:
            # Discard the polluted serialized weights and deterministically
            # rebuild from sanitized reversible events.
            model._weights.clear()
            model._evidence.clear()
            model._learned_sequences.clear()
            for key, value in parsed_events:
                model.learn(
                    _example_from_dict(value["example"]),
                    now_unix_ms=int(value["now_unix_ms"]),
                    event_id=key,
                    lane=value.get("lane"),
                    lifetime=value.get("lifetime", "global"),
                )
        else:
            model._events = dict(parsed_events)
        model._revision = max(
            model._revision,
            max(0, int(state.get("revision", 0))),
        )
        with model._lock:
            model._publish_locked()
        return model

    def _publish_locked(self) -> None:
        """Atomically expose one complete generation to the live ranker."""

        self._published_weights = dict(self._weights)
        self._published_evidence = dict(self._evidence)
        self._published_revision = self._revision
        self._published_learned_sequences = tuple(
            self._learned_sequences[key]
            for key in sorted(self._learned_sequences)
        )
        self._published_learned_sequence_scopes = {
            key: frozenset(value)
            for key, value in self._learned_sequence_scopes.items()
        }

    def _update_vector(
        self, features: dict[str, float], amount: float
    ) -> None:
        for name, value in features.items():
            delta = amount * value
            self._weights[name] = self._weights.get(name, 0.0) + delta
            self._evidence[name] = (
                self._evidence.get(name, 0.0) + abs(delta)
            )

    def _update_metric(
        self,
        context_tokens: tuple[str, ...],
        metric: str,
        amount: float,
        *,
        lane: str | None = None,
        lifetime: str = "global",
    ) -> None:
        normalized_lifetime = _normalized_learning_lifetime(lifetime)
        if normalized_lifetime == "global":
            names = [f"metric:{metric}"]
        else:
            prefix = f"{normalized_lifetime}:"
            names = [
                f"context:{token}|metric:{metric}"
                for token in context_tokens if token.startswith(prefix)
            ]
        if lane is not None:
            names = [f"lane:{lane}|{name}" for name in names]
        for name in names:
            self._weights[name] = self._weights.get(name, 0.0) + amount
            self._evidence[name] = (
                self._evidence.get(name, 0.0) + abs(amount)
            )


class BoundarySequencePlanner:
    """Hold one sequence until a phrase/section boundary changes."""

    def __init__(self, model: SequencePreferenceModel) -> None:
        self.model = model
        self._active: PlanSelection | None = None
        self._last_opening_routine: str | None = None
        self._opening_routine_streak = 0
        self._context_signature: tuple[str, str, str] | None = None
        self._lock = threading.RLock()

    @property
    def active(self) -> PlanSelection | None:
        with self._lock:
            return self._active

    def choose(
        self,
        *,
        boundary_id: str,
        context: MusicalContext,
        candidates: Iterable[ChoreographySequence],
        recent_dmx: Iterable[DmxHistorySample] = (),
        lane: str | None = None,
    ) -> PlanSelection:
        if not boundary_id.strip():
            raise ValueError("boundary_id must not be empty")
        with self._lock:
            if (
                self._active is not None
                and self._active.boundary_id == boundary_id
            ):
                return PlanSelection(
                    boundary_id=boundary_id,
                    sequence=self._active.sequence,
                    changed=False,
                    held_for_boundary=True,
                    model_revision=self.model.revision,
                    confidence=self._active.confidence,
                    reason=(
                        "Active choreography is leased until the next musical "
                        "boundary; feedback was retained for the next choice."
                    ),
                )
            ranked = self.model.rank(
                context, candidates, recent_dmx=recent_dmx, lane=lane
            )
            winner = ranked[0]
            context_signature = (
                context.functional_label.casefold(),
                context.energy_label.casefold(),
                context.content_label.casefold(),
            )
            if context_signature != self._context_signature:
                # A new musical role starts its own development arc. Repetition
                # limits must not make a breakdown inherit a release's streak.
                self._last_opening_routine = None
                self._opening_routine_streak = 0
                self._context_signature = context_signature
            opening_routine = winner.sequence.steps[0].routine.casefold()
            repetition_limited = False
            if (
                opening_routine == self._last_opening_routine
                and self._opening_routine_streak >= 2
                and not winner.sequence.source.startswith("operator_song_timeline")
            ):
                # Automatic and preference-ranked material gets at most two
                # consecutive phrases with the same opening routine. Exact
                # owner timeline placements remain authoritative, and a
                # one-candidate pool is necessarily allowed to continue.
                alternate = next(
                    (
                        row for row in ranked[1:]
                        if row.sequence.steps[0].routine.casefold()
                        != opening_routine
                    ),
                    None,
                )
                if alternate is not None:
                    winner = alternate
                    opening_routine = (
                        winner.sequence.steps[0].routine.casefold()
                    )
                    repetition_limited = True
            previous_id = (
                self._active.sequence.sequence_id
                if self._active is not None
                else None
            )
            selection = PlanSelection(
                boundary_id=boundary_id,
                sequence=winner.sequence,
                changed=winner.sequence.sequence_id != previous_id,
                held_for_boundary=False,
                model_revision=winner.model_revision,
                confidence=winner.confidence,
                reason=(
                    "Selected at a new musical boundary after the hard "
                    "routine repetition limit."
                    if repetition_limited
                    else "Selected at a new musical boundary from the complete "
                    "sequence preference ranking."
                ),
            )
            self._active = selection
            if opening_routine == self._last_opening_routine:
                self._opening_routine_streak += 1
            else:
                self._last_opening_routine = opening_routine
                self._opening_routine_streak = 1
            return selection

    def release(self) -> None:
        """Explicitly clear the lease when the live engine stops."""

        with self._lock:
            self._active = None
            self._last_opening_routine = None
            self._opening_routine_streak = 0
            self._context_signature = None


class ParallelBoundarySequencePlanner:
    """Two independent leases advanced by one shared musical boundary clock."""

    def __init__(self, model: SequencePreferenceModel) -> None:
        self.model = model
        self._planners = {
            lane: BoundarySequencePlanner(model)
            for lane in CHOREOGRAPHY_LANES
        }

    @property
    def active(self) -> PlanSelection | None:
        """Compatibility view: the movers lane is the primary rig decision."""

        return self._planners["movers"].active

    def active_for(self, lane: str) -> PlanSelection | None:
        return self._planners[_normalized_lane(lane) or "movers"].active

    def choose_lane(
        self,
        lane: str,
        *,
        boundary_id: str,
        context: MusicalContext,
        candidates: Iterable[ChoreographySequence],
        recent_dmx: Iterable[DmxHistorySample] = (),
    ) -> PlanSelection:
        normalized = _normalized_lane(lane)
        assert normalized is not None
        return self._planners[normalized].choose(
            boundary_id=boundary_id,
            context=context,
            candidates=candidates,
            recent_dmx=recent_dmx,
            lane=normalized,
        )

    def release(self) -> None:
        for planner in self._planners.values():
            planner.release()


def _feedback_mass(
    signal: FeedbackSignal,
    *,
    now_unix_ms: int,
    half_life_days: float,
) -> float:
    magnitude = abs(signal.value)
    # Repeated taps communicate urgency but are one correlated observation,
    # not N independent training examples. Log growth keeps ten phones (or
    # ten rapid taps) meaningful without letting them dominate indefinitely.
    repeated = (
        1.0 + math.log2(max(1, signal.occurrences))
    ) * (0.5 + 0.5 * signal.urgency)
    decay = 1.0
    if signal.created_unix_ms is not None:
        age_days = max(
            0.0,
            (now_unix_ms - signal.created_unix_ms) / 86_400_000.0,
        )
        decay = 0.5 ** (age_days / half_life_days)
    return magnitude * repeated * decay


def _context_tokens(
    context: MusicalContext,
    dmx: DmxHistorySummary,
) -> tuple[str, ...]:
    tokens = [
        f"functional:{context.functional_label.casefold()}",
        f"energy_label:{context.energy_label.casefold()}",
        f"content:{context.content_label.casefold()}",
        f"energy_bin:{min(4, int(context.energy * 5))}",
        f"motion_bin:{min(4, int(context.motion * 5))}",
        f"tension_bin:{min(4, int(context.tension * 5))}",
    ]
    if context.bpm is not None:
        tokens.append(f"bpm_bin:{int(context.bpm // 20) * 20}")
    if context.song_key:
        tokens.append(f"song:{context.song_key.casefold()}")
    if context.artist:
        tokens.append(f"artist:{context.artist.casefold().strip()}")
    if context.cue_key:
        tokens.append(f"cue:{context.cue_key.casefold().strip()}")
    if dmx.sample_count:
        tokens.extend(
            (
                f"dmx_movement_bin:{min(4, int(dmx.movement * 5))}",
                f"dmx_intensity_bin:{min(4, int(dmx.intensity * 5))}",
                f"dmx_strobe_bin:{min(4, int(dmx.strobe * 5))}",
            )
        )
    return tuple(tokens)


def _sequence_metrics(
    sequence: ChoreographySequence,
) -> dict[str, float]:
    total_duration = sum(step.duration_beats for step in sequence.steps)
    movement = sum(
        _MOVEMENT_ROUTINES.get(step.routine.casefold(), 0.5)
        * step.duration_beats
        * step.intensity
        for step in sequence.steps
    ) / total_duration
    intensity = sum(
        step.intensity * step.duration_beats for step in sequence.steps
    ) / total_duration
    motion_speed = sum(
        _ROUTINE_SPEED.get(step.routine.casefold(), 0.5)
        * (0.5 + step.motion_speed)
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    travel_size = sum(
        _ROUTINE_TRAVEL.get(step.routine.casefold(), 0.5)
        * step.travel_size
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    activity_density = sum(
        _ROUTINE_ACTIVITY.get(step.routine.casefold(), 0.5)
        * step.activity_density
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    brightness = sum(
        (step.intensity if step.brightness is None else step.brightness)
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    strobe_enabled = sum(
        float(
            step.strobe > 0.05
            if step.strobe_enabled is None
            else step.strobe_enabled
        )
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    strobe_rate = sum(
        (step.strobe if step.strobe_rate is None else step.strobe_rate)
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    beat_sync = sum(
        step.beat_sync * step.duration_beats for step in sequence.steps
    ) / total_duration
    blackout = sum(
        step.duration_beats
        for step in sequence.steps
        if step.routine.casefold() == "blackout_accent"
    ) / total_duration
    unique_routines = len(
        {step.routine.casefold() for step in sequence.steps}
    )
    transitions = sum(
        1
        for previous, current in zip(sequence.steps, sequence.steps[1:])
        if previous.routine.casefold() != current.routine.casefold()
    )
    variety = min(
        1.0,
        (
            unique_routines / max(1, len(sequence.steps))
            + transitions / max(1, len(sequence.steps) - 1)
        )
        / 2.0,
    )
    cue_timing = sum(
        step.cue_timing * step.duration_beats for step in sequence.steps
    ) / total_duration
    palette_warmth = sum(
        _PALETTE_WARMTH.get((step.palette or "auto").casefold(), 0.5)
        * step.duration_beats
        for step in sequence.steps
    ) / total_duration
    return {
        # Legacy aliases preserve already-learned v1-v5 weights. New feedback
        # is written only to the literal axes below.
        "movement": movement,
        "intensity": intensity,
        "strobe": strobe_rate,
        "motion_speed": min(1.0, motion_speed),
        "travel_size": min(1.0, travel_size),
        "activity_density": min(1.0, activity_density),
        "brightness": brightness,
        "strobe_enabled": strobe_enabled,
        "strobe_rate": strobe_rate,
        "beat_sync": beat_sync,
        "cue_timing": cue_timing,
        "palette_warmth": palette_warmth,
        "blackout": blackout,
        "variety": variety,
    }


def _joint_features(
    context_tokens: tuple[str, ...],
    sequence: ChoreographySequence,
    *,
    lane: str | None = None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    signature = sequence.semantic_signature
    metrics = _sequence_metrics(sequence)
    total_duration = sum(step.duration_beats for step in sequence.steps)
    features[f"sequence:{signature}"] = 1.0
    for step in sequence.steps:
        fraction = step.duration_beats / total_duration
        routine = step.routine.casefold()
        scope = step.fixture_scope.casefold()
        features[f"routine:{routine}"] = (
            features.get(f"routine:{routine}", 0.0) + fraction
        )
        features[f"scope:{scope}"] = (
            features.get(f"scope:{scope}", 0.0) + fraction
        )
        if step.palette:
            palette = step.palette.casefold()
            features[f"palette:{palette}"] = (
                features.get(f"palette:{palette}", 0.0) + fraction
            )
    for previous, current in zip(sequence.steps, sequence.steps[1:]):
        transition = (
            f"transition:{previous.routine.casefold()}>"
            f"{current.routine.casefold()}"
        )
        features[transition] = features.get(transition, 0.0) + 1.0
    for metric, value in metrics.items():
        features[f"metric:{metric}"] = value
    # Exact sequence and routine interactions let preferences specialize to a
    # song or section, while the global features above remain useful fallbacks.
    routine_features = {
        name: value
        for name, value in features.items()
        if name.startswith("routine:")
    }
    for token in context_tokens:
        features[f"context:{token}|sequence:{signature}"] = 1.0
        for name, value in routine_features.items():
            features[f"context:{token}|{name}"] = value
        for metric, value in metrics.items():
            features[f"context:{token}|metric:{metric}"] = value
    if lane is None:
        return features
    # Lane-namespaced weights prevent a mover correction from incidentally
    # teaching the center lane merely because both sequences use a similarly
    # named routine or have comparable movement metrics.
    lane_features = {
        f"lane:{lane}|{name}": value for name, value in features.items()
    }
    return lane_features


def _example_as_dict(
    example: CapturedChoreographyExample,
) -> dict[str, Any]:
    return {
        "context": asdict(example.context),
        "performed": example.performed.as_dict(),
        "feedback": [asdict(signal) for signal in example.feedback],
        "preferred": (
            example.preferred.as_dict()
            if example.preferred is not None
            else None
        ),
        "preferred_strength": example.preferred_strength,
        "dmx_history": [asdict(sample) for sample in example.dmx_history],
    }


def _example_from_dict(
    value: dict[str, Any],
) -> CapturedChoreographyExample:
    preferred = value.get("preferred")
    return CapturedChoreographyExample(
        context=MusicalContext(**dict(value["context"])),
        performed=ChoreographySequence.from_dict(
            dict(value["performed"])
        ),
        feedback=tuple(
            FeedbackSignal(**dict(signal))
            for signal in value.get("feedback", ())
        ),
        preferred=(
            ChoreographySequence.from_dict(dict(preferred))
            if isinstance(preferred, dict)
            else None
        ),
        preferred_strength=float(value.get("preferred_strength", 1.0)),
        dmx_history=tuple(
            DmxHistorySample(**dict(sample))
            for sample in value.get("dmx_history", ())
        ),
    )
