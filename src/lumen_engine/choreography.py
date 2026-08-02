"""Dependency-light learning for complete semantic choreography sequences.

This module deliberately does not write DMX or control the live runtime.
``SequencePreferenceModel`` learns ranking weights from captured performances,
while ``BoundarySequencePlanner`` keeps the chosen sequence leased until the
caller announces a new musical boundary. Feedback can therefore update the
next choice without interrupting motion already in progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChoreographyStep:
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
            strobe=float(value.get("strobe", 0.0)),
            beat_sync=float(value.get("beat_sync", 1.0)),
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

_DIRECTIONAL_FEEDBACK: dict[str, tuple[str, float]] = {
    "increase_movement": ("movement", 1.0),
    "more_movement": ("movement", 1.0),
    "pick_it_up": ("movement", 1.0),
    "not_busy_enough": ("movement", 1.0),
    "faster": ("movement", 1.0),
    "faster_side_arms": ("movement", 1.0),
    "decrease_movement": ("movement", -1.0),
    "less_movement": ("movement", -1.0),
    "calm_down": ("movement", -1.0),
    "too_busy": ("movement", -1.0),
    "slower": ("movement", -1.0),
    "slower_side_arms": ("movement", -1.0),
    "brighter": ("intensity", 1.0),
    "more_intensity": ("intensity", 1.0),
    "too_dim": ("intensity", 1.0),
    "dimmer": ("intensity", -1.0),
    "too_bright": ("intensity", -1.0),
    "strobe": ("strobe", 1.0),
    "more_strobe": ("strobe", 1.0),
    "faster_strobe": ("strobe", 1.0),
    "no_strobe": ("strobe", -1.0),
    "no_strobes": ("strobe", -1.0),
    "less_strobe": ("strobe", -1.0),
    "slower_strobe": ("strobe", -1.0),
    "more_variety": ("variety", 1.0),
    "too_repetitive": ("variety", 1.0),
    "less_variety": ("variety", -1.0),
    "too_varied": ("variety", -1.0),
    "more_blackout": ("blackout", 1.0),
    "less_blackout": ("blackout", -1.0),
    "better_beat_sync": ("beat_sync", 1.0),
    "great_timing": ("beat_sync", 1.0),
    "bad_timing": ("beat_sync", -1.0),
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


class SequencePreferenceModel:
    """Online linear ranker over whole semantic choreography sequences."""

    STATE_VERSION = 3

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
        self._events: dict[str, dict[str, Any]] = {}
        self._revision = 0
        self._lock = threading.RLock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def learn(
        self,
        example: CapturedChoreographyExample,
        *,
        now_unix_ms: int | None = None,
        event_id: str | None = None,
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
            context_tokens, example.performed
        )
        with self._lock:
            if normalized_event_id is not None:
                # Keep event replacement atomic for the real-time ranker.  It
                # must never observe a temporary model with this event absent.
                self.forget(normalized_event_id)
            if example.preferred is not None:
                self._learned_sequences[
                    example.preferred.semantic_signature
                ] = example.preferred
                preferred_features = _joint_features(
                    context_tokens, example.preferred
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
                    )
                    # A correction means the exact performed sequence did not
                    # satisfy the operator, even when no alternative was named.
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
                }
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
        """Remove one identified update and replay every remaining event."""

        normalized = str(event_id).strip()
        if not normalized:
            raise ValueError("event_id must not be empty")
        with self._lock:
            if normalized not in self._events:
                return False
            previous_revision = self._revision
            del self._events[normalized]
            retained = [
                (key, dict(value))
                for key, value in self._events.items()
            ]
            self._weights.clear()
            self._evidence.clear()
            self._learned_sequences.clear()
            self._events.clear()
            self._revision = 0
            # RLock permits the replay to use the normal validation/update
            # path while rankers remain excluded until the rebuild is whole.
            for key, value in retained:
                self.learn(
                    _example_from_dict(value["example"]),
                    now_unix_ms=int(value["now_unix_ms"]),
                    event_id=key,
                )
            self._revision = max(
                self._revision, previous_revision
            ) + 1
        return True

    def learned_candidates(self) -> tuple[ChoreographySequence, ...]:
        """Return operator-authored sequences for future boundary choices."""

        with self._lock:
            return tuple(
                self._learned_sequences[key]
                for key in sorted(self._learned_sequences)
            )

    def rank(
        self,
        context: MusicalContext,
        candidates: Iterable[ChoreographySequence],
        *,
        recent_dmx: Iterable[DmxHistorySample] = (),
    ) -> tuple[RankedSequence, ...]:
        """Rank candidates without retaining or activating a selection."""

        choices = tuple(candidates)
        if not choices:
            raise ValueError("at least one choreography candidate is required")
        dmx_summary = summarize_dmx_history(recent_dmx)
        context_tokens = _context_tokens(context, dmx_summary)
        with self._lock:
            weights = dict(self._weights)
            evidence = dict(self._evidence)
            revision = self._revision
        ranked: list[tuple[int, RankedSequence]] = []
        for index, sequence in enumerate(choices):
            features = _joint_features(context_tokens, sequence)
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
        if version not in {1, 2, cls.STATE_VERSION}:
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
            model._learned_sequences[
                sequence.semantic_signature
            ] = sequence
        events = state.get("events", {})
        if not isinstance(events, dict):
            raise ValueError("model events must be an object")
        for key, value in events.items():
            if not isinstance(value, dict):
                raise ValueError("model event entries must be objects")
            example = value.get("example")
            if not isinstance(example, dict):
                raise ValueError("model event example must be an object")
            _example_from_dict(example)
            model._events[str(key)] = {
                "example": example,
                "now_unix_ms": int(value["now_unix_ms"]),
            }
        model._revision = max(0, int(state.get("revision", 0)))
        return model

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
    ) -> None:
        names = [f"metric:{metric}"]
        names.extend(
            f"context:{token}|metric:{metric}" for token in context_tokens
        )
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
                context, candidates, recent_dmx=recent_dmx
            )
            winner = ranked[0]
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
                    "Selected at a new musical boundary from the complete "
                    "sequence preference ranking."
                ),
            )
            self._active = selection
            return selection

    def release(self) -> None:
        """Explicitly clear the lease when the live engine stops."""

        with self._lock:
            self._active = None


def _feedback_mass(
    signal: FeedbackSignal,
    *,
    now_unix_ms: int,
    half_life_days: float,
) -> float:
    magnitude = abs(signal.value)
    repeated = signal.occurrences * (0.5 + 0.5 * signal.urgency)
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
    strobe = sum(
        step.strobe * step.duration_beats for step in sequence.steps
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
    return {
        "movement": movement,
        "intensity": intensity,
        "strobe": strobe,
        "beat_sync": beat_sync,
        "blackout": blackout,
        "variety": variety,
    }


def _joint_features(
    context_tokens: tuple[str, ...],
    sequence: ChoreographySequence,
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
    return features


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
