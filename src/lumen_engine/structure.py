"""Dependency-free types for normalized musical-structure annotations.

The public structure datasets used by Lumen describe different things.  Some
name functional sections (verse/chorus), some name changes in energy
(build/drop), and some describe musical content (vocal/instrumental).  These
types deliberately keep those axes independent instead of forcing every source
label into one mutually-exclusive class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Iterable


class FunctionalSection(StrEnum):
    UNKNOWN = "unknown"
    SILENCE = "silence"
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    POST_CHORUS = "post_chorus"
    BRIDGE = "bridge"
    INTERLUDE = "interlude"
    TRANSITION = "transition"
    INSTRUMENTAL = "instrumental"
    SOLO = "solo"
    THEME = "theme"
    DEVELOPMENT = "development"
    OUTRO = "outro"


class TechnoSection(StrEnum):
    """Canonical sustained musical states used by Lumen's techno model."""

    SILENCE = "silence"
    INTRO = "intro"
    GROOVE = "groove"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"


CANONICAL_TECHNO_SECTIONS = tuple(section.value for section in TechnoSection)


class EnergySection(StrEnum):
    """Canonical techno state plus ``unknown`` for missing annotations.

    ``unknown`` is an annotation sentinel, not a musical state. Historical
    ``low``, ``sustained``, and ``release`` values intentionally are not part
    of this ontology; old timelines remain auditable under their old version.
    """

    UNKNOWN = "unknown"
    SILENCE = "silence"
    INTRO = "intro"
    GROOVE = "groove"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"


class TransitionEvent(StrEnum):
    """Instantaneous events kept separate from sustained section state."""

    SECTION_START = "section_start"
    ENERGY_RISE = "energy_rise"
    ENERGY_FALL = "energy_fall"
    BUILD_START = "build_start"
    DROP_ONSET = "drop_onset"
    BREAKDOWN_ONSET = "breakdown_onset"
    GROOVE_RETURN = "groove_return"
    OUTRO_START = "outro_start"
    TRACK_END = "track_end"

    # Source-compatible member names read old code and serialized intent into
    # the one canonical vocabulary. Their values are aliases, so new output
    # never creates a second spelling for the same event.
    TRACK_START = "section_start"
    SECTION_CHANGE = "section_start"
    BUILD_ONSET = "build_start"
    OUTRO_ONSET = "outro_start"


def transition_event_for(
    previous: str | TechnoSection | EnergySection | None,
    current: str | TechnoSection | EnergySection | None,
    *,
    terminal: bool = False,
) -> TransitionEvent:
    """Name the boundary event without turning it into a sustained state."""

    if terminal:
        return TransitionEvent.TRACK_END
    current_value = str(current or "unknown")
    if isinstance(current, (TechnoSection, EnergySection)):
        current_value = current.value
    if previous is None:
        return TransitionEvent.SECTION_START
    return {
        TechnoSection.BUILD.value: TransitionEvent.BUILD_START,
        TechnoSection.DROP.value: TransitionEvent.DROP_ONSET,
        TechnoSection.BREAKDOWN.value: TransitionEvent.BREAKDOWN_ONSET,
        TechnoSection.GROOVE.value: TransitionEvent.GROOVE_RETURN,
        TechnoSection.OUTRO.value: TransitionEvent.OUTRO_START,
    }.get(current_value, TransitionEvent.SECTION_START)


class ContentRole(StrEnum):
    UNKNOWN = "unknown"
    SILENCE = "silence"
    VOCAL = "vocal"
    INSTRUMENTAL = "instrumental"
    SOLO = "solo"
    TRANSITION = "transition"
    APPLAUSE = "applause"


@dataclass(frozen=True, slots=True)
class AnnotationProvenance:
    """Where an annotation came from and how much authority it carries."""

    source: str
    annotation_type: str = "ground_truth"
    source_version: str | None = None
    source_file: str | None = None
    annotator: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("provenance source must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("provenance confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    """Stable identity of a source recording, independent of a listening run."""

    dataset: str
    source_track_id: str
    title: str | None = None
    artists: tuple[str, ...] = ()
    recording_id: str | None = None
    audio_filename: str | None = None
    external_ids: dict[str, str] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must not be empty")
        if not self.source_track_id.strip():
            raise ValueError("source_track_id must not be empty")

    @property
    def group_key(self) -> str:
        """Key used to keep all annotations of one recording in one split."""

        return f"{self.dataset.casefold()}:{self.source_track_id}"


@dataclass(frozen=True, slots=True)
class StructuralLabel:
    raw: str
    functional: FunctionalSection = FunctionalSection.UNKNOWN
    energy: EnergySection = EnergySection.UNKNOWN
    content: ContentRole = ContentRole.UNKNOWN
    normalized: str = "unknown"
    qualifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw.strip():
            raise ValueError("raw structural label must not be empty")
        if not self.normalized.strip():
            raise ValueError("normalized structural label must not be empty")


@dataclass(frozen=True, slots=True)
class StructureBoundary:
    time_s: float
    label: StructuralLabel
    provenance: AnnotationProvenance
    terminal: bool = False
    event: TransitionEvent | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("boundary time must be finite and non-negative")
        if self.event is None:
            object.__setattr__(
                self,
                "event",
                (
                    TransitionEvent.TRACK_END
                    if self.terminal
                    else TransitionEvent.SECTION_CHANGE
                ),
            )
        elif self.terminal and self.event is not TransitionEvent.TRACK_END:
            raise ValueError("terminal boundary event must be track_end")

    @property
    def confidence(self) -> float:
        return self.provenance.confidence


@dataclass(frozen=True, slots=True)
class StructureSegment:
    start_s: float
    end_s: float
    label: StructuralLabel
    provenance: AnnotationProvenance

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("segment start must be finite and non-negative")
        if not math.isfinite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError("segment end must be finite and after its start")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def confidence(self) -> float:
        return self.provenance.confidence


@dataclass(frozen=True, slots=True)
class BeatEvent:
    time_s: float
    position_in_bar: int | None = None
    bar_number: int | None = None
    confidence: float = 1.0
    provenance: AnnotationProvenance | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("beat time must be finite and non-negative")
        if self.position_in_bar is not None and self.position_in_bar < 1:
            raise ValueError("position_in_bar must be positive")
        if self.bar_number is not None and self.bar_number < 0:
            raise ValueError("bar_number must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("beat confidence must be in [0, 1]")

    @property
    def downbeat(self) -> bool:
        return self.position_in_bar == 1


@dataclass(frozen=True, slots=True)
class StructureTrack:
    identity: TrackIdentity
    segments: tuple[StructureSegment, ...]
    boundaries: tuple[StructureBoundary, ...]
    duration_s: float | None = None
    beats: tuple[BeatEvent, ...] = ()
    metadata: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        validate_track(self)


class StructureValidationError(ValueError):
    """Raised when source annotations cannot form a reliable timeline."""


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    track_count: int
    segment_count: int
    boundary_count: int
    beat_count: int
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return True


def validate_track(track: StructureTrack, *, tolerance_s: float = 1e-6) -> None:
    """Validate ordering, contiguity, duration, boundaries, and beat timing."""

    if track.duration_s is not None:
        if not math.isfinite(track.duration_s) or track.duration_s <= 0:
            raise StructureValidationError(
                f"{track.identity.group_key}: duration must be positive"
            )

    previous_end: float | None = None
    for index, segment in enumerate(track.segments):
        if previous_end is not None:
            if segment.start_s < previous_end - tolerance_s:
                raise StructureValidationError(
                    f"{track.identity.group_key}: segments {index - 1} and "
                    f"{index} overlap"
                )
            if segment.start_s > previous_end + tolerance_s:
                raise StructureValidationError(
                    f"{track.identity.group_key}: gap before segment {index}"
                )
        previous_end = segment.end_s

    if track.duration_s is not None and track.segments:
        if track.segments[-1].end_s > track.duration_s + tolerance_s:
            raise StructureValidationError(
                f"{track.identity.group_key}: segments exceed track duration"
            )

    previous_boundary = -1.0
    terminal_count = 0
    for index, boundary in enumerate(track.boundaries):
        if boundary.time_s <= previous_boundary:
            raise StructureValidationError(
                f"{track.identity.group_key}: boundaries must be strictly "
                "increasing"
            )
        if boundary.terminal:
            terminal_count += 1
            if index != len(track.boundaries) - 1:
                raise StructureValidationError(
                    f"{track.identity.group_key}: terminal boundary must be last"
                )
        previous_boundary = boundary.time_s
    if track.boundaries and terminal_count != 1:
        raise StructureValidationError(
            f"{track.identity.group_key}: exactly one terminal boundary required"
        )

    previous_beat = -1.0
    for beat in track.beats:
        if beat.time_s <= previous_beat:
            raise StructureValidationError(
                f"{track.identity.group_key}: beats must be strictly increasing"
            )
        # Beat annotations can land a few milliseconds beyond an independently
        # rounded terminal boundary in the same upstream dataset.
        if (
            track.duration_s is not None
            and beat.time_s > track.duration_s + max(tolerance_s, 0.05)
        ):
            raise StructureValidationError(
                f"{track.identity.group_key}: beat exceeds track duration"
            )
        previous_beat = beat.time_s


def validate_dataset(
    tracks: Iterable[StructureTrack],
) -> DatasetValidationReport:
    """Validate tracks and report useful aggregate counts.

    Multiple SALAMI annotators may describe the same source track.  Duplicate
    source identities are therefore allowed, but exact duplicate annotation
    keys are reported as warnings.
    """

    materialized = tuple(tracks)
    warnings: list[str] = []
    annotation_keys: set[tuple[str, str | None]] = set()
    for track in materialized:
        validate_track(track)
        annotator = (
            track.segments[0].provenance.annotator if track.segments else None
        )
        annotation_key = (track.identity.group_key, annotator)
        if annotation_key in annotation_keys:
            warnings.append(
                f"duplicate annotation for {track.identity.group_key} "
                f"(annotator={annotator or 'unknown'})"
            )
        annotation_keys.add(annotation_key)
    return DatasetValidationReport(
        track_count=len(materialized),
        segment_count=sum(len(track.segments) for track in materialized),
        boundary_count=sum(len(track.boundaries) for track in materialized),
        beat_count=sum(len(track.beats) for track in materialized),
        warnings=tuple(warnings),
    )
