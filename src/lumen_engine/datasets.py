"""Adapters from public structure datasets to Lumen's normalized timeline.

This module intentionally uses only the Python standard library.  Downloading
audio, extracting neural embeddings, and running teacher models belong in
optional worker environments; reading and validating annotations must remain
available to Lumen's dependency-free core.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping
import unicodedata

from .structure import (
    AnnotationProvenance,
    BeatEvent,
    ContentRole,
    EnergySection,
    FunctionalSection,
    StructuralLabel,
    StructureBoundary,
    StructureSegment,
    StructureTrack,
    StructureValidationError,
    TrackIdentity,
    transition_event_for,
    validate_dataset,
)


_TERMINATORS = {"end", "eof", "terminal"}


def _words(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    # Upstream sets append occurrence identifiers freely: ``chorus2``,
    # ``verse1a``, ``solo3``.  They are not semantic label differences.
    value = re.sub(r"\b([a-z][a-z_-]*?)(?:\d+[a-z]?)\b", r"\1", value)
    compact_aliases = {
        "prechorus": "pre chorus",
        "postchorus": "post chorus",
        "altchorus": "chorus",
        "instchorus": "instrumental chorus",
        "chorusinst": "chorus instrumental",
        "quietchorus": "quiet chorus",
        "intchorus": "chorus",
        "instrumentalverse": "instrumental verse",
        "introverse": "intro verse",
        "guitarsolo": "guitar solo",
        "vocaloutro": "vocal outro",
        "verseinst": "verse instrumental",
        "slowverse": "slow verse",
        "rhythmlessintro": "rhythmless intro",
        "miniverse": "verse",
        "postverse": "verse",
        "fadein": "fade in",
        "fadeout": "fade out",
        "gtr": "guitar",
        "inst": "instrumental",
    }
    for alias, replacement in compact_aliases.items():
        value = re.sub(rf"\b{alias}\b", replacement, value)
    value = value.replace("&", " and ")
    value = re.sub(r"[_/\\-]+", " ", value)
    value = re.sub(r"[\"'’]", "", value)
    value = re.sub(r"[\(\)\[\]\{\},;:]+", " ", value)
    value = re.sub(
        r"\b((?:re\s*)?(?:intro|verse|chorus|bridge|drop))"
        r"(?:\s*\d+|\s+[a-z](?=\s|$))\b",
        r"\1",
        value,
    )
    return " ".join(value.split())


def normalize_structure_label(raw_label: str) -> StructuralLabel:
    """Normalize noisy source vocabulary without collapsing independent axes."""

    raw = raw_label.strip()
    if not raw:
        raise ValueError("source label must not be empty")
    words = _words(raw)
    padded = f" {words} "

    functional = FunctionalSection.UNKNOWN
    if re.search(r"\b(silence|silent|pause)\b", words):
        functional = FunctionalSection.SILENCE
    elif re.search(r"\b(re ?intro|intro|introduction|opening|fade in)\b", words):
        functional = FunctionalSection.INTRO
    elif re.search(r"\b(pre ?chorus|pre ?refrain)\b", words):
        functional = FunctionalSection.PRE_CHORUS
    elif re.search(r"\b(post ?chorus|post ?refrain)\b", words):
        functional = FunctionalSection.POST_CHORUS
    elif re.search(r"\b(chorus|refrain|hook)\b", words):
        functional = FunctionalSection.CHORUS
    elif re.search(r"\b(pre ?verse)\b", words):
        functional = FunctionalSection.TRANSITION
    elif re.search(r"\bverse\b", words):
        functional = FunctionalSection.VERSE
    elif re.search(r"\b(bridge|middle eight|contrasting middle)\b", words):
        functional = FunctionalSection.BRIDGE
    elif re.search(r"\b(interlude|interruption|break)\b", words):
        functional = FunctionalSection.INTERLUDE
    elif re.search(r"\b(transition|link|riser)\b", words):
        functional = FunctionalSection.TRANSITION
    elif re.search(r"\b(solo|improv|improvisation|cadenza)\b", words):
        functional = FunctionalSection.SOLO
    elif re.search(r"\b(instrumental)\b", words):
        functional = FunctionalSection.INSTRUMENTAL
    elif re.search(r"\b(development|exposition)\b", words):
        functional = FunctionalSection.DEVELOPMENT
    elif re.search(r"\b(theme|head)\b", words):
        functional = FunctionalSection.THEME
    elif re.search(
        r"\b(outro|ending|closing|coda|codetta|finale|fade out)\b", words
    ):
        functional = FunctionalSection.OUTRO

    energy = EnergySection.UNKNOWN
    if re.search(r"\b(silence|silent|pause)\b", words):
        energy = EnergySection.SILENCE
    elif re.search(r"\b(re ?intro|intro|introduction|opening|fade in)\b", words):
        energy = EnergySection.INTRO
    elif re.search(
        r"\b(outro|ending|closing|coda|codetta|finale|fade out)\b", words
    ):
        energy = EnergySection.OUTRO
    elif re.search(r"\b(breakdown|break down|break)\b", words):
        energy = EnergySection.BREAKDOWN
    elif re.search(r"\b(buildup|build up|build|riser|rising)\b", words):
        energy = EnergySection.BUILD
    elif re.search(r"\b(drop|release|climax)\b", words):
        energy = EnergySection.DROP
    elif re.search(r"\b(groove)\b", words):
        energy = EnergySection.GROOVE
    elif re.search(r"\b(quiet|calm|soft|slow|low energy)\b", words):
        energy = EnergySection.BREAKDOWN
    elif re.search(r"\b(sustain|sustained|plateau)\b", words):
        energy = EnergySection.GROOVE

    content = ContentRole.UNKNOWN
    if re.search(r"\b(silence|silent|pause)\b", words):
        content = ContentRole.SILENCE
    elif re.search(r"\b(applause|crowd)\b", words):
        content = ContentRole.APPLAUSE
    elif re.search(r"\b(solo|improv|improvisation|cadenza)\b", words):
        content = ContentRole.SOLO
    elif re.search(r"\b(vocal|vocals|voice|singer|singing|rap|raps)\b", words):
        content = ContentRole.VOCAL
    elif re.search(
        r"\b(instrumental|guitar|piano|drums?|strings?|synth|orchestra|"
        r"flute|organ|bass|banjo|horns?|riff)\b",
        words,
    ):
        content = ContentRole.INSTRUMENTAL
    elif functional in {
        FunctionalSection.TRANSITION,
        FunctionalSection.INTERLUDE,
    }:
        content = ContentRole.TRANSITION

    normalized = next(
        (
            value
            for value in (
                functional.value
                if functional is not FunctionalSection.UNKNOWN
                else None,
                energy.value if energy is not EnergySection.UNKNOWN else None,
                content.value if content is not ContentRole.UNKNOWN else None,
            )
            if value is not None
        ),
        words or "unknown",
    )
    qualifiers = tuple(
        qualifier
        for qualifier in ("repeated", "instrumental", "vocal", "fade")
        if f" {qualifier} " in padded
    )
    return StructuralLabel(
        raw=raw,
        functional=functional,
        energy=energy,
        content=content,
        normalized=normalized,
        qualifiers=qualifiers,
    )


def normalize_techno_structure_label(raw_label: str) -> StructuralLabel:
    """Map one EDM/techno label onto the canonical sustained-state axis.

    EDMFormer and EDM-98 describe energy form, not pop functional form or
    vocal content. Keep those unrelated axes unknown even when a raw token
    such as ``intro`` happens to overlap their vocabulary.
    """

    normalized = normalize_structure_label(raw_label)
    return StructuralLabel(
        raw=normalized.raw,
        functional=FunctionalSection.UNKNOWN,
        energy=normalized.energy,
        content=ContentRole.UNKNOWN,
        normalized=(
            normalized.energy.value
            if normalized.energy is not EnergySection.UNKNOWN
            else "unknown"
        ),
        qualifiers=normalized.qualifiers,
    )


def _is_terminal(label: str) -> bool:
    return _words(label) in _TERMINATORS


def _timeline_from_boundaries(
    identity: TrackIdentity,
    entries: Iterable[tuple[float, str]],
    provenance: AnnotationProvenance,
    *,
    duration_s: float | None = None,
    beats: tuple[BeatEvent, ...] = (),
    metadata: dict[str, object] | None = None,
    inherit_unknown_axes: bool = False,
    label_normalizer: Callable[[str], StructuralLabel] = normalize_structure_label,
) -> StructureTrack:
    ordered = sorted(
        ((float(time_s), str(label).strip()) for time_s, label in entries),
        key=lambda item: item[0],
    )
    # Some SALAMI parsed function files legitimately emit two functions at the
    # same boundary (for example "Instrumental" and "Solo").  Preserve both
    # pieces of information as one composite source label.
    materialized: list[tuple[float, str]] = []
    for time_s, label in ordered:
        if materialized and time_s == materialized[-1][0]:
            previous_time, previous_label = materialized[-1]
            materialized[-1] = (
                previous_time,
                (
                    "end"
                    if _is_terminal(previous_label) or _is_terminal(label)
                    else f"{previous_label}, {label}"
                ),
            )
        else:
            materialized.append((time_s, label))
    if len(materialized) < 2:
        raise StructureValidationError(
            f"{identity.group_key}: timeline requires at least two boundaries"
        )
    for index in range(1, len(materialized)):
        if materialized[index][0] <= materialized[index - 1][0]:
            raise StructureValidationError(
                f"{identity.group_key}: source times must be strictly increasing"
            )

    if not _is_terminal(materialized[-1][1]):
        if duration_s is None:
            raise StructureValidationError(
                f"{identity.group_key}: timeline has no terminal boundary"
            )
        if duration_s <= materialized[-1][0]:
            raise StructureValidationError(
                f"{identity.group_key}: duration is not after final boundary"
            )
        materialized.append((duration_s, "end"))
    terminal_time = materialized[-1][0]
    catalog_duration = duration_s
    observed_end = max(
        terminal_time,
        beats[-1].time_s if beats else terminal_time,
    )
    if catalog_duration is None:
        duration_s = observed_end
    else:
        # SALAMI catalog durations are rounded, while Harmonix occasionally
        # marks "end" before the final beat or trailing audio.  Never truncate
        # either annotation stream.
        duration_s = max(catalog_duration, observed_end)
    if catalog_duration is not None and abs(catalog_duration - terminal_time) > 1e-6:
        extra = dict(metadata or {})
        extra["catalog_duration_s"] = catalog_duration
        extra["annotation_terminal_s"] = terminal_time
        metadata = extra

    labels: list[StructuralLabel] = []
    previous: StructuralLabel | None = None
    for _, raw_label in materialized:
        label = label_normalizer(raw_label)
        if inherit_unknown_axes and previous is not None and not _is_terminal(raw_label):
            label = StructuralLabel(
                raw=label.raw,
                functional=(
                    previous.functional
                    if label.functional is FunctionalSection.UNKNOWN
                    else label.functional
                ),
                energy=(
                    previous.energy
                    if label.energy is EnergySection.UNKNOWN
                    else label.energy
                ),
                content=(
                    previous.content
                    if label.content is ContentRole.UNKNOWN
                    else label.content
                ),
                normalized=(
                    previous.normalized
                    if (
                        label.functional is FunctionalSection.UNKNOWN
                        and label.energy is EnergySection.UNKNOWN
                        and label.content is ContentRole.UNKNOWN
                    )
                    else label.normalized
                ),
                qualifiers=label.qualifiers,
            )
        labels.append(label)
        if not _is_terminal(raw_label):
            previous = label

    boundaries = tuple(
        StructureBoundary(
            time_s=time_s,
            label=label,
            provenance=provenance,
            terminal=index == len(materialized) - 1,
            event=transition_event_for(
                (
                    None
                    if index == 0
                    else labels[index - 1].energy
                ),
                label.energy,
                terminal=index == len(materialized) - 1,
            ),
        )
        for index, ((time_s, _), label) in enumerate(zip(materialized, labels))
    )
    segments = tuple(
        StructureSegment(
            start_s=materialized[index][0],
            end_s=materialized[index + 1][0],
            label=labels[index],
            provenance=provenance,
        )
        for index in range(len(materialized) - 1)
    )
    return StructureTrack(
        identity=identity,
        segments=segments,
        boundaries=boundaries,
        duration_s=duration_s,
        beats=beats,
        metadata=dict(metadata or {}),
    )


def parse_edm98(dataset_path: str | Path) -> list[StructureTrack]:
    """Read EDM-98's canonical JSONL records."""

    path = Path(dataset_path)
    if path.is_dir():
        candidates = (
            path / "dataset.jsonl",
            path / "src" / "edm98" / "resources" / "dataset.jsonl",
        )
        path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )
    tracks: list[StructureTrack] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StructureValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise StructureValidationError(
                    f"{path}:{line_number}: record must be an object"
                )
            track_id = str(payload.get("id", "")).strip()
            if not track_id:
                raise StructureValidationError(
                    f"{path}:{line_number}: missing track id"
                )
            if track_id in seen_ids:
                raise StructureValidationError(
                    f"{path}:{line_number}: duplicate track id {track_id}"
                )
            seen_ids.add(track_id)
            raw_labels = payload.get("labels")
            if not isinstance(raw_labels, list):
                raise StructureValidationError(
                    f"{path}:{line_number}: labels must be a list"
                )
            entries: list[tuple[float, str]] = []
            for entry in raw_labels:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise StructureValidationError(
                        f"{path}:{line_number}: labels must be [time, label] pairs"
                    )
                entries.append((float(entry[0]), str(entry[1])))
            filename = payload.get("file_path")
            identity = TrackIdentity(
                dataset="edm98",
                source_track_id=track_id,
                recording_id=f"deezer:{track_id}",
                audio_filename=str(filename) if filename else None,
                external_ids={"deezer": track_id},
            )
            provenance = AnnotationProvenance(
                source="edm98",
                source_version="1",
                source_file=str(path),
                details={"line_number": line_number},
            )
            tracks.append(
                _timeline_from_boundaries(
                    identity,
                    entries,
                    provenance,
                    label_normalizer=normalize_techno_structure_label,
                )
            )
    validate_dataset(tracks)
    return tracks


def _read_harmonix_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("File", "")).strip(): {
                str(key): str(value or "").strip() for key, value in row.items()
            }
            for row in rows
            if str(row.get("File", "")).strip()
        }


def _read_boundary_text(path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.match(
                r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))[\t ]+(.+?)\s*$",
                stripped,
            )
            if match is None:
                raise StructureValidationError(
                    f"{path}:{line_number}: expected '<seconds> <label>'"
                )
            entries.append((float(match.group(1)), match.group(2)))
    return entries


def _read_harmonix_beats(path: Path) -> tuple[BeatEvent, ...]:
    provenance = AnnotationProvenance(
        source="harmonix",
        source_version="1",
        source_file=str(path),
    )
    beats: list[BeatEvent] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise StructureValidationError(
                    f"{path}:{line_number}: expected time, beat position, bar"
                )
            beats.append(
                BeatEvent(
                    time_s=float(parts[0]),
                    position_in_bar=int(parts[1]),
                    bar_number=int(parts[2]),
                    provenance=provenance,
                )
            )
    return tuple(beats)


def parse_harmonix(
    metadata_path: str | Path,
    segments_dir: str | Path,
    beats_dir: str | Path | None = None,
) -> list[StructureTrack]:
    """Read Harmonix metadata.csv, segment files, and optional beat files."""

    metadata_file = Path(metadata_path)
    metadata = _read_harmonix_metadata(metadata_file)
    segment_root = Path(segments_dir)
    beat_root = Path(beats_dir) if beats_dir is not None else None
    tracks: list[StructureTrack] = []
    for path in sorted(segment_root.glob("*.txt")):
        track_id = path.stem
        row = metadata.get(track_id)
        if row is None:
            raise StructureValidationError(
                f"{path}: no matching Harmonix metadata row"
            )
        duration = float(row["Duration"]) if row.get("Duration") else None
        artists = (row["Artist"],) if row.get("Artist") else ()
        external_ids = {
            key: value
            for key, value in {
                "musicbrainz": row.get("MusicBrainz Id", ""),
                "acoustid": row.get("Acoustid Id", ""),
            }.items()
            if value
        }
        identity = TrackIdentity(
            dataset="harmonix",
            source_track_id=track_id,
            title=row.get("Title") or None,
            artists=artists,
            recording_id=external_ids.get("musicbrainz"),
            external_ids=external_ids,
        )
        beats = ()
        if beat_root is not None:
            beat_path = beat_root / path.name
            if beat_path.exists():
                beats = _read_harmonix_beats(beat_path)
        provenance = AnnotationProvenance(
            source="harmonix",
            source_version="1",
            source_file=str(path),
        )
        tracks.append(
            _timeline_from_boundaries(
                identity,
                _read_boundary_text(path),
                provenance,
                duration_s=duration,
                beats=beats,
                metadata={
                    "release": row.get("Release") or None,
                    "bpm": float(row["BPM"]) if row.get("BPM") else None,
                    "genre": row.get("Genre") or None,
                    "time_signature": row.get("Time Signature") or None,
                },
            )
        )
    validate_dataset(tracks)
    return tracks


def _parse_ccmusic_file(
    path: Path,
    *,
    time_scale: float,
    source_version: str,
    provenance_details: Mapping[str, object] | None = None,
) -> StructureTrack | None:
    rows: list[tuple[float, float, str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.match(
                r"^\"?(\d+(?:\.\d+)?)\"?(?:\s*[\t,]\s*|\s+)"
                r"\"?(\d+(?:\.\d+)?)\"?(?:\s*[\t,]\s*|\s+)"
                r"\"?(.+?)\"?\s*$",
                stripped,
            )
            if match is None:
                # Real distributions may include one heading row.
                if not rows and re.search(r"start|structure|label", stripped, re.I):
                    continue
                raise StructureValidationError(
                    f"{path}:{line_number}: expected start, end, label"
                )
            label = match.group(3).strip().strip('"')
            rows.append(
                (
                    float(match.group(1)) * time_scale,
                    float(match.group(2)) * time_scale,
                    label,
                )
            )
    if not rows:
        return None
    identity = TrackIdentity(
        dataset="ccmusic",
        source_track_id=path.stem,
        title=path.stem,
    )
    return _ccmusic_track_from_rows(
        identity,
        rows,
        source_version=source_version,
        source_file=str(path),
        provenance_details=provenance_details,
    )


def _ccmusic_track_from_rows(
    identity: TrackIdentity,
    rows: list[tuple[float, float, str]],
    *,
    source_version: str,
    source_file: str | None,
    provenance_details: Mapping[str, object] | None = None,
) -> StructureTrack:
    """Build one strict Lumen timeline without hiding upstream defects.

    CCMusic contains a small number of genuine gaps and overlaps.  Positive
    gaps become explicit low-confidence ``unannotated gap`` segments.  For an
    overlap, the later annotation onset is the normalized transition point.
    Both cases retain their exact source-unit ranges in provenance and track
    metadata so evaluation never mistakes the normalization for ground truth.
    """

    rows.sort(key=lambda row: row[0])
    discontinuities: list[dict[str, object]] = []
    entries: list[tuple[float, str]] = []
    for index, (start, end, label) in enumerate(rows):
        entries.append((start, label))
        if index == len(rows) - 1:
            continue
        next_start = rows[index + 1][0]
        delta = next_start - end
        if abs(delta) <= 1e-6:
            continue
        source_previous_end = end / 0.01
        source_next_start = next_start / 0.01
        source_delta = delta / 0.01
        discontinuities.append(
            {
                "previous_segment_index": index,
                "previous_offset": int(round(source_previous_end)),
                "next_onset": int(round(source_next_start)),
                "delta_centiseconds": int(round(source_delta)),
                "kind": "gap" if delta > 0 else "overlap",
            }
        )
        if delta > 0:
            entries.append((end, "unannotated gap"))
    entries.append((rows[-1][1], "end"))
    details = {
        **dict(provenance_details or {}),
        "timeline_discontinuities": discontinuities,
        "discontinuity_policy": (
            "explicit_unknown_gap_later_onset_wins_overlap"
        ),
    }
    provenance = AnnotationProvenance(
        source="ccmusic",
        source_version=source_version,
        source_file=source_file,
        details=details,
    )
    track = _timeline_from_boundaries(
        identity,
        entries,
        provenance,
        metadata={"timeline_discontinuities": discontinuities},
    )
    if not any(segment.label.raw == "unannotated gap" for segment in track.segments):
        return track
    gap_provenance = AnnotationProvenance(
        source="ccmusic",
        annotation_type="derived_gap",
        source_version=source_version,
        source_file=source_file,
        confidence=0.0,
        details=details,
    )
    segments = tuple(
        StructureSegment(
            start_s=segment.start_s,
            end_s=segment.end_s,
            label=segment.label,
            provenance=(
                gap_provenance
                if segment.label.raw == "unannotated gap"
                else segment.provenance
            ),
        )
        for segment in track.segments
    )
    boundaries = tuple(
        StructureBoundary(
            time_s=boundary.time_s,
            label=boundary.label,
            provenance=(
                gap_provenance
                if boundary.label.raw == "unannotated gap"
                else boundary.provenance
            ),
            terminal=boundary.terminal,
            event=boundary.event,
        )
        for boundary in track.boundaries
    )
    return StructureTrack(
        identity=track.identity,
        segments=segments,
        boundaries=boundaries,
        duration_s=track.duration_s,
        beats=track.beats,
        metadata=track.metadata,
    )


def parse_ccmusic(
    source: str | Path,
    *,
    time_unit: str = "centiseconds",
    source_version: str = "song_structure",
    provenance_details: Mapping[str, object] | None = None,
) -> list[StructureTrack]:
    """Read CCMusic song-structure text files.

    The official description expresses start/end values in 0.01 seconds.
    ``time_unit='seconds'`` is available for converted mirrors.
    """

    if time_unit not in {"centiseconds", "seconds"}:
        raise ValueError("time_unit must be 'centiseconds' or 'seconds'")
    time_scale = 0.01 if time_unit == "centiseconds" else 1.0
    path = Path(source)
    files = [path] if path.is_file() else sorted(path.rglob("*.txt"))
    tracks = [
        track
        for track in (
            _parse_ccmusic_file(
                file_path,
                time_scale=time_scale,
                source_version=source_version,
                provenance_details=provenance_details,
            )
            for file_path in files
        )
        if track is not None
    ]
    validate_dataset(tracks)
    return tracks


def parse_ccmusic_label_records(
    records: Iterable[Mapping[str, object]],
    *,
    source_version: str,
    time_unit: str = "centiseconds",
) -> list[StructureTrack]:
    """Normalize projected Hugging Face Arrow ``label`` records.

    The gated dataset's Arrow schema also declares ``audio`` and ``mel``.
    Callers should project the label column before invoking this adapter; this
    function reads and retains only onset, offset, and structure values even if
    an upstream row mapping contains additional fields.
    """

    if time_unit not in {"centiseconds", "seconds"}:
        raise ValueError("time_unit must be 'centiseconds' or 'seconds'")
    time_scale = 0.01 if time_unit == "centiseconds" else 1.0
    tracks: list[StructureTrack] = []
    for row_index, record in enumerate(records):
        labels = record.get("label")
        if not isinstance(labels, (list, tuple)) or not labels:
            raise StructureValidationError(
                f"CCMusic Arrow row {row_index} has no label sequence"
            )
        rows: list[tuple[float, float, str]] = []
        for segment_index, item in enumerate(labels):
            if not isinstance(item, Mapping):
                raise StructureValidationError(
                    f"CCMusic Arrow row {row_index} label {segment_index} "
                    "must be an object"
                )
            try:
                start = float(item["onset_time"]) * time_scale
                end = float(item["offset_time"]) * time_scale
                label = str(item["structure"]).strip()
            except (KeyError, TypeError, ValueError) as error:
                raise StructureValidationError(
                    f"CCMusic Arrow row {row_index} label {segment_index} "
                    "does not match the audited schema"
                ) from error
            if not label or end <= start:
                raise StructureValidationError(
                    f"CCMusic Arrow row {row_index} label {segment_index} "
                    "has an invalid interval or structure"
                )
            rows.append((start, end, label))
        identity = TrackIdentity(
            dataset="ccmusic",
            source_track_id=f"row-{row_index:06d}",
            title=f"CCMusic row {row_index}",
        )
        tracks.append(
            _ccmusic_track_from_rows(
                identity,
                rows,
                source_version=source_version,
                source_file=None,
                provenance_details={
                    "source_format": "huggingface_arrow_label_projection",
                    "row_index": row_index,
                    "retained_fields": (
                        "label.onset_time",
                        "label.offset_time",
                        "label.structure",
                    ),
                    "excluded_fields": ("audio", "mel"),
                    "contains_audio_or_links": False,
                },
            )
        )
    validate_dataset(tracks)
    return tracks


def _salami_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("SONG_ID", "")).strip(): {
                str(key): str(value or "").strip() for key, value in row.items()
            }
            for row in csv.DictReader(handle)
            if str(row.get("SONG_ID", "")).strip()
        }


def _parse_salami_annotation(
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> StructureTrack:
    try:
        track_id = path.parents[1].name if path.parent.name == "parsed" else path.parent.name
    except IndexError as exc:
        raise StructureValidationError(f"{path}: cannot derive SALAMI id") from exc
    annotation_match = re.search(r"textfile(\d+)", path.name)
    annotation_number = annotation_match.group(1) if annotation_match else "1"
    row = dict(metadata or {})
    annotator = row.get(f"ANNOTATOR{annotation_number}") or annotation_number
    entries = _read_boundary_text(path)
    identity = TrackIdentity(
        dataset="salami",
        source_track_id=track_id,
        title=row.get("SONG_TITLE") or None,
        artists=(row["ARTIST"],) if row.get("ARTIST") else (),
    )
    provenance = AnnotationProvenance(
        source="salami",
        source_version="2.0",
        source_file=str(path),
        annotator=annotator,
        details={"annotation_number": annotation_number},
    )
    duration = float(row["SONG_DURATION"]) if row.get("SONG_DURATION") else None
    return _timeline_from_boundaries(
        identity,
        entries,
        provenance,
        duration_s=duration,
        metadata={
            "class": row.get("CLASS") or None,
            "genre": row.get("GENRE") or None,
            "annotation_number": annotation_number,
        },
        inherit_unknown_axes=path.parent.name != "parsed",
    )


def parse_salami(
    source: str | Path,
    metadata_path: str | Path | None = None,
) -> list[StructureTrack]:
    """Read SALAMI v2 raw annotations or its parsed functional timelines.

    For a complete checkout, Lumen prefers
    ``annotations/<id>/parsed/textfileN_functions.txt`` because those files are
    the upstream project's canonical functional layer.  A direct raw
    ``textfileN.txt`` is also supported and retains its fine/coarse boundaries.
    """

    source_path = Path(source)
    if source_path.is_dir() and (source_path / "annotations").is_dir():
        source_path = source_path / "annotations"
    metadata_file = Path(metadata_path) if metadata_path is not None else None
    if source_path.is_dir() and metadata_file is None:
        candidate = source_path.parent / "metadata" / "metadata.csv"
        if candidate.exists():
            metadata_file = candidate
        else:
            candidate = source_path / "metadata" / "metadata.csv"
            if candidate.exists():
                metadata_file = candidate
    metadata = _salami_metadata(metadata_file)
    if source_path.is_file():
        files = [source_path]
    else:
        parsed = sorted(
            source_path.glob("*/parsed/textfile*_functions.txt")
        )
        files = parsed or sorted(source_path.glob("*/textfile*.txt"))
    tracks = [
        _parse_salami_annotation(
            path,
            metadata=metadata.get(
                path.parents[1].name
                if path.parent.name == "parsed"
                else path.parent.name
            ),
        )
        for path in files
    ]
    validate_dataset(tracks)
    return tracks


def deterministic_track_split(
    tracks: Iterable[StructureTrack],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: str = "lumen-structure-v1",
) -> dict[str, tuple[StructureTrack, ...]]:
    """Split by source recording, never by window or annotator.

    Sorting a stable SHA-256 digest makes the result independent of filesystem
    order and Python's randomized hash seed.
    """

    split_ratios = dict(
        ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    if not split_ratios:
        raise ValueError("at least one split ratio is required")
    if any(value < 0 for value in split_ratios.values()):
        raise ValueError("split ratios must not be negative")
    total = sum(split_ratios.values())
    if total <= 0:
        raise ValueError("split ratios must have a positive sum")
    normalized = {
        name: value / total for name, value in split_ratios.items()
    }

    groups: dict[str, list[StructureTrack]] = {}
    for track in tracks:
        groups.setdefault(track.identity.group_key, []).append(track)
    ranked = sorted(
        groups,
        key=lambda key: (
            hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
            key,
        ),
    )
    split_names = tuple(normalized)
    remaining = len(ranked)
    counts: dict[str, int] = {}
    allocated = 0
    for name in split_names[:-1]:
        count = int(len(ranked) * normalized[name])
        counts[name] = count
        allocated += count
        remaining -= count
    counts[split_names[-1]] = len(ranked) - allocated

    output: dict[str, list[StructureTrack]] = {
        name: [] for name in split_names
    }
    cursor = 0
    for name in split_names:
        for group_key in ranked[cursor : cursor + counts[name]]:
            output[name].extend(
                sorted(
                    groups[group_key],
                    key=lambda track: (
                        track.segments[0].provenance.annotator or ""
                        if track.segments
                        else ""
                    ),
                )
            )
        cursor += counts[name]
    return {name: tuple(items) for name, items in output.items()}


class EDM98Adapter:
    def load(self, dataset_path: str | Path) -> list[StructureTrack]:
        return parse_edm98(dataset_path)


class HarmonixAdapter:
    def load(
        self,
        metadata_path: str | Path,
        segments_dir: str | Path,
        beats_dir: str | Path | None = None,
    ) -> list[StructureTrack]:
        return parse_harmonix(metadata_path, segments_dir, beats_dir)


class CCMusicAdapter:
    def load(
        self, source: str | Path, *, time_unit: str = "centiseconds"
    ) -> list[StructureTrack]:
        return parse_ccmusic(source, time_unit=time_unit)


class SalamiAdapter:
    def load(
        self,
        source: str | Path,
        metadata_path: str | Path | None = None,
    ) -> list[StructureTrack]:
        return parse_salami(source, metadata_path)


# Dataset names are acronyms upstream; provide acronym-preserving spellings for
# callers while retaining conventional class capitalization above.
SALAMIAdapter = SalamiAdapter
parse_harmonix_set = parse_harmonix
parse_ccmusic_song_structure = parse_ccmusic
