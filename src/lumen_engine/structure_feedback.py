"""Turn timestamped listener structure calls into auditable consensus.

Raw annotations are immutable evidence.  This module derives a replaceable,
song-level view that can be recalled by Live and overlaid on teacher targets.
It deliberately has no database or numerical-library dependency.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from lumen_engine.structure import ContentRole, TransitionEvent


OPERATOR_CONSENSUS_VERSION = "lumen_operator_consensus_v1"
CONSENSUS_WINDOW_MS = 3_000

_ENERGY_STATES = {
    "silence", "intro", "groove", "breakdown", "build", "drop", "outro",
}
_CONTENT_STATES = {
    role.value for role in ContentRole if role.value != "unknown"
}
_EVENTS = {event.value for event in TransitionEvent}
_EVENT_ENERGY = {
    "build_start": "build",
    "drop_onset": "drop",
    "breakdown_onset": "breakdown",
    "groove_return": "groove",
    "outro_start": "outro",
    "track_end": "silence",
}


def annotation_target(label: str) -> tuple[str, str] | None:
    """Return the canonical target axis for a feedback vocabulary label."""

    normalized = str(label).strip().casefold()
    if normalized in _EVENTS:
        return "event", normalized
    if normalized in _ENERGY_STATES:
        return "energy", normalized
    if normalized in _CONTENT_STATES:
        return "content", normalized
    return None


def consensus_anchors(
    annotations: Iterable[dict[str, Any]],
    *,
    window_ms: int = CONSENSUS_WINDOW_MS,
) -> list[dict[str, Any]]:
    """Collapse nearby participant calls without counting rapid repeats.

    Each participant has one vote per axis in a cluster.  Their latest call is
    their vote, while its intensity remains an urgency/confidence qualifier.
    Conflicting ties are retained for audit but are not accepted as authority.
    """

    normalized: list[dict[str, Any]] = []
    for annotation in annotations:
        if str(annotation.get("kind") or "") != "musical_context":
            continue
        target = annotation_target(str(annotation.get("label") or ""))
        position = annotation.get("position_ms")
        if target is None or position is None:
            continue
        participant = str(annotation.get("participant_id") or "").strip()
        if not participant:
            # Anonymous inputs are independent evidence, never one artificial
            # super-user whose later click erases everyone else's vote.
            participant = f"anonymous:{annotation.get('id')}"
        normalized.append({
            "id": int(annotation.get("id") or 0),
            "song_id": annotation.get("song_id"),
            "position_ms": max(0, int(position)),
            "axis": target[0],
            "label": target[1],
            "participant_id": participant,
            "participant_name": annotation.get("participant_name"),
            "intensity": max(0.1, min(1.0, float(annotation.get("intensity") or 1.0))),
            "created_unix_ms": int(annotation.get("created_unix_ms") or 0),
        })
    normalized.sort(key=lambda item: (str(item["song_id"]), item["axis"], item["position_ms"], item["id"]))

    clusters: list[list[dict[str, Any]]] = []
    for item in normalized:
        if (
            not clusters
            or clusters[-1][0]["song_id"] != item["song_id"]
            or clusters[-1][0]["axis"] != item["axis"]
            or item["position_ms"] - clusters[-1][-1]["position_ms"] > window_ms
        ):
            clusters.append([item])
        else:
            clusters[-1].append(item)

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        participant_votes: dict[str, dict[str, Any]] = {}
        for item in cluster:
            prior = participant_votes.get(item["participant_id"])
            if prior is None or (item["created_unix_ms"], item["id"]) >= (
                prior["created_unix_ms"], prior["id"]
            ):
                participant_votes[item["participant_id"]] = item
        counts = Counter(item["label"] for item in participant_votes.values())
        winner_count = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == winner_count)
        # A tie is evidence of disagreement. Pick a stable display value but
        # keep it below the acceptance threshold.
        winner = winners[0]
        winner_votes = [
            item for item in participant_votes.values() if item["label"] == winner
        ]
        total = len(participant_votes)
        agreement = winner_count / max(1, total)
        if len(winners) > 1:
            confidence = 0.50 * agreement
            accepted = False
        else:
            confidence = min(
                0.99,
                (0.74 + 0.08 * min(3, winner_count))
                * agreement
                * (0.85 + 0.15 * max(item["intensity"] for item in winner_votes)),
            )
            accepted = confidence >= 0.60
        positions = sorted(item["position_ms"] for item in winner_votes)
        position_ms = positions[len(positions) // 2]
        result.append({
            "song_id": cluster[0]["song_id"],
            "axis": cluster[0]["axis"],
            "label": winner,
            "position_ms": position_ms,
            "confidence": confidence,
            "agreement": agreement,
            "accepted": accepted,
            "participant_count": total,
            "winning_participant_count": winner_count,
            "participants": sorted(participant_votes),
            "annotation_ids": sorted(item["id"] for item in cluster),
            "vote_counts": dict(sorted(counts.items())),
            "repeated_inputs_collapsed": len(cluster) - total,
        })
    return result


def consensus_segments(
    anchors: Iterable[dict[str, Any]],
    *,
    duration_ms: int,
    base_segments: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Create sparse correction segments; untouched teacher axes stay null."""

    duration = max(1, int(duration_ms))
    base_starts = sorted({
        int(segment["start_ms"])
        for segment in base_segments
        if 0 < int(segment.get("start_ms") or 0) < duration
    })
    accepted = [
        dict(anchor) for anchor in anchors
        if anchor.get("accepted") and 0 <= int(anchor["position_ms"]) < duration
    ]
    intervals: list[dict[str, Any]] = []
    for index, anchor in enumerate(accepted):
        start = int(anchor["position_ms"])
        axis = str(anchor["axis"])
        next_base = next((point for point in base_starts if point > start), None)
        next_axis = next(
            (
                int(other["position_ms"])
                for other in accepted[index + 1 :]
                if other["axis"] == axis and int(other["position_ms"]) > start
            ),
            None,
        )
        candidates = [value for value in (next_base, next_axis, duration) if value is not None]
        end = min(candidates)
        if end <= start:
            continue
        label = str(anchor["label"])
        interval_axis = axis
        interval_label = label
        if axis == "event":
            interval_axis = "energy" if label in _EVENT_ENERGY else "event"
            interval_label = _EVENT_ENERGY.get(label, label)
            # Events are precise cues, not states that should occupy a whole
            # teacher section. Their inferred state lasts only to the next
            # boundary (and at most eight seconds).
            end = min(end, start + 8_000)
        intervals.append({
            "start_ms": start,
            "end_ms": end,
            "axis": interval_axis,
            "label": interval_label,
            "event": label if axis == "event" else None,
            "anchor": anchor,
        })
    if not intervals:
        return []

    points = sorted({value for item in intervals for value in (item["start_ms"], item["end_ms"])})
    segments: list[dict[str, Any]] = []
    for start, end in zip(points, points[1:], strict=False):
        active = [item for item in intervals if item["start_ms"] <= start < item["end_ms"]]
        if not active or end <= start:
            continue
        labels: dict[str, str | None] = {
            "functional_label": None,
            "energy_label": None,
            "content_label": None,
        }
        for axis in ("functional", "energy", "content"):
            candidates = [item for item in active if item["axis"] == axis]
            if candidates:
                selected = max(candidates, key=lambda item: (item["start_ms"], item["anchor"]["confidence"]))
                labels[f"{axis}_label"] = selected["label"]
        events = [item for item in active if item.get("event") and item["start_ms"] == start]
        evidence = [item["anchor"] for item in active]
        segments.append({
            "segment_index": len(segments),
            "start_ms": start,
            "end_ms": end,
            **labels,
            "boundary_confidence": max(
                [float(item["anchor"]["confidence"]) for item in events] or [0.0]
            ),
            "label_confidence": max(float(item["anchor"]["confidence"]) for item in active),
            "raw_label": "+".join(sorted({item["anchor"]["label"] for item in active})),
            "provenance": {
                "source": "operator_annotation_consensus",
                "transition_events": sorted({item["event"] for item in events}),
                "annotation_ids": sorted({annotation_id for item in evidence for annotation_id in item["annotation_ids"]}),
                "participant_count": len({participant for item in evidence for participant in item["participants"]}),
                "agreement": min(float(item["agreement"]) for item in evidence),
                "anchors": evidence,
            },
        })
    return segments
