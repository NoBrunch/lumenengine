"""Editable, deterministic motion paths shared by rehearsal and performance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from lumen_engine.models import clamp


RELATIONSHIPS = ("synchronized", "opposed", "mirrored", "chase", "counter")


@dataclass(frozen=True, slots=True)
class MotionTuning:
    cycle_beats: float
    pan_size: float
    tilt_size: float
    pan_center: float = 0.5
    tilt_center: float = 0.5
    relationship: str = "synchronized"
    direction: int = 1
    beat_motion: float = 0.0
    body_size: float = 0.75
    arm_size: float = 0.85

    def patch(self, values: dict[str, Any]) -> "MotionTuning":
        relationship = str(values.get("relationship", self.relationship))
        if relationship not in RELATIONSHIPS:
            raise ValueError("unknown fixture relationship")
        direction = int(values.get("direction", self.direction))
        if direction not in {-1, 1}:
            raise ValueError("motion direction must be -1 or 1")
        return replace(
            self,
            cycle_beats=clamp(float(values.get("cycle_beats", self.cycle_beats)), 1.0, 64.0),
            pan_size=clamp(float(values.get("pan_size", self.pan_size)), 0.0, 1.0),
            tilt_size=clamp(float(values.get("tilt_size", self.tilt_size)), 0.0, 1.0),
            pan_center=clamp(float(values.get("pan_center", self.pan_center)), 0.0, 1.0),
            tilt_center=clamp(float(values.get("tilt_center", self.tilt_center)), 0.0, 1.0),
            relationship=relationship,
            direction=direction,
            beat_motion=clamp(float(values.get("beat_motion", self.beat_motion)), 0.0, 1.0),
            body_size=clamp(float(values.get("body_size", self.body_size)), 0.0, 1.0),
            arm_size=clamp(float(values.get("arm_size", self.arm_size)), 0.0, 1.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MOTION_TUNINGS: dict[str, MotionTuning] = {
    "breathe": MotionTuning(16.0, 0.72, 0.20, relationship="synchronized", body_size=0.25, arm_size=0.30),
    "fan_sweep": MotionTuning(8.0, 1.0, 0.08, tilt_center=0.62, relationship="mirrored", body_size=0.55, arm_size=0.80),
    "figure_eight": MotionTuning(16.0, 1.0, 0.82, relationship="opposed", body_size=1.0, arm_size=0.90),
    "opposing_chase": MotionTuning(8.0, 1.0, 0.30, relationship="opposed", body_size=0.70, arm_size=1.0),
    "beat_nod": MotionTuning(4.0, 0.0, 0.78, relationship="opposed", beat_motion=1.0, body_size=0.15, arm_size=0.95),
    "counter_rotate": MotionTuning(16.0, 0.92, 0.76, relationship="counter", body_size=0.85, arm_size=1.0),
}


def merged_motion_tunings(payload: dict[str, Any] | None) -> dict[str, MotionTuning]:
    result = dict(DEFAULT_MOTION_TUNINGS)
    for routine, values in (payload or {}).items():
        if routine in result and isinstance(values, dict):
            result[routine] = result[routine].patch(values)
    return result


def motion_coordinates(
    routine: str,
    beat_position: float,
    fixture_index: int,
    fixture_count: int,
    tuning: MotionTuning,
) -> tuple[float, float]:
    """Return normalized -1..1 pan/tilt on one shared musical clock."""
    theta = math.tau * beat_position / tuning.cycle_beats * tuning.direction
    relationship = tuning.relationship
    if relationship == "opposed":
        theta += fixture_index * math.pi
    elif relationship == "chase":
        theta += fixture_index / max(1, fixture_count) * math.tau
    elif relationship == "counter" and fixture_index % 2:
        theta = -theta

    if routine == "figure_eight":
        pan, tilt = math.sin(theta), math.sin(2.0 * theta)
    elif routine == "fan_sweep":
        pan, tilt = math.sin(theta), 0.18 * math.sin(theta)
    elif routine == "opposing_chase":
        pan, tilt = math.sin(theta), 0.35 * math.cos(theta)
    elif routine == "beat_nod":
        pan, tilt = 0.0, -math.cos(theta)
    elif routine == "counter_rotate":
        pan, tilt = math.cos(theta), math.sin(theta)
    else:  # breathe
        pan, tilt = math.sin(theta), 0.32 * math.sin(theta * 0.5)
    if relationship == "mirrored" and fixture_index % 2:
        pan = -pan
    return clamp(pan, -1.0, 1.0), clamp(tilt, -1.0, 1.0)


def normalized_position(
    routine: str,
    beat_position: float,
    fixture_index: int,
    fixture_count: int,
    tuning: MotionTuning,
    size: float = 1.0,
) -> tuple[float, float]:
    pan, tilt = motion_coordinates(
        routine, beat_position, fixture_index, fixture_count, tuning
    )
    amplitude = clamp(size, 0.0, 1.0)
    return (
        clamp(tuning.pan_center + pan * tuning.pan_size * amplitude * 0.5, 0.0, 1.0),
        clamp(tuning.tilt_center + tilt * tuning.tilt_size * amplitude * 0.5, 0.0, 1.0),
    )


def preview_paths(routine: str, tuning: MotionTuning, samples: int = 129) -> list[list[list[float]]]:
    return [
        [
            list(normalized_position(routine, tuning.cycle_beats * point / (samples - 1), fixture, 2, tuning))
            for point in range(samples)
        ]
        for fixture in range(2)
    ]


def required_axis_speeds(
    routine: str,
    tuning: MotionTuning,
    *,
    bpm: float,
    fixture_index: int,
    fixture_count: int,
    pan_range_deg: float,
    tilt_range_deg: float,
    size: float = 1.0,
    sample_rate_hz: float = 96.0,
) -> tuple[float, float]:
    duration_s = tuning.cycle_beats * 60.0 / max(1.0, bpm)
    count = max(8, round(duration_s * sample_rate_hz))
    prior = normalized_position(
        routine, 0.0, fixture_index, fixture_count, tuning, size
    )
    max_pan = max_tilt = 0.0
    for index in range(1, count + 1):
        seconds = duration_s * index / count
        beat = seconds * bpm / 60.0
        current = normalized_position(
            routine, beat, fixture_index, fixture_count, tuning, size
        )
        dt = duration_s / count
        max_pan = max(max_pan, abs(current[0] - prior[0]) * pan_range_deg / dt)
        max_tilt = max(max_tilt, abs(current[1] - prior[1]) * tilt_range_deg / dt)
        prior = current
    return max_pan, max_tilt
