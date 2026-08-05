"""Editable, deterministic motion paths shared by rehearsal and performance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from lumen_engine.models import clamp


RELATIONSHIPS = ("synchronized", "opposed", "mirrored", "chase", "counter")

# These are the only public fixture scopes used by rehearsal, authored
# choreography, and feedback.  Fixture IDs remain an output-layer concern.
MOTION_SCOPES = ("movers", "center", "overall")
MOTION_SCOPE_LABELS = {
    "movers": "Movers",
    "center": "Center",
    "overall": "Whole rig",
}

CENTER_EMITTER_PATTERNS = ("both", "alternate", "ball", "arms", "chase")
CENTER_COLOR_PATTERNS = ("palette", "alternate", "opposed", "pulse")
CENTER_LASER_MODES = ("off", "steady", "beat", "alternate")


def canonical_motion_scope(value: Any) -> str:
    """Normalize old scope spellings without exposing individual fixtures."""

    normalized = str(value or "overall").strip().lower().replace(" ", "_")
    aliases = {
        "group:movers": "movers",
        "moving_heads": "movers",
        "group:center": "center",
        "multi_effect": "center",
        "center_effect": "center",
        "whole_rig": "overall",
        "all": "overall",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MOTION_SCOPES:
        raise ValueError("motion scope must be movers, center, or overall")
    return normalized


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


@dataclass(frozen=True, slots=True)
class CenterMotionTuning:
    """Authored controls for the characterized 19-channel center fixture.

    Travel and output levels are normalized 0..1.  Phase is expressed in
    cycles (0..1), speed is a musical-rate multiplier, and direction is -1 or
    1.  Keeping these values semantic lets another characterized center
    fixture translate them at its profile boundary later.
    """

    cycle_beats: float = 8.0
    relationship: str = "synchronized"
    body_direction: int = 1
    body_speed: float = 1.0
    body_travel: float = 0.75
    body_phase: float = 0.0
    arm_1_direction: int = 1
    arm_1_speed: float = 1.0
    arm_1_travel: float = 0.85
    arm_1_phase: float = 0.0
    arm_2_direction: int = 1
    arm_2_speed: float = 1.0
    arm_2_travel: float = 0.85
    arm_2_phase: float = 0.0
    emitter_pattern: str = "both"
    color_pattern: str = "alternate"
    laser_mode: str = "beat"
    laser_level: float = 0.65
    strip_program: int = 0
    strip_speed: float = 0.55
    strobe_level: float = 0.0
    intensity: float = 1.0
    blackout_accent: float = 0.0

    def patch(self, values: dict[str, Any]) -> "CenterMotionTuning":
        relationship = str(values.get("relationship", self.relationship))
        if relationship not in RELATIONSHIPS:
            raise ValueError("unknown center arm relationship")
        emitter = str(values.get("emitter_pattern", self.emitter_pattern))
        if emitter not in CENTER_EMITTER_PATTERNS:
            raise ValueError("unknown center emitter pattern")
        color = str(values.get("color_pattern", self.color_pattern))
        if color not in CENTER_COLOR_PATTERNS:
            raise ValueError("unknown center color pattern")
        laser = str(values.get("laser_mode", self.laser_mode))
        if laser not in CENTER_LASER_MODES:
            raise ValueError("unknown center laser mode")

        def direction(name: str, current: int) -> int:
            result = int(values.get(name, current))
            if result not in {-1, 1}:
                raise ValueError(f"{name} must be -1 or 1")
            return result

        def unit(name: str, current: float) -> float:
            return clamp(float(values.get(name, current)), 0.0, 1.0)

        def rate(name: str, current: float) -> float:
            return clamp(float(values.get(name, current)), 0.125, 4.0)

        return replace(
            self,
            cycle_beats=clamp(float(values.get("cycle_beats", self.cycle_beats)), 1.0, 64.0),
            relationship=relationship,
            body_direction=direction("body_direction", self.body_direction),
            body_speed=rate("body_speed", self.body_speed),
            body_travel=unit("body_travel", self.body_travel),
            body_phase=float(values.get("body_phase", self.body_phase)) % 1.0,
            arm_1_direction=direction("arm_1_direction", self.arm_1_direction),
            arm_1_speed=rate("arm_1_speed", self.arm_1_speed),
            arm_1_travel=unit("arm_1_travel", self.arm_1_travel),
            arm_1_phase=float(values.get("arm_1_phase", self.arm_1_phase)) % 1.0,
            arm_2_direction=direction("arm_2_direction", self.arm_2_direction),
            arm_2_speed=rate("arm_2_speed", self.arm_2_speed),
            arm_2_travel=unit("arm_2_travel", self.arm_2_travel),
            arm_2_phase=float(values.get("arm_2_phase", self.arm_2_phase)) % 1.0,
            emitter_pattern=emitter,
            color_pattern=color,
            laser_mode=laser,
            laser_level=unit("laser_level", self.laser_level),
            strip_program=max(0, min(255, int(values.get("strip_program", self.strip_program)))),
            strip_speed=unit("strip_speed", self.strip_speed),
            strobe_level=unit("strobe_level", self.strobe_level),
            intensity=unit("intensity", self.intensity),
            blackout_accent=unit("blackout_accent", self.blackout_accent),
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


DEFAULT_CENTER_MOTION_TUNINGS: dict[str, CenterMotionTuning] = {
    "breathe": CenterMotionTuning(
        cycle_beats=16.0, body_speed=0.5, body_travel=0.25,
        arm_1_speed=0.5, arm_2_speed=0.5,
        arm_1_travel=0.30, arm_2_travel=0.30,
        relationship="mirrored", laser_mode="off", strip_speed=0.18,
        intensity=0.72,
    ),
    "fan_sweep": CenterMotionTuning(
        cycle_beats=8.0, body_speed=0.75, body_travel=0.55,
        arm_1_speed=1.5, arm_2_speed=1.5,
        arm_1_travel=0.80, arm_2_travel=0.80,
        relationship="chase", emitter_pattern="alternate",
    ),
    "figure_eight": CenterMotionTuning(
        cycle_beats=16.0, body_speed=2.0, body_travel=1.0,
        arm_1_speed=2.0, arm_2_speed=2.0,
        arm_1_travel=0.90, arm_2_travel=0.90,
        arm_2_phase=0.25, relationship="counter", emitter_pattern="chase",
    ),
    "opposing_chase": CenterMotionTuning(
        cycle_beats=8.0, body_travel=0.70,
        arm_1_travel=1.0, arm_2_travel=1.0,
        relationship="opposed", emitter_pattern="alternate",
        color_pattern="opposed", blackout_accent=0.16,
    ),
    "beat_nod": CenterMotionTuning(
        cycle_beats=4.0, body_speed=0.5, body_travel=0.15,
        arm_1_speed=2.0, arm_2_speed=2.0,
        arm_1_travel=0.95, arm_2_travel=0.95,
        relationship="opposed", emitter_pattern="alternate",
    ),
    "counter_rotate": CenterMotionTuning(
        cycle_beats=16.0, body_speed=1.0, body_travel=0.85,
        arm_1_speed=0.75, arm_2_speed=0.75,
        arm_1_travel=1.0, arm_2_travel=1.0,
        relationship="counter", emitter_pattern="chase",
    ),
}


@dataclass(frozen=True, slots=True)
class GroupMotionTunings:
    movers: dict[str, MotionTuning]
    center: dict[str, CenterMotionTuning]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "groups": {
                "movers": {
                    "routines": {
                        name: tuning.as_dict()
                        for name, tuning in self.movers.items()
                    },
                },
                "center": {
                    "routines": {
                        name: tuning.as_dict()
                        for name, tuning in self.center.items()
                    },
                },
            },
        }


def _routine_bucket(payload: dict[str, Any], scope: str) -> dict[str, Any]:
    groups = payload.get("groups")
    if isinstance(groups, dict):
        group = groups.get(scope, {})
    else:
        group = payload.get(scope, {})
    if not isinstance(group, dict):
        return {}
    routines = group.get("routines", group)
    return routines if isinstance(routines, dict) else {}


def merged_group_motion_tunings(payload: dict[str, Any] | None) -> GroupMotionTunings:
    """Load version-2 group settings or migrate the existing flat JSON.

    The old file tuned both fixture families with one object. Its pan/tilt
    fields continue to tune movers, while body/arm fields seed the equivalent
    center controls. Nothing in an owner's current file is discarded.
    """

    source = payload if isinstance(payload, dict) else {}
    has_groups = isinstance(source.get("groups"), dict) or any(
        key in source for key in ("movers", "center")
    )
    mover_values = _routine_bucket(source, "movers") if has_groups else source
    center_values = _routine_bucket(source, "center") if has_groups else source
    movers = dict(DEFAULT_MOTION_TUNINGS)
    center = dict(DEFAULT_CENTER_MOTION_TUNINGS)
    for routine in DEFAULT_MOTION_TUNINGS:
        values = mover_values.get(routine)
        if isinstance(values, dict):
            movers[routine] = movers[routine].patch(values)
        values = center_values.get(routine)
        if isinstance(values, dict):
            if not has_groups:
                values = {
                    "cycle_beats": values.get("cycle_beats", center[routine].cycle_beats),
                    "relationship": values.get("relationship", center[routine].relationship),
                    "body_direction": values.get("direction", center[routine].body_direction),
                    "body_travel": values.get("body_size", center[routine].body_travel),
                    "arm_1_direction": values.get("direction", center[routine].arm_1_direction),
                    "arm_2_direction": values.get("direction", center[routine].arm_2_direction),
                    "arm_1_travel": values.get("arm_size", center[routine].arm_1_travel),
                    "arm_2_travel": values.get("arm_size", center[routine].arm_2_travel),
                }
            center[routine] = center[routine].patch(values)
    return GroupMotionTunings(movers=movers, center=center)


def merged_motion_tunings(payload: dict[str, Any] | None) -> dict[str, MotionTuning]:
    """Compatibility loader returning the mover half of grouped settings."""

    return merged_group_motion_tunings(payload).movers


def merged_center_motion_tunings(
    payload: dict[str, Any] | None,
) -> dict[str, CenterMotionTuning]:
    return merged_group_motion_tunings(payload).center


def center_motion_coordinates(
    routine: str,
    beat_position: float,
    tuning: CenterMotionTuning,
) -> tuple[float, float, float]:
    """Return independent normalized body/arm positions on a musical clock."""

    cycles = beat_position / tuning.cycle_beats

    def wave(
        speed: float,
        direction: int,
        phase: float,
        *,
        body_axis: bool = False,
    ) -> float:
        angle = math.tau * (cycles * speed * direction + phase)
        if routine == "figure_eight":
            return math.sin(angle * 2.0)
        if routine == "beat_nod":
            return -math.cos(angle * 2.0)
        if routine == "counter_rotate" and body_axis:
            return math.cos(angle)
        return math.sin(angle)

    body = wave(
        tuning.body_speed,
        tuning.body_direction,
        tuning.body_phase,
        body_axis=True,
    )
    arm_1 = wave(tuning.arm_1_speed, tuning.arm_1_direction, tuning.arm_1_phase)
    arm_2_phase = tuning.arm_2_phase
    arm_2_direction = tuning.arm_2_direction
    invert = False
    if tuning.relationship == "opposed":
        arm_2_phase += 0.5
    elif tuning.relationship == "mirrored":
        invert = True
    elif tuning.relationship == "chase":
        arm_2_phase += 0.25
    elif tuning.relationship == "counter":
        arm_2_direction *= -1
    arm_2 = wave(tuning.arm_2_speed, arm_2_direction, arm_2_phase)
    if invert:
        arm_2 = -arm_2
    return (
        clamp(body * tuning.body_travel, -1.0, 1.0),
        clamp(arm_1 * tuning.arm_1_travel, -1.0, 1.0),
        clamp(arm_2 * tuning.arm_2_travel, -1.0, 1.0),
    )


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
