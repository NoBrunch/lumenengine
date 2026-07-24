"""Declarative fixture knowledge migrated from Party Parrot.

The initial registry covers every fixture type exposed by Party Parrot's show
editor and carries exact channel semantics for the active garage fixtures.
Additional channel maps can be filled in incrementally without changing the
runtime or spatial engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MotionKind(StrEnum):
    NONE = "none"
    MOVING_HEAD = "moving_head"
    ROTATING_MULTI_EFFECT = "rotating_multi_effect"
    MOTION_STRIP = "motion_strip"


@dataclass(frozen=True, slots=True)
class FixtureProfile:
    key: str
    label: str
    dmx_footprint: int
    channels: dict[str, int] = field(default_factory=dict)
    motion_kind: MotionKind = MotionKind.NONE
    pan_degrees: float | None = None
    tilt_degrees: float | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.dmx_footprint <= 512:
            raise ValueError("fixture footprint must be in [1, 512]")
        for name, relative_channel in self.channels.items():
            if not 1 <= relative_channel <= self.dmx_footprint:
                raise ValueError(
                    f"{self.key}.{name} channel {relative_channel} is outside "
                    f"its {self.dmx_footprint}-channel footprint"
                )


PARTY_PARROT_PROFILES: dict[str, FixtureProfile] = {
    "manual_dimmer_channel": FixtureProfile(
        "manual_dimmer_channel", "Manual Dimmer Channel", 1, {"dimmer": 1}
    ),
    "par_rgb": FixtureProfile("par_rgb", "LED Par RGB", 7),
    "par_rgbawu": FixtureProfile("par_rgbawu", "Par RGBAWU", 9),
    "generic_rgbw_moving_head_11ch": FixtureProfile(
        key="generic_rgbw_moving_head_11ch",
        label="Generic RGBW Moving Head (11ch)",
        dmx_footprint=11,
        channels={
            "pan_coarse": 1,
            "pan_fine": 2,
            "tilt_coarse": 3,
            "tilt_fine": 4,
            "movement_speed": 5,
            "dimmer": 6,
            "strobe": 7,
            "red": 8,
            "green": 9,
            "blue": 10,
            "white": 11,
        },
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540.0,
        tilt_degrees=270.0,
        capabilities=("pan", "tilt", "dimmer", "strobe", "rgbw"),
    ),
    "generic_multi_effect_19ch": FixtureProfile(
        key="generic_multi_effect_19ch",
        label="Triple-Emitter Rotating Multi Effect (19ch)",
        dmx_footprint=19,
        channels={
            "body_rotation": 1,
            "body_rotation_speed": 2,
            "arm_1_motor": 3,
            "arm_2_motor": 4,
            "master_dimmer": 5,
            "strobe": 6,
            "magic_ball_red": 7,
            "magic_ball_green": 8,
            "magic_ball_blue": 9,
            "magic_ball_white": 10,
            "arm_beams_red": 11,
            "arm_beams_green": 12,
            "arm_beams_blue": 13,
            "arm_beams_white": 14,
            "red_laser": 15,
            "green_laser": 16,
            "strip_program": 17,
            "strip_speed": 18,
            "macro": 19,
        },
        motion_kind=MotionKind.ROTATING_MULTI_EFFECT,
        pan_degrees=300.0,
        tilt_degrees=180.0,
        capabilities=(
            "body_rotation",
            "dual_arm",
            "dimmer",
            "strobe",
            "rgbw",
            "laser",
            "strip",
            "macro",
        ),
    ),
    "chauvet_spot_110": FixtureProfile(
        "chauvet_spot_110",
        "Chauvet Intimidator 110/120",
        12,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "chauvet_spot_160": FixtureProfile(
        "chauvet_spot_160",
        "Chauvet Intimidator 160",
        11,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "chauvet_rogue_beam_r2x": FixtureProfile(
        "chauvet_rogue_beam_r2x",
        "Chauvet Rogue Beam R2X",
        18,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "chauvet_rogue_hybrid_rh1": FixtureProfile(
        "chauvet_rogue_hybrid_rh1",
        "Chauvet Rogue Hybrid RH1 (20ch)",
        20,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "chauvet_rogue_hybrid_rh1_25ch": FixtureProfile(
        "chauvet_rogue_hybrid_rh1_25ch",
        "Chauvet Rogue Hybrid RH1 (25ch)",
        25,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "motionstrip_38": FixtureProfile(
        "motionstrip_38",
        "Motionstrip (38ch)",
        38,
        motion_kind=MotionKind.MOTION_STRIP,
    ),
    "five_beam_laser": FixtureProfile("five_beam_laser", "Five Beam Laser", 13),
    "two_beam_laser": FixtureProfile("two_beam_laser", "Two Beam Laser", 10),
    "chauvet_slimpar_pro_q_5ch": FixtureProfile(
        "chauvet_slimpar_pro_q_5ch", "Chauvet SlimPAR Pro Q", 5
    ),
    "chauvet_slimpar_pro_h_7ch": FixtureProfile(
        "chauvet_slimpar_pro_h_7ch", "Chauvet SlimPAR Pro H", 7
    ),
    "chauvet_par_rgbawu": FixtureProfile(
        "chauvet_par_rgbawu", "Chauvet Par RGBAWU", 7
    ),
    "chauvet_derby": FixtureProfile("chauvet_derby", "Chauvet Derby", 6),
    "chauvet_rotosphere_28ch": FixtureProfile(
        "chauvet_rotosphere_28ch", "Chauvet Rotosphere", 28
    ),
    "chauvet_move_9ch": FixtureProfile(
        "chauvet_move_9ch",
        "Chauvet Move",
        12,
        motion_kind=MotionKind.MOVING_HEAD,
        pan_degrees=540,
        tilt_degrees=270,
    ),
    "chauvet_colorband_pix_36ch": FixtureProfile(
        "chauvet_colorband_pix_36ch", "Chauvet Colorband PiX", 36
    ),
    "mirrorball": FixtureProfile("mirrorball", "Mirrorball", 4),
}

# Historical aliases found in older Party Parrot databases.
PARTY_PARROT_PROFILE_ALIASES = {
    "chauvet_rogue_beam_r2": "chauvet_rogue_beam_r2x",
}


def party_parrot_profile(key: str) -> FixtureProfile | None:
    normalized = PARTY_PARROT_PROFILE_ALIASES.get(key, key)
    return PARTY_PARROT_PROFILES.get(normalized)


def profile_summary(profile: FixtureProfile) -> dict[str, Any]:
    return {
        "key": profile.key,
        "label": profile.label,
        "dmx_footprint": profile.dmx_footprint,
        "channels": dict(profile.channels),
        "motion_kind": profile.motion_kind.value,
        "pan_degrees": profile.pan_degrees,
        "tilt_degrees": profile.tilt_degrees,
        "capabilities": list(profile.capabilities),
    }

