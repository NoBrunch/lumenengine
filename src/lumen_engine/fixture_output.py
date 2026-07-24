"""Turn semantic Lumen decisions into Party Parrot-compatible fixture channels."""

from __future__ import annotations

import math

from lumen_engine.dmx import DMXFrame
from lumen_engine.models import (
    FixturePatch,
    Gesture,
    PerformanceDecision,
    ProfileFixturePatch,
    clamp,
)
from lumen_engine.profiles import party_parrot_profile


def expression_rgb(decision: PerformanceDecision) -> tuple[float, float, float]:
    """A small baseline palette that can later be replaced by learned choices."""

    energy = decision.expression.energy
    tension = decision.expression.tension
    intimacy = decision.expression.intimacy
    red = clamp(0.08 + 0.82 * tension + 0.20 * energy, 0.0, 1.0)
    green = clamp(0.10 + 0.34 * intimacy + 0.18 * energy, 0.0, 1.0)
    blue = clamp(0.30 + 0.62 * (1.0 - tension) + 0.10 * energy, 0.0, 1.0)
    if decision.gesture is Gesture.RELEASE:
        return 1.0, clamp(green + 0.28, 0, 1), clamp(blue + 0.18, 0, 1)
    return red, green, blue


def rgb_to_rgbw(rgb: tuple[float, float, float]) -> tuple[int, int, int, int]:
    red, green, blue = (clamp(value, 0.0, 1.0) for value in rgb)
    white = min(red, green, blue)
    return (
        round((red - white) * 255),
        round((green - white) * 255),
        round((blue - white) * 255),
        round(white * 255),
    )


def apply_moving_head_profile(
    frame: DMXFrame,
    fixture: FixturePatch,
    decision: PerformanceDecision,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_rgbw_moving_head_11ch":
        channels = profile.channels
        _set_relative(frame, fixture.universe, fixture.address, channels["movement_speed"], 0)
        _set_relative(frame, fixture.universe, fixture.address, channels["strobe"], 0)
        for name, value in zip(
            ("red", "green", "blue", "white"),
            rgb_to_rgbw(expression_rgb(decision)),
        ):
            _set_relative(
                frame, fixture.universe, fixture.address, channels[name], value
            )


def apply_auxiliary_fixture(
    frame: DMXFrame,
    fixture: ProfileFixturePatch,
    decision: PerformanceDecision,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_multi_effect_19ch":
        _apply_generic_multi_effect(frame, fixture, decision)
        return
    dimmer = profile.channels.get("dimmer")
    if dimmer is not None:
        _set_relative(
            frame,
            fixture.universe,
            fixture.address,
            dimmer,
            round(decision.brightness * 255),
        )


def _apply_generic_multi_effect(
    frame: DMXFrame,
    fixture: ProfileFixturePatch,
    decision: PerformanceDecision,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    assert profile is not None
    channels = profile.channels
    timestamp = decision.timestamp_s
    motion = decision.expression.motion
    energy = decision.expression.energy

    body = round((0.5 + 0.48 * math.sin(timestamp * (0.25 + motion))) * 255)
    arm_1 = round((0.5 + 0.46 * math.sin(timestamp * 0.72)) * 255)
    arm_2 = 255 - arm_1
    body_speed = round(clamp(225.0 - motion * 195.0, 0.0, 255.0))
    strobe = 0
    if decision.gesture is Gesture.PULSE:
        strobe = 36
    elif decision.gesture is Gesture.RELEASE:
        strobe = 92

    values = {
        "body_rotation": body,
        "body_rotation_speed": body_speed,
        "arm_1_motor": arm_1,
        "arm_2_motor": arm_2,
        "master_dimmer": round(decision.brightness * 255),
        "strobe": strobe,
        "strip_speed": round(motion * 255),
        "macro": 0,
    }
    rgbw = rgb_to_rgbw(expression_rgb(decision))
    for prefix in ("magic_ball", "arm_beams"):
        for color_name, value in zip(("red", "green", "blue", "white"), rgbw):
            values[f"{prefix}_{color_name}"] = value
    red, green, _blue = expression_rgb(decision)
    values["red_laser"] = round(red * decision.brightness * 255)
    values["green_laser"] = round(green * decision.brightness * 255)
    values["strip_program"] = _strip_color_value(expression_rgb(decision))

    for channel_name, value in values.items():
        _set_relative(
            frame,
            fixture.universe,
            fixture.address,
            channels[channel_name],
            max(0, min(255, int(value))),
        )


def _strip_color_value(rgb: tuple[float, float, float]) -> int:
    red, green, blue = rgb
    if min(rgb) >= max(rgb) * 0.72:
        return 71
    if red >= green * 1.35 and red >= blue * 1.35:
        return 17
    if green >= red * 1.35 and green >= blue * 1.35:
        return 25
    if blue >= red * 1.35 and blue >= green * 1.35:
        return 34
    if red >= blue and green >= blue:
        return 44
    if red >= green:
        return 53
    return 62


def _set_relative(
    frame: DMXFrame,
    universe: int,
    address: int,
    relative_channel: int,
    value: int,
) -> None:
    frame.set_channel(universe, address + relative_channel - 1, value)

