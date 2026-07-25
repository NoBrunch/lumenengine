"""Turn semantic Lumen decisions into Party Parrot-compatible fixture channels."""

from __future__ import annotations

import math

from lumen_engine.dmx import DMXFrame
from lumen_engine.models import (
    FixturePatch,
    Gesture,
    MusicalObservation,
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
    observation: MusicalObservation | None = None,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_rgbw_moving_head_11ch":
        channels = profile.channels
        _set_relative(frame, fixture.universe, fixture.address, channels["movement_speed"], 0)
        beat_pulse = 0.0 if observation is None else observation.beat_pulse
        strobe = (
            round(22 + 76 * beat_pulse)
            if beat_pulse >= 0.52 and decision.expression.energy >= 0.42
            else 0
        )
        _set_relative(
            frame,
            fixture.universe,
            fixture.address,
            channels["strobe"],
            strobe,
        )
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
    observation: MusicalObservation | None = None,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_multi_effect_19ch":
        _apply_generic_multi_effect(
            frame,
            fixture,
            decision,
            observation,
        )
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
    observation: MusicalObservation | None = None,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    assert profile is not None
    channels = profile.channels
    timestamp = decision.timestamp_s
    motion = decision.expression.motion
    energy = decision.expression.energy
    beat_pulse = 0.0 if observation is None else observation.beat_pulse
    bar_phase = 0.0 if observation is None else observation.bar_phase
    beat_confidence = (
        0.0 if observation is None else observation.beat_confidence
    )
    activity = clamp(
        0.20 + 0.48 * energy + 0.42 * motion,
        0.0,
        1.0,
    )
    phase = (
        bar_phase * math.tau
        if beat_confidence >= 0.18
        else timestamp * (0.48 + 0.92 * activity)
    )
    beat_index = int(bar_phase * 4.0) % 4
    accent_side = -1.0 if beat_index % 2 else 1.0

    body_amplitude = 50.0 + 66.0 * activity
    arm_amplitude = 42.0 + 58.0 * activity
    body = 128.0 + body_amplitude * math.sin(phase)
    arm_1 = 128.0 + arm_amplitude * math.sin(phase * 1.25)
    arm_2 = 128.0 - arm_amplitude * math.sin(phase * 1.25)
    if beat_pulse > 0.02:
        body += accent_side * 46.0 * beat_pulse
        arm_1 += accent_side * 54.0 * beat_pulse
        arm_2 -= accent_side * 54.0 * beat_pulse
    body = round(clamp(body, 8.0, 247.0))
    arm_1 = round(clamp(arm_1, 12.0, 243.0))
    arm_2 = round(clamp(arm_2, 12.0, 243.0))
    # This fixture's CH2 is inverted: zero is fastest and 255 is slowest.
    body_speed = round(clamp(112.0 - activity * 88.0, 18.0, 116.0))
    strobe = 0
    if beat_pulse >= 0.52 and energy >= 0.38:
        strobe = round(18.0 + 92.0 * beat_pulse * activity)
    elif decision.gesture is Gesture.RELEASE:
        strobe = 92

    values = {
        "body_rotation": body,
        "body_rotation_speed": body_speed,
        "arm_1_motor": arm_1,
        "arm_2_motor": arm_2,
        "master_dimmer": round(decision.brightness * 255),
        "strobe": strobe,
        "strip_speed": 0,
        "macro": 0,
    }
    rgbw = rgb_to_rgbw(expression_rgb(decision))
    even_beat = beat_index % 2 == 0
    contrast = beat_pulse * (0.45 + 0.45 * activity)
    ball_scale = clamp(0.78 + (0.22 if even_beat else -0.48) * contrast, 0.24, 1.0)
    arm_scale = clamp(0.78 + (-0.48 if even_beat else 0.22) * contrast, 0.24, 1.0)
    for prefix, scale in (
        ("magic_ball", ball_scale),
        ("arm_beams", arm_scale),
    ):
        for color_name, value in zip(("red", "green", "blue", "white"), rgbw):
            values[f"{prefix}_{color_name}"] = round(value * scale)

    red, green, _blue = expression_rgb(decision)
    laser_floor = 26.0 + 118.0 * activity
    red_laser = laser_floor * max(0.32, red)
    green_laser = laser_floor * max(0.32, green)
    if beat_pulse >= 0.35:
        if even_beat:
            red_laser += 118.0 * beat_pulse
        else:
            green_laser += 118.0 * beat_pulse
    values["red_laser"] = round(clamp(red_laser, 0.0, 255.0))
    values["green_laser"] = round(clamp(green_laser, 0.0, 255.0))

    if activity >= 0.45:
        effect = 1 + beat_index + (4 if energy >= 0.70 else 0)
        values["strip_program"] = 76 + (effect - 1) * 9 + 4
        values["strip_speed"] = round(58 + 186 * activity)
    else:
        values["strip_program"] = _strip_color_value(
            expression_rgb(decision)
        )

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
