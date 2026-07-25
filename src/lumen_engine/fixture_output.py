"""Turn semantic Lumen decisions into Party Parrot-compatible fixture channels."""

from __future__ import annotations

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
            round(70 + 150 * beat_pulse)
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
    if beat_confidence >= 0.12:
        pattern_phase = bar_phase
    else:
        assumed_bpm = (
            120.0
            if observation is None or observation.bpm is None
            else observation.bpm
        )
        pattern_phase = (timestamp * assumed_bpm / 60.0 / 4.0) % 1.0
    beat_index = int(pattern_phase * 4.0) % 4
    # Hold widely separated motor targets for a complete beat instead of
    # sending a continuously moving sine target that the inexpensive motors
    # can never catch. The fixture itself supplies the physical easing.
    body_pattern = (18.0, 18.0, 237.0, 237.0)
    arm_1_pattern = (20.0, 232.0, 232.0, 20.0)
    arm_2_pattern = (235.0, 235.0, 23.0, 23.0)
    motion_scale = clamp(0.78 + 0.26 * activity, 0.78, 1.0)
    if decision.gesture is Gesture.BREATHE:
        motion_scale *= 0.56
    body = round(128.0 + (body_pattern[beat_index] - 128.0) * motion_scale)
    arm_1 = round(
        128.0 + (arm_1_pattern[beat_index] - 128.0) * motion_scale
    )
    arm_2 = round(
        128.0 + (arm_2_pattern[beat_index] - 128.0) * motion_scale
    )
    # This fixture's CH2 is inverted: zero is fastest and 255 is slowest.
    body_speed = 0
    strobe = 0
    if beat_pulse >= 0.52 and energy >= 0.38:
        strobe = round(70.0 + 150.0 * beat_pulse * activity)

    values = {
        "body_rotation": body,
        "body_rotation_speed": body_speed,
        "arm_1_motor": arm_1,
        "arm_2_motor": arm_2,
        "master_dimmer": round(
            clamp(
                max(decision.brightness, 0.76 + 0.18 * energy)
                + 0.10 * beat_pulse,
                0.0,
                1.0,
            )
            * 255
        ),
        "strobe": strobe,
        "strip_speed": 0,
        "macro": 0,
    }
    rgbw = rgb_to_rgbw(expression_rgb(decision))
    even_beat = beat_index % 2 == 0
    contrast = beat_pulse * (0.45 + 0.45 * activity)
    ball_scale = clamp(
        0.92 + (0.18 if even_beat else -0.72) * contrast,
        0.18,
        1.0,
    )
    arm_scale = clamp(
        0.92 + (-0.72 if even_beat else 0.18) * contrast,
        0.18,
        1.0,
    )
    for prefix, scale in (
        ("magic_ball", ball_scale),
        ("arm_beams", arm_scale),
    ):
        for color_name, value in zip(("red", "green", "blue", "white"), rgbw):
            values[f"{prefix}_{color_name}"] = round(value * scale)

    red, green, _blue = expression_rgb(decision)
    laser_floor = 72.0 + 138.0 * activity
    red_laser = laser_floor * max(0.32, red)
    green_laser = laser_floor * max(0.32, green)
    if beat_pulse >= 0.35:
        if even_beat:
            red_laser += 150.0 * beat_pulse
        else:
            green_laser += 150.0 * beat_pulse
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
