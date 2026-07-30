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


# One authoritative palette vocabulary shared by the desktop controls and the
# DMX resolver.  ``midnight_teal`` is retained as a real family because it was
# already persisted in this installation; it is not allowed to silently fall
# through to automatic selection.
PALETTE_FAMILIES: dict[str, tuple[int, ...] | None] = {
    "auto": None,
    "party_vivid": (0, 1, 2, 3, 4, 5, 6),
    "midnight_teal": (1, 2, 5),
    "cool": (1, 2, 5, 6),
    "warm": (3, 4, 6, 0),
    "magenta_blue": (0, 1, 5),
    "cyan_violet": (1, 2, 5),
    "red_amber": (3, 4, 6),
}


def expression_rgb(decision: PerformanceDecision, palette_bias: float = 0.0) -> tuple[float, float, float]:
    """Resolve Lumen expressions through Party Parrot's saturated show palette.

    Party Parrot's useful character came from rotating three-color schemes,
    not from continuously mixing toward gray.  Keep that behavior here while
    allowing the learned tension/intimacy values to choose how boldly a scheme
    is presented.
    """

    energy = decision.expression.energy
    tension = decision.expression.tension
    intimacy = decision.expression.intimacy
    palettes = (
        (1.00, 0.00, 0.42),  # magenta / blue / purple family
        (0.18, 0.00, 1.00),  # blue / purple
        (0.05, 0.78, 1.00),  # cyan / blue
        (1.00, 0.08, 0.02),  # red / blue contrast
        (1.00, 0.55, 0.02),  # orange / coral
        (0.55, 0.08, 1.00),  # violet
        (0.95, 0.95, 1.00),  # white hit
    )
    mode = PALETTE_FAMILIES.get(decision.palette_hint)
    palette_index = int(max(0.0, decision.timestamp_s) / 8.0 + tension * 2.0 + palette_bias * 2.0)
    if mode:
        palette_index = mode[palette_index % len(mode)]
    else:
        palette_index %= len(palettes)
    red, green, blue = palettes[palette_index]
    # Keep lower-energy passages atmospheric without washing out the palette.
    saturation = clamp(0.62 + 0.38 * energy + 0.10 * intimacy, 0.0, 1.0)
    red *= saturation
    green *= saturation
    blue *= saturation
    # Release changes movement/brightness, not color to white. This keeps a
    # release from being mistaken for a strobe hit.
    if energy < 0.46 and decision.palette_hint in {"auto", "cool", "cyan_violet"}:
        # Soft acoustic/jazz material defaults to cool, low-saturation color.
        red = 0.18
        green = 0.08
        blue = 0.62
    if palette_bias < -0.05:
        red = clamp(red * 0.55, 0.0, 1.0)
        green = clamp(green * 0.75, 0.0, 1.0)
        blue = clamp(blue + 0.22, 0.0, 1.0)
    elif palette_bias > 0.05:
        red = clamp(red + 0.20, 0.0, 1.0)
        green = clamp(green + 0.08, 0.0, 1.0)
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
    idle_amount: float = 0.0,
    motion_feedback: float = 0.0,
    strobe_feedback: float = 0.0,
    palette_bias: float = 0.0,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_rgbw_moving_head_11ch":
        channels = profile.channels
        speed = round(200.0 * clamp(idle_amount, 0.0, 1.0))
        _set_relative(frame, fixture.universe, fixture.address, channels["movement_speed"], speed)
        beat_pulse = 0.0 if observation is None else observation.beat_pulse
        strobe = (
            round(clamp(80 + 135 * beat_pulse + 60 * strobe_feedback, 0.0, 255.0))
            if (
                idle_amount < 1.0
                and strobe_feedback > 0.25
                and beat_pulse >= 0.78
                and decision.expression.energy >= 0.70
            )
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
            rgb_to_rgbw(expression_rgb(decision, palette_bias)),
        ):
            _set_relative(
                frame, fixture.universe, fixture.address, channels[name], value
            )


def apply_auxiliary_fixture(
    frame: DMXFrame,
    fixture: ProfileFixturePatch,
    decision: PerformanceDecision,
    observation: MusicalObservation | None = None,
    idle_amount: float = 0.0,
    motion_feedback: float = 0.0,
    strobe_feedback: float = 0.0,
    palette_bias: float = 0.0,
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
            idle_amount,
            motion_feedback,
            strobe_feedback,
            palette_bias,
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
    idle_amount: float = 0.0,
    motion_feedback: float = 0.0,
    strobe_feedback: float = 0.0,
    palette_bias: float = 0.0,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    assert profile is not None
    channels = profile.channels
    if idle_amount >= 1.0:
        rgbw = rgb_to_rgbw(expression_rgb(decision, palette_bias))
        values = {
            "body_rotation": 128,
            "body_rotation_speed": 200,
            "arm_1_motor": 128,
            "arm_2_motor": 128,
            "master_dimmer": 24,
            "strobe": 0,
            "red_laser": 0,
            "green_laser": 0,
            "strip_program": _strip_color_value(expression_rgb(decision)),
            "strip_speed": 0,
            "macro": 0,
        }
        for prefix in ("magic_ball", "arm_beams"):
            for color_name, value in zip(("red", "green", "blue", "white"), rgbw):
                values[f"{prefix}_{color_name}"] = round(value * 0.20)
        for channel_name, value in values.items():
            _set_relative(
                frame,
                fixture.universe,
                fixture.address,
                channels[channel_name],
                max(0, min(255, int(value))),
            )
        return
    timestamp = decision.timestamp_s
    motion = decision.expression.motion
    energy = decision.expression.energy
    beat_pulse = 0.0 if observation is None else observation.beat_pulse
    bar_phase = 0.0 if observation is None else observation.bar_phase
    beat_confidence = (
        0.0 if observation is None else observation.beat_confidence
    )
    activity = clamp(
        0.20 + 0.48 * energy + 0.42 * motion + 0.55 * motion_feedback,
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
    # Change the gesture every bar: opposing sweeps, circles, alternating
    # arms, and a restrained nod. Motor speed follows energy so the effect
    # visibly settles with the music instead of racing through quiet parts.
    bar_number = int(timestamp * (observation.bpm or 120.0) / 60.0 / 4.0)
    pattern = bar_number % 6
    # Give the side arms the full 8-bit travel at performance energy; the
    # resolver no longer compresses them into a small nod.
    motion_scale = clamp(0.35 + 0.90 * activity, 0.30, 1.0)
    if decision.gesture is Gesture.BREATHE:
        motion_scale *= 0.56
    # Use the detected bar phase when available so the compound fixture is
    # locked to the same beat grid as the movers instead of free-running from
    # the process clock.
    motion_phase = (
        bar_phase * math.tau
        if beat_confidence >= 0.12
        else timestamp * (observation.bpm or 120.0) / 60.0 * math.tau / 4.0
    )
    if pattern == 0:  # chase: each arm follows the other by a quarter cycle
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase), math.sin(motion_phase), math.sin(motion_phase + math.pi / 2)
    elif pattern == 1:  # true opposing sweep
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase * .5), math.sin(motion_phase), -math.sin(motion_phase)
    elif pattern == 2:  # figure-eight style independent phases
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase * 2), math.sin(motion_phase * 2), -math.cos(motion_phase * 2)
    elif pattern == 3:  # beat alternation, reaching endpoints on every other beat
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase * 2), (1 if beat_index % 2 == 0 else -1), (-1 if beat_index % 2 == 0 else 1)
    elif pattern == 4:  # broad fan: body and arms use different rates
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase * .5), math.sin(motion_phase * 1.5), math.cos(motion_phase * 1.5)
    else:  # counter-rotating circles
        body_motion, arm_1_motion, arm_2_motion = math.cos(motion_phase), math.sin(motion_phase * .75), -math.sin(motion_phase * .75)
    body = round(128.0 + 127.0 * motion_scale * body_motion)
    arm_1 = round(128.0 + 127.0 * motion_scale * arm_1_motion)
    arm_2 = round(128.0 + 127.0 * motion_scale * arm_2_motion)
    # This fixture's CH2 is inverted: zero is fastest and 255 is slowest.
    body_speed = round(235.0 - 210.0 * activity)
    strobe = 0
    # The center unit keeps Party Parrot's beat-flash behavior; mover strobes
    # remain explicitly off unless requested. This is the fixture that owns
    # the center beat chase, rather than every light flashing together.
    if beat_pulse >= 0.78 and energy >= 0.70 and strobe_feedback > -0.2:
        strobe = round(clamp(70.0 + 150.0 * beat_pulse * activity + 80.0 * strobe_feedback, 0.0, 255.0))

    values = {
        "body_rotation": body,
        "body_rotation_speed": body_speed,
        "arm_1_motor": arm_1,
        "arm_2_motor": arm_2,
        "master_dimmer": round(
            clamp(
                max(decision.brightness, 0.20 + 0.70 * energy)
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
    even_beat = beat_index % 2 == 0
    ball_rgbw = rgb_to_rgbw(expression_rgb(decision, palette_bias + (0.45 if even_beat else -0.45)))
    arm_rgbw = rgb_to_rgbw(expression_rgb(decision, palette_bias + (-0.45 if even_beat else 0.45)))
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
    for prefix, scale, color_values in (
        ("magic_ball", ball_scale, ball_rgbw),
        ("arm_beams", arm_scale, arm_rgbw),
    ):
        for color_name, value in zip(("red", "green", "blue", "white"), color_values):
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
        # Party Parrot exposes a larger bank of built-in ring programs; walk
        # that bank by bar instead of repeating the same four programs.
        effect = 1 + (bar_number % 20)
        values["strip_program"] = 76 + (effect - 1) * 9 + 4
        values["strip_speed"] = round(58 + 186 * activity)
    else:
        values["strip_program"] = _strip_color_value(
            expression_rgb(decision, palette_bias)
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
