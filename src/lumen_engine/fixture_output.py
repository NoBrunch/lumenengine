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
from lumen_engine.motion import MotionTuning
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


def _strobe_request(
    observation: MusicalObservation | None,
    energy: float,
    strobe_feedback: float,
    choreography_strobe: float,
    *,
    automatic: bool,
) -> float:
    """Return a sustained, normalized request for a fixture strobe channel.

    A fixture strobe channel is a rate command, not a one-frame flash trigger.
    Explicit rehearsal/choreography and positive operator feedback therefore
    remain active for the whole frame sequence in which they apply.  Automatic
    beat strobe uses a visible section of the detected beat instead of the
    much narrower onset envelope; repeatedly writing a non-zero value for only
    one analysis frame merely looks like an irregular flicker on the physical
    fixtures.
    """

    # Strong negative feedback is an operator veto even when the remembered
    # sequence contains a strobe step. Milder negative/positive feedback trims
    # that step's hardware rate, so slower/faster have literal meaning.
    if strobe_feedback <= -0.60:
        return 0.0
    if choreography_strobe > 0.05:
        return clamp(
            choreography_strobe + 0.5 * strobe_feedback,
            0.06,
            1.0,
        )
    if strobe_feedback > 0.05:
        return clamp(strobe_feedback, 0.0, 1.0)
    if not automatic or strobe_feedback < -0.2 or observation is None:
        return 0.0
    if (
        energy < 0.70
        or observation.beat_confidence < 0.25
        or observation.beat_phase > 0.42
    ):
        return 0.0
    return clamp(0.48 + 0.42 * energy, 0.0, 1.0)


def _direct_strobe_dmx(request: float) -> int:
    """Map a normalized request to the direct 0=off, 1..255 mover channel."""

    if request <= 0.0:
        return 0
    return round(18.0 + 237.0 * clamp(request, 0.0, 1.0))


def _multi_effect_strobe_dmx(request: float) -> int:
    """Map to the characterized center fixture range: 0 off, 10..255 active."""

    if request <= 0.0:
        return 0
    logical = round(clamp(request, 0.0, 1.0) * 255.0)
    return 10 + round(logical / 255.0 * 245.0)


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
    choreography_strobe: float = 0.0,
    palette_bias: float = 0.0,
    enabled: bool = True,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    if profile is None:
        return
    if fixture.profile_key == "generic_rgbw_moving_head_11ch":
        channels = profile.channels
        if not enabled:
            for name in ("strobe", "red", "green", "blue", "white"):
                _set_relative(
                    frame, fixture.universe, fixture.address,
                    channels[name], 0,
                )
            _set_relative(
                frame, fixture.universe, fixture.address,
                channels["movement_speed"], 200,
            )
            return
        speed = round(200.0 * clamp(idle_amount, 0.0, 1.0))
        _set_relative(frame, fixture.universe, fixture.address, channels["movement_speed"], speed)
        strobe_request = (
            0.0
            if idle_amount >= 1.0
            else _strobe_request(
                observation,
                decision.expression.energy,
                strobe_feedback,
                choreography_strobe,
                automatic=False,
            )
        )
        strobe = _direct_strobe_dmx(strobe_request)
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
    choreography_strobe: float = 0.0,
    palette_bias: float = 0.0,
    enabled: bool = True,
    motion_tuning: MotionTuning | None = None,
    motion_timestamp_s: float | None = None,
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
            choreography_strobe,
            palette_bias,
            enabled,
            motion_tuning,
            motion_timestamp_s,
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
    choreography_strobe: float = 0.0,
    palette_bias: float = 0.0,
    enabled: bool = True,
    motion_tuning: MotionTuning | None = None,
    motion_timestamp_s: float | None = None,
) -> None:
    profile = party_parrot_profile(fixture.profile_key)
    assert profile is not None
    channels = profile.channels
    if not enabled:
        values = {
            "body_rotation": 128,
            "body_rotation_speed": 255,
            "arm_1_motor": 128,
            "arm_2_motor": 128,
            "master_dimmer": 0,
            "strobe": 0,
            "red_laser": 0,
            "green_laser": 0,
            "strip_program": 0,
            "strip_speed": 0,
            "macro": 0,
        }
        for prefix in ("magic_ball", "arm_beams"):
            for color_name in ("red", "green", "blue", "white"):
                values[f"{prefix}_{color_name}"] = 0
        for channel_name, value in values.items():
            _set_relative(
                frame, fixture.universe, fixture.address,
                channels[channel_name], value,
            )
        return
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
    timestamp = (
        decision.timestamp_s
        if motion_timestamp_s is None
        else motion_timestamp_s
    )
    motion = decision.expression.motion
    energy = decision.expression.energy
    beat_pulse = 0.0 if observation is None else observation.beat_pulse
    bar_phase = 0.0 if observation is None else observation.bar_phase
    beat_confidence = (
        0.0 if observation is None else observation.beat_confidence
    )
    activity = clamp(
        0.02 + 0.85 * energy + 0.35 * motion + 0.55 * motion_feedback,
        0.0,
        1.0,
    )
    tuning = motion_tuning
    cycle_beats = tuning.cycle_beats if tuning is not None else 4.0
    if beat_confidence >= 0.12:
        pattern_phase = (
            timestamp * (observation.bpm or 120.0) / 60.0 / cycle_beats
        ) % 1.0
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
    pattern = {
        "breathe": 4,
        "opposing_chase": 1,
        "figure_eight": 2,
        "beat_nod": 3,
        "fan_sweep": 4,
        "counter_rotate": 5,
    }.get(decision.routine, bar_number % 6)
    # Give the side arms the full 8-bit travel at performance energy; the
    # resolver no longer compresses them into a small nod.
    motion_scale = clamp(0.08 + 0.92 * activity, 0.05, 1.0)
    if decision.gesture is Gesture.BREATHE:
        motion_scale *= 0.38
    # Use the detected bar phase when available so the compound fixture is
    # locked to the same beat grid as the movers instead of free-running from
    # the process clock.
    motion_phase = (
        pattern_phase * math.tau
        if beat_confidence >= 0.12
        else timestamp * (observation.bpm or 120.0) / 60.0 * math.tau / 4.0
    )
    if pattern == 0:  # chase: each arm follows the other by a quarter cycle
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase), math.sin(motion_phase), math.sin(motion_phase + math.pi / 2)
    elif pattern == 1:  # true opposing sweep
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase), math.sin(motion_phase), -math.sin(motion_phase)
    elif pattern == 2:  # figure-eight style independent phases
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase * 2), math.sin(motion_phase * 2), -math.cos(motion_phase * 2)
    elif pattern == 3:  # smooth beat nod with opposing arms
        body_motion = math.sin(motion_phase)
        arm_1_motion = math.sin(motion_phase * 2.0)
        arm_2_motion = -arm_1_motion
    elif pattern == 4:  # broad fan: body and arms use different rates
        body_motion, arm_1_motion, arm_2_motion = math.sin(motion_phase), math.sin(motion_phase * 1.5), math.cos(motion_phase * 1.5)
    else:  # counter-rotating circles
        body_motion, arm_1_motion, arm_2_motion = math.cos(motion_phase), math.sin(motion_phase * .75), -math.sin(motion_phase * .75)
    body_scale = tuning.body_size if tuning is not None else 1.0
    arm_scale_motion = tuning.arm_size if tuning is not None else 1.0
    body = round(128.0 + 127.0 * motion_scale * body_scale * body_motion)
    arm_1 = round(128.0 + 127.0 * motion_scale * arm_scale_motion * arm_1_motion)
    arm_2 = round(128.0 + 127.0 * motion_scale * arm_scale_motion * arm_2_motion)
    # This fixture's CH2 is inverted: zero is fastest and 255 is slowest.
    body_speed = round(235.0 - 210.0 * activity)
    # Channel 6 is the fixture's internal strobe-rate control. Keep it active
    # for the requested interval; do not pulse the channel itself for a single
    # analysis frame. The fixture then produces a clean, regular strobe at the
    # requested rate instead of an irregular software flicker.
    strobe = _multi_effect_strobe_dmx(
        _strobe_request(
            observation,
            energy,
            strobe_feedback,
            choreography_strobe,
            automatic=True,
        )
    )

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
