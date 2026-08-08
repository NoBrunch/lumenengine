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
from lumen_engine.motion import (
    CenterMotionTuning,
    MotionTuning,
    center_motion_coordinates,
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

# User-authored colors are kept separate from palette families.  A palette is
# useful for developing a show, but a fixture test needs one unambiguous color
# that does not change because the beat clock advanced.  The control layer
# replaces this registry when the operator edits Color Studio.
_CUSTOM_COLORS: dict[str, tuple[float, float, float]] = {}
_CUSTOM_PALETTES: dict[str, tuple[tuple[float, float, float], ...]] = {}


def configure_color_library(
    colors: dict[str, str] | None = None,
    palettes: dict[str, list[str]] | None = None,
) -> None:
    """Install validated operator colors for the current Lumen process."""

    def parse(value: str) -> tuple[float, float, float] | None:
        raw = str(value).strip().lstrip("#")
        if len(raw) != 6:
            return None
        try:
            return tuple(int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            return None

    parsed_colors: dict[str, tuple[float, float, float]] = {}
    for name, value in (colors or {}).items():
        key = str(name).strip()[:48]
        rgb = parse(str(value))
        if key and rgb is not None:
            parsed_colors[key] = rgb
    parsed_palettes: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for name, values in (palettes or {}).items():
        key = str(name).strip()[:48]
        if not key or not isinstance(values, list):
            continue
        entries = tuple(rgb for item in values if (rgb := parse(str(item))) is not None)
        if entries:
            parsed_palettes[key] = entries[:16]
    _CUSTOM_COLORS.clear()
    _CUSTOM_COLORS.update(parsed_colors)
    _CUSTOM_PALETTES.clear()
    _CUSTOM_PALETTES.update(parsed_palettes)


def color_library_snapshot() -> dict[str, object]:
    return {
        "colors": {
            name: "#%02x%02x%02x" % tuple(round(channel * 255) for channel in rgb)
            for name, rgb in _CUSTOM_COLORS.items()
        },
        "palettes": {
            name: [
                "#%02x%02x%02x" % tuple(round(channel * 255) for channel in rgb)
                for rgb in values
            ]
            for name, values in _CUSTOM_PALETTES.items()
        },
    }


def _strobe_request(
    observation: MusicalObservation | None,
    energy: float,
    strobe_feedback: float,
    choreography_strobe: float,
    *,
    automatic: bool,
    strobe_rate_feedback: float | None = None,
    choreography_strobe_enabled: bool | None = None,
) -> float:
    """Return a sustained, normalized request for a fixture strobe channel.

    A fixture strobe channel is a rate command, not a one-frame flash trigger.
    Explicit rehearsal/choreography remains active only for its current step.
    Positive operator feedback can tune or rank such a cue but cannot create
    an unbounded hardware command. Automatic beat strobe uses a visible,
    musically gated section of the detected beat instead of the much narrower
    onset envelope.
    """

    rate_feedback = (
        strobe_feedback
        if strobe_rate_feedback is None else strobe_rate_feedback
    )
    explicit_enabled = (
        choreography_strobe > 0.05
        if choreography_strobe_enabled is None
        else choreography_strobe_enabled
    )
    # Strong negative enable feedback is an operator veto. Positive enable
    # feedback deliberately does not create a cue: it changes future sequence
    # preference while an authored step or a short musical gate remains the
    # only way to energize the physical strobe channel.
    if strobe_feedback <= -0.60:
        return 0.0
    if explicit_enabled and choreography_strobe > 0.0:
        return clamp(
            choreography_strobe + 0.5 * rate_feedback,
            0.06,
            1.0,
        )
    if not automatic or strobe_feedback < -0.2 or observation is None:
        return 0.0
    if observation.section in {
        "silence", "intro", "breakdown", "low", "outro"
    }:
        return 0.0
    energy_threshold = 0.70 - 0.06 * max(0.0, strobe_feedback)
    if (
        energy < energy_threshold
        or observation.beat_confidence < 0.25
        or observation.beat_phase > 0.42
    ):
        return 0.0
    return clamp(0.48 + 0.42 * energy + 0.25 * rate_feedback, 0.0, 1.0)


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


def expression_rgb(
    decision: PerformanceDecision,
    palette_bias: float = 0.0,
    color_activity: float = 1.0,
) -> tuple[float, float, float]:
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
    palette_hint = str(decision.palette_hint or "auto")
    if palette_hint.startswith("solid:"):
        raw = palette_hint[6:].strip().lstrip("#")
        if len(raw) == 6:
            try:
                return tuple(int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                pass
    custom = _CUSTOM_COLORS.get(palette_hint)
    if custom is not None:
        return custom
    custom_palette = _CUSTOM_PALETTES.get(palette_hint)
    if custom_palette:
        # Custom palette selection is still deterministic. Runtime color
        # latching decides when this index may advance.
        index = int(max(0.0, decision.timestamp_s) / 8.0 + tension * 2.0 + palette_bias * 2.0)
        return custom_palette[index % len(custom_palette)]
    mode = PALETTE_FAMILIES.get(palette_hint)
    # Structure controls how frequently the palette family develops. Quiet
    # states hold a color long enough to read; drops can advance normally.
    palette_clock = max(0.0, decision.timestamp_s) * clamp(
        color_activity, 0.10, 1.0
    )
    palette_index = int(
        palette_clock / 8.0 + tension * 2.0 + palette_bias * 2.0
    )
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
    if energy < 0.46 and palette_hint in {"auto", "cool", "cyan_violet"}:
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
    strobe_rate_feedback: float | None = None,
    choreography_strobe: float = 0.0,
    choreography_strobe_enabled: bool | None = None,
    palette_bias: float = 0.0,
    color_activity: float = 1.0,
    enabled: bool = True,
    fixture_index: int = 0,
    fixture_count: int = 1,
    chase_beat_position: float | None = None,
    latched_rgb: tuple[float, float, float] | None = None,
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
        chase_active: bool | None = None
        if (
            decision.routine == "opposing_chase"
            and observation is not None
            and observation.beat_confidence >= 0.20
            and fixture_count > 1
        ):
            # A chase is an exchange, not two continuously illuminated heads
            # following the same color. Hold one mover for the complete beat,
            # then hand the beam to the next mover. The fixture's characterized
            # dimmer remains the only illumination control; no software strobe
            # or calibration channel is involved.
            beat_position = chase_beat_position
            if beat_position is None:
                beat_position = (
                    max(0.0, observation.timestamp_s)
                    * (observation.bpm or 120.0)
                    / 60.0
                )
            beat_number = math.floor(max(0.0, beat_position) + 1e-6)
            chase_active = (
                beat_number % fixture_count == fixture_index % fixture_count
            )
            _set_relative(
                frame,
                fixture.universe,
                fixture.address,
                channels["dimmer"],
                round(decision.brightness * 255.0)
                if chase_active
                else 0,
            )
        strobe_request = (
            0.0
            if idle_amount >= 1.0
            else _strobe_request(
                observation,
                decision.expression.energy,
                strobe_feedback,
                choreography_strobe,
                automatic=False,
                strobe_rate_feedback=strobe_rate_feedback,
                choreography_strobe_enabled=choreography_strobe_enabled,
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
        chase_palette_bias = palette_bias
        if chase_active is not None:
            # Give each side a stable saturated color. Because illumination
            # trades sides every beat, the visible color trades with it rather
            # than both movers flickering through one shared RGB value.
            chase_palette_bias += (
                0.70 if fixture_index % 2 == 0 else -0.70
            )
        for name, value in zip(
            ("red", "green", "blue", "white"),
            rgb_to_rgbw(
                latched_rgb
                if latched_rgb is not None
                else expression_rgb(decision, chase_palette_bias, color_activity)
            ),
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
    motion_speed: float = 0.5,
    travel_size: float = 1.0,
    activity_density: float = 1.0,
    strobe_feedback: float = 0.0,
    strobe_rate_feedback: float | None = None,
    choreography_strobe: float = 0.0,
    choreography_strobe_enabled: bool | None = None,
    palette_bias: float = 0.0,
    color_activity: float = 1.0,
    enabled: bool = True,
    motion_tuning: MotionTuning | CenterMotionTuning | None = None,
    motion_timestamp_s: float | None = None,
    latched_rgb: tuple[float, float, float] | None = None,
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
            motion_speed,
            travel_size,
            activity_density,
            strobe_feedback,
            strobe_rate_feedback,
            choreography_strobe,
            choreography_strobe_enabled,
            palette_bias,
            color_activity,
            enabled,
            motion_tuning,
            motion_timestamp_s,
            latched_rgb,
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
    motion_speed: float = 0.5,
    travel_size: float = 1.0,
    activity_density: float = 1.0,
    strobe_feedback: float = 0.0,
    strobe_rate_feedback: float | None = None,
    choreography_strobe: float = 0.0,
    choreography_strobe_enabled: bool | None = None,
    palette_bias: float = 0.0,
    color_activity: float = 1.0,
    enabled: bool = True,
    motion_tuning: MotionTuning | CenterMotionTuning | None = None,
    motion_timestamp_s: float | None = None,
    latched_rgb: tuple[float, float, float] | None = None,
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
        rgbw = rgb_to_rgbw(
            latched_rgb
            if latched_rgb is not None
            else expression_rgb(decision, palette_bias, color_activity)
        )
        values = {
            "body_rotation": 128,
            # This characterized channel is inverted: 255 is the slowest
            # possible command.  Sending 200 left enough motor authority for
            # the center housing to hunt around its neutral command.
            "body_rotation_speed": 255,
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
    raw_activity = clamp(
        0.02 + 0.85 * energy + 0.35 * motion + 0.55 * motion_feedback,
        0.0,
        1.0,
    )
    # The resolved section density is an actual fixture instruction, not just
    # a duty-cycle hint. This lets a breakdown slow the motor and reduce its
    # travel even when the instantaneous spectral energy remains elevated.
    activity = clamp(
        raw_activity * (0.25 + 0.75 * clamp(activity_density, 0.0, 1.0)),
        0.0,
        1.0,
    )
    tuning = motion_tuning
    cycle_beats = tuning.cycle_beats if tuning is not None else 4.0
    speed_multiplier = 0.5 + clamp(motion_speed, 0.0, 1.0)
    resolved_bpm = (
        120.0
        if observation is None or observation.bpm is None
        else observation.bpm
    )
    musical_beat = (
        timestamp
        * resolved_bpm
        / 60.0
        * speed_multiplier
    )
    cycle_start = math.floor(musical_beat / cycle_beats) * cycle_beats
    musical_beat = min(
        musical_beat,
        cycle_start + max(0.02, cycle_beats * clamp(activity_density, 0.0, 1.0)),
    )
    if beat_confidence >= 0.12:
        pattern_phase = (musical_beat / cycle_beats) % 1.0
    else:
        assumed_bpm = (
            120.0
            if observation is None or observation.bpm is None
            else observation.bpm
        )
        free_beat = timestamp * assumed_bpm / 60.0 * speed_multiplier
        free_start = math.floor(free_beat / 4.0) * 4.0
        free_beat = min(
            free_beat,
            free_start + max(0.02, 4.0 * clamp(activity_density, 0.0, 1.0)),
        )
        pattern_phase = (free_beat / 4.0) % 1.0
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
    motion_scale = clamp(
        (0.08 + 0.92 * activity) * clamp(travel_size, 0.0, 1.35),
        0.05,
        1.0,
    )
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
    center_tuning = tuning if isinstance(tuning, CenterMotionTuning) else None
    if center_tuning is not None:
        body_motion, arm_1_motion, arm_2_motion = center_motion_coordinates(
            decision.routine,
            musical_beat,
            center_tuning,
        )
        body_scale = arm_scale_motion = 1.0
    else:
        body_scale = tuning.body_size if isinstance(tuning, MotionTuning) else 1.0
        arm_scale_motion = tuning.arm_size if isinstance(tuning, MotionTuning) else 1.0
    body = round(128.0 + 127.0 * motion_scale * body_scale * body_motion)
    arm_1 = round(128.0 + 127.0 * motion_scale * arm_scale_motion * arm_1_motion)
    arm_2 = round(128.0 + 127.0 * motion_scale * arm_scale_motion * arm_2_motion)
    # This fixture's CH2 is inverted: zero is fastest and 255 is slowest.
    authored_body_rate = 1.0 if center_tuning is None else center_tuning.body_speed
    body_speed = round(
        255.0
        - 230.0 * clamp(
            activity
            * math.sqrt(authored_body_rate)
            * speed_multiplier,
            0.0,
            1.0,
        )
    )
    # Channel 6 is the fixture's internal strobe-rate control. Keep it active
    # for the requested interval; do not pulse the channel itself for a single
    # analysis frame. The fixture then produces a clean, regular strobe at the
    # requested rate instead of an irregular software flicker.
    authored_strobe = (
        choreography_strobe
        if center_tuning is None
        else max(center_tuning.strobe_level, choreography_strobe)
    )
    strobe = _multi_effect_strobe_dmx(
        _strobe_request(
            observation,
            energy,
            strobe_feedback,
            authored_strobe,
            automatic=center_tuning is None,
            strobe_rate_feedback=strobe_rate_feedback,
            choreography_strobe_enabled=choreography_strobe_enabled,
        )
    )

    values = {
        "body_rotation": body,
        "body_rotation_speed": body_speed,
        "arm_1_motor": arm_1,
        "arm_2_motor": arm_2,
        "master_dimmer": round(
            clamp(
                max(
                    decision.brightness,
                    0.12
                    + 0.45
                    * energy
                    * clamp(activity_density, 0.0, 1.0),
                )
                + 0.10 * beat_pulse,
                0.0,
                1.0,
            )
            * (1.0 if center_tuning is None else center_tuning.intensity)
            * (
                1.0
                if center_tuning is None
                else 1.0 - center_tuning.blackout_accent * beat_pulse
            )
            * 255
        ),
        "strobe": strobe,
        "strip_speed": 0,
        "macro": 0,
    }
    even_beat = beat_index % 2 == 0
    color_pattern = (
        "alternate" if center_tuning is None else center_tuning.color_pattern
    )
    if color_pattern == "palette":
        ball_bias = arm_bias = palette_bias
    elif color_pattern == "pulse":
        ball_bias = arm_bias = palette_bias + (0.35 if beat_pulse >= 0.5 else -0.15)
    elif color_pattern == "opposed":
        ball_bias, arm_bias = palette_bias + 0.55, palette_bias - 0.55
    else:
        ball_bias = palette_bias + (0.45 if even_beat else -0.45)
        arm_bias = palette_bias + (-0.45 if even_beat else 0.45)
    ball_rgbw = rgb_to_rgbw(
        latched_rgb
        if latched_rgb is not None
        else expression_rgb(decision, ball_bias, color_activity)
    )
    arm_rgbw = rgb_to_rgbw(
        latched_rgb
        if latched_rgb is not None
        else expression_rgb(decision, arm_bias, color_activity)
    )
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
    emitter_pattern = (
        "alternate" if center_tuning is None else center_tuning.emitter_pattern
    )
    if emitter_pattern == "ball":
        ball_scale, arm_scale = 1.0, 0.0
    elif emitter_pattern == "arms":
        ball_scale, arm_scale = 0.0, 1.0
    elif emitter_pattern == "both":
        ball_scale = arm_scale = 1.0
    elif emitter_pattern == "chase":
        chase = 0.5 + 0.5 * math.sin(pattern_phase * math.tau)
        ball_scale, arm_scale = chase, 1.0 - chase
    for prefix, scale, color_values in (
        ("magic_ball", ball_scale, ball_rgbw),
        ("arm_beams", arm_scale, arm_rgbw),
    ):
        for color_name, value in zip(("red", "green", "blue", "white"), color_values):
            values[f"{prefix}_{color_name}"] = round(value * scale)

    red, green, _blue = (
        latched_rgb if latched_rgb is not None else expression_rgb(decision)
    )
    laser_mode = "beat" if center_tuning is None else center_tuning.laser_mode
    laser_level = 1.0 if center_tuning is None else center_tuning.laser_level
    laser_floor = (72.0 + 138.0 * activity) * laser_level
    red_laser = laser_floor * max(0.32, red)
    green_laser = laser_floor * max(0.32, green)
    if laser_mode == "off":
        red_laser = green_laser = 0.0
    elif laser_mode == "beat":
        red_laser *= beat_pulse
        green_laser *= beat_pulse
    elif laser_mode == "alternate":
        if even_beat:
            green_laser = 0.0
        else:
            red_laser = 0.0
    if laser_mode in {"beat", "alternate"} and beat_pulse >= 0.35:
        if even_beat:
            red_laser += 150.0 * beat_pulse * laser_level
        else:
            green_laser += 150.0 * beat_pulse * laser_level
    values["red_laser"] = round(clamp(red_laser, 0.0, 255.0))
    values["green_laser"] = round(clamp(green_laser, 0.0, 255.0))

    if center_tuning is not None and center_tuning.strip_program > 0:
        values["strip_program"] = center_tuning.strip_program
        values["strip_speed"] = round(center_tuning.strip_speed * 255.0)
    elif activity >= 0.45:
        # Party Parrot exposes a larger bank of built-in ring programs; walk
        # that bank by bar instead of repeating the same four programs.
        effect = 1 + (bar_number % 20)
        values["strip_program"] = 76 + (effect - 1) * 9 + 4
        strip_scale = 1.0 if center_tuning is None else center_tuning.strip_speed
        values["strip_speed"] = round((58 + 186 * activity) * strip_scale)
    else:
        values["strip_program"] = _strip_color_value(
            latched_rgb
            if latched_rgb is not None
            else expression_rgb(decision, palette_bias, color_activity)
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
