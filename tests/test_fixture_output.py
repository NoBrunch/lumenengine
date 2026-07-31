from __future__ import annotations

import unittest

from lumen_engine.dmx import DMXFrame
from lumen_engine.fixture_output import (
    apply_auxiliary_fixture,
    apply_moving_head_profile,
    expression_rgb,
)
from lumen_engine.models import (
    EulerXYZ,
    ExpressionState,
    FixtureCalibration,
    FixturePatch,
    Gesture,
    MusicalObservation,
    PerformanceDecision,
    ProfileFixturePatch,
    Vec3,
)


def decision() -> PerformanceDecision:
    return PerformanceDecision(
        timestamp_s=2.0,
        gesture=Gesture.RELEASE,
        expression=ExpressionState(
            energy=0.8,
            tension=0.7,
            motion=0.6,
            intimacy=0.2,
            confidence=0.9,
        ),
        target=Vec3(0, 0, 1),
        brightness=0.75,
        reason="test",
        confidence=0.9,
    )


class FixtureOutputTests(unittest.TestCase):
    def test_legacy_midnight_teal_is_an_explicit_palette_family(self) -> None:
        selected = decision()
        from dataclasses import replace
        teal = expression_rgb(replace(selected, palette_hint="midnight_teal"))
        automatic = expression_rgb(replace(selected, palette_hint="auto"))
        self.assertNotEqual(teal, automatic)
        self.assertGreater(teal[2], teal[0])

    def test_generic_rgbw_mover_writes_party_parrot_layout(self) -> None:
        fixture = FixturePatch(
            fixture_id="mover",
            name="Mover",
            profile_key="generic_rgbw_moving_head_11ch",
            universe=0,
            address=31,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
            calibration=FixtureCalibration(-270, 270, -135, 135),
            dimmer_channel=6,
        )
        frame = DMXFrame()
        apply_moving_head_profile(frame, fixture, decision())
        self.assertEqual(frame.get_channel(0, 35), 0)  # movement speed
        self.assertEqual(frame.get_channel(0, 37), 0)  # strobe
        self.assertTrue(any(frame.get_channel(0, channel) for channel in range(38, 42)))

    def test_multi_effect_writes_all_active_systems(self) -> None:
        fixture = ProfileFixturePatch(
            fixture_id="multi",
            name="Multi",
            profile_key="generic_multi_effect_19ch",
            universe=0,
            address=1,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
        )
        frame = DMXFrame()
        apply_auxiliary_fixture(
            frame,
            fixture,
            decision(),
            MusicalObservation(
                timestamp_s=2.0,
                loudness=0.8,
                onset_strength=0.9,
                low_energy=0.6,
                mid_energy=0.3,
                high_energy=0.1,
                beat_pulse=1.0,
                beat_confidence=0.9,
                bpm=128.0,
            ),
        )
        self.assertNotEqual(frame.get_channel(0, 1), 0)  # body
        self.assertGreaterEqual(
            frame.get_channel(0, 5), round(0.75 * 255)
        )  # reinforced master dimmer
        self.assertGreater(frame.get_channel(0, 6), 150)  # beat strobe
        self.assertTrue(any(frame.get_channel(0, channel) for channel in range(7, 15)))
        self.assertTrue(any(frame.get_channel(0, channel) for channel in (15, 16)))
        self.assertEqual(frame.get_channel(0, 19), 0)  # no internal macro

    def test_multi_effect_motion_consumes_phrase_routine(self) -> None:
        from dataclasses import replace

        fixture = ProfileFixturePatch(
            fixture_id="multi",
            name="Multi",
            profile_key="generic_multi_effect_19ch",
            universe=0,
            address=1,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
        )
        observation = MusicalObservation(
            timestamp_s=3.0,
            loudness=0.8,
            onset_strength=0.3,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.2,
            bar_phase=0.37,
            beat_confidence=0.9,
            bpm=120.0,
        )
        opposing = DMXFrame()
        fan = DMXFrame()
        apply_auxiliary_fixture(
            opposing, fixture, replace(decision(), routine="opposing_chase"), observation
        )
        apply_auxiliary_fixture(
            fan, fixture, replace(decision(), routine="fan_sweep"), observation
        )
        self.assertNotEqual(
            (opposing.get_channel(0, 3), opposing.get_channel(0, 4)),
            (fan.get_channel(0, 3), fan.get_channel(0, 4)),
        )


if __name__ == "__main__":
    unittest.main()
