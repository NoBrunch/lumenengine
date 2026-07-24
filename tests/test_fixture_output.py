from __future__ import annotations

import unittest

from lumen_engine.dmx import DMXFrame
from lumen_engine.fixture_output import (
    apply_auxiliary_fixture,
    apply_moving_head_profile,
)
from lumen_engine.models import (
    EulerXYZ,
    ExpressionState,
    FixtureCalibration,
    FixturePatch,
    Gesture,
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
        apply_auxiliary_fixture(frame, fixture, decision())
        self.assertNotEqual(frame.get_channel(0, 1), 0)  # body
        self.assertEqual(frame.get_channel(0, 5), round(0.75 * 255))  # dimmer
        self.assertEqual(frame.get_channel(0, 6), 92)  # release strobe
        self.assertTrue(any(frame.get_channel(0, channel) for channel in range(7, 15)))
        self.assertTrue(any(frame.get_channel(0, channel) for channel in (15, 16)))
        self.assertEqual(frame.get_channel(0, 19), 0)  # no internal macro


if __name__ == "__main__":
    unittest.main()

