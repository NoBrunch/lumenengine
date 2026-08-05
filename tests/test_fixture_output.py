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
from lumen_engine.motion import CenterMotionTuning


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

    def test_opposing_chase_trades_mover_beam_and_color_each_beat(
        self,
    ) -> None:
        from dataclasses import replace

        fixtures = tuple(
            FixturePatch(
                fixture_id=f"mover-{index}",
                name=f"Mover {index}",
                profile_key="generic_rgbw_moving_head_11ch",
                universe=0,
                address=address,
                position_m=Vec3(0, 0, 2),
                housing_rotation=EulerXYZ(),
                calibration=FixtureCalibration(-270, 270, -135, 135),
                dimmer_channel=6,
            )
            for index, address in enumerate((31, 43))
        )
        selected = replace(decision(), routine="opposing_chase")

        def chase_frame(timestamp: float) -> DMXFrame:
            frame = DMXFrame()
            observation = MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.8,
                onset_strength=0.5,
                low_energy=0.7,
                mid_energy=0.5,
                high_energy=0.3,
                beat_pulse=0.8,
                beat_confidence=0.9,
                bpm=120.0,
            )
            for index, fixture in enumerate(fixtures):
                apply_moving_head_profile(
                    frame,
                    fixture,
                    selected,
                    observation,
                    fixture_index=index,
                    fixture_count=len(fixtures),
                )
            return frame

        first = chase_frame(0.0)
        second = chase_frame(0.5)
        first_dimmers = tuple(
            first.get_channel(0, fixture.address + 5)
            for fixture in fixtures
        )
        second_dimmers = tuple(
            second.get_channel(0, fixture.address + 5)
            for fixture in fixtures
        )
        self.assertGreater(first_dimmers[0], 0)
        self.assertEqual(first_dimmers[1], 0)
        self.assertEqual(second_dimmers[0], 0)
        self.assertGreater(second_dimmers[1], 0)
        first_color = tuple(
            first.get_channel(0, fixtures[0].address + relative)
            for relative in range(7, 11)
        )
        second_color = tuple(
            second.get_channel(0, fixtures[1].address + relative)
            for relative in range(7, 11)
        )
        self.assertNotEqual(first_color, second_color)

    def test_explicit_mover_strobe_stays_on_between_beat_peaks(self) -> None:
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
        apply_moving_head_profile(
            frame,
            fixture,
            decision(),
            MusicalObservation(
                timestamp_s=2.2,
                loudness=0.8,
                onset_strength=0.1,
                low_energy=0.5,
                mid_energy=0.4,
                high_energy=0.3,
                beat_pulse=0.05,
                beat_phase=0.75,
                beat_confidence=0.9,
                bpm=120.0,
            ),
            choreography_strobe=0.6,
        )
        self.assertGreater(frame.get_channel(0, 37), 0)

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

    def test_explicit_multi_effect_strobe_uses_internal_rate_channel(self) -> None:
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
                timestamp_s=2.2,
                loudness=0.8,
                onset_strength=0.1,
                low_energy=0.5,
                mid_energy=0.4,
                high_energy=0.3,
                beat_pulse=0.05,
                beat_phase=0.75,
                beat_confidence=0.9,
                bpm=120.0,
            ),
            choreography_strobe=0.6,
        )
        # Characterized personality: channel 6 is 0 off and 10..255 active.
        self.assertGreaterEqual(frame.get_channel(0, 6), 10)

    def test_strobe_feedback_changes_hardware_rate_and_can_veto_step(self) -> None:
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
            timestamp_s=2.2,
            loudness=0.8,
            onset_strength=0.1,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            beat_pulse=0.05,
            beat_phase=0.75,
            beat_confidence=0.9,
            bpm=120.0,
        )
        slower = DMXFrame()
        faster = DMXFrame()
        vetoed = DMXFrame()
        apply_auxiliary_fixture(
            slower, fixture, decision(), observation,
            strobe_feedback=-0.3, choreography_strobe=0.6,
        )
        apply_auxiliary_fixture(
            faster, fixture, decision(), observation,
            strobe_feedback=0.3, choreography_strobe=0.6,
        )
        apply_auxiliary_fixture(
            vetoed, fixture, decision(), observation,
            strobe_feedback=-0.8, choreography_strobe=0.6,
        )
        self.assertLess(
            slower.get_channel(0, 6), faster.get_channel(0, 6)
        )
        self.assertEqual(vetoed.get_channel(0, 6), 0)

    def test_positive_strobe_preference_cannot_bypass_context_gate(self) -> None:
        fixture = ProfileFixturePatch(
            fixture_id="multi",
            name="Multi",
            profile_key="generic_multi_effect_19ch",
            universe=0,
            address=1,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
        )
        quiet_context = MusicalObservation(
            timestamp_s=2.0,
            loudness=0.8,
            onset_strength=0.9,
            low_energy=0.6,
            mid_energy=0.3,
            high_energy=0.1,
            beat_pulse=1.0,
            beat_phase=0.0,
            beat_confidence=0.9,
            bpm=128.0,
            section="breakdown",
        )
        frame = DMXFrame()
        apply_auxiliary_fixture(
            frame,
            fixture,
            decision(),
            quiet_context,
            strobe_feedback=1.0,
            strobe_rate_feedback=1.0,
        )
        self.assertEqual(frame.get_channel(0, 6), 0)

    def test_automatic_strobe_is_bounded_to_current_beat_cue(self) -> None:
        fixture = ProfileFixturePatch(
            fixture_id="multi",
            name="Multi",
            profile_key="generic_multi_effect_19ch",
            universe=0,
            address=1,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
        )

        def observation(phase: float) -> MusicalObservation:
            return MusicalObservation(
                timestamp_s=2.0 + phase,
                loudness=0.8,
                onset_strength=0.9,
                low_energy=0.6,
                mid_energy=0.3,
                high_energy=0.1,
                beat_pulse=1.0,
                beat_phase=phase,
                beat_confidence=0.9,
                bpm=128.0,
                section="drop",
            )

        cue = DMXFrame()
        outside = DMXFrame()
        apply_auxiliary_fixture(cue, fixture, decision(), observation(0.1))
        apply_auxiliary_fixture(
            outside,
            fixture,
            decision(),
            observation(0.8),
            strobe_feedback=1.0,
            strobe_rate_feedback=1.0,
        )
        self.assertGreater(cue.get_channel(0, 6), 0)
        self.assertEqual(outside.get_channel(0, 6), 0)

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

    def test_center_tuning_drives_characterized_channels_without_movers(self) -> None:
        fixture = ProfileFixturePatch(
            fixture_id="multi",
            name="Multi",
            profile_key="generic_multi_effect_19ch",
            universe=0,
            address=1,
            position_m=Vec3(0, 0, 2),
            housing_rotation=EulerXYZ(),
        )
        selected = CenterMotionTuning(
            cycle_beats=8.0,
            body_direction=-1,
            body_speed=2.0,
            body_travel=0.9,
            body_phase=0.125,
            arm_1_speed=1.0,
            arm_1_travel=0.25,
            arm_1_phase=0.0,
            arm_2_direction=-1,
            arm_2_speed=2.0,
            arm_2_travel=0.95,
            arm_2_phase=0.25,
            relationship="synchronized",
            emitter_pattern="ball",
            color_pattern="palette",
            laser_mode="off",
            strip_program=123,
            strip_speed=0.4,
            strobe_level=0.75,
            intensity=0.5,
            blackout_accent=0.5,
        )
        frame = DMXFrame()
        apply_auxiliary_fixture(
            frame,
            fixture,
            decision(),
            MusicalObservation(
                timestamp_s=2.0,
                loudness=0.8,
                onset_strength=0.4,
                low_energy=0.5,
                mid_energy=0.4,
                high_energy=0.2,
                beat_pulse=1.0,
                beat_phase=0.0,
                beat_confidence=0.9,
                bpm=120.0,
            ),
            motion_tuning=selected,
        )
        self.assertNotEqual(frame.get_channel(0, 1), 128)  # body position
        self.assertLess(frame.get_channel(0, 2), 128)  # inverted fast body speed
        self.assertNotEqual(
            frame.get_channel(0, 3), frame.get_channel(0, 4)
        )  # independent arm positions
        self.assertLessEqual(frame.get_channel(0, 5), 64)  # intensity + blackout
        self.assertGreater(frame.get_channel(0, 6), 10)  # characterized strobe
        self.assertTrue(any(frame.get_channel(0, ch) for ch in range(7, 11)))
        self.assertFalse(
            any(frame.get_channel(0, ch) for ch in range(11, 15))
        )  # arm emitters off
        self.assertEqual(frame.get_channel(0, 15), 0)
        self.assertEqual(frame.get_channel(0, 16), 0)
        self.assertEqual(frame.get_channel(0, 17), 123)
        self.assertEqual(frame.get_channel(0, 18), 102)
        self.assertEqual(frame.get_channel(0, 19), 0)

    def test_center_arm_phase_changes_only_center_motor_channels(self) -> None:
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
            timestamp_s=1.0,
            loudness=0.8,
            onset_strength=0.3,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.2,
            beat_pulse=0.0,
            beat_confidence=0.9,
            bpm=120.0,
        )
        first = DMXFrame()
        second = DMXFrame()
        base = CenterMotionTuning(
            relationship="synchronized",
            laser_mode="off",
            color_pattern="palette",
            emitter_pattern="both",
        )
        apply_auxiliary_fixture(
            first, fixture, decision(), observation,
            motion_tuning=base.patch({"arm_2_phase": 0.0}),
        )
        apply_auxiliary_fixture(
            second, fixture, decision(), observation,
            motion_tuning=base.patch({"arm_2_phase": 0.25}),
        )
        self.assertEqual(first.get_channel(0, 1), second.get_channel(0, 1))
        self.assertEqual(first.get_channel(0, 3), second.get_channel(0, 3))
        self.assertNotEqual(first.get_channel(0, 4), second.get_channel(0, 4))


if __name__ == "__main__":
    unittest.main()
