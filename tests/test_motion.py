from __future__ import annotations

import unittest

from lumen_engine.motion import (
    CenterMotionTuning,
    DEFAULT_MOTION_TUNINGS,
    canonical_motion_scope,
    center_motion_coordinates,
    merged_group_motion_tunings,
    normalized_position,
    preview_paths,
    required_axis_speeds,
)
from lumen_engine.config import load_rig


class MotionPathTests(unittest.TestCase):
    def test_public_scopes_are_permanent_groups_not_fixture_ids(self) -> None:
        self.assertEqual(canonical_motion_scope("group:movers"), "movers")
        self.assertEqual(canonical_motion_scope("multi effect"), "center")
        self.assertEqual(canonical_motion_scope("whole rig"), "overall")
        with self.assertRaises(ValueError):
            canonical_motion_scope("fixture:mover-31")

    def test_grouped_routine_tuning_keeps_movers_and_center_independent(self) -> None:
        library = merged_group_motion_tunings({
            "version": 2,
            "groups": {
                "movers": {"routines": {"fan_sweep": {"pan_size": 0.22}}},
                "center": {"routines": {"fan_sweep": {
                    "body_travel": 0.91,
                    "arm_1_travel": 0.33,
                    "arm_2_travel": 0.77,
                }}},
            },
        })
        self.assertEqual(library.movers["fan_sweep"].pan_size, 0.22)
        self.assertEqual(library.center["fan_sweep"].body_travel, 0.91)
        self.assertEqual(library.center["fan_sweep"].arm_1_travel, 0.33)
        self.assertEqual(library.center["fan_sweep"].arm_2_travel, 0.77)
        self.assertEqual(library.movers["fan_sweep"].body_size, 0.55)

    def test_existing_flat_file_migrates_body_and_arm_values(self) -> None:
        library = merged_group_motion_tunings({
            "breathe": {
                "cycle_beats": 12,
                "pan_size": 0.71,
                "body_size": 0.42,
                "arm_size": 0.63,
                "relationship": "chase",
                "direction": -1,
            },
        })
        self.assertEqual(library.movers["breathe"].pan_size, 0.71)
        center = library.center["breathe"]
        self.assertEqual(center.cycle_beats, 12)
        self.assertEqual(center.body_travel, 0.42)
        self.assertEqual(center.arm_1_travel, 0.63)
        self.assertEqual(center.arm_2_travel, 0.63)
        self.assertEqual(center.relationship, "chase")
        self.assertEqual(center.body_direction, -1)

    def test_center_arms_have_independent_rate_phase_direction_and_travel(self) -> None:
        tuning = CenterMotionTuning(
            cycle_beats=8,
            relationship="synchronized",
            body_travel=0.0,
            arm_1_speed=1.0,
            arm_1_phase=0.0,
            arm_1_direction=1,
            arm_1_travel=0.25,
            arm_2_speed=2.0,
            arm_2_phase=0.25,
            arm_2_direction=-1,
            arm_2_travel=0.9,
        )
        body, arm_1, arm_2 = center_motion_coordinates(
            "fan_sweep", 1.0, tuning
        )
        self.assertEqual(body, 0.0)
        self.assertAlmostEqual(arm_1, 0.25 * 2 ** -0.5, places=6)
        self.assertAlmostEqual(arm_2, 0.0, places=6)
        later = center_motion_coordinates("fan_sweep", 2.0, tuning)
        self.assertNotAlmostEqual(later[1], later[2], places=4)

    def test_center_counter_relationship_reverses_second_arm(self) -> None:
        tuning = CenterMotionTuning(
            cycle_beats=8,
            relationship="counter",
            body_travel=0.0,
            arm_1_speed=1.0,
            arm_2_speed=1.0,
            arm_1_phase=0.0,
            arm_2_phase=0.0,
        )
        _body, arm_1, arm_2 = center_motion_coordinates(
            "counter_rotate", 1.0, tuning
        )
        self.assertAlmostEqual(arm_1, -arm_2, places=7)

    def test_figure_eight_uses_literal_full_pan_and_two_tilt_lobes(self) -> None:
        tuning = DEFAULT_MOTION_TUNINGS["figure_eight"]
        center = normalized_position("figure_eight", 0.0, 0, 2, tuning)
        right = normalized_position(
            "figure_eight", tuning.cycle_beats / 4.0, 0, 2, tuning
        )
        upper_lobe = normalized_position(
            "figure_eight", tuning.cycle_beats / 8.0, 0, 2, tuning
        )
        lower_lobe = normalized_position(
            "figure_eight", tuning.cycle_beats * 3.0 / 8.0, 0, 2, tuning
        )
        self.assertAlmostEqual(center[0], 0.5)
        self.assertAlmostEqual(right[0], 1.0)
        self.assertGreater(upper_lobe[1], 0.85)
        self.assertLess(lower_lobe[1], 0.15)

    def test_opposed_relationship_is_exact_not_arbitrary(self) -> None:
        tuning = DEFAULT_MOTION_TUNINGS["opposing_chase"]
        for beat in (0.25, 1.0, 2.75, 5.5):
            first = normalized_position(
                "opposing_chase", beat, 0, 2, tuning
            )
            second = normalized_position(
                "opposing_chase", beat, 1, 2, tuning
            )
            self.assertAlmostEqual(first[0] + second[0], 1.0, places=7)

    def test_preview_contains_complete_closed_path_for_both_movers(self) -> None:
        paths = preview_paths(
            "counter_rotate", DEFAULT_MOTION_TUNINGS["counter_rotate"]
        )
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(paths[0]), 129)
        self.assertAlmostEqual(paths[0][0][0], paths[0][-1][0])
        self.assertAlmostEqual(paths[0][0][1], paths[0][-1][1])
        self.assertAlmostEqual(paths[1][0][0], paths[1][-1][0])
        self.assertAlmostEqual(paths[1][0][1], paths[1][-1][1])

    def test_authored_defaults_fit_saved_mover_velocity(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        for routine, tuning in DEFAULT_MOTION_TUNINGS.items():
            for fixture_index, fixture in enumerate(rig.fixtures):
                calibration = fixture.calibration
                pan, tilt = required_axis_speeds(
                    routine,
                    tuning,
                    bpm=120.0,
                    fixture_index=fixture_index,
                    fixture_count=len(rig.fixtures),
                    pan_range_deg=calibration.pan_max_deg - calibration.pan_min_deg,
                    tilt_range_deg=calibration.tilt_max_deg - calibration.tilt_min_deg,
                )
                self.assertLessEqual(pan, calibration.max_pan_speed_deg_s)
                self.assertLessEqual(tilt, calibration.max_tilt_speed_deg_s)


if __name__ == "__main__":
    unittest.main()
