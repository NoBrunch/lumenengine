from __future__ import annotations

import unittest

from lumen_engine.motion import (
    DEFAULT_MOTION_TUNINGS,
    normalized_position,
    preview_paths,
    required_axis_speeds,
)
from lumen_engine.config import load_rig


class MotionPathTests(unittest.TestCase):
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
