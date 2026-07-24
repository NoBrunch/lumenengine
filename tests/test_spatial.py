from __future__ import annotations

import unittest

from lumen_engine.models import (
    EulerXYZ,
    FixtureCalibration,
    FixturePatch,
    Vec3,
)
from lumen_engine.spatial import SpatialTargetingEngine, UnreachableTargetError


def fixture(
    *,
    position: Vec3 = Vec3(0, 0, 0),
    rotation: EulerXYZ = EulerXYZ(),
    calibration: FixtureCalibration | None = None,
) -> FixturePatch:
    return FixturePatch(
        fixture_id="test",
        name="Test head",
        universe=0,
        address=1,
        position_m=position,
        housing_rotation=rotation,
        calibration=calibration
        or FixtureCalibration(
            pan_min_deg=-540,
            pan_max_deg=540,
            tilt_min_deg=-270,
            tilt_max_deg=270,
        ),
    )


class SpatialTargetingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SpatialTargetingEngine()

    def test_canonical_axes(self) -> None:
        x = self.engine.solve(fixture(), Vec3(1, 0, 0))
        y = self.engine.solve(fixture(), Vec3(0, 1, 0))
        up = self.engine.solve(fixture(), Vec3(1, 0, 1))
        self.assertAlmostEqual(x.pan_deg, 0.0)
        self.assertAlmostEqual(x.tilt_deg, 0.0)
        self.assertAlmostEqual(y.pan_deg, 90.0)
        self.assertAlmostEqual(y.tilt_deg, 0.0)
        self.assertAlmostEqual(up.pan_deg, 0.0)
        self.assertAlmostEqual(up.tilt_deg, 45.0)
        self.assertLess(up.aim_error_deg, 1e-6)

    def test_housing_rotation_is_removed_before_solving(self) -> None:
        rotated = fixture(rotation=EulerXYZ(z_deg=90))
        solution = self.engine.solve(rotated, Vec3(0, 2, 0))
        self.assertAlmostEqual(solution.pan_deg, 0.0, places=7)
        self.assertAlmostEqual(solution.tilt_deg, 0.0, places=7)

    def test_flipped_branch_reaches_behind_with_tilt(self) -> None:
        restricted = fixture(
            calibration=FixtureCalibration(
                pan_min_deg=-90,
                pan_max_deg=90,
                tilt_min_deg=-270,
                tilt_max_deg=270,
            )
        )
        solution = self.engine.solve(restricted, Vec3(-1, 0, 0))
        self.assertAlmostEqual(solution.pan_deg, 0.0, places=7)
        self.assertAlmostEqual(abs(solution.tilt_deg), 180.0, places=7)
        self.assertTrue(solution.branch.startswith("flipped"))
        self.assertLess(solution.aim_error_deg, 1e-6)

    def test_continuity_prefers_nearby_pan_turn(self) -> None:
        solution = self.engine.solve(
            fixture(),
            Vec3(-1, -0.1763269807, 0),
            previous_pan_deg=190.0,
            previous_tilt_deg=0.0,
        )
        self.assertAlmostEqual(solution.pan_deg, 190.0, places=5)
        self.assertLess(solution.movement_cost_deg, 1e-4)

    def test_coincident_target_is_rejected(self) -> None:
        placed = fixture(position=Vec3(1, 2, 3))
        with self.assertRaises(UnreachableTargetError):
            self.engine.solve(placed, Vec3(1, 2, 3))


if __name__ == "__main__":
    unittest.main()

