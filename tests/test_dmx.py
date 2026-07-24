from __future__ import annotations

import unittest

from lumen_engine.dmx import (
    DMXFrame,
    OutputSafetyGate,
    VirtualDMXOutput,
    apply_moving_head_solution,
)
from lumen_engine.models import (
    EulerXYZ,
    FixtureCalibration,
    FixturePatch,
    Vec3,
)
from lumen_engine.spatial import TargetingSolution


def patched_fixture() -> FixturePatch:
    return FixturePatch(
        fixture_id="head",
        name="Head",
        universe=2,
        address=101,
        position_m=Vec3(0, 0, 2),
        housing_rotation=EulerXYZ(),
        calibration=FixtureCalibration(
            pan_min_deg=-270,
            pan_max_deg=270,
            tilt_min_deg=-135,
            tilt_max_deg=135,
        ),
    )


class DMXTests(unittest.TestCase):
    def test_solution_maps_to_coarse_fine_and_dimmer(self) -> None:
        fixture = patched_fixture()
        frame = DMXFrame()
        solution = TargetingSolution(
            fixture_id="head",
            target=Vec3(1, 0, 2),
            pan_deg=0,
            tilt_deg=0,
            distance_m=1,
            movement_cost_deg=0,
            aim_error_deg=0,
            branch="direct",
        )
        apply_moving_head_solution(frame, fixture, solution, brightness=0.5)
        self.assertEqual(frame.get_channel(2, 101), 128)
        self.assertEqual(frame.get_channel(2, 102), 0)
        self.assertEqual(frame.get_channel(2, 103), 128)
        self.assertEqual(frame.get_channel(2, 104), 0)
        self.assertEqual(frame.get_channel(2, 105), 128)

    def test_safety_gate_drops_frames_until_armed(self) -> None:
        output = VirtualDMXOutput()
        gate = OutputSafetyGate(output, watchdog_timeout_s=1.0)
        frame = DMXFrame()
        frame.set_channel(0, 1, 255)
        self.assertFalse(gate.send(frame))
        self.assertEqual(output.frame_count, 0)
        gate.arm()
        self.assertTrue(gate.send(frame))
        self.assertEqual(output.frame_count, 1)
        gate.disarm()
        self.assertEqual(output.last_frame.get_channel(0, 1), 0)


if __name__ == "__main__":
    unittest.main()

