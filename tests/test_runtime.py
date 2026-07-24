from __future__ import annotations

import unittest

from lumen_engine.config import load_rig
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import MusicalObservation
from lumen_engine.runtime import PerformanceRuntime


class RuntimeTests(unittest.TestCase):
    def test_observation_reaches_virtual_dmx(self) -> None:
        rig = load_rig("config/example-rig.json")
        output = VirtualDMXOutput()
        runtime = PerformanceRuntime(rig.fixtures, output)
        result = runtime.step(
            MusicalObservation(
                timestamp_s=0,
                loudness=0.4,
                onset_strength=0.5,
                low_energy=0.6,
                mid_energy=0.4,
                high_energy=0.2,
                beat_confidence=0.7,
                section="verse",
                section_confidence=0.8,
            )
        )
        self.assertEqual(output.frame_count, 1)
        self.assertEqual(len(result.solutions), 2)
        self.assertEqual(result.warnings, ())
        self.assertIn(0, result.dmx.universes)


if __name__ == "__main__":
    unittest.main()

