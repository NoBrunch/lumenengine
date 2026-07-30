from __future__ import annotations

import unittest

from lumen_engine.config import load_rig
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import MusicalObservation
from lumen_engine.models import Vec3
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

    def test_real_silence_parks_center_and_holds_movers(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        output = VirtualDMXOutput()
        runtime = PerformanceRuntime(
            rig.fixtures,
            output,
            auxiliary_fixtures=rig.auxiliary_fixtures,
        )
        runtime.step(
            MusicalObservation(
                timestamp_s=0.0,
                loudness=0.8,
                onset_strength=0.8,
                low_energy=0.6,
                mid_energy=0.4,
                high_energy=0.2,
                beat_pulse=1.0,
                beat_confidence=0.8,
                bpm=120.0,
            )
        )
        for index in range(1, 15):
            result = runtime.step(
                MusicalObservation(
                    timestamp_s=index * 0.5,
                    loudness=0.0,
                    onset_strength=0.0,
                    low_energy=0.0,
                    mid_energy=0.0,
                    high_energy=0.0,
                )
            )
        self.assertTrue(result.solutions[0].branch.startswith("quiet-hold"))
        self.assertEqual(result.dmx.get_channel(0, 1), 128)
        self.assertEqual(result.dmx.get_channel(0, 2), 200)
        self.assertEqual(result.dmx.get_channel(0, 3), 128)
        self.assertEqual(result.dmx.get_channel(0, 4), 128)
        self.assertEqual(result.dmx.get_channel(0, 5), 24)
        self.assertEqual(result.dmx.get_channel(0, 6), 0)
        self.assertEqual(result.dmx.get_channel(0, 15), 0)
        self.assertEqual(result.dmx.get_channel(0, 16), 0)

    def test_active_garage_rig_uses_room_and_center_fixture_on_beats(
        self,
    ) -> None:
        rig = load_rig("config/party-parrot-active.json")
        output = VirtualDMXOutput()
        runtime = PerformanceRuntime(
            rig.fixtures,
            output,
            auxiliary_fixtures=rig.auxiliary_fixtures,
            motion_extents=Vec3(1.2, 3.6, 2.6),
        )
        angles: dict[str, list[list[float]]] = {
            fixture.fixture_id: [[], []]
            for fixture in rig.fixtures
        }
        center_body: list[int] = []
        center_arm_1: list[int] = []
        center_arm_2: list[int] = []
        center_strobe: list[int] = []
        warnings: list[str] = []
        for index in range(120):
            beat = index % 4 == 0
            result = runtime.step(
                MusicalObservation(
                    timestamp_s=index * 0.12,
                    loudness=0.72,
                    onset_strength=0.86 if beat else 0.18,
                    low_energy=0.62,
                    mid_energy=0.58,
                    high_energy=0.38,
                    beat_phase=(index % 4) / 4.0,
                    bar_phase=(index % 16) / 16.0,
                    beat_pulse=1.0 if beat else 0.12,
                    beat_confidence=0.82,
                    bpm=125.0,
                    section="chorus",
                    section_confidence=0.8,
                    novelty=0.7 if beat else 0.2,
                )
            )
            warnings.extend(result.warnings)
            for solution in result.solutions:
                angles[solution.fixture_id][0].append(solution.pan_deg)
                angles[solution.fixture_id][1].append(solution.tilt_deg)
            center_body.append(result.dmx.get_channel(0, 1))
            center_arm_1.append(result.dmx.get_channel(0, 3))
            center_arm_2.append(result.dmx.get_channel(0, 4))
            center_strobe.append(result.dmx.get_channel(0, 6))

        self.assertEqual(warnings, [])
        for pan, tilt in angles.values():
            self.assertGreater(max(pan) - min(pan), 50.0)
            self.assertGreater(max(tilt) - min(tilt), 70.0)
        self.assertGreater(max(center_body) - min(center_body), 200)
        self.assertGreater(max(center_arm_1) - min(center_arm_1), 190)
        self.assertGreater(max(center_arm_2) - min(center_arm_2), 190)
        self.assertGreater(max(center_strobe), 150)

    def test_phrase_routine_is_stable_within_bar_and_changes_between_bars(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        routines = []
        for index in range(96):
            timestamp = index / 12.0
            result = runtime.step(
                MusicalObservation(
                    timestamp_s=timestamp,
                    loudness=0.82,
                    onset_strength=0.72 if index % 3 == 0 else 0.25,
                    low_energy=0.65,
                    mid_energy=0.60,
                    high_energy=0.55,
                    bar_phase=((index / 3.0) % 4.0) / 4.0,
                    beat_pulse=1.0 if index % 3 == 0 else 0.15,
                    beat_confidence=0.9,
                    bpm=120.0,
                    section="drop",
                    section_confidence=0.9,
                )
            )
            routines.append(result.decision.routine)
        self.assertEqual(len(set(routines[:24])), 1)
        self.assertGreaterEqual(len(set(routines)), 3)

    def test_learned_routine_preference_reaches_decision(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        runtime.replace_feedback({"overall": {"motion": 0.0, "intensity": 0.0,
                                                "strobe": 0.0, "palette": 0.0,
                                                "routines": {"counter_rotate": 0.9}}})
        result = runtime.step(
            MusicalObservation(
                timestamp_s=0.0,
                loudness=0.8,
                onset_strength=0.7,
                low_energy=0.6,
                mid_energy=0.6,
                high_energy=0.5,
                beat_pulse=1.0,
                beat_confidence=0.8,
                bpm=120.0,
                section="groove",
                section_confidence=0.8,
            )
        )
        self.assertEqual(result.decision.routine, "counter_rotate")

    def test_media_change_does_not_carry_previous_routine(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        runtime.replace_feedback({"overall": {"routines": {"counter_rotate": 0.9}}})
        observation = MusicalObservation(
            timestamp_s=0.0, loudness=0.8, onset_strength=0.7,
            low_energy=0.6, mid_energy=0.6, high_energy=0.5,
            beat_pulse=1.0, beat_confidence=0.8, bpm=120.0,
            section="groove", section_confidence=0.8,
        )
        self.assertEqual(runtime.step(observation).decision.routine, "counter_rotate")
        runtime.set_media_context(99, "groove", "new artist")
        runtime.replace_feedback({})
        self.assertNotEqual(runtime.step(observation).decision.routine, "counter_rotate")


if __name__ == "__main__":
    unittest.main()
