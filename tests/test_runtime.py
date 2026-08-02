from __future__ import annotations

import unittest

from lumen_engine.config import load_rig
from lumen_engine.choreography import SequencePreferenceModel
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import MusicalObservation
from lumen_engine.models import Vec3
from lumen_engine.runtime import PerformanceRuntime


class RuntimeTests(unittest.TestCase):
    def test_rehearsal_can_isolate_one_mover_and_fully_disable_center_light(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
        )
        selected = rig.fixtures[0]
        runtime.set_rehearsal(
            "figure_eight",
            scope=f"fixture:{selected.fixture_id}",
            intensity=0.7,
            size=1.0,
            palette="party_vivid",
            isolate=True,
        )
        frame = runtime.step(MusicalObservation(
            timestamp_s=0.0, loudness=0.7, onset_strength=0.8,
            low_energy=0.6, mid_energy=0.5, high_energy=0.4,
            beat_phase=0.0, bar_phase=0.0, beat_pulse=1.0,
            beat_confidence=1.0, bpm=120.0, section="groove",
            section_confidence=1.0,
        )).dmx
        universe = frame.universe_data(0)
        selected_start = selected.address - 1
        other_start = rig.fixtures[1].address - 1
        center_start = rig.auxiliary_fixtures[0].address - 1
        self.assertGreater(universe[selected_start + 5], 0)  # mover dimmer
        self.assertEqual(universe[other_start + 5], 0)
        self.assertEqual(universe[other_start + 7:other_start + 11], bytes(4))
        self.assertEqual(universe[center_start + 4], 0)  # master dimmer
        self.assertEqual(universe[center_start + 5], 0)  # strobe
        self.assertEqual(universe[center_start + 6:center_start + 19], bytes(13))

    def test_rehearsal_forces_selected_routine_without_song_planner(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
        )
        runtime.set_rehearsal(
            "figure_eight", scope="movers", intensity=0.7,
            size=0.9, palette="party_vivid", isolate=True,
        )
        result = runtime.step(MusicalObservation(
            timestamp_s=0.0, loudness=0.7, onset_strength=0.8,
            low_energy=0.6, mid_energy=0.5, high_energy=0.4,
            beat_phase=0.0, bar_phase=0.0, beat_pulse=1.0,
            beat_confidence=1.0, bpm=120.0, section="groove",
            section_confidence=1.0,
        ))
        self.assertEqual(result.decision.routine, "figure_eight")
        self.assertIn("Rehearsal isolates", result.decision.reason)
        self.assertEqual(
            runtime.choreography_snapshot()["rehearsal"]["scope"], "movers"
        )

    def test_performance_paths_follow_physical_axis_direction(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        runtime.set_rehearsal(
            "breathe", scope="movers", intensity=0.7,
            size=1.0, palette="party_vivid", isolate=True,
        )

        def observation(timestamp: float) -> MusicalObservation:
            return MusicalObservation(
                timestamp_s=timestamp, loudness=0.7, onset_strength=0.5,
                low_energy=0.6, mid_energy=0.5, high_energy=0.4,
                beat_phase=0.0, bar_phase=0.0, beat_pulse=0.4,
                beat_confidence=1.0, bpm=120.0, section="groove",
                section_confidence=1.0,
            )

        runtime.step(observation(0.0))
        result = runtime.step(observation(1.0))
        semantic_tilts = []
        for fixture, solution in zip(rig.fixtures, result.solutions):
            calibration = fixture.calibration
            numeric = (
                solution.tilt_deg - calibration.tilt_min_deg
            ) / (calibration.tilt_max_deg - calibration.tilt_min_deg)
            semantic_tilts.append(
                numeric if calibration.tilt_direction > 0 else 1.0 - numeric
            )
        # Both movers receive the same room-semantic breathe path even though
        # channel 31's captured high/low DMX order is reversed.
        self.assertAlmostEqual(
            semantic_tilts[0], semantic_tilts[1], delta=0.02
        )

    def test_scalar_feedback_is_staged_until_next_phrase_boundary(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )

        def observation(timestamp: float, phase: float) -> MusicalObservation:
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.72,
                onset_strength=0.4,
                low_energy=0.6,
                mid_energy=0.5,
                high_energy=0.4,
                bar_phase=phase,
                beat_confidence=0.9,
                bpm=120.0,
                section="groove",
                section_confidence=0.8,
            )

        runtime.step(observation(0.0, 0.9))
        runtime.replace_feedback(
            {
                "overall": {
                    "motion": 1.0,
                    "intensity": 0.8,
                    "strobe": 0.6,
                }
            }
        )
        self.assertNotIn("overall", runtime._feedback_motion)
        self.assertIsNotNone(runtime._pending_feedback_biases)

        # First bar wrap is still within the leased two-bar phrase.
        runtime.step(observation(0.2, 0.1))
        self.assertNotIn("overall", runtime._feedback_motion)
        runtime.step(observation(1.8, 0.9))
        runtime.step(observation(2.0, 0.1))
        self.assertEqual(runtime._feedback_motion["overall"], 1.0)
        self.assertIsNone(runtime._pending_feedback_biases)

    def test_preferred_action_enters_next_phrase_without_interrupting(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        model = SequencePreferenceModel()
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=model,
        )

        def observation(timestamp: float, bar_phase: float) -> MusicalObservation:
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.78,
                onset_strength=0.6,
                low_energy=0.7,
                mid_energy=0.6,
                high_energy=0.5,
                bar_phase=bar_phase,
                beat_pulse=0.8,
                beat_confidence=0.9,
                bpm=120.0,
                section="drop",
                section_confidence=0.9,
            )

        first = runtime.step(observation(0.0, 0.9))
        self.assertIsNotNone(runtime._choreography_planner)
        active_before = runtime._choreography_planner.active
        self.assertIsNotNone(active_before)
        learning = runtime.learn_choreography_feedback(
            label="more_like_this",
            value=1.0,
            occurrences=3,
            preferred_routine="counter_rotate",
        )
        self.assertIsNotNone(learning)
        runtime.step(observation(0.2, 0.1))
        active_held = runtime._choreography_planner.active
        self.assertEqual(
            active_held.sequence.semantic_signature,
            active_before.sequence.semantic_signature,
        )
        self.assertEqual(active_held.boundary_id, active_before.boundary_id)

        runtime.step(observation(1.8, 0.9))
        next_phrase = runtime.step(observation(2.0, 0.1))
        self.assertEqual(next_phrase.decision.routine, "counter_rotate")
        self.assertTrue(model.learned_candidates())

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
                    bar_phase=(timestamp / 2.0) % 1.0,
                    beat_pulse=1.0 if index % 3 == 0 else 0.15,
                    beat_confidence=0.9,
                    bpm=120.0,
                    section="drop",
                    section_confidence=0.9,
                )
            )
            routines.append(result.decision.routine)
        self.assertEqual(len(set(routines[:48])), 1)
        self.assertGreaterEqual(len(set(routines)), 2)

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

    def test_contextual_preference_breaks_conflicting_global_tie(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        runtime.set_media_context(7, "groove", "artist")
        runtime.replace_feedback(
            {
                "overall": {"routines": {"breathe": 0.4, "opposing_chase": 0.4}},
                "song:7": {"routines": {"opposing_chase": 0.6}},
            }
        )
        result = runtime.step(
            MusicalObservation(
                timestamp_s=100.0, loudness=0.65, onset_strength=0.25,
                low_energy=0.5, mid_energy=0.5, high_energy=0.4,
                bar_phase=0.3, beat_confidence=0.8, bpm=120.0,
                section="groove", section_confidence=0.8,
            )
        )
        self.assertEqual(result.decision.routine, "opposing_chase")

    def test_bpm_nudges_do_not_advance_phrase_without_bar_wrap(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        routines = []
        for index in range(12):
            result = runtime.step(
                MusicalObservation(
                    timestamp_s=500_000 + index * 0.2,
                    loudness=0.8, onset_strength=0.25,
                    low_energy=0.6, mid_energy=0.5, high_energy=0.4,
                    bar_phase=0.10 + index * 0.05,
                    beat_confidence=0.9, bpm=126.0 + index * 0.1,
                    section="groove", section_confidence=0.8,
                )
            )
            routines.append(result.decision.routine)
        self.assertEqual(len(set(routines)), 1)


if __name__ == "__main__":
    unittest.main()
