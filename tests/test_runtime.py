from __future__ import annotations

from dataclasses import replace
import math
import unittest

from lumen_engine.config import load_rig
from lumen_engine.choreography import MusicalContext, SequencePreferenceModel
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import MusicalObservation
from lumen_engine.models import Vec3
from lumen_engine.runtime import PerformanceRuntime, _choreography_candidates


class RuntimeTests(unittest.TestCase):
    def test_trusted_structure_shapes_fixture_dynamics_without_creating_strobe(self) -> None:
        rig = load_rig("config/party-parrot-active.json")

        def output_for(section: str):
            runtime = PerformanceRuntime(
                rig.fixtures,
                VirtualDMXOutput(),
                auxiliary_fixtures=rig.auxiliary_fixtures,
                choreography_model=SequencePreferenceModel(),
            )
            runtime.set_structure_context(
                energy=section,
                confidence=0.95,
                boundary_probability=0.0,
                resolution={
                    "axes": {
                        "energy": {
                            "source": "cached_offline_teacher",
                            "provenance": {
                                "source": "operator_annotation_consensus"
                            },
                        }
                    }
                },
            )
            return runtime.step(self._lane_observation(
                0.0, 0.0, section=section, loudness=0.65
            )).effective_outputs[0]

        breakdown = output_for("breakdown")
        drop = output_for("drop")
        self.assertLess(breakdown.motion_speed, drop.motion_speed)
        self.assertLess(breakdown.travel_size, drop.travel_size)
        self.assertLess(breakdown.activity_density, drop.activity_density)
        self.assertLess(breakdown.color_activity, drop.color_activity)
        self.assertEqual(breakdown.palette, "cool")
        self.assertEqual(drop.palette, "party_vivid")
        self.assertFalse(breakdown.strobe_enabled)

    def test_canonical_teacher_drop_selects_drop_choreography_without_loudness(self) -> None:
        candidates = _choreography_candidates(MusicalContext(
            energy_label="drop",
            energy=0.5,
            motion=0.5,
            tension=0.6,
        ))
        self.assertTrue(candidates[0].sequence_id.startswith("movers-release-"))

    @staticmethod
    def _lane_observation(
        timestamp: float,
        phase: float,
        *,
        section: str = "groove",
        loudness: float = 0.72,
    ) -> MusicalObservation:
        return MusicalObservation(
            timestamp_s=timestamp,
            loudness=loudness,
            onset_strength=0.55 if loudness else 0.0,
            low_energy=0.6 if loudness else 0.0,
            mid_energy=0.5 if loudness else 0.0,
            high_energy=0.4 if loudness else 0.0,
            bar_phase=phase,
            beat_pulse=0.8 if loudness else 0.0,
            beat_confidence=0.9,
            bpm=120.0,
            section=section,
            section_confidence=0.9,
        )

    def test_live_planner_runs_distinct_movers_and_center_lanes(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures, VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.step(self._lane_observation(0.0, 0.0))
        snapshot = runtime.choreography_snapshot()
        movers = snapshot["lanes"]["movers"]
        center = snapshot["lanes"]["center"]
        self.assertEqual(movers["active_step"]["routine"], "figure_eight")
        self.assertEqual(center["active_step"]["routine"], "counter_rotate")
        self.assertEqual(
            movers["active_boundary_id"].replace(":movers:", ":"),
            center["active_boundary_id"].replace(":center:", ":"),
        )
        self.assertNotEqual(
            movers["active_sequence_id"], center["active_sequence_id"]
        )
        self.assertIn("Selected at a new", movers["reason"])
        self.assertIn("Selected at a new", center["reason"])

    def test_effective_output_trace_exposes_literal_feedback_axes(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.replace_feedback({
            "overall": {
                "motion_speed": 0.6,
                "travel_size": -0.3,
                "activity_density": -0.4,
                "brightness": 0.2,
                "strobe_enabled": 1.0,
                "strobe_rate": 0.8,
                "beat_sync": 0.4,
                "cue_timing": -0.5,
            }
        })
        frame = runtime.step(self._lane_observation(
            0.0, 0.0, section="breakdown", loudness=0.45
        ))
        self.assertEqual(len(frame.effective_outputs), 3)
        mover = frame.effective_outputs[0]
        self.assertGreater(mover.motion_speed, 0.5)
        self.assertLess(mover.travel_size, 1.0)
        self.assertLess(mover.activity_density, 1.0)
        # Positive preference cannot create a strobe outside an authored cue.
        self.assertFalse(mover.strobe_enabled)
        snapshot = runtime.choreography_snapshot()
        self.assertIn(
            rig.fixtures[0].fixture_id, snapshot["effective_outputs"]
        )
        self.assertEqual(
            snapshot["effective_outputs"][rig.fixtures[0].fixture_id][
                "strobe"
            ]["rate"],
            0.0,
        )
    def test_continuous_mover_routine_does_not_flash_its_dimmer_on_beats(
        self,
    ) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        dimmers: list[int] = []
        strobes: list[int] = []
        # The first eight-beat authored step is a continuous figure eight;
        # stop before the next intentionally beam-gated opposing chase.
        for index in range(20):
            pulse = 1.0 if index % 5 == 0 else 0.05
            result = runtime.step(replace(
                self._lane_observation(
                    index * 0.1, (index % 20) / 20.0
                ),
                beat_pulse=pulse,
                onset_strength=0.8 if pulse > 0.5 else 0.15,
            ))
            # First mover starts at 31: dimmer is relative channel 6 and
            # hardware strobe is relative channel 7.
            dimmers.append(result.dmx.get_channel(0, 36))
            strobes.append(result.dmx.get_channel(0, 37))
        deltas = [
            abs(current - previous)
            for previous, current in zip(dimmers[5:], dimmers[6:])
        ]
        self.assertLessEqual(max(deltas), 12)
        self.assertEqual(max(strobes), 0)

    def test_same_section_develops_only_after_active_sequence_finishes(
        self,
    ) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.step(self._lane_observation(0.0, 0.0))
        initial = runtime.choreography_snapshot()["lanes"]
        initial_ids = {
            lane: initial[lane]["active_sequence_id"]
            for lane in ("movers", "center")
        }

        runtime.step(self._lane_observation(0.8, 0.9))
        runtime.step(self._lane_observation(1.0, 0.1))
        held = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(
            {
                lane: held[lane]["active_sequence_id"]
                for lane in ("movers", "center")
            },
            initial_ids,
        )

        for timestamp in (1.8, 2.0, 2.8, 3.0, 3.8, 4.0):
            runtime.step(self._lane_observation(timestamp, 0.9 if timestamp % 1.0 > 0.5 else 0.1))
        developed = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(
            developed["movers"]["active_sequence_id"],
            "movers-groove-wide-answer",
        )
        self.assertEqual(
            developed["center"]["active_sequence_id"],
            "center-groove-answer",
        )

    def test_section_change_waits_for_active_sequence_boundary(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.step(self._lane_observation(0.0, 0.0, section="groove"))
        before = runtime.choreography_snapshot()["lanes"]
        before_ids = {
            lane: before[lane]["active_sequence_id"]
            for lane in ("movers", "center")
        }

        runtime.step(self._lane_observation(0.5, 0.5, section="build"))
        held = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(
            {
                lane: held[lane]["active_sequence_id"]
                for lane in ("movers", "center")
            },
            before_ids,
        )
        runtime.step(self._lane_observation(0.8, 0.9, section="build"))
        runtime.step(self._lane_observation(1.0, 0.1, section="build"))
        for timestamp in (1.8, 2.0, 2.8, 3.0, 3.8, 4.0):
            runtime.step(self._lane_observation(timestamp, 0.9 if timestamp % 1.0 > 0.5 else 0.1, section="build"))
        after = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(
            after["movers"]["active_sequence_id"],
            "movers-build-and-answer",
        )
        self.assertEqual(
            after["center"]["active_sequence_id"],
            "center-build-chase",
        )

    def test_opposing_chase_trades_calibrated_position_and_beam(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
        )
        runtime.set_rehearsal("opposing_chase", scope="movers")

        def run(timestamp: float, phase: float):
            frame = runtime.step(self._lane_observation(
                timestamp, phase, section="release", loudness=0.88,
            ))
            semantic_pan = []
            for fixture, solution in zip(rig.fixtures, frame.solutions):
                calibration = fixture.calibration
                numeric = (
                    solution.pan_deg - calibration.pan_min_deg
                ) / (calibration.pan_max_deg - calibration.pan_min_deg)
                semantic_pan.append(
                    numeric
                    if calibration.pan_direction > 0
                    else 1.0 - numeric
                )
            dimmers = tuple(
                frame.dmx.get_channel(
                    fixture.universe,
                    fixture.address + fixture.dimmer_channel - 1,
                )
                for fixture in rig.fixtures
            )
            return semantic_pan, dimmers

        _, first_dimmers = run(0.0, 0.0)
        second_pan, second_dimmers = run(0.5, 0.25)
        self.assertGreater(abs(second_pan[0] - second_pan[1]), 0.20)
        self.assertGreater(first_dimmers[0], 0)
        self.assertEqual(first_dimmers[1], 0)
        self.assertEqual(second_dimmers[0], 0)
        self.assertGreater(second_dimmers[1], 0)

    def test_repeated_scoped_feedback_waits_and_changes_only_movers_lane(
        self,
    ) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures, VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.step(self._lane_observation(0.0, 0.9))
        before = runtime.choreography_snapshot()["lanes"]
        for index in range(4):
            result = runtime.learn_choreography_feedback(
                label="pick_it_up", value=1.0, urgency=1.0,
                occurrences=3, scope="group", fixture_id="movers",
                preferred_routine="beat_nod", event_id=f"crowd:{index}",
            )
            self.assertEqual(tuple(result["lanes"]), ("movers",))
        runtime.step(self._lane_observation(0.2, 0.1))
        held = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(
            held["movers"]["active_sequence_id"],
            before["movers"]["active_sequence_id"],
        )
        self.assertEqual(
            held["center"]["active_sequence_id"],
            before["center"]["active_sequence_id"],
        )
        runtime.step(self._lane_observation(1.8, 0.9))
        runtime.step(self._lane_observation(2.0, 0.1))
        after = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(after["movers"]["active_step"]["routine"], "beat_nod")
        self.assertEqual(
            after["center"]["active_sequence_id"],
            before["center"]["active_sequence_id"],
        )

    def test_whole_rig_preference_addresses_both_lanes_at_boundary(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures, VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.step(self._lane_observation(0.0, 0.9))
        learned = runtime.learn_choreography_feedback(
            label="more_like_this", value=1.0, occurrences=5,
            scope="overall", preferred_routine="counter_rotate",
            event_id="crowd:whole-rig",
        )
        self.assertEqual(set(learned["lanes"]), {"movers", "center"})
        runtime.step(self._lane_observation(0.2, 0.1))
        runtime.step(self._lane_observation(1.8, 0.9))
        runtime.step(self._lane_observation(2.0, 0.1))
        lanes = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(lanes["movers"]["active_step"]["routine"], "counter_rotate")
        self.assertEqual(lanes["center"]["active_step"]["routine"], "counter_rotate")

    def test_whole_rig_feedback_accepts_independent_lane_consensus(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        model = SequencePreferenceModel()
        runtime = PerformanceRuntime(
            rig.fixtures, VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=model,
        )
        runtime.step(self._lane_observation(0.0, 0.9))
        learned = runtime.learn_choreography_feedback(
            label="pick_it_up",
            value=1.0,
            scope="overall",
            event_ids_by_lane={
                "movers": "batch:movers-context",
                "center": "batch:center-context",
            },
            occurrences_by_lane={"movers": 4, "center": 2},
            urgency_by_lane={"movers": 0.9, "center": 0.55},
        )
        assert learned is not None
        events = model.state_dict()["events"]
        self.assertEqual(
            set(events), {"batch:movers-context", "batch:center-context"}
        )
        self.assertEqual(
            events["batch:movers-context"]["example"]["feedback"][0][
                "occurrences"
            ],
            4,
        )
        self.assertEqual(
            events["batch:center-context"]["example"]["feedback"][0][
                "occurrences"
            ],
            2,
        )
        self.assertEqual(
            events["batch:movers-context"]["example"]["feedback"][0][
                "urgency"
            ],
            0.9,
        )
        self.assertEqual(
            events["batch:center-context"]["example"]["feedback"][0][
                "urgency"
            ],
            0.55,
        )

    def test_characteristic_feedback_never_becomes_a_choreography_sequence(
        self,
    ) -> None:
        rig = load_rig("config/party-parrot-active.json")
        model = SequencePreferenceModel()
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=model,
        )
        runtime.step(self._lane_observation(0.0, 0.9))
        learned = runtime.learn_choreography_feedback(
            label="no_strobes",
            value=1.0,
            occurrences=4,
            scope="overall",
            event_id="characteristic-only",
        )
        assert learned is not None
        self.assertFalse(learned["preferred_sequence_learned"])
        self.assertIsNone(learned["preferred_sequence"])
        self.assertEqual(model.learned_candidates(), ())

    def test_silence_selects_calm_step_for_each_lane(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures, VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.set_structure_context(energy="silence", confidence=0.9)
        runtime.step(self._lane_observation(
            0.0, 0.0, section="silence", loudness=0.0
        ))
        lanes = runtime.choreography_snapshot()["lanes"]
        self.assertEqual(lanes["movers"]["active_step"]["routine"], "breathe")
        self.assertEqual(lanes["center"]["active_step"]["routine"], "breathe")

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

    def test_scalar_feedback_updates_without_replanning_active_phrase(self) -> None:
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
        original_boundary = runtime.choreography_snapshot()["lanes"][
            "movers"
        ]["active_boundary_id"]
        runtime.replace_feedback(
            {
                "overall": {
                    "motion": 1.0,
                    "intensity": 0.8,
                    "strobe": 0.6,
                }
            },
            replan_lanes=(),
        )
        self.assertEqual(runtime._feedback_motion["overall"], 1.0)
        self.assertIsNone(runtime._pending_feedback_biases)
        self.assertEqual(
            runtime.choreography_snapshot()["replan_pending_lanes"], []
        )
        runtime.step(observation(0.2, 0.1))
        self.assertEqual(
            runtime.choreography_snapshot()["lanes"]["movers"][
                "active_boundary_id"
            ],
            original_boundary,
        )

    def test_more_movement_overrides_structure_baseline_without_replan(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.set_structure_context(
            energy="breakdown",
            confidence=0.95,
            resolution={"axes": {"energy": {"source": "cached_teacher"}}},
        )
        first = runtime.step(self._lane_observation(
            0.0, 0.2, section="breakdown", loudness=0.45,
        ))
        baseline = first.effective_outputs[0].travel_size
        boundary = runtime.choreography_snapshot()["lanes"]["movers"][
            "active_boundary_id"
        ]
        runtime.replace_feedback(
            {"overall": {"travel_size": 1.0}}, replan_lanes=()
        )
        second = runtime.step(self._lane_observation(
            0.02, 0.21, section="breakdown", loudness=0.45,
        ))
        self.assertGreater(second.effective_outputs[0].travel_size, baseline)
        self.assertEqual(
            runtime.choreography_snapshot()["lanes"]["movers"][
                "active_boundary_id"
            ],
            boundary,
        )

    def test_motion_clock_does_not_jump_when_bpm_estimate_changes(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())

        def observation(timestamp: float, bpm: float, phase: float):
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.7,
                onset_strength=0.4,
                low_energy=0.6,
                mid_energy=0.5,
                high_energy=0.4,
                bpm=bpm,
                beat_confidence=0.9,
                bar_phase=phase,
            )

        self.assertAlmostEqual(
            runtime._continuous_motion_beat(observation(0.0, 120.0, 0.0)),
            0.0,
        )
        before = runtime._continuous_motion_beat(
            observation(1.0, 120.0, 0.5)
        )
        # Both movers ask for the clock at the same timestamp. A changed BPM
        # estimate must not change the already-published path position.
        same_frame = runtime._continuous_motion_beat(
            observation(1.0, 180.0, 0.5)
        )
        self.assertAlmostEqual(same_frame, before)
        after = runtime._continuous_motion_beat(
            observation(1.01, 180.0, 0.505)
        )
        self.assertGreater(after, before)
        self.assertLess(after - before, 0.03)
        self.assertLessEqual(runtime._motion_clock_bpm, 120.12 + 1e-9)

    def test_motion_speed_change_alters_velocity_without_phase_jump(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())

        def observation(timestamp: float, phase: float):
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.7,
                onset_strength=0.4,
                low_energy=0.6,
                mid_energy=0.5,
                high_energy=0.4,
                bpm=120.0,
                beat_confidence=0.9,
                bar_phase=phase,
            )

        runtime._continuous_motion_path_beat(
            observation(0.0, 0.0), 0.75
        )
        before = runtime._continuous_motion_path_beat(
            observation(1.0, 0.5), 0.75
        )
        same_frame = runtime._continuous_motion_path_beat(
            observation(1.0, 0.5), 1.5
        )
        self.assertAlmostEqual(same_frame, before)
        after = runtime._continuous_motion_path_beat(
            observation(1.1, 0.55), 1.5
        )
        self.assertGreater(after, before)
        self.assertLess(after - before, 0.35)

    def test_every_rehearsal_routine_emits_both_movers_continuously(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        for routine in (
            "breathe",
            "fan_sweep",
            "figure_eight",
            "opposing_chase",
            "beat_nod",
            "counter_rotate",
        ):
            with self.subTest(routine=routine):
                runtime = PerformanceRuntime(
                    rig.fixtures,
                    VirtualDMXOutput(),
                    auxiliary_fixtures=rig.auxiliary_fixtures,
                )
                runtime.set_rehearsal(
                    routine, scope="movers", size=1.0
                )
                prior: dict[str, tuple[float, float]] = {}
                for index in range(240):
                    timestamp = index / 50.0
                    beat_position = timestamp * 124.0 / 60.0
                    result = runtime.step(MusicalObservation(
                        timestamp_s=timestamp,
                        loudness=0.72,
                        onset_strength=0.4,
                        low_energy=0.6,
                        mid_energy=0.5,
                        high_energy=0.4,
                        bpm=124.0 + 12.0 * math.sin(timestamp * 1.7),
                        beat_confidence=0.9,
                        bar_phase=(beat_position % 4.0) / 4.0,
                        section="groove",
                        section_confidence=0.8,
                    ))
                    self.assertEqual(len(result.solutions), len(rig.fixtures))
                    self.assertFalse(result.warnings)
                    for solution in result.solutions:
                        previous = prior.get(solution.fixture_id)
                        if previous is not None:
                            self.assertLessEqual(
                                abs(solution.pan_deg - previous[0]),
                                180.0 / 50.0 + 1e-6,
                            )
                            self.assertLessEqual(
                                abs(solution.tilt_deg - previous[1]),
                                180.0 / 50.0 + 1e-6,
                            )
                        prior[solution.fixture_id] = (
                            solution.pan_deg, solution.tilt_deg
                        )

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
        self.assertEqual(result.dmx.get_channel(0, 2), 255)
        self.assertEqual(result.dmx.get_channel(0, 3), 128)
        self.assertEqual(result.dmx.get_channel(0, 4), 128)
        self.assertEqual(result.dmx.get_channel(0, 5), 24)
        self.assertEqual(result.dmx.get_channel(0, 6), 0)
        self.assertEqual(result.dmx.get_channel(0, 15), 0)
        self.assertEqual(result.dmx.get_channel(0, 16), 0)
        center_output = next(
            item
            for item in runtime.choreography_snapshot()["effective_outputs"].values()
            if item["lane"] == "center"
        )
        self.assertEqual(center_output["routine"], "parked")
        self.assertEqual(center_output["activity_density"], 0.0)

    def test_confirmed_silence_parks_center_without_second_delay(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
        )
        runtime.step(MusicalObservation(
            timestamp_s=0.0,
            loudness=0.8,
            onset_strength=0.5,
            low_energy=0.5,
            mid_energy=0.3,
            high_energy=0.2,
        ))
        result = runtime.step(MusicalObservation(
            timestamp_s=0.6,
            loudness=0.0,
            onset_strength=0.0,
            low_energy=0.0,
            mid_energy=0.0,
            high_energy=0.0,
            section="silence",
            section_confidence=1.0,
        ))
        self.assertEqual(result.dmx.get_channel(0, 1), 128)
        self.assertEqual(result.dmx.get_channel(0, 2), 255)
        self.assertEqual(result.dmx.get_channel(0, 3), 128)
        self.assertEqual(result.dmx.get_channel(0, 4), 128)

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
        # Center strobe is opt-in through an authored center routine or
        # feedback; high musical energy alone must not reintroduce the old
        # always-strobing behavior.
        self.assertEqual(max(center_strobe), 0)

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

    def test_playback_seek_releases_active_causal_choreography(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(
            rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=rig.auxiliary_fixtures,
            choreography_model=SequencePreferenceModel(),
        )
        runtime.set_media_context(99, "groove", "artist")
        runtime.step(self._lane_observation(10.0, 0.2))
        self.assertIsNotNone(
            runtime.choreography_snapshot()["lanes"]["movers"]
            ["active_sequence_id"]
        )

        runtime.notify_timeline_discontinuity()

        lanes = runtime.choreography_snapshot()["lanes"]
        self.assertIsNone(lanes["movers"]["active_sequence_id"])
        self.assertIsNone(lanes["center"]["active_sequence_id"])
        self.assertIsNone(runtime._last_bar_phase)

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

    def test_tempo_reacquisition_cannot_create_a_false_bar_wrap(self) -> None:
        rig = load_rig("config/party-parrot-active.json")
        runtime = PerformanceRuntime(rig.fixtures, VirtualDMXOutput())
        runtime.step(self._lane_observation(0.0, 0.90))
        self.assertEqual(runtime._routine_bar_counter, 0)

        unlocked = replace(
            self._lane_observation(0.4, 0.0),
            beat_confidence=0.0,
            bpm=None,
        )
        runtime.step(unlocked)
        self.assertIsNone(runtime._last_bar_phase)

        runtime.step(self._lane_observation(0.8, 0.10))
        self.assertEqual(runtime._routine_bar_counter, 0)


if __name__ == "__main__":
    unittest.main()
