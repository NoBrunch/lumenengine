from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from lumen_engine.control import GatedOutput, LumenApplication
from lumen_engine.choreography import ChoreographySequence, ChoreographyStep
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import (
    ExpressionState,
    Gesture,
    MediaIdentity,
    MusicalObservation,
    PerformanceDecision,
    Vec3,
)
from lumen_engine.runtime import PerformanceRuntime, _apply_choreography_step


class GoalIntegrationTests(unittest.TestCase):
    """Cross-subsystem contracts required by the finished teaching workflow."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.rig_path = root / "rig.json"
        self.rig_path.write_text(
            Path("config/party-parrot-active.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "memory.sqlite3",
            settings_path=root / "settings.json",
        )

    def tearDown(self) -> None:
        self.application.close()
        self.temporary.cleanup()

    def test_center_group_scalar_feedback_reaches_center_fixture(self) -> None:
        """A permanent group target must affect the fixture in that group."""

        center = self.application.rig.auxiliary_fixtures[0]
        result = self.application.add_feedback(
            {
                "label": "increase_movement",
                "value": 1.0,
                "scope": "group",
                "group_id": "center",
                "participant_id": "listener-a",
                "client_event_id": "center-motion-1",
            }
        )
        self.assertTrue(result["created"])

        runtime = PerformanceRuntime(
            self.application.rig.fixtures,
            VirtualDMXOutput(),
            auxiliary_fixtures=self.application.rig.auxiliary_fixtures,
        )
        runtime.replace_feedback(self.application._feedback_biases)
        center_motion, _intensity, _strobe, _palette = runtime._feedback_for(
            center.fixture_id
        )
        self.assertGreater(center_motion, 0.0)

    def test_preferred_action_retry_is_idempotent_per_listener_event(self) -> None:
        """Network retries must not duplicate supervision or learning weight."""

        payload = {
            "kind": "preferred_action",
            "label": "figure_eight",
            "scope": "group",
            "group_id": "movers",
            "participant_id": "listener-a",
            "participant_name": "Alex",
            "client_event_id": "preferred-action-42",
        }

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda _index: self.application.add_training_annotation(
                        dict(payload)
                    ),
                    range(20),
                )
            )

        self.assertEqual(
            len({int(result["annotation_id"]) for result in results}), 1
        )
        self.assertEqual(
            self.application.memory.training_summary()["annotations"], 1
        )

    def test_simultaneous_listeners_aggregate_without_interrupting_phrase(
        self,
    ) -> None:
        """Distinct phones add agreement while the active lane lease holds."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        self.application._remember_media_identity(MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:multiuser-lifetime",
            title="Multiuser lifetime",
            artists=("Test Artist",),
            is_playing=False,
        ))
        assert self.application.song_id is not None
        runtime.set_media_context(
            self.application.song_id, "groove", "Test Artist"
        )
        observation = MusicalObservation(
            timestamp_s=100.0,
            loudness=0.78,
            onset_strength=0.62,
            low_energy=0.70,
            mid_energy=0.58,
            high_energy=0.44,
            beat_phase=0.2,
            bar_phase=0.25,
            beat_pulse=0.7,
            beat_confidence=0.9,
            bpm=124.0,
            section="groove",
            section_confidence=0.8,
        )
        frame = runtime.step(observation)
        before = {
            lane: runtime.choreography_snapshot()["lanes"][lane][
                "active_sequence_id"
            ]
            for lane in ("movers", "center")
        }
        self.application._runtime = runtime
        self.application.observation = observation
        self.application.frame = frame

        def submit(index: int) -> dict[str, object]:
            return self.application.add_feedback(
                {
                    "label": "increase_movement",
                    "value": 1.0,
                    "scope": "group",
                    "group_id": "movers",
                    "participant_id": f"listener-{index}",
                    "client_event_id": f"group-tap-{index}",
                }
            )

        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(submit, range(12)))

        after_snapshot = runtime.choreography_snapshot()
        after = {
            lane: after_snapshot["lanes"][lane]["active_sequence_id"]
            for lane in ("movers", "center")
        }
        self.assertEqual(after, before)
        self.assertEqual(
            max(int(result["participant_agreement"]) for result in results),
            12,
        )
        self.assertEqual(
            max(int(result["feedback_occurrences"]) for result in results),
            12,
        )
        self.assertIsNone(runtime._pending_feedback_biases)
        self.assertEqual(after_snapshot["replan_pending_lanes"], [])
        self.assertGreater(
            max(runtime._feedback_travel_size.values(), default=0.0),
            0.0,
        )
        events = self.application._choreography_model.state_dict()["events"]
        mover_events = [
            event for event in events.values()
            if event.get("lane") == "movers"
        ]
        self.assertEqual(len(mover_events), 1)
        self.assertEqual(mover_events[0]["lifetime"], "cue")
        self.assertIn(
            "section:groove",
            mover_events[0]["example"]["context"]["cue_key"],
        )
        self.assertEqual(
            mover_events[0]["example"]["feedback"][0]["occurrences"],
            12,
        )
        self.application._runtime = None
        runtime.close()

    def test_center_feedback_queues_only_the_center_lane(self) -> None:
        """A center correction cannot force a movers sequence reselection."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        runtime.step(MusicalObservation(
            timestamp_s=0.0,
            loudness=0.7,
            onset_strength=0.5,
            low_energy=0.6,
            mid_energy=0.5,
            high_energy=0.4,
            bar_phase=0.1,
            beat_confidence=0.9,
            bpm=120.0,
            section="groove",
            section_confidence=0.8,
        ))
        runtime.replace_feedback(
            {"center-fixture": {"motion": 0.4}},
            replan_lanes=("center",),
        )
        self.assertEqual(
            runtime.choreography_snapshot()["replan_pending_lanes"],
            ["center"],
        )
        runtime.close()

    def test_new_section_boundary_starts_each_lane_sequence_at_beat_zero(
        self,
    ) -> None:
        """A newly leased sequence must not inherit the old phrase offset."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        runtime.set_media_context(77)

        def observation(timestamp: float, section: str, bar_phase: float):
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.82,
                onset_strength=0.65,
                low_energy=0.72,
                mid_energy=0.60,
                high_energy=0.45,
                beat_phase=bar_phase * 4.0 % 1.0,
                bar_phase=bar_phase,
                beat_pulse=0.7,
                beat_confidence=0.9,
                bpm=124.0,
                section=section,
                section_confidence=0.9,
            )

        runtime.step(observation(10.0, "groove", 0.10))
        runtime.step(observation(11.0, "drop", 0.75))
        lanes = runtime.choreography_snapshot()["lanes"]

        self.assertEqual(lanes["movers"]["active_step"]["start_beat"], 0.0)
        self.assertEqual(lanes["center"]["active_step"]["start_beat"], 0.0)
        runtime.close()

    def test_authored_sequence_can_reach_steps_after_eight_beats(self) -> None:
        """A complete taught phrase may span more than one planner boundary."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        runtime.set_media_context(88)
        runtime.set_recalled_choreography(
            (
                ChoreographySequence(
                    sequence_id="sixteen-beat-movers",
                    source="operator_song_timeline",
                    base_priority=1.0,
                    steps=(
                        ChoreographyStep(
                            0.0, 8.0, "movers", "opposing_chase"
                        ),
                        ChoreographyStep(
                            8.0, 8.0, "movers", "fan_sweep"
                        ),
                    ),
                ),
            )
        )

        def observation(timestamp: float, bar_phase: float):
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.8,
                onset_strength=0.6,
                low_energy=0.7,
                mid_energy=0.6,
                high_energy=0.4,
                bar_phase=bar_phase,
                beat_confidence=0.9,
                bpm=120.0,
                section="groove",
                section_confidence=0.9,
            )

        for timestamp, phase in (
            (0.0, 0.0),
            (1.0, 0.9),
            (2.0, 0.1),
            (3.0, 0.9),
            (4.0, 0.1),
        ):
            runtime.step(observation(timestamp, phase))

        active = runtime.choreography_snapshot()["lanes"]["movers"][
            "active_step"
        ]
        self.assertEqual(active["start_beat"], 8.0)
        self.assertEqual(active["routine"], "fan_sweep")
        runtime.close()

    def test_new_recalled_sequence_waits_for_boundary_then_replans_lane(self) -> None:
        """A timeline call arriving mid-phrase is pending, never immediate."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        sequence = ChoreographySequence(
            sequence_id="new-center-call",
            source="operator_song_timeline",
            base_priority=1.0,
            steps=(
                ChoreographyStep(0.0, 8.0, "center", "opposing_chase"),
            ),
        )
        runtime.set_recalled_choreography((sequence,))
        snapshot = runtime.choreography_snapshot()
        self.assertIn("center", snapshot["replan_pending_lanes"])
        self.assertNotIn("movers", snapshot["replan_pending_lanes"])

        # Once recall leaves its polling window, it must not request another
        # replan that would cut short the sequence already admitted.
        runtime.set_recalled_choreography(())
        snapshot = runtime.choreography_snapshot()
        self.assertEqual(snapshot["replan_pending_lanes"], ["center"])
        runtime.close()

    def test_recalled_timeline_placement_is_not_replayed_during_poll_grace(
        self,
    ) -> None:
        """The discovery grace may admit a call once, never loop the call."""

        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )

        def observation(timestamp: float, phase: float) -> MusicalObservation:
            return MusicalObservation(
                timestamp_s=timestamp,
                loudness=0.72,
                onset_strength=0.55,
                low_energy=0.60,
                mid_energy=0.50,
                high_energy=0.40,
                bar_phase=phase,
                beat_pulse=0.70,
                beat_confidence=0.90,
                bpm=120.0,
                section="groove",
                section_confidence=0.90,
            )

        runtime.step(observation(0.0, 0.1))
        recalled = ChoreographySequence(
            sequence_id="one-shot-movers",
            source="operator_song_timeline",
            base_priority=1.0,
            steps=(
                ChoreographyStep(
                    0.0, 8.0, "movers", "counter_rotate"
                ),
            ),
        )
        runtime.set_recalled_choreography((recalled,))
        for timestamp, phase in (
            (1.0, 0.9),
            (2.0, 0.1),
            (3.0, 0.9),
            (4.0, 0.1),
        ):
            runtime.step(observation(timestamp, phase))
        admitted = runtime.choreography_snapshot()["lanes"]["movers"]
        self.assertEqual(
            admitted["active_sequence_id"], "one-shot-movers@movers"
        )
        admitted_boundary = admitted["active_boundary_id"]

        # The background poll can still expose the placement during its grace
        # interval. Completing the authored eight beats must nevertheless
        # return the lane to normal planning instead of restarting beat zero.
        for timestamp, phase in (
            (5.0, 0.9),
            (6.0, 0.1),
            (7.0, 0.9),
            (8.0, 0.1),
            # Advance beyond the exact floating-point boundary edge.
            (8.1, 0.2),
        ):
            runtime.step(observation(timestamp, phase))
        completed = runtime.choreography_snapshot()["lanes"]["movers"]
        self.assertNotEqual(completed["active_boundary_id"], admitted_boundary)
        self.assertNotEqual(
            completed["active_sequence_id"], "one-shot-movers@movers"
        )

        # Leaving the placement window clears its consumption marker. Re-entry
        # on a later play can therefore schedule this same stable placement ID.
        runtime.set_recalled_choreography(())
        runtime.set_recalled_choreography((recalled,))
        self.assertIn(
            "movers", runtime.choreography_snapshot()["replan_pending_lanes"]
        )
        for timestamp, phase in (
            (9.0, 0.9),
            (10.0, 0.1),
            (11.0, 0.9),
            (12.0, 0.2),
        ):
            runtime.step(observation(timestamp, phase))
        replay = runtime.choreography_snapshot()["lanes"]["movers"]
        self.assertEqual(
            replay["active_sequence_id"], "one-shot-movers@movers"
        )
        runtime.close()

    def test_choreography_timing_controls_change_the_resolved_output(self) -> None:
        """Persisted timing controls must not be inert metadata."""

        base = PerformanceDecision(
            timestamp_s=10.0,
            gesture=Gesture.SWEEP,
            expression=ExpressionState(
                energy=0.72,
                tension=0.55,
                motion=0.60,
                intimacy=0.30,
                confidence=0.90,
            ),
            target=Vec3(0.0, 0.0, 1.2),
            brightness=0.70,
            reason="integration test",
            confidence=0.90,
        )

        unsynced = ChoreographyStep(
            0.0, 8.0, "movers", "fan_sweep", beat_sync=0.0
        )
        synced = ChoreographyStep(
            0.0, 8.0, "movers", "fan_sweep", beat_sync=1.0
        )
        without_beat = _apply_choreography_step(
            base, unsynced, beat_pulse=1.0, step_elapsed_beats=2.0
        )
        with_beat = _apply_choreography_step(
            base, synced, beat_pulse=1.0, step_elapsed_beats=2.0
        )
        self.assertGreater(with_beat.brightness, without_beat.brightness)
        self.assertGreater(
            with_beat.expression.motion, without_beat.expression.motion
        )

        ordinary_entry = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep",
                entry_behavior="phrase_boundary",
            ),
            step_elapsed_beats=0.0,
        )
        soft_entry = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep", entry_behavior="soft"
            ),
            step_elapsed_beats=0.0,
        )
        accent_entry = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep", entry_behavior="accent"
            ),
            step_elapsed_beats=0.0,
        )
        self.assertLess(soft_entry.brightness, ordinary_entry.brightness)
        self.assertGreater(accent_entry.brightness, ordinary_entry.brightness)

        ordinary_exit = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep", exit_behavior="resolve"
            ),
            step_elapsed_beats=7.75,
        )
        crossfade_exit = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep", exit_behavior="crossfade"
            ),
            step_elapsed_beats=7.75,
        )
        blackout_exit = _apply_choreography_step(
            base,
            ChoreographyStep(
                0.0, 8.0, "movers", "fan_sweep", exit_behavior="blackout"
            ),
            step_elapsed_beats=7.75,
        )
        self.assertLess(crossfade_exit.brightness, ordinary_exit.brightness)
        self.assertLess(blackout_exit.brightness, crossfade_exit.brightness)


if __name__ == "__main__":
    unittest.main()
