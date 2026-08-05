from __future__ import annotations

import json
import threading
import time
import unittest

from lumen_engine.choreography import (
    BoundarySequencePlanner,
    CapturedChoreographyExample,
    ChoreographySequence,
    ChoreographyStep,
    DmxHistorySample,
    FeedbackSignal,
    MusicalContext,
    ParallelBoundarySequencePlanner,
    SequencePreferenceModel,
    _sequence_metrics,
    sequence_for_lane,
    summarize_dmx_history,
)


def _calm_sequence() -> ChoreographySequence:
    return ChoreographySequence(
        sequence_id="calm-blue",
        steps=(
            ChoreographyStep(
                start_beat=0,
                duration_beats=8,
                fixture_scope="movers",
                routine="breathe",
                intensity=0.3,
                palette="blue_violet",
                beat_sync=0.7,
            ),
            ChoreographyStep(
                start_beat=8,
                duration_beats=8,
                fixture_scope="all",
                routine="fan_sweep",
                intensity=0.4,
                palette="blue_violet",
                beat_sync=0.8,
            ),
        ),
    )


def _rave_sequence() -> ChoreographySequence:
    return ChoreographySequence(
        sequence_id="rave-release",
        steps=(
            ChoreographyStep(
                start_beat=0,
                duration_beats=8,
                fixture_scope="movers",
                routine="opposing_chase",
                intensity=0.9,
                palette="saturated_jewel",
                strobe=0.7,
            ),
            ChoreographyStep(
                start_beat=8,
                duration_beats=4,
                fixture_scope="center",
                routine="counter_rotate",
                intensity=1.0,
                palette="saturated_jewel",
                strobe=0.8,
            ),
            ChoreographyStep(
                start_beat=12,
                duration_beats=4,
                fixture_scope="all",
                routine="blackout_accent",
                intensity=0.8,
                palette="saturated_jewel",
            ),
        ),
    )


class SequencePreferenceTests(unittest.TestCase):
    def test_step_characteristics_are_serialized_and_scored_independently(self) -> None:
        step = ChoreographyStep(
            start_beat=0,
            duration_beats=8,
            fixture_scope="movers",
            routine="figure_eight",
            intensity=0.8,
            motion_speed=0.2,
            travel_size=0.4,
            activity_density=0.6,
            brightness=0.7,
            strobe_enabled=True,
            strobe_rate=0.3,
            beat_sync=0.5,
            cue_timing=0.9,
        )
        self.assertEqual(ChoreographyStep.from_dict(step.as_dict()), step)
        base = ChoreographySequence("base", (step,))
        baseline = _sequence_metrics(base)
        dimensions = {
            "motion_speed": 0.8,
            "travel_size": 0.9,
            "activity_density": 0.9,
            "brightness": 0.2,
            "strobe_rate": 0.8,
            "beat_sync": 0.9,
            "cue_timing": 0.2,
        }
        from dataclasses import replace
        for metric, value in dimensions.items():
            changed = _sequence_metrics(ChoreographySequence(
                metric, (replace(step, **{metric: value}),)
            ))
            self.assertNotEqual(changed[metric], baseline[metric], metric)

    def test_one_step_preferred_action_trains_but_is_not_a_live_candidate(self) -> None:
        context = MusicalContext(energy_label="groove", energy=0.6)
        performed = _rave_sequence()
        one_step = ChoreographySequence(
            sequence_id="tap:figure-eight",
            source="operator_preferred_action",
            steps=(ChoreographyStep(
                start_beat=0,
                duration_beats=8,
                fixture_scope="movers",
                routine="figure_eight",
            ),),
        )
        model = SequencePreferenceModel()
        receipt = model.learn(CapturedChoreographyExample(
            context=context,
            performed=performed,
            preferred=one_step,
            feedback=(FeedbackSignal(label="more_like_this"),),
        ), event_id="preferred-action:one")
        self.assertTrue(receipt.preferred_sequence_learned)
        self.assertEqual(model.learned_candidates(), ())
        self.assertEqual(
            list(model.state_dict()["events"]), ["preferred-action:one"]
        )

    def test_feedback_vocabulary_writes_only_its_literal_metric_axis(self) -> None:
        expected = {
            "increase_movement": "travel_size",
            "faster": "motion_speed",
            "not_busy_enough": "activity_density",
            "brighter": "brightness",
            "strobe": "strobe_enabled",
            "faster_strobe": "strobe_rate",
            "better_beat_sync": "beat_sync",
            "great_timing": "cue_timing",
            "warmer_color": "palette_warmth",
        }
        context = MusicalContext(energy_label="groove", energy=0.6)
        for label, metric in expected.items():
            model = SequencePreferenceModel()
            model.learn(CapturedChoreographyExample(
                context=context,
                performed=_calm_sequence(),
                feedback=(FeedbackSignal(label=label),),
            ))
            weights = model.state_dict()["weights"]
            self.assertIn(f"metric:{metric}", weights, label)

    def test_live_ranking_never_waits_for_feedback_model_mutation(self) -> None:
        model = SequencePreferenceModel()
        entered = threading.Event()
        release = threading.Event()

        def hold_writer_lock() -> None:
            with model._lock:
                entered.set()
                release.wait(timeout=2.0)

        worker = threading.Thread(target=hold_writer_lock)
        worker.start()
        self.assertTrue(entered.wait(timeout=1.0))
        started = time.monotonic()
        ranked = model.rank(
            MusicalContext(energy_label="groove", energy=0.6),
            (_calm_sequence(),),
            lane="movers",
        )
        elapsed = time.monotonic() - started
        release.set()
        worker.join(timeout=1.0)
        self.assertEqual(ranked[0].sequence.sequence_id, "calm-blue")
        self.assertLess(elapsed, 0.05)

    def test_planner_hard_limits_three_identical_opening_routines(
        self,
    ) -> None:
        model = SequencePreferenceModel()
        planner = BoundarySequencePlanner(model)
        context = MusicalContext(energy_label="groove", energy=0.6)
        repeated = ChoreographySequence(
            "preferred-repeat",
            (ChoreographyStep(0, 8, "movers", "opposing_chase"),),
            base_priority=1.0,
        )
        alternate = ChoreographySequence(
            "authored-alternate",
            (ChoreographyStep(0, 8, "movers", "fan_sweep"),),
        )
        choices = (repeated, alternate)

        first = planner.choose(
            boundary_id="phrase:1", context=context, candidates=choices
        )
        held = planner.choose(
            boundary_id="phrase:1", context=context, candidates=choices
        )
        second = planner.choose(
            boundary_id="phrase:2", context=context, candidates=choices
        )
        third = planner.choose(
            boundary_id="phrase:3", context=context, candidates=choices
        )

        self.assertTrue(held.held_for_boundary)
        self.assertEqual(first.sequence.sequence_id, "preferred-repeat")
        self.assertEqual(second.sequence.sequence_id, "preferred-repeat")
        self.assertEqual(third.sequence.sequence_id, "authored-alternate")
        self.assertIn("hard routine repetition limit", third.reason)

    def test_exact_owner_timeline_is_exempt_from_repetition_limit(
        self,
    ) -> None:
        planner = BoundarySequencePlanner(SequencePreferenceModel())
        context = MusicalContext(energy_label="release", energy=0.9)
        placed = ChoreographySequence(
            "owner-call",
            (ChoreographyStep(0, 8, "movers", "opposing_chase"),),
            source="operator_song_timeline",
            base_priority=1.0,
        )
        alternate = ChoreographySequence(
            "automatic-call",
            (ChoreographyStep(0, 8, "movers", "fan_sweep"),),
        )
        selections = [
            planner.choose(
                boundary_id=f"phrase:{index}",
                context=context,
                candidates=(placed, alternate),
            )
            for index in range(4)
        ]
        self.assertTrue(all(
            item.sequence.sequence_id == "owner-call"
            for item in selections
        ))

    def test_mixed_and_whole_rig_sequences_project_to_independent_lanes(
        self,
    ) -> None:
        sequence = _rave_sequence()
        movers = sequence_for_lane(sequence, "movers")
        center = sequence_for_lane(sequence, "center")
        self.assertIsNotNone(movers)
        self.assertIsNotNone(center)
        self.assertEqual(
            [step.routine for step in movers.steps],
            ["opposing_chase", "blackout_accent"],
        )
        self.assertEqual(
            [step.routine for step in center.steps],
            ["counter_rotate", "blackout_accent"],
        )
        self.assertTrue(all(
            step.fixture_scope == "movers" for step in movers.steps
        ))
        self.assertTrue(all(
            step.fixture_scope == "center" for step in center.steps
        ))

    def test_parallel_planner_leases_each_lane_independently(self) -> None:
        model = SequencePreferenceModel()
        planner = ParallelBoundarySequencePlanner(model)
        context = MusicalContext(energy_label="groove", energy=0.6)
        movers_calm = sequence_for_lane(_calm_sequence(), "movers")
        movers_rave = sequence_for_lane(_rave_sequence(), "movers")
        center_rave = sequence_for_lane(_rave_sequence(), "center")
        self.assertIsNotNone(movers_calm)
        self.assertIsNotNone(movers_rave)
        self.assertIsNotNone(center_rave)
        movers = planner.choose_lane(
            "movers", boundary_id="phrase:1", context=context,
            candidates=(movers_calm, movers_rave),
        )
        center = planner.choose_lane(
            "center", boundary_id="phrase:1", context=context,
            candidates=(center_rave,),
        )
        self.assertNotEqual(
            movers.sequence.sequence_id, center.sequence.sequence_id
        )
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=movers_calm,
                preferred=movers_rave,
            ),
            lane="movers",
        )
        held_movers = planner.choose_lane(
            "movers", boundary_id="phrase:1", context=context,
            candidates=(movers_calm, movers_rave),
        )
        held_center = planner.choose_lane(
            "center", boundary_id="phrase:1", context=context,
            candidates=(center_rave,),
        )
        self.assertEqual(
            held_movers.sequence.sequence_id, movers.sequence.sequence_id
        )
        self.assertEqual(
            held_center.sequence.sequence_id, center.sequence.sequence_id
        )
        self.assertTrue(held_movers.held_for_boundary)
        self.assertTrue(held_center.held_for_boundary)

    def test_lane_namespaced_feedback_does_not_rank_other_lane(self) -> None:
        model = SequencePreferenceModel()
        context = MusicalContext(energy_label="release", energy=0.8)
        movers_calm = sequence_for_lane(_calm_sequence(), "movers")
        movers_rave = sequence_for_lane(_rave_sequence(), "movers")
        center_calm = ChoreographySequence(
            "center-calm", (ChoreographyStep(
                0, 8, "center", "breathe", intensity=0.3
            ),)
        )
        center_rave = sequence_for_lane(_rave_sequence(), "center")
        self.assertIsNotNone(movers_calm)
        self.assertIsNotNone(movers_rave)
        self.assertIsNotNone(center_rave)
        before = model.rank(
            context, (center_calm, center_rave), lane="center"
        )
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=movers_calm,
                preferred=movers_rave,
                feedback=(FeedbackSignal(
                    label="pick_it_up", occurrences=8, scope="movers"
                ),),
            ),
            lane="movers",
        )
        after = model.rank(
            context, (center_calm, center_rave), lane="center"
        )
        self.assertEqual(
            [(row.sequence.sequence_id, row.score) for row in before],
            [(row.sequence.sequence_id, row.score) for row in after],
        )

    def test_whole_rig_event_base_id_forgets_both_lane_updates(self) -> None:
        model = SequencePreferenceModel()
        context = MusicalContext(energy_label="release", energy=0.8)
        for lane in ("movers", "center"):
            performed = ChoreographySequence(
                f"performed:{lane}",
                (ChoreographyStep(0, 8, lane, "breathe"),),
            )
            preferred = ChoreographySequence(
                f"preferred:{lane}",
                (ChoreographyStep(0, 8, lane, "counter_rotate"),),
            )
            model.learn(
                CapturedChoreographyExample(
                    context=context,
                    performed=performed,
                    preferred=preferred,
                ),
                event_id=f"feedback:42:{lane}",
                lane=lane,
            )
        self.assertTrue(model.learned_candidates())
        self.assertTrue(model.forget("feedback:42"))
        self.assertEqual(model.learned_candidates(), ())

    def test_version_three_directional_feedback_migration_removes_implicit_preferred_routine(
        self,
    ) -> None:
        model = SequencePreferenceModel()
        calm = _calm_sequence()
        raw_rave = _rave_sequence()
        rave = ChoreographySequence(
            sequence_id=raw_rave.sequence_id,
            steps=raw_rave.steps,
            source="operator_preferred_action",
        )
        context = MusicalContext(energy_label="groove", energy=0.5)
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                preferred=rave,
                feedback=(FeedbackSignal(label="pick_it_up"),),
            ),
            event_id="feedback:legacy:movers",
            lane="movers",
        )
        legacy = json.loads(json.dumps(model.state_dict()))
        legacy["version"] = 3
        restored = SequencePreferenceModel.from_state_dict(legacy)
        event = restored.state_dict()["events"]["feedback:legacy:movers"]
        self.assertIsNone(event["example"]["preferred"])
        self.assertEqual(restored.learned_candidates(), ())

    def test_preferred_complete_sequence_is_learned_and_round_trips(self) -> None:
        model = SequencePreferenceModel()
        calm = _calm_sequence()
        rave = _rave_sequence()
        context = MusicalContext(
            functional_label="transition",
            energy_label="build",
            content_label="instrumental",
            energy=0.7,
            motion=0.7,
            tension=0.8,
            bpm=126,
            song_key="spotify:track:test",
        )
        receipt = model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                preferred=rave,
                preferred_strength=1.0,
                feedback=(
                    FeedbackSignal(
                        label="pick_it_up",
                        urgency=0.8,
                        occurrences=3,
                    ),
                ),
            )
        )
        self.assertTrue(receipt.preferred_sequence_learned)
        self.assertEqual(receipt.output_action, "none")
        ranked = model.rank(context, (calm, rave))
        self.assertEqual(ranked[0].sequence.sequence_id, rave.sequence_id)
        self.assertGreater(ranked[0].score, ranked[1].score)

        restored = SequencePreferenceModel.from_state_dict(
            json.loads(json.dumps(model.state_dict()))
        )
        restored_ranked = restored.rank(context, (calm, rave))
        self.assertEqual(
            restored_ranked[0].sequence.sequence_id, rave.sequence_id
        )
        self.assertAlmostEqual(restored_ranked[0].score, ranked[0].score)
        self.assertEqual(
            restored.learned_candidates()[0].semantic_signature,
            rave.semantic_signature,
        )

    def test_interface_feedback_vocabulary_changes_sequence_metrics(
        self,
    ) -> None:
        context = MusicalContext(
            energy_label="groove",
            energy=0.6,
            motion=0.5,
            tension=0.4,
            bpm=118,
        )
        calm = _calm_sequence()
        rave = _rave_sequence()

        calmer = SequencePreferenceModel()
        calmer.learn(
            CapturedChoreographyExample(
                context=context,
                performed=rave,
                feedback=(
                    FeedbackSignal(
                        label="no_strobes",
                        occurrences=3,
                    ),
                    FeedbackSignal(
                        label="too_busy",
                        occurrences=3,
                    ),
                ),
            )
        )
        self.assertEqual(
            calmer.rank(context, (rave, calm))[0].sequence.sequence_id,
            calm.sequence_id,
        )

        livelier = SequencePreferenceModel()
        livelier.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                feedback=(
                    FeedbackSignal(
                        label="not_busy_enough",
                        occurrences=3,
                    ),
                    FeedbackSignal(
                        label="too_dim",
                        occurrences=3,
                    ),
                ),
            )
        )
        self.assertEqual(
            livelier.rank(context, (calm, rave))[0].sequence.sequence_id,
            rave.sequence_id,
        )

    def test_version_four_characteristic_candidates_are_removed_on_load(
        self,
    ) -> None:
        context = MusicalContext(energy_label="groove", energy=0.6)
        performed = _rave_sequence()
        characteristic = ChoreographySequence(
            sequence_id="preferred:no_strobes:movers",
            source="operator_preferred_characteristic",
            steps=performed.steps,
        )
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=performed,
                feedback=(FeedbackSignal(label="no_strobes"),),
                preferred=characteristic,
            ),
            event_id="feedback:legacy:movers",
            lane="movers",
        )
        state = model.state_dict()
        state["version"] = 4
        restored = SequencePreferenceModel.from_state_dict(state)
        self.assertEqual(restored.learned_candidates(), ())
        event = restored.state_dict()["events"]["feedback:legacy:movers"]
        self.assertIsNone(event["example"]["preferred"])

    def test_great_timing_reinforces_instead_of_penalizing_performed_sequence(
        self,
    ) -> None:
        model = SequencePreferenceModel()
        calm = _calm_sequence()
        rave = _rave_sequence()
        context = MusicalContext(energy_label="groove", energy=0.5)
        before = {
            row.sequence.sequence_id: row.score
            for row in model.rank(context, (calm, rave))
        }
        model.learn(CapturedChoreographyExample(
            context=context,
            performed=calm,
            feedback=(FeedbackSignal(label="great_timing"),),
        ))
        after = {
            row.sequence.sequence_id: row.score
            for row in model.rank(context, (calm, rave))
        }
        self.assertGreater(
            after[calm.sequence_id] - before[calm.sequence_id],
            after[rave.sequence_id] - before[rave.sequence_id],
        )

    def test_legacy_global_feedback_still_affects_lane_ranking(self) -> None:
        model = SequencePreferenceModel()
        calm = _calm_sequence()
        rave = _rave_sequence()
        context = MusicalContext(energy_label="groove", energy=0.6)
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                feedback=(FeedbackSignal(label="pick_it_up"),),
            ),
            event_id="feedback:legacy-global",
            lane=None,
        )
        state = json.loads(json.dumps(model.state_dict()))
        state["version"] = 3
        restored = SequencePreferenceModel.from_state_dict(state)
        ranked = restored.rank(context, (calm, rave), lane="movers")
        self.assertNotEqual(ranked[0].score, ranked[1].score)

    def test_identified_feedback_can_be_removed_and_replayed(self) -> None:
        context = MusicalContext(
            energy_label="release",
            energy=0.8,
            motion=0.7,
            tension=0.6,
        )
        calm = _calm_sequence()
        rave = _rave_sequence()
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                preferred=rave,
            ),
            event_id="feedback:10",
        )
        self.assertEqual(
            model.rank(context, (calm, rave))[0].sequence.sequence_id,
            rave.sequence_id,
        )
        self.assertTrue(model.forget("feedback:10"))
        self.assertFalse(model.forget("feedback:10"))
        self.assertEqual(model.learned_candidates(), ())
        ranked = model.rank(context, (calm, rave))
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)

    def test_incremental_forget_matches_model_built_from_retained_events(
        self,
    ) -> None:
        context = MusicalContext(
            energy_label="release", energy=0.8, motion=0.7
        )
        calm = _calm_sequence()
        rave = _rave_sequence()
        retained = CapturedChoreographyExample(
            context=context,
            performed=calm,
            feedback=(FeedbackSignal(label="increase_movement"),),
        )
        removed = CapturedChoreographyExample(
            context=context,
            performed=calm,
            preferred=rave,
            feedback=(FeedbackSignal(label="strobe"),),
        )
        model = SequencePreferenceModel()
        model.learn(
            retained, event_id="feedback:keep", now_unix_ms=1_000_000
        )
        model.learn(
            removed, event_id="feedback:remove", now_unix_ms=1_100_000
        )
        self.assertTrue(model.forget("feedback:remove"))

        expected = SequencePreferenceModel()
        expected.learn(
            retained, event_id="feedback:keep", now_unix_ms=1_000_000
        )
        actual_state = model.state_dict()
        expected_state = expected.state_dict()
        for field in ("weights", "evidence"):
            names = set(actual_state[field]) | set(expected_state[field])
            for name in names:
                self.assertAlmostEqual(
                    actual_state[field].get(name, 0.0),
                    expected_state[field].get(name, 0.0),
                )
        self.assertEqual(model.learned_candidates(), ())

    def test_incremental_consensus_revision_matches_direct_learning(
        self,
    ) -> None:
        context = MusicalContext(
            energy_label="groove", energy=0.6, motion=0.5
        )
        calm = _calm_sequence()
        initial = CapturedChoreographyExample(
            context=context,
            performed=calm,
            feedback=(FeedbackSignal(label="increase_movement"),),
        )
        model = SequencePreferenceModel()
        model.learn(
            initial, event_id="feedback:consensus", now_unix_ms=2_000_000
        )
        self.assertTrue(model.revise_feedback_event(
            "feedback:consensus", occurrences=6, urgency=0.95
        ))

        expected = SequencePreferenceModel()
        expected.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                feedback=(FeedbackSignal(
                    label="increase_movement",
                    occurrences=6,
                    urgency=0.95,
                ),),
            ),
            event_id="feedback:consensus",
            now_unix_ms=2_000_000,
        )
        actual_state = model.state_dict()
        expected_state = expected.state_dict()
        self.assertEqual(actual_state["weights"], expected_state["weights"])
        self.assertEqual(
            actual_state["evidence"], expected_state["evidence"]
        )

    def test_step_order_is_learned_as_a_sequence_not_a_routine_bag(
        self,
    ) -> None:
        first_then_second = ChoreographySequence(
            sequence_id="chase-then-sweep",
            steps=(
                ChoreographyStep(
                    0, 8, "movers", "opposing_chase", intensity=0.7
                ),
                ChoreographyStep(
                    8, 8, "movers", "fan_sweep", intensity=0.7
                ),
            ),
        )
        second_then_first = ChoreographySequence(
            sequence_id="sweep-then-chase",
            steps=(
                ChoreographyStep(
                    0, 8, "movers", "fan_sweep", intensity=0.7
                ),
                ChoreographyStep(
                    8, 8, "movers", "opposing_chase", intensity=0.7
                ),
            ),
        )
        context = MusicalContext(
            functional_label="transition",
            energy_label="build",
            content_label="instrumental",
            energy=0.7,
            motion=0.6,
            tension=0.8,
        )
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=second_then_first,
                preferred=first_then_second,
            )
        )
        ranked = model.rank(
            context, (second_then_first, first_then_second)
        )
        self.assertEqual(
            ranked[0].sequence.sequence_id, "chase-then-sweep"
        )

    def test_feedback_updates_next_boundary_without_interrupting_motion(
        self,
    ) -> None:
        model = SequencePreferenceModel()
        planner = BoundarySequencePlanner(model)
        calm = _calm_sequence()
        rave = _rave_sequence()
        context = MusicalContext(
            functional_label="chorus",
            energy_label="release",
            content_label="instrumental",
            energy=0.85,
            motion=0.8,
            tension=0.7,
            bpm=128,
        )
        first = planner.choose(
            boundary_id="phrase:1",
            context=context,
            candidates=(calm, rave),
        )
        self.assertEqual(first.sequence.sequence_id, calm.sequence_id)

        receipt = model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=calm,
                preferred=rave,
                feedback=(
                    FeedbackSignal(
                        label="pick_it_up",
                        urgency=1.0,
                        occurrences=6,
                    ),
                ),
            )
        )
        # Repeated taps raise urgency logarithmically; they are one correlated
        # consensus window rather than six independent examples.
        self.assertGreater(receipt.urgency, 0.95)
        held = planner.choose(
            boundary_id="phrase:1",
            context=context,
            candidates=(calm, rave),
        )
        self.assertEqual(held.sequence.sequence_id, calm.sequence_id)
        self.assertTrue(held.held_for_boundary)
        self.assertFalse(held.changed)

        next_phrase = planner.choose(
            boundary_id="phrase:2",
            context=context,
            candidates=(calm, rave),
        )
        self.assertEqual(next_phrase.sequence.sequence_id, rave.sequence_id)
        self.assertFalse(next_phrase.held_for_boundary)

    def test_dmx_history_and_corrective_feedback_change_sequence_metrics(
        self,
    ) -> None:
        model = SequencePreferenceModel()
        calm = _calm_sequence()
        rave = _rave_sequence()
        context = MusicalContext(
            functional_label="verse",
            energy_label="restrained",
            content_label="vocal_focus",
            energy=0.3,
            motion=0.25,
            tension=0.2,
            bpm=88,
        )
        history = tuple(
            DmxHistorySample(
                beat=float(index),
                fixture_scope="movers",
                dimmer=0.9,
                pan=0.0 if index % 2 == 0 else 1.0,
                tilt=1.0 if index % 2 == 0 else 0.0,
                strobe=0.8,
                color=(1.0, 0.0, 0.0)
                if index % 2 == 0
                else (0.0, 0.0, 1.0),
            )
            for index in range(8)
        )
        summary = summarize_dmx_history(history)
        self.assertEqual(summary.sample_count, 8)
        self.assertGreater(summary.movement, 0.9)
        self.assertGreater(summary.strobe, 0.7)
        self.assertGreater(summary.color_change, 0.7)

        receipt = model.learn(
            CapturedChoreographyExample(
                context=context,
                performed=rave,
                dmx_history=history,
                feedback=(
                    FeedbackSignal(
                        label="no_strobe",
                        urgency=0.9,
                        occurrences=4,
                    ),
                    FeedbackSignal(
                        label="too_busy",
                        urgency=0.9,
                        occurrences=4,
                    ),
                ),
            )
        )
        self.assertEqual(receipt.dmx_summary, summary)
        ranked = model.rank(
            context, (rave, calm), recent_dmx=history[-2:]
        )
        self.assertEqual(ranked[0].sequence.sequence_id, calm.sequence_id)

    def test_feedback_occurrences_raise_urgency_and_old_feedback_decays(
        self,
    ) -> None:
        context = MusicalContext()
        performed = _calm_sequence()
        now = 2_000_000_000_000
        single = SequencePreferenceModel(feedback_half_life_days=10)
        single_receipt = single.learn(
            CapturedChoreographyExample(
                context=context,
                performed=performed,
                feedback=(
                    FeedbackSignal(
                        label="good_motion",
                        occurrences=1,
                        created_unix_ms=now,
                    ),
                ),
            ),
            now_unix_ms=now,
        )
        repeated = SequencePreferenceModel(feedback_half_life_days=10)
        repeated_receipt = repeated.learn(
            CapturedChoreographyExample(
                context=context,
                performed=performed,
                feedback=(
                    FeedbackSignal(
                        label="good_motion",
                        occurrences=5,
                        created_unix_ms=now,
                    ),
                ),
            ),
            now_unix_ms=now,
        )
        old = SequencePreferenceModel(feedback_half_life_days=10)
        old_receipt = old.learn(
            CapturedChoreographyExample(
                context=context,
                performed=performed,
                feedback=(
                    FeedbackSignal(
                        label="good_motion",
                        occurrences=5,
                        created_unix_ms=now - 30 * 86_400_000,
                    ),
                ),
            ),
            now_unix_ms=now,
        )
        self.assertGreater(
            repeated_receipt.urgency, single_receipt.urgency
        )
        self.assertGreater(
            repeated_receipt.effective_strength,
            old_receipt.effective_strength,
        )

    def test_cue_lifetime_never_ranks_another_song_or_section(self) -> None:
        calm = _calm_sequence()
        rave = _rave_sequence()
        exact = MusicalContext(
            energy_label="build",
            energy=0.7,
            song_key="song-a",
            artist="artist-a",
            cue_key="song:song-a|section:build",
        )
        other_section = MusicalContext(
            energy_label="groove",
            energy=0.7,
            song_key="song-a",
            artist="artist-a",
            cue_key="song:song-a|section:groove",
        )
        other_song = MusicalContext(
            energy_label="build",
            energy=0.7,
            song_key="song-b",
            artist="artist-b",
            cue_key="song:song-b|section:build",
        )
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=exact,
                performed=calm,
                feedback=(FeedbackSignal(label="liked_this"),),
            ),
            event_id="cue-feedback:movers",
            lane="movers",
            lifetime="cue",
        )

        def advantage(context: MusicalContext) -> float:
            scores = {
                row.sequence.sequence_id: row.score
                for row in model.rank(context, (calm, rave), lane="movers")
            }
            return scores[calm.sequence_id] - scores[rave.sequence_id]

        self.assertGreater(advantage(exact), 0.0)
        self.assertAlmostEqual(advantage(other_section), 0.0)
        self.assertAlmostEqual(advantage(other_song), 0.0)

    def test_explicit_global_lifetime_crosses_song_and_section(self) -> None:
        calm = _calm_sequence()
        rave = _rave_sequence()
        source = MusicalContext(
            energy_label="build", energy=0.7, song_key="song-a",
            cue_key="song:song-a|section:build",
        )
        destination = MusicalContext(
            energy_label="groove", energy=0.7, song_key="song-b",
            cue_key="song:song-b|section:groove",
        )
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=source,
                performed=calm,
                feedback=(FeedbackSignal(label="liked_this"),),
            ),
            lane="movers",
            lifetime="global",
        )
        scores = {
            row.sequence.sequence_id: row.score
            for row in model.rank(
                destination, (calm, rave), lane="movers"
            )
        }
        self.assertGreater(scores[calm.sequence_id], scores[rave.sequence_id])

    def test_song_and_artist_lifetimes_use_only_their_namespaces(self) -> None:
        calm = _calm_sequence()
        rave = _rave_sequence()
        source = MusicalContext(
            energy_label="build", energy=0.7, song_key="song-a",
            artist="shared-artist",
            cue_key="song:song-a|section:build",
        )
        same_song = MusicalContext(
            energy_label="groove", energy=0.7, song_key="song-a",
            artist="shared-artist",
            cue_key="song:song-a|section:groove",
        )
        same_artist = MusicalContext(
            energy_label="groove", energy=0.7, song_key="song-b",
            artist="shared-artist",
            cue_key="song:song-b|section:groove",
        )
        unrelated = MusicalContext(
            energy_label="groove", energy=0.7, song_key="song-c",
            artist="other-artist",
            cue_key="song:song-c|section:groove",
        )

        def trained(lifetime: str) -> SequencePreferenceModel:
            model = SequencePreferenceModel()
            model.learn(
                CapturedChoreographyExample(
                    context=source,
                    performed=calm,
                    feedback=(FeedbackSignal(label="liked_this"),),
                ),
                lane="movers",
                lifetime=lifetime,
            )
            return model

        def advantage(
            model: SequencePreferenceModel, context: MusicalContext
        ) -> float:
            scores = {
                row.sequence.sequence_id: row.score
                for row in model.rank(
                    context, (calm, rave), lane="movers"
                )
            }
            return scores[calm.sequence_id] - scores[rave.sequence_id]

        song_model = trained("song")
        self.assertGreater(advantage(song_model, same_song), 0.0)
        self.assertAlmostEqual(advantage(song_model, same_artist), 0.0)
        artist_model = trained("artist")
        self.assertGreater(advantage(artist_model, same_song), 0.0)
        self.assertGreater(advantage(artist_model, same_artist), 0.0)
        self.assertAlmostEqual(advantage(artist_model, unrelated), 0.0)

    def test_cue_lifetime_round_trip_revision_and_delete_remain_local(
        self,
    ) -> None:
        calm = _calm_sequence()
        rave = _rave_sequence()
        exact = MusicalContext(
            energy_label="drop", energy=0.9, song_key="song-a",
            cue_key="song:song-a|timeline:teacher:45000",
        )
        other = MusicalContext(
            energy_label="drop", energy=0.9, song_key="song-a",
            cue_key="song:song-a|timeline:teacher:90000",
        )
        event_id = "listener-window:movers"
        model = SequencePreferenceModel()
        first = model.learn(
            CapturedChoreographyExample(
                context=exact,
                performed=calm,
                preferred=rave,
                feedback=(FeedbackSignal(
                    label="more_like_this", occurrences=1, urgency=0.45,
                ),),
            ),
            event_id=event_id,
            lane="movers",
            lifetime="cue",
        )
        state = json.loads(json.dumps(model.state_dict()))
        self.assertEqual(state["events"][event_id]["lifetime"], "cue")
        restored = SequencePreferenceModel.from_state_dict(state)
        self.assertEqual(
            [item.sequence_id for item in restored.learned_candidates(exact)],
            [rave.sequence_id],
        )
        self.assertEqual(restored.learned_candidates(other), ())

        def advantage(context: MusicalContext) -> float:
            scores = {
                row.sequence.sequence_id: row.score
                for row in restored.rank(
                    context, (calm, rave), lane="movers"
                )
            }
            return scores[rave.sequence_id] - scores[calm.sequence_id]

        initial = advantage(exact)
        self.assertGreater(initial, 0.0)
        self.assertAlmostEqual(advantage(other), 0.0)
        self.assertTrue(restored.revise_feedback_event(
            event_id, occurrences=6, urgency=0.95
        ))
        self.assertGreater(advantage(exact), initial)
        self.assertAlmostEqual(advantage(other), 0.0)
        self.assertTrue(restored.forget(event_id))
        self.assertAlmostEqual(advantage(exact), 0.0)
        self.assertEqual(restored.learned_candidates(exact), ())
        self.assertGreater(first.urgency, 0.0)

    def test_v6_model_loads_as_explicit_legacy_global(self) -> None:
        calm = _calm_sequence()
        rave = _rave_sequence()
        source = MusicalContext(song_key="old-song", energy=0.5)
        model = SequencePreferenceModel()
        model.learn(
            CapturedChoreographyExample(
                context=source,
                performed=calm,
                feedback=(FeedbackSignal(label="liked_this"),),
            ),
            event_id="legacy:movers",
            lane="movers",
        )
        state = json.loads(json.dumps(model.state_dict()))
        state["version"] = 6
        state.pop("learned_sequence_scopes", None)
        state["events"]["legacy:movers"].pop("lifetime", None)
        restored = SequencePreferenceModel.from_state_dict(state)
        destination = MusicalContext(song_key="new-song", energy=0.5)
        scores = {
            row.sequence.sequence_id: row.score
            for row in restored.rank(
                destination, (calm, rave), lane="movers"
            )
        }
        self.assertGreater(scores[calm.sequence_id], scores[rave.sequence_id])


if __name__ == "__main__":
    unittest.main()
