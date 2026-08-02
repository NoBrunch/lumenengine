from __future__ import annotations

import json
import unittest

from lumen_engine.choreography import (
    BoundarySequencePlanner,
    CapturedChoreographyExample,
    ChoreographySequence,
    ChoreographyStep,
    DmxHistorySample,
    FeedbackSignal,
    MusicalContext,
    SequencePreferenceModel,
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
        self.assertGreater(receipt.urgency, 0.99)
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


if __name__ == "__main__":
    unittest.main()
