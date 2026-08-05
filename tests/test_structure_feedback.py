from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.models import MediaIdentity
from lumen_engine.offline import _apply_operator_consensus_rows
from lumen_engine.structure_feedback import consensus_anchors, consensus_segments


def annotation(
    identifier: int,
    participant: str,
    label: str,
    position_ms: int,
    *,
    created_ms: int | None = None,
) -> dict:
    return {
        "id": identifier,
        "song_id": 1,
        "kind": "musical_context",
        "label": label,
        "position_ms": position_ms,
        "participant_id": participant,
        "intensity": 1.0,
        "created_unix_ms": identifier if created_ms is None else created_ms,
    }


class StructureFeedbackTests(unittest.TestCase):
    def test_consensus_collapses_repeats_and_counts_people(self) -> None:
        anchors = consensus_anchors([
            annotation(1, "alice", "drop", 10_000),
            annotation(2, "alice", "drop", 10_200),
            annotation(3, "bob", "drop", 10_400),
        ])
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["label"], "drop")
        self.assertEqual(anchors[0]["participant_count"], 2)
        self.assertEqual(anchors[0]["repeated_inputs_collapsed"], 1)
        self.assertTrue(anchors[0]["accepted"])
        self.assertGreater(anchors[0]["confidence"], 0.85)

    def test_conflicting_listener_tie_is_not_authoritative(self) -> None:
        anchors = consensus_anchors([
            annotation(1, "alice", "build", 10_000),
            annotation(2, "bob", "drop", 10_200),
        ])
        self.assertFalse(anchors[0]["accepted"])
        self.assertLess(anchors[0]["confidence"], 0.60)

    def test_state_correction_expires_at_next_teacher_boundary(self) -> None:
        anchors = consensus_anchors([
            annotation(1, "alice", "breakdown", 12_000),
        ])
        segments = consensus_segments(
            anchors,
            duration_ms=30_000,
            base_segments=[
                {"start_ms": 0, "end_ms": 20_000},
                {"start_ms": 20_000, "end_ms": 30_000},
            ],
        )
        self.assertEqual(segments[0]["start_ms"], 12_000)
        self.assertEqual(segments[0]["end_ms"], 20_000)
        self.assertEqual(segments[0]["energy_label"], "breakdown")
        self.assertEqual(segments[0]["boundary_confidence"], 0.0)

    def test_transition_event_teaches_boundary_and_following_state(self) -> None:
        anchors = consensus_anchors([
            annotation(1, "alice", "drop_onset", 12_000),
            annotation(2, "bob", "drop_onset", 12_200),
        ])
        segments = consensus_segments(
            anchors,
            duration_ms=30_000,
            base_segments=[
                {"start_ms": 0, "end_ms": 20_000},
                {"start_ms": 20_000, "end_ms": 30_000},
            ],
        )
        self.assertEqual(segments[0]["energy_label"], "drop")
        self.assertGreater(segments[0]["boundary_confidence"], 0.8)
        self.assertIn(
            "drop_onset", segments[0]["provenance"]["transition_events"]
        )

    def test_consensus_overlays_training_and_recalls_across_captures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SongMemoryStore(Path(temporary) / "memory.sqlite3")
            song_id = store.remember_media(MediaIdentity(
                provider="spotify",
                provider_item_id="track:consensus",
                title="Consensus",
                artists=("Lumen",),
                duration_ms=30_000,
            ))
            first = store.remember_recording_version(
                provider="spotify", provider_item_id="track:consensus",
                song_id=song_id, duration_ms=30_000,
                audio_fingerprint="capture-one",
            )
            second = store.remember_recording_version(
                provider="spotify", provider_item_id="track:consensus",
                song_id=song_id, duration_ms=30_100,
                audio_fingerprint="capture-two",
            )
            run_id = store.begin_teacher_run(
                teacher_name="EDMFormer", teacher_version="test",
                device="cpu",
                preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                recording_id=first,
            )
            store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.9,
                recording_id=first,
                song_id=song_id,
                teacher_run_id=run_id,
                segments=[
                    {"start_ms": 0, "end_ms": 20_000, "energy_label": "groove"},
                    {"start_ms": 20_000, "end_ms": 30_000, "energy_label": "drop"},
                ],
            )
            store.finish_teacher_run(run_id, status="complete")
            store.add_training_annotation(
                song_id=song_id, position_ms=12_000,
                kind="musical_context", label="breakdown",
                scope="overall", fixture_id=None, intensity=1.0,
                note=None, capture_session_id=None, audio_frame_index=None,
                context={}, participant_id="alice",
            )
            report = store.refresh_operator_structure_consensus(
                song_ids={song_id}
            )
            self.assertEqual(report["songs"], 1)

            recalled = store.cached_structure_at(
                recording_id=second, playback_position_ms=15_000
            )
            assert recalled is not None
            self.assertEqual(recalled["axes"]["energy"]["label"], "breakdown")
            self.assertEqual(
                recalled["axes"]["energy"]["recall_authority"],
                "operator_consensus",
            )
            self.assertIn(
                "operator",
                recalled["axes"]["energy"]["provenance"]["source"],
            )

            rows, overlay = _apply_operator_consensus_rows(store, [{
                "recording_id": second,
                "capture_session_id": "capture",
                "audio_frame_index": 1,
                "position_ms": 15_000,
                "functional": "unknown",
                "energy": "groove",
                "content": "unknown",
                "target_confidence": 0.9,
            }])
            self.assertEqual(rows[0]["energy"], "breakdown")
            self.assertEqual(overlay["rows_corrected"], 1)


if __name__ == "__main__":
    unittest.main()
