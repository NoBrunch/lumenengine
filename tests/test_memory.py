from __future__ import annotations

from contextlib import closing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
import tempfile
import unittest

from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.models import Feedback, MediaIdentity


class MemoryTests(unittest.TestCase):
    def test_structure_timeline_catalog_exposes_unplayed_review_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="spotify:track:catalog",
                duration_ms=180_000,
                metadata={
                    "track_identity": {
                        "title": "Catalog Track",
                        "artists": ["Example Artist"],
                        "album": "Local Review",
                    }
                },
            )
            run_id = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="test",
                device="cpu",
                preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                recording_id=recording_id,
            )
            timeline_id = store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.0,
                recording_id=recording_id,
                teacher_run_id=run_id,
                segments=[{
                    "start_ms": 0,
                    "end_ms": 180_000,
                    "energy_label": "drop",
                }],
            )
            store.finish_teacher_run(run_id, status="complete")

            catalog = store.structure_timeline_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["recording_id"], recording_id)
            self.assertEqual(catalog[0]["title"], "Catalog Track")
            self.assertEqual(catalog[0]["artists"], ["Example Artist"])
            self.assertEqual(catalog[0]["review_status"], "needs_review")
            self.assertFalse(catalog[0]["reviewed"])
            self.assertEqual(catalog[0]["teacher_sources"], ["EDMFormer"])
            self.assertTrue(catalog[0]["training_eligible"])

            store.review_structure_timeline(
                timeline_id=timeline_id,
                status="approved",
            )
            self.assertEqual(
                store.structure_timeline_catalog()[0]["review_status"],
                "approved",
            )
            store.review_structure_timeline(
                timeline_id=timeline_id,
                status="unreviewed",
            )
            self.assertEqual(
                store.structure_timeline_catalog()[0]["review_status"],
                "needs_review",
            )

    def test_live_store_defers_wal_checkpoints_until_explicit_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            with closing(store._connect()) as connection:
                automatic = connection.execute(
                    "PRAGMA wal_autocheckpoint"
                ).fetchone()[0]
            self.assertEqual(automatic, 0)
            result = store.checkpoint("PASSIVE")
            self.assertEqual(
                set(result), {"busy", "log_pages", "checkpointed_pages"}
            )

    def test_schema_five_memory_migrates_to_eight_with_listener_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO metadata VALUES('schema_version', '5');
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL,
                    position_ms INTEGER, label TEXT NOT NULL,
                    value REAL NOT NULL, note TEXT,
                    scope TEXT NOT NULL DEFAULT 'overall', fixture_id TEXT,
                    gesture TEXT, section TEXT, energy REAL, motion REAL,
                    tension REAL, confidence REAL, bpm REAL, routine TEXT,
                    capture_session_id TEXT, audio_frame_index INTEGER,
                    created_unix_ms INTEGER NOT NULL
                );
                CREATE TABLE choreography_sequences (
                    id TEXT PRIMARY KEY, song_id INTEGER, timeline_id TEXT,
                    source TEXT NOT NULL, confidence REAL NOT NULL,
                    context_json TEXT NOT NULL, created_unix_ms INTEGER NOT NULL
                );
                CREATE TABLE choreography_steps (
                    sequence_id TEXT NOT NULL, step_index INTEGER NOT NULL,
                    start_beat REAL NOT NULL, duration_beats REAL NOT NULL,
                    fixture_scope TEXT NOT NULL, routine TEXT NOT NULL,
                    intensity REAL NOT NULL, palette TEXT,
                    parameters_json TEXT NOT NULL,
                    PRIMARY KEY(sequence_id, step_index)
                );
                INSERT INTO choreography_sequences VALUES(
                    'legacy', NULL, NULL, 'operator', 1.0, '{}', 123
                );
                INSERT INTO choreography_steps VALUES(
                    'legacy', 0, 0, 4, 'movers', 'sweep', 1, NULL, '{}'
                );
                """
            )
            connection.commit()
            connection.close()

            store = SongMemoryStore(path)
            legacy = store.choreography_sequence("legacy")
            assert legacy is not None
            self.assertEqual(legacy["revision"], 1)
            self.assertEqual(legacy["updated_unix_ms"], 123)
            self.assertEqual(legacy["steps"][0]["strobe"], {})
            with closing(sqlite3.connect(path)) as migrated:
                version = migrated.execute(
                    "SELECT value FROM metadata WHERE key='schema_version'"
                ).fetchone()
                feedback_columns = {
                    row[1] for row in migrated.execute("PRAGMA table_info(feedback)")
                }
                annotation_columns = {
                    row[1] for row in migrated.execute(
                        "PRAGMA table_info(training_annotations)"
                    )
                }
            self.assertEqual(version, ("8",))
            self.assertTrue(
                {
                    "participant_id", "participant_name", "client_event_id",
                    "listening_session_id", "lane_context_json",
                }
                <= feedback_columns
            )
            self.assertTrue(
                {
                    "participant_id", "participant_name", "client_event_id",
                    "listening_session_id",
                }
                <= annotation_columns
            )

    def test_concurrent_listener_feedback_is_idempotent_per_client_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            song_id = store.remember_media(
                MediaIdentity(
                    provider="spotify",
                    provider_item_id="spotify:track:group-session",
                    title="Group Session",
                )
            )
            feedback = Feedback(
                song_id=song_id,
                position_ms=12_000,
                label="more_motion",
                value=1.0,
            )

            def submit(_: int) -> int:
                return store.add_feedback(
                    feedback,
                    participant_id="listener-a",
                    client_event_id="tap-42",
                    listening_session_id="session-friday",
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                event_ids = list(executor.map(submit, range(24)))
            self.assertEqual(len(set(event_ids)), 1)
            self.assertEqual(len(store.list_feedback(song_id)), 1)
            retry = store.add_feedback_event(
                feedback,
                participant_id="listener-a",
                participant_name="Alex",
                client_event_id="tap-42",
                listening_session_id="session-friday",
            )
            self.assertEqual(retry, {"id": event_ids[0], "created": False})
            created = store.add_feedback_event(
                feedback,
                participant_id="listener-a",
                participant_name="Alex",
                client_event_id="tap-43",
                listening_session_id="session-friday",
            )
            self.assertTrue(created["created"])
            second_listener = store.add_feedback(
                feedback,
                participant_id="listener-b",
                client_event_id="tap-42",
                listening_session_id="session-friday",
            )
            next_session = store.add_feedback(
                feedback,
                participant_id="listener-a",
                client_event_id="tap-42",
                listening_session_id="session-saturday",
            )
            self.assertNotEqual(second_listener, event_ids[0])
            self.assertNotEqual(next_session, event_ids[0])
            rows = store.training_feedback(feedback.capture_session_id or "missing")
            self.assertEqual(rows, [])
            self.assertEqual(store.summary()["totals"]["feedback_participants"], 2)
            self.assertEqual(store.summary()["totals"]["listening_sessions"], 2)

    def test_abandoned_analysis_job_is_requeued_without_touching_completed_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            job_id = store.enqueue_analysis_job(
                job_type="teacher.edmformer",
                payload={},
            )
            claimed = store.claim_analysis_job(
                worker_id="dead-worker", worker_pid=999_999_999
            )
            assert claimed is not None
            self.assertEqual(claimed["id"], job_id)
            abandoned_run = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="test",
                device="cpu",
                preprocessing_version="test",
                analysis_job_id=job_id,
            )
            completed_run = store.begin_teacher_run(
                teacher_name="SongFormer",
                teacher_version="test",
                device="cpu",
                preprocessing_version="test",
            )
            store.finish_teacher_run(completed_run, status="complete")

            recovered = store.recover_abandoned_analysis_jobs()

            self.assertEqual([item["job_id"] for item in recovered], [job_id])
            jobs = {item["id"]: item for item in store.list_analysis_jobs()}
            self.assertEqual(jobs[job_id]["status"], "queued")
            runs = {item["id"]: item for item in store.list_teacher_runs()}
            self.assertEqual(runs[abandoned_run]["status"], "failed")
            self.assertIn("previous offline worker", runs[abandoned_run]["error"])
            self.assertEqual(runs[completed_run]["status"], "complete")

    def test_live_analysis_job_lease_is_not_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            store.enqueue_analysis_job(job_type="teacher.songformer", payload={})
            claimed = store.claim_analysis_job(
                worker_id="live-worker", worker_pid=os.getpid()
            )
            assert claimed is not None
            self.assertTrue(
                store.heartbeat_analysis_job(
                    claimed["id"],
                    worker_id="live-worker",
                    progress={"rss_bytes": 1234},
                )
            )
            self.assertEqual(store.recover_abandoned_analysis_jobs(), [])
            job = store.list_analysis_jobs()[0]
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["result"]["rss_bytes"], 1234)

    def test_song_identity_analysis_feedback_and_routine_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            media = MediaIdentity(
                provider="spotify",
                provider_item_id="spotify:track:abc",
                title="Example",
                artists=("One", "Two"),
                album="Album",
                duration_ms=123_000,
                is_playing=True,
            )
            song_id = store.remember_media(media, count_play=True)
            same_id = store.remember_media(media, count_play=False)
            self.assertEqual(song_id, same_id)
            song = store.get_song(song_id)
            assert song is not None
            self.assertEqual(song["artists"], ("One", "Two"))
            self.assertEqual(song["play_count"], 1)

            store.save_analysis(song_id, 1, {"sections": [{"start": 0, "type": "intro"}]})
            analysis = store.latest_analysis(song_id)
            assert analysis is not None
            self.assertEqual(analysis["analysis_version"], 1)
            self.assertEqual(analysis["payload"]["sections"][0]["type"], "intro")

            store.add_feedback(
                Feedback(
                    song_id=song_id,
                    position_ms=31_000,
                    label="too_busy",
                    value=-1,
                    note="Preserve the vocal.",
                )
            )
            self.assertEqual(store.list_feedback(song_id)[0]["label"], "too_busy")

            store.save_routine(song_id, 2, {"strategy": "adaptive"})
            routine = store.get_routine(song_id)
            assert routine is not None
            self.assertEqual(routine["routine_version"], 2)
            self.assertEqual(routine["payload"]["strategy"], "adaptive")

            store.log_performance_sample(
                "session-1",
                {"observation": {"section": "groove"}, "decision": {"routine": "fan_sweep"}},
                song_id=song_id,
                position_ms=32_000,
            )
            samples = store.latest_performance_session()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["session_id"], "session-1")
            self.assertEqual(samples[0]["payload"]["decision"]["routine"], "fan_sweep")

    def test_research_provenance_timeline_jobs_and_choreography_round_trip(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            song_id = store.remember_media(
                MediaIdentity(
                    provider="spotify",
                    provider_item_id="spotify:track:structure",
                    title="Structure",
                    artists=("Lumen",),
                    duration_ms=180_000,
                )
            )
            store.begin_training_session(
                session_id="capture:one",
                mode="monitor",
                sample_rate=48_000,
                channels=2,
                sample_width=2,
                relative_path="audio/capture-one",
                metadata={"purpose": "test"},
            )
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="spotify:track:structure",
                song_id=song_id,
                duration_ms=180_000,
            )
            self.assertEqual(
                recording_id,
                store.remember_recording_version(
                    provider="spotify",
                    provider_item_id="spotify:track:structure",
                    song_id=song_id,
                    duration_ms=180_000,
                ),
            )
            store.add_capture_track_span(
                capture_session_id="capture:one",
                recording_id=recording_id,
                song_id=song_id,
                start_audio_frame=0,
                end_audio_frame=48_000,
                start_position_ms=0,
                end_position_ms=1_000,
                identity_source="spotify",
                identity_confidence=0.95,
            )
            inventory = store.capture_track_spans()
            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0]["recording_id"], recording_id)
            self.assertEqual(inventory[0]["recording_duration_ms"], 180_000)
            self.assertEqual(
                store.capture_spans_for_recording(recording_id), inventory
            )

            store.register_dataset_source(
                source_id="edm98",
                display_name="EDM-98",
                role="edm_structure",
                status="ready",
                version="1",
                revision="abc123",
                license_name="CC BY 4.0 / MIT metadata",
            )
            track_id = store.upsert_dataset_track(
                source_id="edm98",
                source_track_id="1060564312",
                title="Airwalk",
                split="train",
                metadata={"labels": 4},
            )
            self.assertGreater(track_id, 0)

            run_id = store.begin_teacher_run(
                teacher_name="edmformer",
                teacher_version="1",
                device="cpu",
                preprocessing_version="muq-musicfm-v1",
                recording_id=recording_id,
                capture_session_id="capture:one",
            )
            timeline_id = store.save_structure_timeline(
                provenance="edmformer",
                timeline_version="lumen-structure-v1",
                confidence=0.8,
                recording_id=recording_id,
                song_id=song_id,
                capture_session_id="capture:one",
                dataset_source_id="edm98",
                teacher_run_id=run_id,
                segments=[
                    {
                        "start_ms": 0,
                        "end_ms": 30_000,
                        "functional_label": "intro",
                        "energy_label": "restrained",
                        "content_label": "instrumental",
                        "raw_label": "intro",
                    },
                    {
                        "start_ms": 30_000,
                        "end_ms": 45_000,
                        "functional_label": "transition",
                        "energy_label": "build",
                        "content_label": "instrumental",
                        "raw_label": "buildup",
                    },
                ],
            )
            timeline = store.structure_timeline(timeline_id)
            assert timeline is not None
            self.assertEqual(timeline["segments"][1]["energy_label"], "build")
            store.finish_teacher_run(
                run_id, status="complete", metrics={"latency_s": 2.5}
            )

            job_id = store.enqueue_analysis_job(
                job_type="teacher_inference",
                payload={"recording_id": recording_id},
                priority=5,
            )
            store.update_analysis_job(
                job_id,
                status="complete",
                result={"timeline_id": timeline_id},
                increment_attempts=True,
            )
            jobs = store.list_analysis_jobs()
            self.assertEqual(jobs[0]["result"]["timeline_id"], timeline_id)

            sequence_id = store.save_choreography_sequence(
                song_id=song_id,
                timeline_id=timeline_id,
                source="operator_preference",
                confidence=0.9,
                context={"energy_label": "build"},
                steps=[
                    {
                        "start_beat": 0,
                        "duration_beats": 8,
                        "fixture_scope": "movers",
                        "routine": "opposing_chase",
                        "intensity": 0.7,
                    },
                    {
                        "start_beat": 8,
                        "duration_beats": 4,
                        "fixture_scope": "all",
                        "routine": "fan_sweep",
                        "intensity": 1.0,
                        "palette": "saturated_jewel",
                    },
                ],
            )
            sequence = store.choreography_sequence(sequence_id)
            assert sequence is not None
            self.assertEqual(sequence["steps"][0]["routine"], "opposing_chase")
            self.assertEqual(sequence["steps"][1]["duration_beats"], 4.0)
            summary = store.research_summary()
            self.assertEqual(summary["dataset_sources"], 1)
            self.assertEqual(summary["timelines"], 1)
            self.assertEqual(summary["choreography_sequences"], 1)

    def test_cached_structure_fuses_completed_teacher_axes_deterministically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            song_id = store.remember_media(
                MediaIdentity(
                    provider="spotify",
                    provider_item_id="spotify:track:fused",
                    title="Fused",
                    duration_ms=90_000,
                )
            )
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="spotify:track:fused",
                song_id=song_id,
                duration_ms=90_000,
            )
            edm_run = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="1",
                device="cpu",
                preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                recording_id=recording_id,
            )
            store.save_structure_timeline(
                provenance="edmformer",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.9,
                recording_id=recording_id,
                teacher_run_id=edm_run,
                segments=[
                    {"start_ms": 0, "end_ms": 30_000, "energy_label": "groove"},
                    {
                        "start_ms": 30_000,
                        "end_ms": 45_000,
                        "energy_label": "build",
                        "boundary_confidence": 0.94,
                    },
                    {"start_ms": 45_000, "end_ms": 90_000},
                ],
            )
            store.finish_teacher_run(edm_run, status="complete")
            song_run = store.begin_teacher_run(
                teacher_name="SongFormer",
                teacher_version="2",
                device="cpu",
                preprocessing_version=(
                    "songformer_official_features_cpu_windowed_v1:60s:"
                    f"{TEACHER_NORMALIZATION_VERSION}"
                ),
                recording_id=recording_id,
            )
            store.save_structure_timeline(
                provenance="songformer",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.8,
                recording_id=recording_id,
                teacher_run_id=song_run,
                segments=[
                    {
                        "start_ms": 0,
                        "end_ms": 30_000,
                        "functional_label": "verse",
                        "content_label": "vocal",
                    },
                    {
                        "start_ms": 30_000,
                        "end_ms": 60_000,
                        "functional_label": "chorus",
                        "content_label": "vocal",
                    },
                    {
                        "start_ms": 60_000,
                        "end_ms": 90_000,
                        "functional_label": "outro",
                        "energy_label": "drop",
                        "content_label": "instrumental",
                    },
                ],
            )
            store.finish_teacher_run(song_run, status="complete")
            incomplete = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="bad",
                device="cpu",
                preprocessing_version="bad",
                recording_id=recording_id,
            )
            store.save_structure_timeline(
                provenance="operator_correction",
                timeline_version="bad",
                confidence=1.0,
                recording_id=recording_id,
                teacher_run_id=incomplete,
                segments=[
                    {"start_ms": 0, "end_ms": 90_000, "energy_label": "wrong"}
                ],
            )

            context = store.cached_structure_at(
                provider="spotify",
                provider_item_id="fused",
                duration_ms=90_100,
                playback_position_ms=35_000,
            )
            assert context is not None
            self.assertEqual(context["recording"]["id"], recording_id)
            self.assertEqual(context["axes"]["energy"]["label"], "build")
            self.assertEqual(context["axes"]["functional"]["label"], "chorus")
            self.assertEqual(context["axes"]["content"]["label"], "vocal")
            self.assertEqual(context["boundary"]["next"]["time_ms"], 45_000)
            self.assertEqual(context["boundary"]["next"]["in_ms"], 10_000)
            self.assertEqual(context["boundary"]["current_confidence"], 0.0)
            self.assertEqual(context["beat_sync_authority"], "audio_sample_clock")
            self.assertEqual(len(context["provenance"]), 2)
            history = store.structure_timelines_for_recording(recording_id)
            songformer = next(
                item for item in history
                if item["teacher"]["name"] == "SongFormer"
            )
            self.assertTrue(songformer["review_eligible"])
            self.assertTrue(songformer["recall_eligible"])
            self.assertEqual(
                songformer["recall_authority"],
                "scored_teacher",
            )
            boundary = store.cached_structure_at(
                provider="spotify",
                provider_item_id="fused",
                duration_ms=90_100,
                playback_position_ms=30_000,
            )
            assert boundary is not None
            self.assertGreater(
                boundary["boundary"]["current_confidence"], 0.8
            )
            self.assertEqual(
                boundary["boundary"]["timeline_id"],
                boundary["axes"]["energy"]["timeline_id"],
            )
            self.assertEqual(
                boundary["boundary"]["provenance"],
                boundary["axes"]["energy"]["provenance"],
            )
            outro = store.cached_structure_at(
                provider="spotify",
                provider_item_id="fused",
                duration_ms=90_100,
                playback_position_ms=70_000,
            )
            assert outro is not None
            self.assertIsNone(outro["axes"]["energy"])
            self.assertEqual(
                outro["axes"]["functional"]["label"], "outro"
            )

    def test_obsolete_teacher_timeline_is_preserved_but_not_recalled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="obsolete",
                duration_ms=60_000,
            )
            run_id = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="old",
                device="cpu",
                preprocessing_version="edm98_official_pipeline_v1",
                recording_id=recording_id,
            )
            timeline_id = store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version="lumen_normalized_structure_v1",
                confidence=1.0,
                recording_id=recording_id,
                teacher_run_id=run_id,
                segments=[
                    {
                        "start_ms": 0,
                        "end_ms": 60_000,
                        "energy_label": "drop",
                        "label_confidence": 1.0,
                        "boundary_confidence": 1.0,
                    }
                ],
            )
            store.finish_teacher_run(run_id, status="complete")
            mislabeled_run = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="bad-import",
                device="cpu",
                preprocessing_version=(
                    "test:" + TEACHER_NORMALIZATION_VERSION
                ),
                recording_id=recording_id,
            )
            mislabeled_timeline = store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=1.0,
                recording_id=recording_id,
                teacher_run_id=mislabeled_run,
                segments=[{
                    "start_ms": 0,
                    "end_ms": 60_000,
                    "energy_label": "release",
                    "label_confidence": 1.0,
                    "boundary_confidence": 1.0,
                }],
            )
            store.finish_teacher_run(mislabeled_run, status="complete")
            invalid_legacy_correction = store.save_structure_timeline(
                provenance="operator_correction",
                timeline_version="lumen_operator_correction_v1",
                confidence=1.0,
                recording_id=recording_id,
                segments=[{
                    "start_ms": 0,
                    "end_ms": 60_000,
                    "energy_label": "release",
                    "label_confidence": 1.0,
                }],
            )

            self.assertIsNone(
                store.cached_structure_at(
                    recording_id=recording_id,
                    playback_position_ms=0,
                )
            )
            self.assertIsNotNone(store.structure_timeline(timeline_id))
            self.assertIsNotNone(
                store.structure_timeline(mislabeled_timeline)
            )
            self.assertIsNotNone(
                store.structure_timeline(invalid_legacy_correction)
            )

    def test_structure_correction_preserves_teacher_evidence_and_wins_recall(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="spotify:track:correct-me",
                duration_ms=60_000,
            )
            run_id = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="teacher-7",
                device="cpu",
                preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                recording_id=recording_id,
            )
            original_id = store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.73,
                recording_id=recording_id,
                teacher_run_id=run_id,
                segments=[
                    {
                        "start_ms": 0,
                        "end_ms": 30_000,
                        "energy_label": "groove",
                        "raw_label": "Drop",
                        "label_confidence": 0.73,
                        "provenance": {
                            "source": "edmformer",
                            "transition_event": "section_start",
                        },
                    },
                    {
                        "start_ms": 30_000,
                        "end_ms": 60_000,
                        "energy_label": "drop",
                        "raw_label": "Breakdown",
                        "label_confidence": 0.68,
                    },
                ],
            )
            store.finish_teacher_run(run_id, status="complete")

            with self.assertRaisesRegex(
                ValueError, "unknown canonical energy label"
            ):
                store.save_structure_correction(
                    base_timeline_id=original_id,
                    participant_id="operator-1",
                    segments=[
                        {
                            "segment_index": 0,
                            "start_ms": 0,
                            "end_ms": 30_000,
                            "energy_label": "groove",
                        },
                        {
                            "segment_index": 1,
                            "start_ms": 30_000,
                            "end_ms": 60_000,
                            "energy_label": "release",
                        },
                    ],
                )

            corrected_id = store.save_structure_correction(
                base_timeline_id=original_id,
                participant_id="operator-1",
                note="Second half is the quiet breakdown.",
                segments=[
                    {
                        "segment_index": 0,
                        "start_ms": 0,
                        "end_ms": 30_000,
                        "energy_label": "groove",
                    },
                    {
                        "segment_index": 1,
                        "start_ms": 30_000,
                        "end_ms": 60_000,
                        "energy_label": "breakdown",
                        "event": "breakdown_onset",
                    },
                ],
            )

            original = store.structure_timeline(original_id)
            corrected = store.structure_timeline(corrected_id)
            assert original is not None and corrected is not None
            self.assertEqual(original["segments"][1]["energy_label"], "drop")
            self.assertEqual(original["segments"][1]["raw_label"], "Breakdown")
            self.assertEqual(corrected["segments"][1]["energy_label"], "breakdown")
            self.assertEqual(corrected["segments"][1]["raw_label"], "Breakdown")
            self.assertEqual(
                corrected["segments"][0]["provenance"]["transition_event"],
                "section_start",
            )
            self.assertEqual(
                corrected["segments"][1]["provenance"]["transition_event"],
                "breakdown_onset",
            )
            self.assertEqual(
                corrected["metadata"]["corrects_timeline_id"], original_id
            )

            history = store.structure_timelines_for_recording(recording_id)
            self.assertEqual({item["id"] for item in history}, {
                original_id, corrected_id,
            })
            teacher = next(item for item in history if item["id"] == original_id)
            self.assertEqual(teacher["teacher"]["name"], "EDMFormer")
            self.assertEqual(teacher["teacher"]["version"], "teacher-7")
            self.assertTrue(teacher["recall_eligible"])

            recalled = store.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=45_000,
            )
            assert recalled is not None
            self.assertEqual(recalled["axes"]["energy"]["label"], "breakdown")
            self.assertEqual(
                recalled["axes"]["energy"]["timeline_id"], corrected_id
            )

    def test_operator_approval_adds_trust_without_changing_model_confidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="unscored-edm",
                duration_ms=40_000,
            )
            run_id = store.begin_teacher_run(
                teacher_name="EDMFormer",
                teacher_version="unscored",
                device="cpu",
                preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
                recording_id=recording_id,
            )
            timeline_id = store.save_structure_timeline(
                provenance="edmformer_teacher",
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=0.0,
                recording_id=recording_id,
                teacher_run_id=run_id,
                segments=[{
                    "start_ms": 0,
                    "end_ms": 40_000,
                    "energy_label": "build",
                    "raw_label": "Build-up",
                    "label_confidence": 0.0,
                    "boundary_confidence": 0.0,
                }],
            )
            store.finish_teacher_run(run_id, status="complete")

            unreviewed = store.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=10_000,
            )
            assert unreviewed is not None
            self.assertEqual(unreviewed["axes"]["energy"]["confidence"], 0.0)
            self.assertEqual(unreviewed["axes"]["energy"]["operator_trust"], 0.0)
            unreviewed_history = store.structure_timelines_for_recording(
                recording_id
            )
            self.assertTrue(unreviewed_history[0]["review_eligible"])
            self.assertFalse(unreviewed_history[0]["recall_eligible"])
            store.review_structure_timeline(
                timeline_id=timeline_id,
                status="approved",
                participant_id="owner",
            )
            approved = store.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=10_000,
            )
            assert approved is not None
            self.assertEqual(approved["axes"]["energy"]["confidence"], 0.0)
            self.assertEqual(approved["axes"]["energy"]["model_confidence"], 0.0)
            self.assertEqual(approved["axes"]["energy"]["operator_trust"], 1.0)
            self.assertEqual(
                approved["axes"]["energy"]["recall_authority"],
                "operator_approved",
            )
            history = store.structure_timelines_for_recording(recording_id)
            self.assertEqual(history[0]["confidence"], 0.0)
            self.assertEqual(history[0]["recall_authority"], "operator_approved")
            self.assertTrue(history[0]["recall_eligible"])

            store.review_structure_timeline(
                timeline_id=timeline_id,
                status="rejected",
                participant_id="owner",
            )
            self.assertIsNone(store.cached_structure_at(
                recording_id=recording_id,
                playback_position_ms=10_000,
            ))

    def test_choreography_revision_placement_delete_and_undo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            song_id = store.remember_media(
                MediaIdentity(
                    provider="spotify",
                    provider_item_id="spotify:track:choreo",
                    title="Choreo",
                )
            )
            sequence_id = store.save_choreography_sequence(
                song_id=song_id,
                source="operator",
                confidence=1.0,
                context={"energy": "build"},
                name="Mover build",
                fixture_scope="movers",
                participant_id="listener-a",
                participant_name="Alex",
                client_event_id="proposal-1",
                steps=[
                    {
                        "start_beat": 0,
                        "duration_beats": 8,
                        "fixture_scope": "movers",
                        "routine": "fan_sweep",
                        "intensity": 0.75,
                        "palette": "jewel",
                        "strobe": {"enabled": False},
                        "entry_behavior": "phrase_boundary",
                        "exit_behavior": "resolve",
                    }
                ],
            )
            duplicate = store.save_choreography_sequence(
                source="operator",
                confidence=1,
                context={},
                participant_id="listener-a",
                client_event_id="proposal-1",
                steps=[
                    {
                        "start_beat": 0,
                        "duration_beats": 4,
                        "fixture_scope": "center",
                        "routine": "arm_chase",
                    }
                ],
            )
            self.assertEqual(duplicate, sequence_id)
            first = store.choreography_sequence(sequence_id)
            assert first is not None
            self.assertEqual(first["revision"], 1)
            self.assertEqual(first["steps"][0]["strobe"], {"enabled": False})
            store.save_choreography_sequence(
                sequence_id=sequence_id,
                song_id=song_id,
                source="operator",
                confidence=1,
                context={"energy": "drop"},
                fixture_scope="movers",
                steps=[
                    {
                        "start_beat": 0,
                        "duration_beats": 4,
                        "fixture_scope": "movers",
                        "routine": "opposing_chase",
                        "strobe": {"enabled": True, "rate": 0.3},
                    }
                ],
            )
            revised = store.choreography_sequence(sequence_id)
            assert revised is not None
            self.assertEqual(revised["revision"], 2)
            self.assertEqual(revised["steps"][0]["routine"], "opposing_chase")
            self.assertTrue(store.undo_choreography_sequence(sequence_id))
            restored = store.choreography_sequence(sequence_id)
            assert restored is not None
            self.assertEqual(restored["revision"], 1)
            self.assertEqual(restored["steps"][0]["routine"], "fan_sweep")
            self.assertTrue(store.delete_choreography_sequence(sequence_id))
            self.assertIsNone(store.choreography_sequence(sequence_id))
            self.assertTrue(store.undo_choreography_sequence(sequence_id))
            self.assertIsNotNone(store.choreography_sequence(sequence_id))

            placement_id = store.save_choreography_placement(
                sequence_id=sequence_id,
                song_id=song_id,
                fixture_scope="movers",
                source="listener_proposal",
                context={"why": "more motion here"},
                start_ms=30_000,
                end_ms=45_000,
                start_beat=64,
                duration_beats=16,
                section_label="build",
                participant_id="listener-b",
                participant_name="Blair",
                client_event_id="placement-9",
            )
            duplicate_placement = store.save_choreography_placement(
                sequence_id=sequence_id,
                fixture_scope="movers",
                source="listener_proposal",
                context={},
                start_ms=1,
                participant_id="listener-b",
                client_event_id="placement-9",
            )
            self.assertEqual(duplicate_placement, placement_id)
            with ThreadPoolExecutor(max_workers=6) as executor:
                concurrent_ids = list(
                    executor.map(
                        lambda _: store.save_choreography_placement(
                            sequence_id=sequence_id,
                            fixture_scope="movers",
                            source="listener_proposal",
                            context={},
                            start_ms=1,
                            participant_id="listener-b",
                            client_event_id="placement-9",
                        ),
                        range(18),
                    )
                )
            self.assertEqual(set(concurrent_ids), {placement_id})
            self.assertEqual(len(store.list_choreography_placements()), 1)
            placement = store.choreography_placement(placement_id)
            assert placement is not None
            self.assertEqual(placement["participant_name"], "Blair")
            self.assertEqual(placement["revision"], 1)
            store.save_choreography_placement(
                placement_id=placement_id,
                sequence_id=sequence_id,
                song_id=song_id,
                fixture_scope="movers",
                source="operator_edit",
                context={},
                start_ms=32_000,
                end_ms=48_000,
            )
            self.assertEqual(
                store.choreography_placement(placement_id)["revision"], 2
            )
            self.assertTrue(store.undo_choreography_placement(placement_id))
            self.assertEqual(
                store.choreography_placement(placement_id)["start_ms"], 30_000
            )
            self.assertTrue(store.delete_choreography_placement(placement_id))
            self.assertIsNone(store.choreography_placement(placement_id))
            self.assertTrue(store.undo_choreography_placement(placement_id))
            self.assertEqual(
                len(store.list_choreography_placements(song_id=song_id)), 1
            )


if __name__ == "__main__":
    unittest.main()
