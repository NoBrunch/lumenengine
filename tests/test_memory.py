from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from lumen_engine.memory import SongMemoryStore
from lumen_engine.models import Feedback, MediaIdentity


class MemoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
