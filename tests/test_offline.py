import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import wave

from lumen_engine.memory import SongMemoryStore
from lumen_engine.offline import (
    EDMFORMER_JOB,
    MIN_TEACHER_DURATION_MS,
    SONGFORMER_JOB,
    STUDENT_TRAIN_JOB,
    OfflineJobCancelled,
    OfflineMemoryLimitExceeded,
    OfflineResearchWorker,
    PreparedRecording,
    ResearchJobCoordinator,
    _normalize_teacher_segments,
    _merge_teacher_example_rows,
    _validate_teacher_coverage,
    _capture_split_group,
    _recording_split,
    build_student_examples,
    enqueue_student_training,
    training_readiness,
    trusted_student_examples,
)
from lumen_engine.student import FEATURE_NAMES


def _complete_supervision() -> dict:
    return {
        "eligible": True,
        "classification": "complete",
        "reason_codes": [],
        "evidence": {},
    }


class OfflineResearchTests(unittest.TestCase):
    def test_edmformer_worker_uses_bounded_runner_and_saves_output(self):
        class DeterministicEDMFormerWorker(OfflineResearchWorker):
            def _edmformer_paths(self):
                return {
                    "python": Path("/test/edmformer/python"),
                    "runner": Path("/test/edmformer-cpu-runner.py"),
                    "checkpoint": Path("/test/model.pt"),
                    "config": Path("/test/edmformer.yaml"),
                    "musicfm_stat": Path("/test/msd_stats.json"),
                    "musicfm_model": Path("/test/pretrained_msd.pt"),
                    "musicfm_source": Path("/test/musicfm"),
                    "hf_cache": Path("/test/hf-cache"),
                    "revision": "edmformer-test-revision",
                }

            def _run_subprocess(self, command, environment):
                del environment
                self.assert_runner_contract(command)
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        [
                            {"label": "intro", "start": 0.0, "end": 4.0},
                            {"label": "buildup", "start": 4.0, "end": 8.0},
                            {"label": "drop", "start": 8.0, "end": 12.0},
                        ]
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            @staticmethod
            def assert_runner_contract(command):
                assert command[1] == "/test/edmformer-cpu-runner.py"
                assert "-m" not in command
                assert command[command.index("--window-seconds") + 1] == "45"
                assert command[command.index("--threads") + 1] == "4"
                assert command[command.index("--musicfm-source") + 1] == "/test/musicfm"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "recording.wav"
            audio.write_bytes(b"deterministic-test-audio")
            store = SongMemoryStore(root / "lumen.sqlite3")
            recording_id = store.remember_recording_version(
                provider="test",
                provider_item_id="edmformer-track",
                duration_ms=12_000,
            )
            store.begin_training_session(
                session_id="edmformer-capture",
                mode="monitor",
                sample_rate=100,
                channels=1,
                sample_width=2,
                relative_path="audio/edmformer-capture",
                metadata={},
            )
            store.add_training_frames(
                [
                    {
                        "session_id": "edmformer-capture",
                        "audio_frame_index": 50,
                        "segment_index": 0,
                        "segment_frame_index": 50,
                        "created_unix_ms": 1,
                        "song_id": None,
                        "position_ms": 500,
                        "payload": {"observation": {}},
                    }
                ]
            )
            store.add_capture_track_span(
                capture_session_id="edmformer-capture",
                start_audio_frame=0,
                end_audio_frame=1_200,
                recording_id=recording_id,
                start_position_ms=0,
                end_position_ms=12_000,
                identity_source="test",
                identity_confidence=1.0,
                metadata={
                    "split_group_id": "test:edmformer-track",
                    "structure_supervision": _complete_supervision(),
                },
            )
            store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={
                    "recording_id": recording_id,
                    "audio_path": str(audio),
                    "content_sha256": "a" * 64,
                    "duration_ms": 12_000,
                    "song_id": None,
                    "capture_session_id": "edmformer-capture",
                    "structure_supervision": _complete_supervision(),
                    "edmformer_window_seconds": 45,
                },
            )
            result = DeterministicEDMFormerWorker(
                store, research_root=root / "research"
            ).run_once((EDMFORMER_JOB,))
            assert result is not None
            self.assertEqual(result["status"], "complete")
            timeline = store.structure_timeline(result["result"]["timeline_id"])
            assert timeline is not None
            self.assertEqual(
                [row["energy_label"] for row in timeline["segments"]],
                ["unknown", "build", "release"],
            )
            self.assertEqual(timeline["metadata"]["bounded_window_seconds"], 45)
            self.assertEqual(
                timeline["metadata"]["runner"],
                "lumen_edmformer_cpu_runner_v1",
            )

    def test_readiness_reports_capture_eligibility_without_ui_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            complete = _complete_supervision()
            partial = {
                "eligible": False,
                "classification": "partial",
                "reason_codes": ["capture_ended_before_track_end"],
                "evidence": {},
            }
            unknown = {
                "eligible": False,
                "classification": "unknown",
                "reason_codes": ["track_duration_unknown"],
                "evidence": {},
            }
            for job_type in (EDMFORMER_JOB, SONGFORMER_JOB):
                complete_job = store.enqueue_analysis_job(
                    job_type=job_type,
                    payload={
                        "recording_id": "complete-song",
                        "structure_supervision": complete,
                    },
                )
                store.update_analysis_job(complete_job, status="complete")
                store.enqueue_analysis_job(
                    job_type=job_type,
                    payload={
                        "recording_id": "partial-song",
                        "structure_supervision": partial,
                    },
                )
                store.enqueue_analysis_job(
                    job_type=job_type,
                    payload={
                        "recording_id": "unknown-song",
                        "structure_supervision": unknown,
                    },
                )

            readiness = training_readiness(
                store, research_root=root / "research"
            )
            self.assertEqual(readiness["recordings_captured"], 3)
            self.assertEqual(readiness["recordings_eligible"], 1)
            self.assertEqual(readiness["recordings_partial"], 1)
            self.assertEqual(readiness["recordings_unknown"], 1)
            self.assertEqual(readiness["recordings_processed"], 1)
            self.assertEqual(readiness["recordings_planned"], 1)

    def test_export_preparation_reconstructs_audio_and_deduplicates_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            audio_dir = training / "audio" / "capture"
            export = training / "exports" / "dataset-test"
            audio_dir.mkdir(parents=True)
            export.mkdir(parents=True)
            pcm = (b"\x01\x00\x02\x00") * 96_000
            with wave.open(str(audio_dir / "segment.wav"), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8_000)
                output.writeframes(pcm)
            store = SongMemoryStore(root / "lumen.sqlite3")
            store.begin_training_session(
                session_id="capture-1",
                mode="monitor",
                sample_rate=8_000,
                channels=2,
                sample_width=2,
                relative_path="audio/capture",
                metadata={},
            )
            store.finish_training_session(
                session_id="capture-1",
                status="complete",
                frames_received=96_000,
                frames_written=96_000,
                dropped_packets=0,
                dropped_frames=0,
                segment_count=1,
                bytes_written=len(pcm),
            )
            (export / "dataset.json").write_text(
                json.dumps(
                    {
                        "format": "lumen_training_dataset",
                        "version": 2,
                        "validation": {"valid": True, "errors": []},
                    }
                ),
                encoding="utf-8",
            )
            fingerprint = "a" * 64
            recording = {
                "recording_id": "capture-recording",
                "session_id": "capture-1",
                "song_id": None,
                "track_identity": {
                    "provider": "spotify",
                    "provider_item_id": "export-track",
                    "duration_ms": 12_000,
                },
                "split_group_id": "spotify:export-track",
                "split": _recording_split("spotify:export-track"),
                "start_frame": 0,
                "end_frame": 96_000,
                "frame_count": 96_000,
                "sample_rate": 8_000,
                "channels": 2,
                "sample_width": 2,
                "first_position_ms": 0,
                "last_position_ms": 12_000,
                "structure_supervision": _complete_supervision(),
                "content_fingerprint": f"sha256:{fingerprint}",
                "audio_clips": [
                    {
                        "relative_path": "audio/capture/segment.wav",
                        "start_frame": 0,
                        "frame_count": 96_000,
                    }
                ],
            }
            (export / "recordings.jsonl").write_text(
                json.dumps(recording) + "\n", encoding="utf-8"
            )
            coordinator = ResearchJobCoordinator(
                store,
                training_root=training,
                research_root=training / "research",
            )
            first = coordinator.prepare_export(export)
            second = coordinator.prepare_export(export)
            self.assertEqual(first["recordings"], 1)
            self.assertEqual(first["jobs_queued"], 1)
            self.assertEqual(second["jobs_queued"], 0)
            prepared = Path(first["prepared"][0]["audio_path"])
            with wave.open(str(prepared), "rb") as reconstructed:
                self.assertEqual(reconstructed.getnframes(), 96_000)
                self.assertEqual(reconstructed.readframes(96_000), pcm)
            claimed = store.claim_analysis_job((EDMFORMER_JOB,))
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["attempts"], 1)
            self.assertEqual(
                claimed["payload"]["split_group_id"],
                "spotify:export-track",
            )
            self.assertEqual(
                claimed["payload"]["split"],
                _recording_split("spotify:export-track"),
            )
            self.assertIsNone(store.claim_analysis_job((EDMFORMER_JOB,)))
            bad_pcm = (b"\x03\x00\x04\x00") * 96_000
            with wave.open(str(prepared), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8_000)
                output.writeframes(bad_pcm)
            with self.assertRaisesRegex(
                ValueError, "does not match its verified source clips"
            ):
                coordinator.prepare_export(export)

    def test_partial_legacy_capture_is_skipped_and_completed_run_untrusted(
        self,
    ) -> None:
        class NeverLoadTeacher(OfflineResearchWorker):
            def _run_edmformer(self, job):
                del job
                raise AssertionError("partial capture must not load teacher")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research = root / "research"
            examples_root = research / "exports" / "student-examples"
            examples_root.mkdir(parents=True)
            store = SongMemoryStore(root / "lumen.sqlite3")
            store.begin_training_session(
                session_id="partial-capture",
                mode="monitor",
                sample_rate=100,
                channels=1,
                sample_width=2,
                relative_path="audio/partial-capture",
                metadata={},
            )
            store.finish_training_session(
                session_id="partial-capture",
                status="complete",
                frames_received=6_000,
                frames_written=6_000,
                dropped_packets=0,
                dropped_frames=0,
                segment_count=1,
                bytes_written=12_000,
            )
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="partial-song",
                duration_ms=60_000,
                audio_fingerprint="pcm-sha256:" + "a" * 64,
                metadata={
                    "track_identity": {
                        "provider": "spotify",
                        "provider_item_id": "partial-song",
                        "duration_ms": 200_000,
                    }
                },
            )
            store.add_capture_track_span(
                capture_session_id="partial-capture",
                start_audio_frame=0,
                end_audio_frame=6_000,
                recording_id=recording_id,
                start_position_ms=140_000,
                end_position_ms=200_000,
                identity_source="spotify_metadata",
                identity_confidence=0.99,
                metadata={"split_group_id": "spotify:partial-song"},
            )
            store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={
                    "recording_id": recording_id,
                    "capture_session_id": "partial-capture",
                    "duration_ms": 60_000,
                },
            )
            skipped = NeverLoadTeacher(
                store, research_root=research
            ).run_once((EDMFORMER_JOB,))
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(
                skipped["result"]["reason"],
                "recording_incomplete_for_structure_supervision",
            )

            run_id = store.begin_teacher_run(
                teacher_name="legacy",
                teacher_version="v1",
                device="cpu",
                preprocessing_version="legacy",
                recording_id=recording_id,
                capture_session_id="partial-capture",
            )
            path = examples_root / "partial.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "teacher_run_id": run_id,
                        "recording_id": recording_id,
                        "capture_session_id": "partial-capture",
                        "split_group_id": "spotify:partial-song",
                        "split": _recording_split("spotify:partial-song"),
                        "features": [0.2] * len(FEATURE_NAMES),
                        "functional": "intro",
                        "energy": "low",
                        "content": "instrumental",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store.finish_teacher_run(
                run_id,
                status="complete",
                metrics={
                    "student_examples": {
                        "path": str(path),
                        "examples": 1,
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                },
            )
            trusted = trusted_student_examples(
                store, research_root=research
            )
            self.assertEqual(trusted["examples"], 0)
            self.assertEqual(trusted["excluded_examples"], 1)
            self.assertEqual(len(trusted["excluded_teacher_runs"]), 1)

    def test_short_recording_is_prepared_but_teacher_is_explicitly_skipped(self):
        class NeverLoadTeacher(OfflineResearchWorker):
            def _run_edmformer(self, job):
                del job
                raise AssertionError("short recording must not load EDMFormer")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={
                    "recording_id": "identity-fragment",
                    "duration_ms": 405,
                },
            )
            result = NeverLoadTeacher(
                store, research_root=root / "research"
            ).run_once((EDMFORMER_JOB,))
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["result"]["reason"], "recording_too_short")
            self.assertEqual(
                result["result"]["minimum_duration_ms"],
                MIN_TEACHER_DURATION_MS,
            )
            job = store.list_analysis_jobs()[0]
            self.assertEqual(job["status"], "complete")
            self.assertEqual(job["result"]["reason"], "recording_too_short")

    def test_coordinator_does_not_queue_short_identity_fragment(self):
        class ShortRecordingCoordinator(ResearchJobCoordinator):
            def _prepare_recording(self, recording):
                del recording
                return PreparedRecording(
                    recording_id="identity-fragment",
                    audio_path=self.audio_root / "fragment.wav",
                    content_sha256="a" * 64,
                    duration_ms=405,
                    song_id=None,
                    capture_session_id="capture-1",
                    split_group_id="unidentified-session:capture-1",
                    split=_recording_split(
                        "unidentified-session:capture-1"
                    ),
                    structure_supervision={
                        "eligible": False,
                        "classification": "unknown",
                        "reason_codes": ["track_duration_unknown"],
                        "evidence": {},
                    },
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            export.mkdir()
            (export / "dataset.json").write_text(
                json.dumps(
                    {
                        "format": "lumen_training_dataset",
                        "validation": {"valid": True},
                    }
                ),
                encoding="utf-8",
            )
            (export / "recordings.jsonl").write_text("{}\n", encoding="utf-8")
            store = SongMemoryStore(root / "lumen.sqlite3")
            result = ShortRecordingCoordinator(
                store,
                training_root=root / "training",
                research_root=root / "research",
            ).prepare_export(
                export, queue_edmformer=True, queue_songformer=True
            )
            self.assertEqual(result["recordings"], 1)
            self.assertEqual(result["jobs_queued"], 0)
            self.assertEqual(len(result["teachers_skipped"]), 2)
            self.assertEqual(
                {item["job_type"] for item in result["teachers_skipped"]},
                {EDMFORMER_JOB, SONGFORMER_JOB},
            )
            self.assertTrue(
                all(
                    item["reason"] == "recording_too_short"
                    for item in result["teachers_skipped"]
                )
            )
            self.assertEqual(store.list_analysis_jobs(), [])

    def test_coordinator_retains_partial_recording_without_teacher_job(self):
        class PartialRecordingCoordinator(ResearchJobCoordinator):
            def _prepare_recording(self, recording):
                del recording
                return PreparedRecording(
                    recording_id="partial-recording",
                    audio_path=self.audio_root / "partial.wav",
                    content_sha256="a" * 64,
                    duration_ms=60_000,
                    song_id=None,
                    capture_session_id="capture-1",
                    split_group_id="spotify:partial-song",
                    split=_recording_split("spotify:partial-song"),
                    structure_supervision={
                        "eligible": False,
                        "classification": "partial",
                        "reason_codes": [
                            "capture_started_after_track_beginning"
                        ],
                        "evidence": {},
                    },
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export"
            export.mkdir()
            (export / "dataset.json").write_text(
                json.dumps(
                    {
                        "format": "lumen_training_dataset",
                        "validation": {"valid": True},
                    }
                ),
                encoding="utf-8",
            )
            (export / "recordings.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            store = SongMemoryStore(root / "lumen.sqlite3")
            result = PartialRecordingCoordinator(
                store,
                training_root=root / "training",
                research_root=root / "research",
            ).prepare_export(
                export, queue_edmformer=True, queue_songformer=True
            )
            self.assertEqual(result["recordings"], 1)
            self.assertEqual(result["jobs_queued"], 0)
            self.assertEqual(len(result["teachers_skipped"]), 2)
            self.assertTrue(
                all(
                    item["reason"]
                    == "recording_incomplete_for_structure_supervision"
                    for item in result["teachers_skipped"]
                )
            )
            self.assertEqual(store.list_analysis_jobs(), [])

    def test_teacher_labels_are_normalized_on_independent_axes(self):
        rows = _normalize_teacher_segments(
            [
                {"start": 0.0, "end": 8.0, "label": "Intro"},
                {"start": 8.0, "end": 16.0, "label": "Buildup"},
                {"start": 16.0, "end": 24.0, "label": "Drop"},
            ],
            source="test-teacher",
            source_version="v1",
        )
        self.assertEqual(rows[0]["functional_label"], "intro")
        self.assertEqual(rows[1]["energy_label"], "build")
        self.assertEqual(rows[2]["energy_label"], "release")
        self.assertEqual(
            rows[2]["provenance"]["annotation_type"],
            "teacher_prediction",
        )

    def test_teacher_segments_reject_gaps_and_incomplete_coverage(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            _normalize_teacher_segments(
                [
                    {"start": 0.0, "end": 2.0, "label": "intro"},
                    {"start": 2.2, "end": 4.0, "label": "drop"},
                ],
                source="test-teacher",
                source_version="v1",
            )
        segments = _normalize_teacher_segments(
            [{"start": 0.0, "end": 2.0, "label": "intro"}],
            source="test-teacher",
            source_version="v1",
        )
        with self.assertRaisesRegex(ValueError, "ends at"):
            _validate_teacher_coverage(
                segments,
                source="test-teacher",
                duration_ms=20_000,
            )

    def test_teacher_rows_merge_by_axis_without_duplicate_frames(self):
        common = {
            "recording_id": "recording-1",
            "capture_session_id": "capture-1",
            "audio_frame_index": 42,
            "recording_offset_ms": 4_200,
            "position_ms": 4_200,
            "split_group_id": "spotify:track-1",
            "split": "train",
            "features": [0.2] * len(FEATURE_NAMES),
        }
        edm = {
            **common,
            "teacher_run_id": "edm-run",
            "timeline_id": "edm-timeline",
            "target_provenance_details": {"teacher_name": "EDMFormer"},
            "functional": "unknown",
            "energy": "release",
            "content": "unknown",
            "boundary": 0,
            "target_confidence": 0.8,
        }
        song = {
            **common,
            "teacher_run_id": "song-run",
            "timeline_id": "song-timeline",
            "target_provenance_details": {"teacher_name": "SongFormer"},
            "functional": "verse",
            "energy": "unknown",
            "content": "vocal",
            "boundary": 1,
            "milliseconds_since_boundary": 100,
            "target_confidence": 0.7,
        }
        merged, report = _merge_teacher_example_rows([edm, song])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["functional"], "verse")
        self.assertEqual(merged[0]["energy"], "release")
        self.assertEqual(merged[0]["content"], "vocal")
        self.assertEqual(merged[0]["boundary"], 1)
        self.assertEqual(report["duplicates_collapsed"], 1)
        self.assertEqual(
            merged[0]["target_provenance_by_axis"]["energy"][
                "teacher_name"
            ],
            "EDMFormer",
        )
        self.assertEqual(
            merged[0]["target_provenance_by_axis"]["functional"][
                "teacher_name"
            ],
            "SongFormer",
        )

    def test_teacher_merge_fills_legacy_missing_supervision_snapshot(self):
        common = {
            "recording_id": "recording-legacy",
            "capture_session_id": "capture-legacy",
            "audio_frame_index": 42,
            "recording_offset_ms": 4_200,
            "position_ms": 4_200,
            "split_group_id": "spotify:legacy-track",
            "split": "train",
            "features": [0.2] * len(FEATURE_NAMES),
            "functional": "unknown",
            "energy": "unknown",
            "content": "unknown",
            "boundary": 0,
            "target_confidence": 0.8,
        }
        legacy_edm = {
            **common,
            "teacher_run_id": "legacy-edm-run",
            "timeline_id": "legacy-edm-timeline",
            "target_provenance_details": {"source": "EDMFormer"},
            "energy": "release",
        }
        current_song = {
            **common,
            "teacher_run_id": "current-song-run",
            "timeline_id": "current-song-timeline",
            "target_provenance_details": {"source": "SongFormer"},
            "functional": "chorus",
            "structure_supervision": _complete_supervision(),
        }

        merged, report = _merge_teacher_example_rows(
            [legacy_edm, current_song]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(report["duplicates_collapsed"], 1)
        self.assertEqual(merged[0]["energy"], "release")
        self.assertEqual(merged[0]["functional"], "chorus")
        self.assertEqual(
            merged[0]["structure_supervision"], _complete_supervision()
        )

        explicit_edm = {
            **legacy_edm,
            "structure_supervision": _complete_supervision(),
        }
        conflicting_song = {
            **current_song,
            "structure_supervision": {
                "eligible": False,
                "classification": "partial",
                "reason_codes": ["capture_ended_before_track_end"],
                "evidence": {},
            },
        }
        with self.assertRaisesRegex(
            ValueError, "frame invariant structure_supervision"
        ):
            _merge_teacher_example_rows([explicit_edm, conflicting_song])

    def test_student_examples_use_capture_relative_teacher_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            store.begin_training_session(
                session_id="capture",
                mode="monitor",
                sample_rate=48_000,
                channels=2,
                sample_width=2,
                relative_path="audio/capture",
                metadata={},
            )
            payload = {
                "observation": {
                    "loudness": 0.5,
                    "onset_strength": 0.25,
                    "low_energy": 0.6,
                    "mid_energy": 0.3,
                    "high_energy": 0.1,
                    "bpm": 120.0,
                }
            }
            store.add_training_frames(
                [
                    {
                        "session_id": "capture",
                        "audio_frame_index": 48_000,
                        "segment_index": 0,
                        "segment_frame_index": 48_000,
                        "created_unix_ms": 1,
                        "song_id": None,
                        "position_ms": 120_000,
                        "payload": payload,
                    },
                    {
                        "session_id": "capture",
                        "audio_frame_index": 96_000,
                        "segment_index": 0,
                        "segment_frame_index": 96_000,
                        "created_unix_ms": 2,
                        "song_id": None,
                        "position_ms": 121_000,
                        "payload": payload,
                    },
                ]
            )
            recording_id = store.remember_recording_version(
                provider="spotify",
                provider_item_id="partway-through-track",
                duration_ms=2_000,
            )
            store.add_capture_track_span(
                capture_session_id="capture",
                start_audio_frame=48_000,
                end_audio_frame=144_000,
                recording_id=recording_id,
                start_position_ms=120_000,
                identity_source="spotify_metadata",
                identity_confidence=0.99,
                metadata={
                    "split_group_id": "spotify:partway-through-track",
                    "structure_supervision": _complete_supervision(),
                },
            )
            timeline_id = store.save_structure_timeline(
                recording_id=recording_id,
                capture_session_id="capture",
                provenance="test_teacher",
                timeline_version="test",
                confidence=1.0,
                segments=_normalize_teacher_segments(
                    [
                        {"start": 0.0, "end": 1.0, "label": "intro"},
                        {"start": 1.0, "end": 2.0, "label": "drop"},
                    ],
                    source="test",
                    source_version="v1",
                ),
            )

            result = build_student_examples(
                store,
                research_root=root / "research",
                recording_id=recording_id,
                timeline_id=timeline_id,
            )
            rows = [
                json.loads(line)
                for line in Path(result["path"]).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(result["examples"], 2)
            self.assertEqual(
                [row["position_ms"] for row in rows],
                [120_000, 121_000],
            )
            self.assertEqual(
                [row["recording_offset_ms"] for row in rows],
                [0, 1_000],
            )
            self.assertEqual(rows[0]["functional"], "intro")
            self.assertEqual(rows[1]["energy"], "release")
            self.assertEqual(rows[0]["timeline_version"], "test")
            self.assertEqual(
                rows[0]["target_provenance_details"]["source_version"],
                "v1",
            )
            self.assertEqual(
                rows[0]["target_provenance_details"]["annotation_type"],
                "teacher_prediction",
            )
            self.assertEqual(
                {row["split_group_id"] for row in rows},
                {"spotify:partway-through-track"},
            )
            self.assertEqual(
                {row["split"] for row in rows},
                {_recording_split("spotify:partway-through-track")},
            )

    def test_repeated_analog_captures_share_provider_split_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SongMemoryStore(Path(temporary) / "lumen.sqlite3")
            first_recording = store.remember_recording_version(
                provider="spotify",
                provider_item_id="same-track",
                duration_ms=30_000,
                audio_fingerprint="pcm-sha256:" + "a" * 64,
            )
            second_recording = store.remember_recording_version(
                provider="spotify",
                provider_item_id="same-track",
                duration_ms=30_100,
                audio_fingerprint="pcm-sha256:" + "b" * 64,
            )
            self.assertNotEqual(first_recording, second_recording)
        first = {
            "capture_session_id": "capture-a",
            "recording_provider": "spotify",
            "recording_provider_item_id": "same-track",
            "metadata": {},
        }
        second = {
            "capture_session_id": "capture-b",
            "recording_provider": "spotify",
            "recording_provider_item_id": "same-track",
            "metadata": {},
        }
        first_group = _capture_split_group(first)
        second_group = _capture_split_group(second)
        self.assertEqual(first_group, "spotify:same-track")
        self.assertEqual(second_group, first_group)
        self.assertEqual(
            _recording_split(first_group), _recording_split(second_group)
        )

        unidentified_first = {
            "capture_session_id": "capture-a",
            "metadata": {},
        }
        unidentified_second = {
            "capture_session_id": "capture-b",
            "metadata": {},
        }
        self.assertNotEqual(
            _capture_split_group(unidentified_first),
            _capture_split_group(unidentified_second),
        )

    def test_songformer_worker_normalizes_and_saves_runner_output(self):
        class DeterministicSongFormerWorker(OfflineResearchWorker):
            def _songformer_paths(self):
                return {
                    "python": Path("/test/songformer/python"),
                    "runner": Path("/test/songformer-cpu-runner.py"),
                    "revision": "songformer-test-revision",
                }

            def _run_subprocess(self, command, environment):
                del environment
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        [
                            {"label": "intro", "start": 0.0, "end": 4.0},
                            {
                                "label": "pre-chorus",
                                "start": 4.0,
                                "end": 8.0,
                            },
                            {"label": "chorus", "start": 8.0, "end": 12.0},
                        ]
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research = root / "research"
            audio = root / "recording.wav"
            audio.write_bytes(b"deterministic-test-audio")
            store = SongMemoryStore(root / "lumen.sqlite3")
            recording_id = store.remember_recording_version(
                provider="test",
                provider_item_id="songformer-track",
                duration_ms=12_000,
            )
            store.begin_training_session(
                session_id="songformer-capture",
                mode="monitor",
                sample_rate=100,
                channels=1,
                sample_width=2,
                relative_path="audio/songformer-capture",
                metadata={},
            )
            store.add_training_frames(
                [
                    {
                        "session_id": "songformer-capture",
                        "audio_frame_index": 50,
                        "segment_index": 0,
                        "segment_frame_index": 50,
                        "created_unix_ms": 1,
                        "song_id": None,
                        "position_ms": 500,
                        "payload": {"observation": {}},
                    }
                ]
            )
            store.add_capture_track_span(
                capture_session_id="songformer-capture",
                start_audio_frame=0,
                end_audio_frame=1_200,
                recording_id=recording_id,
                start_position_ms=0,
                end_position_ms=12_000,
                identity_source="test",
                identity_confidence=1.0,
                metadata={
                    "split_group_id": "test:songformer-track",
                    "structure_supervision": _complete_supervision(),
                },
            )
            store.enqueue_analysis_job(
                job_type=SONGFORMER_JOB,
                payload={
                    "recording_id": recording_id,
                    "audio_path": str(audio),
                    "content_sha256": "a" * 64,
                    "duration_ms": 12_000,
                    "song_id": None,
                    "capture_session_id": "songformer-capture",
                    "structure_supervision": _complete_supervision(),
                    "songformer_window_seconds": 60,
                },
            )
            result = DeterministicSongFormerWorker(
                store, research_root=research
            ).run_once((SONGFORMER_JOB,))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["status"], "complete")
            timeline = store.structure_timeline(
                result["result"]["timeline_id"]
            )
            self.assertIsNotNone(timeline)
            assert timeline is not None
            self.assertEqual(timeline["provenance"], "songformer_teacher")
            self.assertEqual(
                [row["functional_label"] for row in timeline["segments"]],
                ["intro", "pre_chorus", "chorus"],
            )
            self.assertEqual(
                timeline["metadata"]["cpu_context_window_seconds"], 60
            )

    def test_teacher_subprocess_cancellation_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            cancel_event = threading.Event()
            cancel_event.set()
            worker = OfflineResearchWorker(
                SongMemoryStore(Path(temporary) / "lumen.sqlite3"),
                research_root=Path(temporary) / "research",
                cancel_event=cancel_event,
            )
            with self.assertRaises(OfflineJobCancelled):
                worker._run_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    dict(os.environ),
                )

    def test_teacher_subprocess_timeout_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = OfflineResearchWorker(
                SongMemoryStore(Path(temporary) / "lumen.sqlite3"),
                research_root=Path(temporary) / "research",
                timeout_s=0.01,
            )
            with self.assertRaisesRegex(
                TimeoutError, "offline teacher exceeded"
            ):
                worker._run_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    dict(os.environ),
                )

    def test_teacher_subprocess_memory_limit_stops_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = OfflineResearchWorker(
                SongMemoryStore(Path(temporary) / "lumen.sqlite3"),
                research_root=Path(temporary) / "research",
                max_rss_bytes=1,
            )
            with self.assertRaisesRegex(
                OfflineMemoryLimitExceeded, "local memory limit"
            ):
                worker._run_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    dict(os.environ),
                )
            self.assertGreater(worker._last_subprocess_metrics["peak_rss_bytes"], 0)

    def test_worker_trains_and_saves_cpu_student(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            examples = root / "examples.jsonl"
            rows = []
            for index in range(24):
                features = [0.0] * len(FEATURE_NAMES)
                active = index >= 12
                features[0] = 0.9 if active else 0.1
                features[1] = 0.8 if active else 0.05
                rows.append(
                    {
                        "features": features,
                        "functional": "chorus" if active else "intro",
                        "energy": "release" if active else "restrained",
                        "content": "instrumental",
                    }
                )
            examples.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            model_path = root / "research" / "models" / "student.npz"
            store.enqueue_analysis_job(
                job_type=STUDENT_TRAIN_JOB,
                payload={
                    "examples_path": str(examples),
                    "output_path": str(model_path),
                    "epochs": 10,
                },
            )
            result = OfflineResearchWorker(
                store, research_root=root / "research"
            ).run_once()
            self.assertEqual(result["status"], "complete")
            self.assertTrue(model_path.is_file())
            self.assertEqual(
                store.list_analysis_jobs()[0]["status"], "complete"
            )

    def test_failed_heldout_gate_preserves_active_student(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            examples = root / "examples.jsonl"
            rows = []
            for index in range(40):
                features = [0.1] * len(FEATURE_NAMES)
                rows.append(
                    {
                        "features": features,
                        "functional": "intro",
                        "energy": "low",
                        "content": "instrumental",
                        "boundary": 0,
                        "split": "train",
                        "split_group_id": "training-song",
                    }
                )
            for index in range(20):
                features = [0.1] * len(FEATURE_NAMES)
                rows.append(
                    {
                        "features": features,
                        "functional": "chorus",
                        "energy": "release",
                        "content": "vocal",
                        "boundary": int(index < 5),
                        "split": "validation",
                        "split_group_id": "heldout-song",
                    }
                )
            examples.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            model_path = root / "research" / "models" / "student.npz"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"existing-active-model")
            store.enqueue_analysis_job(
                job_type=STUDENT_TRAIN_JOB,
                payload={
                    "examples_path": str(examples),
                    "output_path": str(model_path),
                    "epochs": 5,
                    "require_activation_gate": True,
                },
            )
            result = OfflineResearchWorker(
                store, research_root=root / "research"
            ).run_once((STUDENT_TRAIN_JOB,))
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["result"]["activated"])
            self.assertEqual(model_path.read_bytes(), b"existing-active-model")
            self.assertTrue(
                Path(result["result"]["candidate_model_path"]).is_file()
            )

    def test_student_training_cancellation_requeues_and_preserves_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "lumen.sqlite3")
            examples = root / "examples.jsonl"
            row = {
                "features": [0.2] * len(FEATURE_NAMES),
                "functional": "intro",
                "energy": "low",
                "content": "instrumental",
                "boundary": 0,
                "split": "train",
                "split_group_id": "training-song",
            }
            examples.write_text(
                "".join(json.dumps(row) + "\n" for _ in range(130)),
                encoding="utf-8",
            )
            model_path = root / "research" / "models" / "student.npz"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"existing-active-model")
            store.enqueue_analysis_job(
                job_type=STUDENT_TRAIN_JOB,
                payload={
                    "examples_path": str(examples),
                    "output_path": str(model_path),
                    "epochs": 10,
                    "require_activation_gate": False,
                },
            )
            cancel_event = threading.Event()
            cancel_event.set()
            result = OfflineResearchWorker(
                store,
                research_root=root / "research",
                cancel_event=cancel_event,
            ).run_once((STUDENT_TRAIN_JOB,))
            self.assertEqual(result["status"], "canceled")
            self.assertIn("requested checkpoint", result["error"])
            self.assertEqual(model_path.read_bytes(), b"existing-active-model")
            self.assertEqual(store.list_analysis_jobs()[0]["status"], "queued")

    def test_student_training_queue_combines_versioned_example_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research = root / "research"
            examples = root / "teacher.jsonl"
            row = {
                "features": [0.2] * len(FEATURE_NAMES),
                "functional": "intro",
                "energy": "low",
                "content": "instrumental",
                "split": "train",
                "split_group_id": "spotify:explicit-test",
            }
            examples.write_text(json.dumps(row) + "\n", encoding="utf-8")
            store = SongMemoryStore(root / "lumen.sqlite3")
            queued = enqueue_student_training(
                store,
                research_root=research,
                example_paths=[examples],
                epochs=1,
            )
            self.assertEqual(queued["examples"], 1)
            job = store.list_analysis_jobs()[0]
            self.assertEqual(job["job_type"], STUDENT_TRAIN_JOB)
            self.assertEqual(job["payload"]["epochs"], 1)
            self.assertIn(str(examples), job["payload"]["source_sha256"])

    def test_automatic_training_ignores_unowned_example_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research = root / "research"
            examples = research / "exports" / "student-examples"
            examples.mkdir(parents=True)
            (examples / "stale.jsonl").write_text(
                json.dumps(
                    {
                        "features": [0.2] * len(FEATURE_NAMES),
                        "functional": "intro",
                        "energy": "low",
                        "content": "instrumental",
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = SongMemoryStore(root / "lumen.sqlite3")
            with self.assertRaisesRegex(RuntimeError, "completed teacher runs"):
                enqueue_student_training(store, research_root=research)
            readiness = training_readiness(store, research_root=research)
            self.assertFalse(readiness["train_ready"])
            self.assertEqual(readiness["usable_examples"], 0)

    def test_automatic_training_accepts_only_verified_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research = root / "research"
            examples_root = research / "exports" / "student-examples"
            examples_root.mkdir(parents=True)
            store = SongMemoryStore(root / "lumen.sqlite3")
            run_id = store.begin_teacher_run(
                teacher_name="test",
                teacher_version="v1",
                device="cpu",
                preprocessing_version="test",
            )
            path = examples_root / "verified.jsonl"
            rows = [
                {
                    "teacher_run_id": run_id,
                    "recording_id": "training-song",
                    "features": [0.2] * len(FEATURE_NAMES),
                    "functional": "intro",
                    "energy": "low",
                    "content": "instrumental",
                    "split": "train",
                    "split_group_id": "training-song",
                },
                {
                    "teacher_run_id": run_id,
                    "recording_id": "training-song-2",
                    "features": [0.4] * len(FEATURE_NAMES),
                    "functional": "verse",
                    "energy": "low",
                    "content": "instrumental",
                    "split": "train",
                    "split_group_id": "training-song-2",
                },
                {
                    "teacher_run_id": run_id,
                    "recording_id": "heldout-song",
                    "features": [0.8] * len(FEATURE_NAMES),
                    "functional": "chorus",
                    "energy": "release",
                    "content": "instrumental",
                    "split": "validation",
                    "split_group_id": "heldout-song",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            store.finish_teacher_run(
                run_id,
                status="complete",
                metrics={
                    "structure_supervision": _complete_supervision(),
                    "student_examples": {
                        "path": str(path),
                        "examples": len(rows),
                        "sha256": checksum,
                    }
                },
            )
            trusted = trusted_student_examples(store, research_root=research)
            self.assertEqual(trusted["paths"], [str(path)])
            self.assertEqual(trusted["examples"], 3)
            queued = enqueue_student_training(store, research_root=research)
            self.assertEqual(queued["teacher_run_ids"], [run_id])
            self.assertEqual(queued["split_counts"]["validation"], 1)


if __name__ == "__main__":
    unittest.main()
