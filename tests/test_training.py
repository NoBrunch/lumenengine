from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock
import wave

from lumen_engine.memory import SongMemoryStore
from lumen_engine.models import Feedback, MediaIdentity
from lumen_engine.training import (
    TrainingCaptureConfig,
    TrainingDataRecorder,
    export_training_dataset,
    structure_supervision_completeness,
)


class TrainingDataTests(unittest.TestCase):
    def test_structure_supervision_requires_a_complete_provider_track(
        self,
    ) -> None:
        complete = structure_supervision_completeness(
            track_duration_ms=200_000,
            start_position_ms=5_000,
            end_position_ms=195_000,
            captured_duration_ms=190_000,
            source_audio_complete=True,
        )
        partial = structure_supervision_completeness(
            track_duration_ms=200_000,
            start_position_ms=143_000,
            end_position_ms=200_000,
            captured_duration_ms=60_000,
            source_audio_complete=True,
        )
        unknown = structure_supervision_completeness(
            track_duration_ms=None,
            start_position_ms=None,
            end_position_ms=None,
            captured_duration_ms=60_000,
            source_audio_complete=True,
        )
        too_long = structure_supervision_completeness(
            track_duration_ms=200_000,
            start_position_ms=0,
            end_position_ms=200_000,
            captured_duration_ms=220_001,
            source_audio_complete=True,
        )

        self.assertTrue(complete["eligible"])
        self.assertEqual(complete["classification"], "complete")
        self.assertFalse(partial["eligible"])
        self.assertEqual(partial["classification"], "partial")
        self.assertIn(
            "capture_started_after_track_beginning",
            partial["reason_codes"],
        )
        self.assertEqual(unknown["classification"], "unknown")
        self.assertIn(
            "captured_audio_too_long_for_track",
            too_long["reason_codes"],
        )

    def test_ten_hz_semantics_use_exact_sample_grid_with_2048_packets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            recorder = TrainingDataRecorder(
                store,
                session_id="ten-hz-grid",
                mode="monitor",
                config=TrainingCaptureConfig(
                    root=root / "training",
                    sample_rate=48_000,
                    channels=1,
                    segment_seconds=10,
                    feature_rate_hz=10.0,
                    max_bytes=1024**3,
                    minimum_free_bytes=0,
                ),
                metadata={},
            )
            recorder.start()
            remaining = 480_000
            while remaining:
                frame_count = min(2_048, remaining)
                recorder.submit(
                    b"\0\0" * frame_count,
                    song_id=None,
                    position_ms=None,
                    payload={"observation": {"section": "groove"}},
                )
                remaining -= frame_count
            recorder.stop()

            rows = store.training_frames("ten-hz-grid")
            indexes = [int(row["audio_frame_index"]) for row in rows]
            self.assertEqual(len(indexes), 100)
            self.assertEqual(indexes[0], 2_400)
            self.assertEqual(indexes[-1], 477_600)
            self.assertEqual(
                {right - left for left, right in zip(indexes, indexes[1:])},
                {4_800},
            )
            self.assertTrue(
                all(
                    row["payload"]["semantic_resampling"]["target_audio_frame"]
                    == row["audio_frame_index"]
                    for row in rows
                )
            )

    def test_pcm_features_feedback_and_export_remain_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            song_id = store.remember_media(
                MediaIdentity(
                    provider="test",
                    provider_item_id="track-1",
                    title="Training Track",
                    artists=("Test Artist",),
                )
            )
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=2,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="session:test",
                mode="monitor",
                config=config,
                metadata={"rig_name": "Test Rig"},
            )
            recorder.start()
            first_pcm = struct.pack("<hh", 1_000, -1_000) * 600
            second_pcm = struct.pack("<hh", 2_000, -2_000) * 600
            first_frame = recorder.submit(
                first_pcm,
                song_id=song_id,
                position_ms=1_000,
                payload={
                    "observation": {"section": "groove"},
                    "decision": {"routine": "breathe"},
                    "fixture_dmx": [
                        {"fixture_id": "movers", "channels": [1, 2, 3]}
                    ],
                },
            )
            second_frame = recorder.submit(
                second_pcm,
                song_id=song_id,
                position_ms=7_000,
                payload={
                    "observation": {"section": "build"},
                    "decision": {"routine": "fan_sweep"},
                    "fixture_dmx": [
                        {"fixture_id": "movers", "channels": [9, 8, 7]}
                    ],
                },
            )
            recorder.stop()

            self.assertEqual(first_frame, 300)
            self.assertEqual(second_frame, 900)
            status = recorder.status()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["frames_written"], 1_200)
            self.assertEqual(status["segments"], 2)
            self.assertEqual(status["dropped_frames"], 0)

            segments = store.training_segments("session:test")
            self.assertEqual(
                [(row["start_frame"], row["frame_count"]) for row in segments],
                [(0, 1_000), (1_000, 200)],
            )
            for expected_frames, segment in zip((1_000, 200), segments):
                path = config.root / segment["relative_path"]
                with wave.open(str(path), "rb") as recording:
                    self.assertEqual(recording.getnframes(), expected_frames)
                    self.assertEqual(recording.getframerate(), 100)
                    self.assertEqual(recording.getnchannels(), 2)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    segment["sha256"],
                )
            reconstructed = bytearray()
            for segment in segments:
                with wave.open(
                    str(config.root / segment["relative_path"]), "rb"
                ) as recording:
                    reconstructed.extend(
                        recording.readframes(recording.getnframes())
                    )
            self.assertEqual(bytes(reconstructed), first_pcm + second_pcm)

            capture = json.loads(
                (
                    config.root
                    / "audio"
                    / next((config.root / "audio").iterdir()).name
                    / "session-test"
                    / "capture.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(capture["timeline"]["sample_accurate"])
            self.assertEqual(capture["timeline"]["gaps"], [])

            store.add_feedback(
                Feedback(
                    song_id=song_id,
                    position_ms=7_000,
                    label="increase_movement",
                    value=1.0,
                    capture_session_id="session:test",
                    audio_frame_index=second_frame,
                )
            )
            store.add_training_annotation(
                song_id=song_id,
                position_ms=7_000,
                kind="preferred_action",
                label="figure_eight",
                scope="group",
                fixture_id="movers",
                intensity=1.0,
                note=None,
                capture_session_id="session:test",
                audio_frame_index=second_frame,
                context={"decision": {"routine": "breathe"}},
            )
            result = export_training_dataset(store, config.root)
            manifest = json.loads(
                (Path(result["path"]) / "dataset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["validation"]["valid"])
            self.assertEqual(manifest["counts"]["sessions"], 1)
            self.assertEqual(manifest["counts"]["segments"], 2)
            self.assertEqual(manifest["counts"]["feedback_examples"], 1)
            self.assertEqual(manifest["counts"]["annotation_examples"], 1)
            self.assertEqual(manifest["counts"]["recordings"], 1)
            self.assertEqual(manifest["counts"]["student_sequences"], 1)
            self.assertEqual(
                manifest["counts"]["choreography_sequences"], 1
            )
            example = json.loads(
                (Path(result["path"]) / "feedback_examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(example["audio_frame_index"], second_frame)
            self.assertEqual(example["feedback"]["label"], "increase_movement")
            self.assertEqual(len(example["audio_clips"]), 2)
            annotation = json.loads(
                (Path(result["path"]) / "annotation_examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                annotation["annotation"]["label"], "figure_eight"
            )
            recording = json.loads(
                (Path(result["path"]) / "recordings.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(recording["split_group_id"], f"lumen-song:{song_id}")
            self.assertEqual(recording["frame_count"], 1_200)
            self.assertFalse(
                recording["structure_supervision"]["eligible"]
            )
            self.assertEqual(
                recording["structure_supervision"]["classification"],
                "unknown",
            )
            self.assertEqual(
                manifest["counts"]["structure_supervision"]["unknown"],
                1,
            )
            self.assertEqual(
                sum(
                    int(clip["frame_count"])
                    for clip in recording["audio_clips"]
                ),
                1_200,
            )
            choreography = json.loads(
                (Path(result["path"]) / "choreography_sequences.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(len(choreography["supervision_events"]), 2)
            self.assertIn("structure_supervision", choreography)
            self.assertEqual(
                choreography["supervision_events"][0][
                    "context_frame_audio_index"
                ],
                895,
            )
            performed = choreography["performed_frames"]
            self.assertEqual(len(performed), 120)
            self.assertEqual(
                performed[60]["fixture_dmx"][0]["channels"], [9, 8, 7]
            )
            self.assertEqual(
                performed[60]["decision"]["routine"], "fan_sweep"
            )
            self.assertEqual(
                [
                    run["routine"]
                    for run in choreography["performed_routine_runs"]
                ],
                ["breathe", "fan_sweep"],
            )
            self.assertEqual(
                choreography["performed_routine_runs"][0][
                    "duration_frames"
                ],
                600,
            )
            self.assertEqual(
                choreography["preferred_action_sequence"][0]["label"],
                "figure_eight",
            )
            self.assertEqual(
                choreography["preferred_sequence_completeness"],
                "ordered_sparse_actions_without_authored_durations",
            )

    def test_queue_loss_is_silent_sample_aligned_and_explicit(self) -> None:
        class DelayedRecorder(TrainingDataRecorder):
            release_writer = threading.Event()

            def _writer(self) -> None:
                self.release_writer.wait(timeout=5.0)
                super()._writer()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=2,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
                queue_packets=1,
            )
            DelayedRecorder.release_writer.clear()
            recorder = DelayedRecorder(
                store,
                session_id="overflow",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            kept_pcm = struct.pack("<hh", 3_000, -3_000) * 100
            dropped_pcm = struct.pack("<hh", 9_000, -9_000) * 100
            recorder.submit(
                kept_pcm, song_id=None, position_ms=None, payload={}
            )
            recorder.submit(
                dropped_pcm, song_id=None, position_ms=None, payload={}
            )
            recorder.submit(
                dropped_pcm, song_id=None, position_ms=None, payload={}
            )
            DelayedRecorder.release_writer.set()
            recorder.stop()

            status = recorder.status()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["frames_received"], 300)
            self.assertEqual(status["frames_written"], 300)
            self.assertEqual(status["dropped_packets"], 2)
            self.assertEqual(status["dropped_frames"], 200)
            segment = store.training_segments("overflow")[0]
            path = config.root / segment["relative_path"]
            with wave.open(str(path), "rb") as recording:
                pcm = recording.readframes(recording.getnframes())
            self.assertEqual(pcm, kept_pcm + bytes(200 * 4))
            capture = json.loads(
                path.with_name("capture.json").read_text(encoding="utf-8")
            )
            self.assertTrue(capture["timeline"]["sample_accurate"])
            self.assertEqual(
                capture["timeline"]["gaps"],
                [
                    {
                        "frame_count": 200,
                        "reason": "capture_queue_overflow",
                        "representation": "pcm_silence",
                        "start_frame": 100,
                    }
                ],
            )
            export = export_training_dataset(store, config.root)
            manifest = export["manifest"]
            self.assertTrue(manifest["validation"]["valid"])
            self.assertEqual(
                manifest["validation"]["warnings"][0]["code"],
                "capture_contains_explicit_gap_ranges",
            )

    def test_export_detects_audio_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="corruption",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            recorder.submit(
                struct.pack("<h", 500) * 100,
                song_id=None,
                position_ms=None,
                payload={},
            )
            recorder.stop()
            segment = store.training_segments("corruption")[0]
            path = config.root / segment["relative_path"]
            with path.open("ab") as target:
                target.write(b"corruption")
            export = export_training_dataset(store, config.root)
            validation = export["manifest"]["validation"]
            self.assertFalse(validation["valid"])
            codes = {item["code"] for item in validation["errors"]}
            self.assertIn("audio_byte_count_mismatch", codes)
            self.assertIn("audio_sha256_mismatch", codes)

    def test_repeated_track_plays_have_distinct_recordings_same_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="repeated-play",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            media = {
                "provider": "spotify",
                "provider_item_id": "stable-track-id",
                "title": "Repeat",
            }
            for position_ms in (1_000, 5_000, 100, 1_100):
                recorder.submit(
                    struct.pack("<h", 700) * 100,
                    song_id=None,
                    position_ms=position_ms,
                    payload={"media": media, "observation": {}},
                )
            recorder.stop()
            export = export_training_dataset(store, config.root)
            recordings = [
                json.loads(line)
                for line in (
                    Path(export["path"]) / "recordings.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(recordings), 2)
            self.assertNotEqual(
                recordings[0]["recording_id"],
                recordings[1]["recording_id"],
            )
            self.assertEqual(
                recordings[0]["split_group_id"],
                "spotify:stable-track-id",
            )
            self.assertEqual(
                recordings[0]["split_group_id"],
                recordings[1]["split_group_id"],
            )
            self.assertEqual(
                recordings[0]["split"], recordings[1]["split"]
            )
            self.assertEqual(
                sum(item["frame_count"] for item in recordings), 400
            )
            second_export = export_training_dataset(store, config.root)
            repeated_recordings = [
                json.loads(line)
                for line in (
                    Path(second_export["path"]) / "recordings.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotEqual(export["path"], second_export["path"])
            self.assertEqual(
                [item["recording_id"] for item in recordings],
                [item["recording_id"] for item in repeated_recordings],
            )

    def test_annotation_failure_never_breaks_pcm_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="annotation-error",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            pcm = struct.pack("<h", 1_234) * 100

            def broken_payload() -> dict[str, object]:
                raise RuntimeError("semantic snapshot failed")

            frame = recorder.submit(
                pcm,
                song_id=None,
                position_ms=None,
                payload=broken_payload,
            )
            self.assertEqual(frame, 50)
            recorder.stop()
            status = recorder.status()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["frames_written"], 100)
            self.assertEqual(status["annotation_errors"], 1)
            self.assertEqual(status["semantic_frames"], 0)
            segment = store.training_segments("annotation-error")[0]
            path = config.root / segment["relative_path"]
            with wave.open(str(path), "rb") as recording:
                self.assertEqual(recording.readframes(100), pcm)
            capture = json.loads(
                path.with_name("capture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                capture["semantic_capture"]["annotation_errors"], 1
            )
            self.assertEqual(
                capture["semantic_capture"]["last_annotation_error"],
                "semantic snapshot failed",
            )

    def test_concurrent_submitters_cannot_reorder_the_pcm_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="concurrent-submit",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            first_pcm = struct.pack("<h", 111) * 100
            second_pcm = struct.pack("<h", 222) * 100
            payload_started = threading.Event()
            release_payload = threading.Event()

            def slow_payload() -> dict[str, object]:
                payload_started.set()
                release_payload.wait(timeout=5)
                return {"source": "first"}

            first_thread = threading.Thread(
                target=lambda: recorder.submit(
                    first_pcm,
                    song_id=None,
                    position_ms=None,
                    payload=slow_payload,
                )
            )
            second_thread = threading.Thread(
                target=lambda: recorder.submit(
                    second_pcm,
                    song_id=None,
                    position_ms=None,
                    payload={"source": "second"},
                )
            )
            first_thread.start()
            self.assertTrue(payload_started.wait(timeout=2))
            second_thread.start()
            time.sleep(0.01)
            release_payload.set()
            first_thread.join(timeout=2)
            second_thread.join(timeout=2)
            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            recorder.stop()
            segment = store.training_segments("concurrent-submit")[0]
            with wave.open(
                str(config.root / segment["relative_path"]), "rb"
            ) as recording:
                reconstructed = recording.readframes(200)
            self.assertEqual(reconstructed, first_pcm + second_pcm)

    def test_stop_waits_for_inflight_submission_before_end_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="stop-handshake",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            pcm = struct.pack("<h", 777) * 100
            payload_started = threading.Event()
            release_payload = threading.Event()

            def slow_payload() -> dict[str, object]:
                payload_started.set()
                release_payload.wait(timeout=5)
                return {"complete": True}

            submit_thread = threading.Thread(
                target=lambda: recorder.submit(
                    pcm,
                    song_id=None,
                    position_ms=None,
                    payload=slow_payload,
                )
            )
            submit_thread.start()
            self.assertTrue(payload_started.wait(timeout=2))
            stop_thread = threading.Thread(
                target=lambda: recorder.stop(timeout=2)
            )
            stop_thread.start()
            time.sleep(0.01)
            self.assertTrue(stop_thread.is_alive())
            release_payload.set()
            submit_thread.join(timeout=2)
            stop_thread.join(timeout=2)
            self.assertFalse(submit_thread.is_alive())
            self.assertFalse(stop_thread.is_alive())
            self.assertEqual(recorder.status()["state"], "complete")
            segment = store.training_segments("stop-handshake")[0]
            path = config.root / segment["relative_path"]
            with wave.open(str(path), "rb") as recording:
                self.assertEqual(recording.readframes(100), pcm)
            capture = json.loads(
                path.with_name("capture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture["timeline"]["gaps"], [])

    def test_storage_quota_accounts_for_open_wav_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                feature_rate_hz=10,
                max_bytes=300,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="quota",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            recorder.submit(
                struct.pack("<h", 1_000) * 100,
                song_id=None,
                position_ms=None,
                payload={},
            )
            recorder.submit(
                struct.pack("<h", 2_000) * 100,
                song_id=None,
                position_ms=None,
                payload={},
            )
            recorder.stop()
            status = recorder.status()
            self.assertEqual(status["state"], "quota")
            self.assertEqual(status["frames_received"], 200)
            self.assertEqual(status["frames_written"], 100)
            self.assertEqual(status["dropped_packets"], 1)
            self.assertEqual(status["dropped_frames"], 100)
            self.assertLessEqual(status["bytes_written"], 300)
            segment = store.training_segments("quota")[0]
            self.assertLessEqual(segment["byte_count"], 300)
            capture = json.loads(
                (
                    config.root / segment["relative_path"]
                ).with_name("capture.json").read_text(encoding="utf-8")
            )
            self.assertFalse(capture["timeline"]["sample_accurate"])
            self.assertEqual(
                capture["timeline"]["gaps"],
                [
                    {
                        "frame_count": 100,
                        "reason": "storage_quota",
                        "representation": "not_written",
                        "start_frame": 100,
                    }
                ],
            )

    def test_failed_export_never_appears_as_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SongMemoryStore(root / "memory.sqlite3")
            config = TrainingCaptureConfig(
                root=root / "training",
                sample_rate=100,
                channels=1,
                segment_seconds=10,
                max_bytes=1024**3,
                minimum_free_bytes=0,
            )
            recorder = TrainingDataRecorder(
                store,
                session_id="export-failure",
                mode="monitor",
                config=config,
                metadata={},
            )
            recorder.start()
            recorder.submit(
                struct.pack("<h", 500) * 100,
                song_id=None,
                position_ms=None,
                payload={},
            )
            recorder.stop()
            with mock.patch(
                "lumen_engine.training._verify_segments",
                side_effect=RuntimeError("forced export failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "forced export failure"
                ):
                    export_training_dataset(store, config.root)
            export_children = list((config.root / "exports").iterdir())
            self.assertEqual(len(export_children), 1)
            self.assertTrue(export_children[0].name.startswith(".dataset-"))
            self.assertTrue(export_children[0].name.endswith(".partial"))
            self.assertFalse(
                (export_children[0] / "dataset.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
