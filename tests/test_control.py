from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest.mock import patch

from lumen_engine.control import (
    GatedOutput,
    LumenApplication,
    LumenHTTPServer,
    OperatorControls,
    OperatorExpressionEngine,
    RehearsalControls,
    _AnalyzedControlFrame,
    _rehearsal_observation,
)
from lumen_engine.audio import AudioCaptureConfig, AudioInputMetrics
from lumen_engine.choreography import (
    ChoreographySequence,
    ChoreographyStep,
    SequencePreferenceModel,
)
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.expression import ExpressionPolicy
from lumen_engine.link import STUDENT_TRAIN_JOB
from lumen_engine.models import MediaIdentity, MusicalObservation
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.student import (
    LABELS,
    StudentPrediction,
    StreamingStructureStudent,
)
from lumen_engine.offline import (
    EDMFORMER_JOB,
    SONGFORMER_JOB,
    STUDENT_ACTIVATION_GATE_VERSION,
)
from lumen_engine.runtime import PerformanceRuntime


class ControlApplicationTests(unittest.TestCase):
    def test_remote_offline_lease_does_not_block_live_engine(self) -> None:
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"execution_target": "threadripper"},
        )
        claimed = self.application.memory.claim_analysis_job_by_id(
            job_id,
            worker_id="lumen-link:test",
            worker_pid=os.getpid(),
        )
        self.assertIsNotNone(claimed)
        with patch.object(
            self.application.lumen_link,
            "live_transition_guard",
            wraps=self.application.lumen_link.live_transition_guard,
        ) as guard:
            snapshot = self.application.start("demo")
        guard.assert_called_once_with()
        self.assertEqual(snapshot["engine"]["mode"], "demo")
        self.application.stop()

    def test_same_track_seek_invalidates_causal_structure_context(self) -> None:
        first = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:seek-test",
            title="Seek Test",
            observed_position_ms=120_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        self.application._remember_media_identity(first)
        generation = self.application._analysis_generation
        self.application._cached_structure_prediction = {
            "axes": {"energy": {"label": "drop", "confidence": 0.9}}
        }

        self.application._remember_media_identity(replace(
            first,
            observed_position_ms=30_000,
            observed_at_unix_ms=round(time.time() * 1000),
        ))

        self.assertEqual(self.application._analysis_generation, generation + 1)
        self.assertIsNone(self.application._cached_structure_prediction)
        self.assertTrue(any(
            "reset causal audio" in event["message"]
            for event in self.application.events
        ))

        # A normal metadata refresh on the same recording is not a seek.
        self.application._remember_media_identity(replace(
            first,
            observed_position_ms=31_000,
            observed_at_unix_ms=round(time.time() * 1000),
        ))
        self.assertEqual(self.application._analysis_generation, generation + 1)

        self.application._remember_media_identity(replace(
            first,
            observed_position_ms=36_000,
            observed_at_unix_ms=round(time.time() * 1000),
        ))
        self.assertEqual(self.application._analysis_generation, generation + 2)

    def test_media_position_can_be_anchored_to_captured_sample_time(self) -> None:
        self.application.media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:sample-time",
            title="Sample Time",
            duration_ms=300_000,
            observed_position_ms=100_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=True,
        )
        current = self.application._media_position_ms()
        captured = self.application._media_position_ms(
            at_monotonic_s=time.monotonic() - 0.30
        )
        self.assertIsNotNone(current)
        self.assertIsNotNone(captured)
        self.assertGreaterEqual(current - captured, 250)
        self.assertLessEqual(current - captured, 350)

    def test_offline_teacher_button_processes_a_resumable_batch(self) -> None:
        for job_type in (EDMFORMER_JOB, SONGFORMER_JOB):
            self.application.memory.enqueue_analysis_job(
                job_type=job_type,
                payload={"recording_id": job_type},
            )

        class FakeWorker:
            responses = [
                {
                    "job_id": "one",
                    "job_type": EDMFORMER_JOB,
                    "status": "complete",
                    "result": {},
                },
            ]

            def __init__(self, *args, **kwargs):
                del args, kwargs

            def run_once(self, job_types):
                self.asserted_types = tuple(job_types)
                return self.responses.pop(0) if self.responses else None

        with patch("lumen_engine.control.OfflineResearchWorker", FakeWorker):
            self.application.start_research_worker()
            thread = self.application._research_worker_thread
            assert thread is not None
            thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        result = self.application._research_worker_last
        assert result is not None
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["failed"], 0)
        queued = {
            job["job_type"]: job["status"]
            for job in self.application.memory.list_analysis_jobs()
        }
        self.assertEqual(queued[SONGFORMER_JOB], "queued")

    def test_analyze_preflight_does_not_rebuild_while_engine_thread_lives(
        self,
    ) -> None:
        release = threading.Event()
        engine = threading.Thread(
            target=lambda: release.wait(timeout=2.0), daemon=True
        )
        engine.start()
        self.application._thread = engine
        # A fault can be published just before the engine owner thread exits.
        # Offline work must still wait for that owner to finish.
        self.application.engine_phase = "fault"
        try:
            with patch.object(
                self.application, "export_training_data"
            ) as export:
                with self.assertRaisesRegex(RuntimeError, "stop Monitor"):
                    self.application.analyze_training_data()
                export.assert_not_called()
        finally:
            release.set()
            engine.join(timeout=1.0)
            self.application._thread = None
            self.application.engine_phase = "ready"

    def test_analyze_preflight_does_not_rebuild_during_active_batch(
        self,
    ) -> None:
        release = threading.Event()
        worker = threading.Thread(
            target=lambda: release.wait(timeout=2.0), daemon=True
        )
        worker.start()
        self.application._research_worker_thread = worker
        try:
            with patch.object(
                self.application, "export_training_data"
            ) as export:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    self.application.analyze_training_data()
                export.assert_not_called()
        finally:
            release.set()
            worker.join(timeout=1.0)

    def test_batch_controller_reports_unexpected_worker_failure(self) -> None:
        self.application.memory.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "controller-failure"},
        )

        class BrokenWorker:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def run_once(self, job_types):
                del job_types
                raise OSError("analysis database became unavailable")

        with patch("lumen_engine.control.OfflineResearchWorker", BrokenWorker):
            self.application.start_research_worker()
            thread = self.application._research_worker_thread
            assert thread is not None
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        last = self.application._research_worker_last
        assert last is not None
        self.assertEqual(last["failed"], 1)
        self.assertIn("database became unavailable", last["error"])
        status = self.application.research_status()
        self.assertFalse(status["worker"]["running"])
        self.assertEqual(status["worker"]["last_result"]["error"], last["error"])
        self.assertTrue(
            any(
                "stopped unexpectedly" in event["message"]
                for event in self.application.events
            )
        )

    def test_application_startup_reports_and_requeues_interrupted_job(
        self,
    ) -> None:
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "interrupted-recording"},
        )
        claimed = self.application.memory.claim_analysis_job(
            worker_id="lost-worker", worker_pid=999_999_999
        )
        assert claimed is not None
        root = Path(self.temporary.name)
        restarted = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "memory.sqlite3",
            settings_path=root / "restarted-settings.json",
        )
        try:
            status = restarted.research_status()
            recovered = status["worker"]["recovered_jobs"]
            self.assertEqual([item["job_id"] for item in recovered], [job_id])
            job = {
                item["id"]: item
                for item in restarted.memory.list_analysis_jobs()
            }[job_id]
            self.assertEqual(job["status"], "queued")
            self.assertTrue(
                any(
                    "ready to resume" in event["message"]
                    for event in restarted.events
                )
            )
        finally:
            restarted.close()

    def test_custom_memory_files_isolate_models_and_training_state(
        self,
    ) -> None:
        root = Path(self.temporary.name)
        other = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "second-memory.sqlite3",
            settings_path=root / "second-settings.json",
        )
        try:
            self.assertNotEqual(
                self.application.training_root, other.training_root
            )
            self.assertNotEqual(
                self.application._choreography_model_path,
                other._choreography_model_path,
            )
            self.assertIn(
                self.application.memory_path.stem,
                self.application.training_root.parent.name,
            )
        finally:
            other.close()

    def test_application_close_joins_performance_trace_worker(self) -> None:
        root = Path(self.temporary.name)
        other = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "close-memory.sqlite3",
            settings_path=root / "close-settings.json",
        )
        trace_thread = other._trace_thread
        model_thread = other._model_save_thread
        feedback_thread = other._feedback_refresh_thread

        other.close()

        self.assertFalse(trace_thread.is_alive())
        self.assertFalse(model_thread.is_alive())
        self.assertFalse(feedback_thread.is_alive())

    def test_choreography_persistence_coalesces_listener_burst(self) -> None:
        root = Path(self.temporary.name)
        other = LumenApplication(
            rig_path=self.rig_path,
            memory_path=root / "coalesce-memory.sqlite3",
            settings_path=root / "coalesce-settings.json",
        )
        try:
            with patch.object(
                other,
                "_write_choreography_model",
                wraps=other._write_choreography_model,
            ) as writer:
                for _ in range(8):
                    other._save_choreography_model()
                with other._model_save_condition:
                    completed = other._model_save_condition.wait_for(
                        lambda: (
                            other._model_save_completed
                            >= other._model_save_requested
                        ),
                        timeout=2.0,
                    )
                self.assertTrue(completed)
                self.assertEqual(writer.call_count, 1)
                saved = json.loads(
                    other._choreography_model_path.read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    saved["format"], "lumen_sequence_preference_model"
                )
        finally:
            other.close()

    def test_research_status_publishes_captured_audio_preparation_stage(self):
        class AlivePreparation:
            @staticmethod
            def is_alive():
                return True

        now_ms = int(time.time() * 1000)
        self.application._training_prepare_thread = AlivePreparation()
        self.application._training_prepare_status.update({
            "running": True,
            "session_id": "session:test",
            "stage": "indexing_capture",
            "progress": 0.20,
            "started_unix_ms": now_ms - 5_000,
            "updated_unix_ms": now_ms,
            "detail": "Verifying audio continuity.",
        })
        try:
            preparation = self.application.research_status(
                wait_for_readiness=False
            )["preparation"]
        finally:
            self.application._training_prepare_thread = None

        self.assertTrue(preparation["running"])
        self.assertEqual(preparation["stage"], "indexing_capture")
        self.assertEqual(preparation["progress"], 0.20)
        self.assertEqual(preparation["detail"], "Verifying audio continuity.")
        self.assertEqual(preparation["started_unix_ms"], now_ms - 5_000)

    def test_student_snapshot_preparation_publishes_live_progress(self):
        observed = []

        def prepare(*args, progress_callback=None, **kwargs):
            del args, kwargs
            assert progress_callback is not None
            progress_callback(
                "materializing_snapshot",
                0.5,
                "Materialized 10 of 20 song groups.",
            )
            observed.append(
                self.application.research_status(
                    wait_for_readiness=False
                )["student_preparation"]
            )
            return {
                "job_id": "job:student-progress",
                "examples": 123,
                "teacher_run_ids": ["run:one"],
            }

        with patch(
            "lumen_engine.control.enqueue_student_training",
            side_effect=prepare,
        ), patch.object(
            self.application.lumen_link, "route_queued_jobs"
        ), patch.object(
            self.application.lumen_link,
            "ready_for_offload",
            return_value=True,
        ), patch.object(self.application.lumen_link, "start"):
            result = self.application.train_structure_student({"epochs": 30})

        self.assertEqual(observed[0]["stage"], "materializing_snapshot")
        self.assertEqual(observed[0]["progress"], 0.5)
        self.assertTrue(observed[0]["running"])
        final = result["research"]["student_preparation"]
        self.assertEqual(final["stage"], "queued")
        self.assertEqual(final["outcome"], "queued")
        self.assertEqual(final["progress"], 1.0)
        self.assertFalse(final["running"])

    def test_student_snapshot_preparation_is_single_flight(self):
        self.application._student_training_prepare_status["running"] = True

        with self.assertRaisesRegex(
            RuntimeError, "snapshot is already being prepared"
        ):
            self.application.train_structure_student({"epochs": 30})

    def test_research_status_projects_remote_student_link_stage(self):
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=STUDENT_TRAIN_JOB,
            payload={"execution_target": "threadripper"},
        )
        with patch.object(
            self.application.lumen_link,
            "status",
            return_value={
                "jobs": [{
                    "job": {"id": job_id, "job_type": STUDENT_TRAIN_JOB},
                    "stage": "student_local_validation",
                    "progress": 0.45,
                    "remote": {
                        "resources": {
                            "elapsed_s": 120.0,
                            "rss_bytes": 2_000_000_000,
                        }
                    },
                }]
            },
        ):
            student_link = self.application.research_status(
                wait_for_readiness=False
            )["student_link"]

        self.assertTrue(student_link["running"])
        self.assertEqual(student_link["job_id"], job_id)
        self.assertEqual(student_link["stage"], "student_local_validation")
        self.assertEqual(student_link["progress"], 0.45)
        self.assertEqual(student_link["resources"]["elapsed_s"], 120.0)

    def test_research_status_reports_manifest_preparation_before_claim(self):
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=STUDENT_TRAIN_JOB,
            payload={"execution_target": "automatic"},
        )
        with patch.object(
            self.application.lumen_link,
            "ready_for_offload",
            return_value=True,
        ), patch.object(
            self.application.lumen_link,
            "status",
            return_value={"jobs": []},
        ):
            student_link = self.application.research_status(
                wait_for_readiness=False
            )["student_link"]

        self.assertTrue(student_link["running"])
        self.assertEqual(student_link["job_id"], job_id)
        self.assertEqual(student_link["stage"], "preparing_manifest")
        self.assertIsNone(student_link["progress"])

    def test_live_listener_burst_coalesces_feedback_bias_rebuild(self) -> None:
        self.application.engine_mode = "live"
        with patch.object(
            self.application,
            "_rebuild_feedback_biases",
            wraps=self.application._rebuild_feedback_biases,
        ) as rebuild:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda index: self.application.add_feedback({
                        "label": "increase_movement",
                        "value": 1,
                        "scope": "overall",
                        "participant_id": f"listener-{index}",
                        "client_event_id": f"listener-tap-{index}",
                    }),
                    range(8),
                ))
            with self.application._feedback_refresh_condition:
                completed = (
                    self.application._feedback_refresh_condition.wait_for(
                        lambda: (
                            self.application._feedback_refresh_completed
                            >= self.application._feedback_refresh_requested
                        ),
                        timeout=2.0,
                    )
                )
            self.assertTrue(completed)
            self.assertEqual(rebuild.call_count, 1)
            self.assertEqual(
                max(row["participant_agreement"] for row in results), 8
            )
            self.assertGreater(
                self.application._feedback_biases["overall"]["motion"],
                0.0,
            )

    def test_feedback_undo_accepts_ui_null_participant_scope(self) -> None:
        created = self.application.add_feedback({
            "label": "great_timing",
            "value": 1,
            "scope": "overall",
            "participant_id": "listener-null-undo",
            "client_event_id": "listener-null-undo-event",
        })
        removed = self.application.delete_feedback({
            "feedback_id": created["feedback_id"],
            "participant_id": None,
        })
        self.assertTrue(removed["deleted"])

    def test_feedback_persistence_never_holds_audio_publication_lock(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = self.application.memory.add_feedback_event

        def delayed(feedback, **kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return original(feedback, **kwargs)

        with patch.object(
            self.application.memory, "add_feedback_event", side_effect=delayed
        ):
            worker = threading.Thread(
                target=self.application.add_feedback,
                args=({"label": "more_like_this", "value": 1.0},),
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            started = time.monotonic()
            with self.application._lock:
                self.application._status_sequence += 1
            elapsed = time.monotonic() - started
            release.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 0.05)

    def test_bootstrap_database_work_never_holds_live_publication_lock(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = self.application.memory.summary

        def delayed(*args, **kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return original(*args, **kwargs)

        with patch.object(
            self.application.memory, "summary", side_effect=delayed
        ):
            worker = threading.Thread(target=self.application.bootstrap)
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            started = time.monotonic()
            with self.application._lock:
                self.application._status_sequence += 1
            elapsed = time.monotonic() - started
            release.set()
            worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 0.05)

    def test_bootstrap_never_runs_exact_training_readiness_inline(self) -> None:
        self.application._research_readiness_cache = None
        with patch.object(
            self.application,
            "_schedule_research_readiness_refresh",
            return_value=True,
        ), patch("lumen_engine.control.training_readiness") as readiness:
            payload = self.application.bootstrap()
        readiness.assert_not_called()
        self.assertTrue(payload["research"]["readiness_cache"]["refreshing"])
        self.assertFalse(payload["research"]["training"]["train_ready"])
        self.assertIn(
            "refreshing",
            payload["research"]["training"]["blockers"][0].lower(),
        )

    def test_large_readiness_refresh_isolated_in_subprocess(self) -> None:
        large_database = Path(self.temporary.name) / "large-memory.sqlite3"
        large_database.touch()
        os.truncate(large_database, 257 * 1024 * 1024)
        original_memory_path = self.application.memory_path
        self.application.memory_path = large_database

        class Completed:
            returncode = 0
            stderr = ""
            stdout = json.dumps(
                {"training": {"train_ready": True, "model": {}}}
            )

        try:
            with patch(
                "lumen_engine.control.subprocess.run",
                return_value=Completed(),
            ) as run, patch(
                "lumen_engine.control.training_readiness"
            ) as inline_readiness:
                self.application._refresh_research_readiness_cache()
            inline_readiness.assert_not_called()
            run.assert_called_once()
            self.assertTrue(
                self.application._research_readiness_cache["training"][
                    "train_ready"
                ]
            )
        finally:
            self.application.memory_path = original_memory_path

    def test_settings_file_write_never_holds_live_publication_lock(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def delayed(_settings):
            entered.set()
            release.wait(timeout=2.0)

        with patch.object(
            self.application, "_save_settings", side_effect=delayed
        ):
            worker = threading.Thread(
                target=self.application.patch_settings,
                args=({"spotify_client_id": "client-test"},),
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            started = time.monotonic()
            with self.application._lock:
                self.application._status_sequence += 1
            elapsed = time.monotonic() - started
            release.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 0.05)

    def test_trace_snapshot_never_holds_live_publication_lock(self) -> None:
        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        observation = MusicalObservation(
            timestamp_s=5.0,
            loudness=0.6,
            onset_strength=0.4,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            bpm=120.0,
            section="groove",
            section_confidence=0.7,
        )
        frame = runtime.step(observation)
        entered = threading.Event()
        release = threading.Event()
        original = runtime.choreography_snapshot

        def delayed():
            entered.set()
            release.wait(timeout=2.0)
            return original()

        self.application._runtime = runtime
        self.application._last_trace_timestamp = None
        with patch.object(
            runtime, "choreography_snapshot", side_effect=delayed
        ):
            worker = threading.Thread(
                target=self.application._accept_runtime_frame,
                args=(observation, frame),
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            started = time.monotonic()
            with self.application._lock:
                self.application._status_sequence += 1
            elapsed = time.monotonic() - started
            release.set()
            worker.join(timeout=2.0)
        self.application._runtime = None
        runtime.close()
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 0.05)

    def test_trace_materialization_uses_the_snapshot_paired_with_its_frame(
        self,
    ) -> None:
        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        observation = MusicalObservation(
            timestamp_s=5.0,
            loudness=0.6,
            onset_strength=0.4,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            bpm=120.0,
            section="groove",
            section_confidence=0.7,
        )
        frame = runtime.step(observation)
        exact_snapshot = runtime.choreography_snapshot()
        item = {
            "_kind": "performance_seed",
            "session_id": "trace-exact",
            "song_id": None,
            "position_ms": None,
            "frame": frame,
            "observation": observation,
            "raw_observation": observation,
            "audio_metrics": None,
            "controls": replace(self.application.controls),
            "structure_model": None,
            "structure_resolution": {
                "source": "test", "section": "groove"
            },
            "choreography_snapshot": exact_snapshot,
        }
        # If materialization consulted live runtime here, this mutation would
        # make the stored planner state differ from the queued frame.
        runtime.notify_timeline_discontinuity()
        runtime.step(replace(observation, timestamp_s=9.0, section="drop"))
        materialized = self.application._materialize_trace_item(item)
        self.assertEqual(
            materialized["payload"]["choreography_runtime"],
            exact_snapshot,
        )
        runtime.close()

    def test_new_runtime_adopts_prepared_recall_when_ids_are_unchanged(
        self,
    ) -> None:
        recalled = ChoreographySequence(
            sequence_id="same-track-restart",
            source="operator_song_timeline",
            steps=(ChoreographyStep(
                start_beat=0.0,
                duration_beats=8.0,
                fixture_scope="movers",
                routine="figure_eight",
            ),),
        )
        self.application._prepared_recalled_choreography = (recalled,)
        self.application._prepared_recalled_ids = ("same-track-v1",)
        # Simulate IDs retained from the runtime that was just stopped.
        self.application._recalled_choreography_ids = ("same-track-v1",)
        previous_runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        self.application._recalled_choreography_runtime = previous_runtime
        new_runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        try:
            self.application._refresh_recalled_choreography(
                new_runtime, self.application.observation
            )
            self.assertEqual(
                new_runtime._recalled_choreography, (recalled,)
            )
        finally:
            previous_runtime.close()
            new_runtime.close()

    def test_status_render_never_holds_live_publication_lock(self) -> None:
        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        observation = MusicalObservation(
            timestamp_s=6.0,
            loudness=0.6,
            onset_strength=0.4,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            bpm=120.0,
            section="groove",
            section_confidence=0.7,
        )
        frame = runtime.step(observation)
        entered = threading.Event()
        release = threading.Event()

        def delayed_status_component():
            entered.set()
            release.wait(timeout=2.0)
            return {"scope": "movers", "groups": {}}

        with patch.object(
            self.application,
            "_motion_editor_snapshot",
            side_effect=delayed_status_component,
        ):
            worker = threading.Thread(target=self.application.snapshot)
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            started = time.monotonic()
            self.application._accept_runtime_frame(
                observation,
                frame,
                audio_metrics=AudioInputMetrics.silence(
                    timestamp_s=observation.timestamp_s
                ),
            )
            elapsed = time.monotonic() - started
            release.set()
            worker.join(timeout=2.0)
        runtime.close()
        self.assertFalse(worker.is_alive())
        self.assertLess(elapsed, 0.05)

    def test_status_distinguishes_capture_from_pipeline_stall(self) -> None:
        class FreshCapture:
            @property
            def diagnostics(self):
                return {
                    "reader_alive": True,
                    "packets_read": 50,
                    "source_frames": 102_400,
                    "queue_depth": 4,
                    "last_packet_age_ms": 12.0,
                }

        self.application.engine_mode = "monitor"
        self.application._thread = threading.current_thread()
        self.application._audio_packets = 2
        self.application._audio_last_packet_at = time.monotonic() - 1.0
        self.application._active_audio_capture = FreshCapture()
        status = self.application._audio_snapshot_unlocked(True)
        self.application._active_audio_capture = None
        self.application._thread = None
        self.assertEqual(status["state"], "pipeline_stale")
        self.assertEqual(status["label"], "ANALYSIS PIPELINE STALLED")
        self.assertEqual(status["last_source_packet_age_ms"], 12.0)
        self.assertGreater(status["last_processed_frame_age_ms"], 750.0)

    def test_semantic_capture_contains_nonzero_student_contract(self) -> None:
        observation = MusicalObservation(
            timestamp_s=2.0,
            loudness=0.5,
            onset_strength=0.4,
            low_energy=0.2,
            mid_energy=0.4,
            high_energy=0.8,
            beat_confidence=0.7,
            bpm=124.0,
            section="groove",
            section_confidence=0.8,
            novelty=0.6,
        )
        metrics = AudioInputMetrics(
            timestamp_s=2.0,
            frame_count=100,
            rms=0.2,
            dbfs=-14.0,
            peak=1.0,
            channel_rms=(0.2, 0.2),
            channel_peak=(1.0, 0.9),
            clipped_samples=4,
            waveform=(0.0,),
        )
        payload = self.application._semantic_audio_payload(
            observation, metrics
        )
        self.assertEqual(payload["observation"]["spectral_flux"], 0.6)
        self.assertGreater(
            payload["observation"]["spectral_brightness"], 0.5
        )
        self.assertEqual(
            payload["observation"]["tempo_confidence"], 0.7
        )
        self.assertAlmostEqual(payload["audio_metrics"]["clipping"], 0.02)

    def test_default_operator_bias_preserves_audio_dynamics(self) -> None:
        engine = OperatorExpressionEngine(OperatorControls(), ExpressionPolicy())
        quiet = None
        for index in range(24):
            quiet = engine.decide(
                MusicalObservation(
                    timestamp_s=index * 0.1,
                    loudness=0.05,
                    onset_strength=0.02,
                    low_energy=0.3,
                    mid_energy=0.3,
                    high_energy=0.2,
                    section="breakdown",
                    section_confidence=0.8,
                )
            )
        assert quiet is not None
        loud = None
        for index in range(24, 64):
            loud = engine.decide(
                MusicalObservation(
                    timestamp_s=index * 0.1,
                    loudness=0.9,
                    onset_strength=0.7,
                    low_energy=0.6,
                    mid_energy=0.5,
                    high_energy=0.4,
                    beat_pulse=0.8,
                    beat_confidence=0.8,
                    section="groove",
                    section_confidence=0.8,
                )
            )
        assert loud is not None
        self.assertGreater(loud.expression.energy - quiet.expression.energy, 0.35)
        self.assertGreater(loud.expression.motion - quiet.expression.motion, 0.20)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.rig_path = root / "rig.json"
        self.rig_path.write_text(
            Path("config/example-rig.json").read_text(encoding="utf-8"),
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

    def test_obsolete_songformer_timeline_cannot_be_approved_for_live_recall(
        self,
    ) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:diagnostic-songformer",
            title="Diagnostic teacher",
            duration_ms=60_000,
            observed_position_ms=10_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        song_id = self.application.memory.remember_media(media)
        recording_id = self.application.memory.remember_recording_version(
            provider=media.provider,
            provider_item_id=media.provider_item_id,
            duration_ms=media.duration_ms,
            song_id=song_id,
        )
        run_id = self.application.memory.begin_teacher_run(
            teacher_name="SongFormer",
            teacher_version="diagnostic",
            device="cpu",
            preprocessing_version="diagnostic",
            recording_id=recording_id,
        )
        timeline_id = self.application.memory.save_structure_timeline(
            provenance="songformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.9,
            recording_id=recording_id,
            teacher_run_id=run_id,
            segments=[{
                "start_ms": 0,
                "end_ms": 60_000,
                "functional_label": "chorus",
            }],
        )
        self.application.memory.finish_teacher_run(run_id, status="complete")
        self.application.media = media
        self.application.song_id = song_id

        with self.assertRaisesRegex(
            ValueError, "diagnostic timeline cannot be approved"
        ):
            self.application.review_structure_timeline({
                "timeline_id": timeline_id,
                "status": "approved",
            })

        self.assertIsNone(
            self.application.memory.structure_timeline_review(timeline_id)
        )
        self.assertIsNone(self.application.memory.cached_structure_at(
            recording_id=recording_id,
            playback_position_ms=10_000,
        ))

    def test_current_songformer_function_can_be_approved_for_recall(
        self,
    ) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:current-songformer",
            title="Current functional teacher",
            duration_ms=60_000,
            observed_position_ms=10_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        song_id = self.application.memory.remember_media(media)
        recording_id = self.application.memory.remember_recording_version(
            provider=media.provider,
            provider_item_id=media.provider_item_id,
            duration_ms=media.duration_ms,
            song_id=song_id,
        )
        run_id = self.application.memory.begin_teacher_run(
            teacher_name="SongFormer",
            teacher_version="current",
            device="cpu",
            preprocessing_version=(
                "songformer_official_features_cpu_windowed_v1:60s:"
                f"{TEACHER_NORMALIZATION_VERSION}"
            ),
            recording_id=recording_id,
        )
        timeline_id = self.application.memory.save_structure_timeline(
            provenance="songformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.0,
            recording_id=recording_id,
            teacher_run_id=run_id,
            segments=[{
                "start_ms": 0,
                "end_ms": 60_000,
                "functional_label": "chorus",
                "content_label": "instrumental",
            }],
        )
        self.application.memory.finish_teacher_run(run_id, status="complete")
        self.application.media = media
        self.application.song_id = song_id

        result = self.application.review_structure_timeline({
            "timeline_id": timeline_id,
            "status": "approved",
        })

        self.assertEqual(result["review"]["status"], "approved")
        cached = self.application.memory.cached_structure_at(
            recording_id=recording_id,
            playback_position_ms=10_000,
        )
        assert cached is not None
        self.assertEqual(cached["axes"]["functional"]["label"], "chorus")
        self.assertEqual(
            cached["axes"]["functional"]["teacher"]["name"],
            "SongFormer",
        )

    def test_timeline_library_can_review_song_without_active_playback(self) -> None:
        recording_id = self.application.memory.remember_recording_version(
            provider="spotify",
            provider_item_id="spotify:track:offline-review",
            duration_ms=90_000,
            metadata={
                "track_identity": {
                    "title": "Offline Review",
                    "artists": ["Lumen Test"],
                }
            },
        )
        run_id = self.application.memory.begin_teacher_run(
            teacher_name="EDMFormer",
            teacher_version="test",
            device="cpu",
            preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
            recording_id=recording_id,
        )
        timeline_id = self.application.memory.save_structure_timeline(
            provenance="edmformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.0,
            recording_id=recording_id,
            teacher_run_id=run_id,
            segments=[{
                "start_ms": 0,
                "end_ms": 90_000,
                "energy_label": "build",
                "raw_label": "Buildup",
            }],
        )
        self.application.memory.finish_teacher_run(run_id, status="complete")
        self.assertIsNone(self.application.media)

        library = self.application.structure_training_library()
        self.assertTrue(library["composite_review_supported"])
        self.assertEqual(library["recordings"], 1)
        self.assertEqual(library["needs_review"], 1)
        self.assertEqual(library["selected_recording_id"], recording_id)
        self.assertEqual(
            library["selected_recording"]["training_status"],
            "ready_for_next_training",
        )
        self.assertEqual(
            library["structure_timelines"][0]["id"], timeline_id
        )

        result = self.application.review_structure_timeline({
            "timeline_id": timeline_id,
            "recording_id": recording_id,
            "status": "approved",
            "participant_id": "desktop-owner",
        })
        self.assertEqual(result["recording_id"], recording_id)
        self.assertIsNone(result["cached_structure"])
        updated = self.application.structure_training_library(recording_id)
        self.assertEqual(updated["needs_review"], 0)
        self.assertEqual(
            updated["selected_recording"]["review_status"], "approved"
        )
        corrected = self.application.correct_structure_timeline({
            "base_timeline_id": timeline_id,
            "recording_id": recording_id,
            "participant_id": "desktop-owner",
            "note": "The build is actually the drop.",
            "segments": [{
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 90_000,
                "energy_label": "drop",
            }],
        })
        self.assertEqual(corrected["recording_id"], recording_id)
        self.assertIsNone(corrected["cached_structure"])
        corrected_library = self.application.structure_training_library(
            recording_id
        )
        self.assertEqual(
            corrected_library["selected_recording"]["review_status"],
            "corrected",
        )

    def test_demo_drives_operator_status_without_hardware(self) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        status = self.application.snapshot()
        while status["decision"] is None and time.monotonic() < deadline:
            time.sleep(0.03)
            status = self.application.snapshot()
        self.assertEqual(status["engine"]["mode"], "demo")
        self.assertEqual(status["output"]["backend"], "Virtual DMX")
        self.assertEqual(status["audio"]["state"], "simulated")
        self.assertEqual(status["audio"]["packets_received"], 0)
        self.assertIsNotNone(status["decision"])
        self.assertGreater(len(status["dmx"]["active_channels"]), 0)
        self.assertEqual(self.application.memory.summary()["totals"]["decisions"], 0)
        self.application.stop()

    def test_rejected_student_is_reported_as_quarantined_not_runtime_error(
        self,
    ) -> None:
        path = self.application._student_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        model = StreamingStructureStudent(hidden_size=8, seed=41)
        model.approved_axes = set()
        model.save(path)
        path.with_name(path.stem + ".evaluation.json").write_text(
            json.dumps(
                {
                    "activated": False,
                    "activation_gate_version": (
                        STUDENT_ACTIVATION_GATE_VERSION
                    ),
                    "teacher_normalization_version": (
                        TEACHER_NORMALIZATION_VERSION
                    ),
                    "edmformer_preprocessing_version": (
                        EDMFORMER_PREPROCESSING_VERSION
                    ),
                    "axis_gate_reasons": {
                        "energy": ["did not beat held-out baseline"]
                    },
                }
            ),
            encoding="utf-8",
        )

        self.application._load_student_model()
        status = self.application.snapshot()["structure_model"]

        self.assertEqual(status["state"], "quarantined")
        self.assertIsNone(status["error"])
        self.assertEqual(
            status["gate_reasons"],
            ["did not beat held-out baseline"],
        )
        self.assertIsNone(self.application._student_model)

    def test_rehearsal_controls_validate_and_clamp(self) -> None:
        controls = RehearsalControls()
        self.assertEqual(controls.palette, "pure_blue")
        controls.patch({
            "routine": "fan_sweep", "scope": "center", "output": "virtual",
            "bpm": 900, "size": -2, "intensity": 2, "strobe": 0.4,
            "movers_strobe_dmx": 900, "center_strobe_dmx": -12,
        })
        self.assertEqual(controls.routine, "fan_sweep")
        self.assertEqual(controls.scope, "center")
        self.assertEqual(controls.bpm, 240.0)
        self.assertEqual(controls.size, 0.0)
        self.assertEqual(controls.intensity, 1.0)
        self.assertEqual(controls.movers_strobe_dmx, 255)
        self.assertEqual(controls.center_strobe_dmx, 0)
        with self.assertRaises(ValueError):
            controls.patch({"routine": "not-a-routine"})

    def test_settled_remote_strobe_is_direct_output_and_contextual_feedback(
        self,
    ) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        while (
            self.application.snapshot()["decision"] is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        interaction_ms = round(time.time() * 1000) - 700
        result = self.application.patch_strobe_control({
            "group": "movers",
            "value": 137,
            "settled": True,
            "lifetime": "cue",
            "participant_id": "owner-test",
            "client_event_id": "strobe-test-137",
            "interaction_unix_ms": interaction_ms,
        })
        self.assertEqual(result["value"], 137)
        self.assertEqual(result["strobe_controls"]["movers"], 137)
        self.assertEqual(result["feedback"]["target_strobe_dmx"], 137)
        self.assertGreaterEqual(result["feedback"]["timing_offset_ms"], 650)
        feedback = self.application.memory.list_feedback(
            result["feedback"]["song_id"]
        )[-1]
        self.assertEqual(feedback["label"], "strobe_level")
        self.assertEqual(feedback["scope"], "group")
        self.assertEqual(feedback["fixture_id"], "movers")
        self.assertEqual(feedback["lane_context"]["target_strobe_dmx"], 137)
        self.assertLessEqual(
            feedback["lane_context"]["interaction_offset_ms"], 1000
        )
        self.application.stop()

    def test_live_performance_defaults_to_automatic_palette(self) -> None:
        self.assertEqual(OperatorControls().palette, "auto")

    def test_interpolated_control_ticks_do_not_create_input_dropouts(self) -> None:
        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        self.application.engine_mode = "live"
        first = MusicalObservation(
            timestamp_s=1.0, loudness=0.5, onset_strength=0.2,
            low_energy=0.4, mid_energy=0.3, high_energy=0.2,
            bpm=120.0, section="groove", section_confidence=0.7,
        )
        metrics = replace(
            AudioInputMetrics.silence(timestamp_s=1.0),
            frame_count=2048, rms=0.12, dbfs=-18.4, peak=0.3,
        )
        self.application._accept_runtime_frame(
            first, runtime.step(first), audio_metrics=metrics,
        )
        interpolated = replace(first, timestamp_s=1.12, loudness=0.52)
        self.application._accept_runtime_frame(
            interpolated, runtime.step(interpolated),
        )
        history = list(self.application._analysis_history)
        runtime.close()
        self.assertEqual(len(history), 2)
        self.assertEqual([item["dbfs"] for item in history], [-18.4, -18.4])
        self.assertTrue(history[0]["input_fresh"])
        self.assertFalse(history[1]["input_fresh"])
        self.assertAlmostEqual(history[1]["input_age_ms"], 120.0)

    def test_analysis_history_is_a_true_twenty_four_second_window(self) -> None:
        runtime = self.application._runtime_for_rig(
            GatedOutput(VirtualDMXOutput(), self.application.controls)
        )
        self.application.engine_mode = "live"
        first = MusicalObservation(
            timestamp_s=5.0, loudness=0.5, onset_strength=0.2,
            low_energy=0.4, mid_energy=0.3, high_energy=0.2,
            bpm=120.0, section="groove", section_confidence=0.7,
        )
        metrics = replace(
            AudioInputMetrics.silence(timestamp_s=5.0),
            frame_count=2048, rms=0.12, dbfs=-18.4, peak=0.3,
        )
        rendered = runtime.step(first)
        for index in range(241):
            observation = replace(first, timestamp_s=5.0 + index * 0.1)
            self.application._accept_runtime_frame(
                observation,
                rendered,
                audio_metrics=metrics if index == 0 else None,
            )
        history = list(self.application._analysis_history)
        runtime.close()
        self.assertEqual(len(history), 240)
        self.assertAlmostEqual(
            history[-1]["timestamp_s"] - history[0]["timestamp_s"],
            23.9,
            places=6,
        )
        self.assertTrue(all(item["dbfs"] == -18.4 for item in history))

    def test_live_show_clock_coalesces_analyzer_burst_to_newest_state(self) -> None:
        control_queue: queue.Queue[_AnalyzedControlFrame] = queue.Queue(16)
        analysis_finished = threading.Event()
        errors: list[BaseException] = []
        metrics = replace(
            AudioInputMetrics.silence(timestamp_s=1.0),
            frame_count=2048, rms=0.12, dbfs=-18.4, peak=0.3,
        )
        for index in range(6):
            observation = MusicalObservation(
                timestamp_s=1.0 + index * 0.04,
                loudness=0.2 + index * 0.1,
                onset_strength=0.2,
                low_energy=0.4,
                mid_energy=0.3,
                high_energy=0.2,
                bpm=120.0,
                section="groove",
            )
            control_queue.put(_AnalyzedControlFrame(
                observation=observation,
                raw_observation=observation,
                audio_metrics=replace(
                    metrics, timestamp_s=observation.timestamp_s
                ),
                audio_bytes=4096,
                training_audio_frame=index * 2048,
                runtime_context={},
                analysis_started_perf_s=time.perf_counter(),
                analysis_stages_ms={},
            ))
        analysis_finished.set()

        class RuntimeProbe:
            def __init__(self) -> None:
                self.observations: list[MusicalObservation] = []

            def set_structure_context(self, **_values) -> None:
                return

            def step(self, observation: MusicalObservation):
                self.observations.append(observation)
                return object()

            def notify_audio_discontinuity(self) -> None:
                return

            def notify_timeline_discontinuity(self) -> None:
                return

        runtime = RuntimeProbe()
        with (
            patch.object(self.application, "_accept_runtime_frame") as accept,
            patch.object(self.application, "_refresh_recalled_choreography"),
            patch.object(self.application, "_record_live_pipeline_timing"),
        ):
            self.application._run_live_control_clock(
                runtime,  # type: ignore[arg-type]
                control_queue,
                analysis_finished,
                errors,
                AudioCaptureConfig(),
            )
        self.assertEqual(errors, [])
        self.assertEqual(len(runtime.observations), 1)
        self.assertAlmostEqual(runtime.observations[0].loudness, 0.7)
        self.assertEqual(accept.call_args.kwargs["audio_packet_count"], 6)
        self.assertEqual(control_queue.unfinished_tasks, 0)

    def test_pcm_rate_uses_source_packets_not_coalesced_show_updates(self) -> None:
        self.application.engine_mode = "live"
        self.application._audio_capture_diagnostics = {
            "packets_read": 240,
            "source_frames": 240 * 2048,
            "last_packet_age_ms": 5.0,
        }
        self.application._audio_packets = 80
        status = self.application._audio_snapshot_unlocked(True)
        self.assertAlmostEqual(
            status["packet_rate_hz"], 48_000 / 2048, places=6
        )

    def test_gesture_movement_editor_persists_live_associations(self) -> None:
        result = self.application.patch_settings({
            "gesture_movements": {
                "hold": ["breathe"],
                "breathe": ["breathe"],
                "converge": ["fan_sweep"],
                "expand": ["figure_eight"],
                "sweep": ["counter_rotate"],
                "pulse": ["beat_nod", "opposing_chase"],
                "release": ["counter_rotate"],
            }
        })
        self.assertEqual(
            result["settings"]["gesture_movements"]["sweep"],
            ["counter_rotate"],
        )
        self.assertEqual(
            self.application.gesture_movements["release"],
            ("counter_rotate",),
        )
        saved = json.loads(self.application.settings_path.read_text())
        self.assertEqual(saved["gesture_movements"]["pulse"], ["beat_nod", "opposing_chase"])

    def test_motion_editor_persists_and_updates_runtime(self) -> None:
        status = self.application.patch_motion_routine({
            "routine": "figure_eight",
            "values": {
                "cycle_beats": 20,
                "pan_size": 0.88,
                "relationship": "counter",
            },
        })
        editor = status["rehearsal"]["motion_editor"]
        self.assertEqual(editor["values"]["cycle_beats"], 20.0)
        self.assertEqual(editor["values"]["relationship"], "counter")
        self.assertTrue(editor["modified"])
        center = editor["groups"]["center"]
        self.assertTrue(center["mechanics"]["base_fixed"])
        self.assertEqual(center["mechanics"]["center_rotation_deg"], 300.0)
        self.assertEqual(center["mechanics"]["pod_rotation_deg"], 180.0)
        self.assertEqual(center["mechanics"]["pod_count"], 2)
        self.assertEqual(center["mechanics"]["mount_orientation"], "ceiling_down")
        self.assertEqual(center["mechanics"]["housing_rotation_deg"][0], 180.0)
        self.assertTrue(self.application.motion_path.is_file())
        restored = self.application._load_motion_tunings()
        self.assertEqual(restored.movers["figure_eight"].pan_size, 0.88)
        reset = self.application.patch_motion_routine({
            "routine": "figure_eight", "action": "reset",
        })
        self.assertFalse(reset["rehearsal"]["motion_editor"]["modified"])

    def test_rehearsal_uses_virtual_output_and_generated_clock(self) -> None:
        self.application.patch_rehearsal({
            "routine": "counter_rotate", "scope": "overall", "bpm": 108,
        })
        self.application.start("rehearsal")
        deadline = time.monotonic() + 2.0
        status = self.application.snapshot()
        while status["decision"] is None and time.monotonic() < deadline:
            time.sleep(0.03)
            status = self.application.snapshot()
        self.assertEqual(status["engine"]["mode"], "rehearsal")
        self.assertEqual(status["output"]["backend"], "Virtual DMX")
        self.assertEqual(status["audio"]["label"], "REHEARSAL — GENERATED CLOCK")
        self.assertEqual(status["decision"]["routine"], "counter_rotate")
        self.assertEqual(status["rehearsal"]["bpm"], 108.0)
        self.assertEqual(len(status["rehearsal"]["routines"]), 6)
        self.application.stop()

    def test_running_rehearsal_rejects_output_switch_atomically(self) -> None:
        self.application.start("rehearsal")
        deadline = time.monotonic() + 1.0
        while not self.application.snapshot()["engine"]["running"] and time.monotonic() < deadline:
            time.sleep(0.02)
        before = self.application.snapshot()["rehearsal"]
        with self.assertRaises(RuntimeError):
            self.application.patch_rehearsal({
                "output": "live", "routine": "breathe", "bpm": 80,
            })
        after = self.application.snapshot()["rehearsal"]
        self.assertEqual(after["output"], before["output"])
        self.assertEqual(after["routine"], before["routine"])
        self.assertEqual(after["bpm"], before["bpm"])
        self.application.stop()

    def test_wrong_look_rejects_active_routine_and_gesture(self) -> None:
        self.assertEqual(
            self.application._feedback_routine_effect("wrong_look", "fan_sweep"),
            {"fan_sweep": -0.50},
        )
        self.assertEqual(
            self.application._feedback_gesture_effect("wrong_look", "sweep"),
            {"sweep": -0.42},
        )

    def test_rehearsal_observation_is_exactly_beat_aligned(self) -> None:
        observation = _rehearsal_observation(1.0, 120.0, 0.6)
        self.assertEqual(observation.beat_phase, 0.0)
        self.assertEqual(observation.bar_phase, 0.5)
        self.assertEqual(observation.beat_pulse, 1.0)
        self.assertEqual(observation.beat_confidence, 1.0)

    def test_trained_student_can_replace_heuristic_section_with_provenance(
        self,
    ) -> None:
        model = StreamingStructureStudent()
        for axis in LABELS:
            model.head_weights[axis].fill(0.0)
            model.head_bias[axis].fill(-8.0)
        model.head_bias["energy"][LABELS["energy"].index("build")] = 8.0
        model.head_bias["functional"][
            LABELS["functional"].index("intro")
        ] = 8.0
        model.head_bias["content"][
            LABELS["content"].index("instrumental")
        ] = 8.0
        model.training_examples = 100
        self.application._student_model = model
        result = self.application._apply_student_structure(
            MusicalObservation(
                timestamp_s=1.0,
                loudness=0.5,
                onset_strength=0.3,
                low_energy=0.4,
                mid_energy=0.4,
                high_energy=0.2,
                section="groove",
                section_confidence=0.4,
            ),
            AudioInputMetrics.silence(),
        )
        self.assertEqual(result.section, "build")
        status = self.application.snapshot()["structure_model"]
        self.assertEqual(
            status["prediction"]["selected_axis"], "student_energy"
        )
        self.assertEqual(
            status["prediction"]["target_provenance"],
            "lumen_streaming_structure_student",
        )
        self.assertEqual(model._last_timestamp_s, 1.0)
        self.assertTrue(status["prediction"]["accepted_axes"]["energy"])

    def test_approved_student_boundary_reaches_the_live_resolver(self) -> None:
        model = StreamingStructureStudent(hidden_size=8, seed=19)
        model.boundary_weights.fill(0.0)
        model.boundary_residual_weights.fill(0.0)
        model.boundary_bias = 20.0
        model.approved_axes = {"boundary"}
        self.application._student_model = model
        raw = MusicalObservation(
            timestamp_s=4.0,
            loudness=0.45,
            onset_strength=0.3,
            low_energy=0.4,
            mid_energy=0.35,
            high_energy=0.25,
            section="groove",
            section_confidence=0.7,
        )
        metrics = AudioInputMetrics(
            timestamp_s=4.0,
            frame_count=2048,
            rms=0.10,
            dbfs=-20.0,
            peak=0.35,
            channel_rms=(0.10, 0.10),
            channel_peak=(0.35, 0.34),
            clipped_samples=0,
            waveform=(0.0,),
        )

        self.application._apply_student_structure(raw, metrics)
        self.application._resolve_structure(raw, metrics)

        prediction = self.application._student_prediction
        self.assertIsNotNone(prediction)
        self.assertTrue(prediction["accepted_axes"]["boundary"])
        boundary = self.application._effective_structure["axes"]["boundary"]
        self.assertEqual(boundary["source"], "streaming_student")
        self.assertGreater(boundary["confidence"], 0.99)

    def test_unapproved_student_energy_head_cannot_control_live(self) -> None:
        model = StreamingStructureStudent()
        for axis in LABELS:
            model.head_weights[axis].fill(0.0)
            model.residual_weights[axis].fill(0.0)
            model.head_bias[axis].fill(-8.0)
        model.head_bias["energy"][LABELS["energy"].index("build")] = 8.0
        model.approved_axes = {"content", "boundary"}
        self.application._student_model = model
        result = self.application._apply_student_structure(
            MusicalObservation(
                timestamp_s=1.0,
                loudness=0.5,
                onset_strength=0.3,
                low_energy=0.4,
                mid_energy=0.4,
                high_energy=0.2,
                section="groove",
                section_confidence=0.7,
            ),
            AudioInputMetrics.silence(),
        )
        prediction = self.application.snapshot()["structure_model"][
            "prediction"
        ]
        self.assertEqual(result.section, "groove")
        self.assertEqual(prediction["selected_axis"], "live_analyzer")
        self.assertFalse(prediction["accepted_axes"]["energy"])
        self.assertNotIn("energy", prediction["approved_axes"])

    def test_new_audio_session_resets_student_and_decoder_state(self) -> None:
        class EmptyCapture:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

            @staticmethod
            def chunks():
                return iter(())

        model = StreamingStructureStudent(hidden_size=8, seed=31)
        self.application._student_model = model
        self.application._apply_student_structure(
            MusicalObservation(
                timestamp_s=12.0,
                loudness=0.7,
                onset_strength=0.5,
                low_energy=0.5,
                mid_energy=0.4,
                high_energy=0.3,
                section="groove",
                section_confidence=0.5,
            ),
            AudioInputMetrics.silence(),
        )
        self.application._student_decoder.update(
            StudentPrediction(
                functional="chorus",
                energy="drop",
                content="instrumental",
                confidence={
                    "functional": 0.9,
                    "energy": 0.9,
                    "content": 0.9,
                },
                probabilities={
                    "functional": {},
                    "energy": {},
                    "content": {},
                },
                boundary_probability=0.8,
            ),
            12.0,
        )
        self.assertTrue(model._started)
        self.assertIsNotNone(self.application._student_prediction)
        self.assertTrue(
            any(
                value is not None
                for value in self.application._student_decoder._selected.values()
            )
        )

        self.application.training_capture_enabled = False
        with (
            patch.object(self.application, "_prepare_dedicated_line_input"),
            patch("lumen_engine.control.AlsaLineIn", return_value=EmptyCapture()),
        ):
            self.application._run_audio(object())

        self.assertFalse(model._started)
        self.assertIsNone(model._last_timestamp_s)
        self.assertIsNone(self.application._student_prediction)
        self.assertTrue(
            all(
                value is None
                for value in self.application._student_decoder._selected.values()
            )
        )

    def test_causal_reset_during_live_silence_preserves_physical_gate(self) -> None:
        self.application._physical_quiet_since_s = 12.0
        self.application._physical_signal_since_s = None
        self.application._physical_silence_active = True

        self.application._reset_student_stream(
            reset_physical_silence=False
        )

        self.assertEqual(self.application._physical_quiet_since_s, 12.0)
        self.assertIsNone(self.application._physical_signal_since_s)
        self.assertTrue(self.application._physical_silence_active)

    def test_weak_student_axes_are_visible_but_cannot_drive_choreography(self) -> None:
        self.application._student_prediction = {
            "functional": "chorus",
            "energy": "drop",
            "content": "vocal",
            "confidence": {
                "functional": 0.59,
                "energy": 0.51,
                "content": 0.54,
            },
            "accepted_axes": {
                "functional": False,
                "energy": False,
                "content": False,
            },
            "boundary_probability": 0.99,
        }

        context = self.application._student_runtime_context()

        self.assertEqual(context["functional"], "unknown")
        self.assertEqual(context["energy"], "unknown")
        self.assertEqual(context["content"], "unknown")
        self.assertEqual(context["confidence"], 0.0)
        self.assertEqual(context["boundary_probability"], 0.0)

    def test_functional_label_does_not_replace_energy_section(self) -> None:
        model = StreamingStructureStudent()
        for axis in LABELS:
            model.head_weights[axis].fill(0.0)
            model.head_bias[axis].fill(-8.0)
        model.head_bias["energy"][LABELS["energy"].index("unknown")] = 8.0
        model.head_bias["functional"][
            LABELS["functional"].index("chorus")
        ] = 8.0
        model.head_bias["content"][
            LABELS["content"].index("vocal")
        ] = 8.0
        self.application._student_model = model

        result = self.application._apply_student_structure(
            MusicalObservation(
                timestamp_s=1.0,
                loudness=0.5,
                onset_strength=0.3,
                low_energy=0.4,
                mid_energy=0.4,
                high_energy=0.2,
                section="groove",
                section_confidence=0.8,
            ),
            AudioInputMetrics.silence(),
        )

        self.assertEqual(result.section, "groove")
        prediction = self.application.snapshot()["structure_model"]["prediction"]
        self.assertTrue(prediction["accepted_axes"]["functional"])
        self.assertEqual(prediction["selected_axis"], "live_analyzer")
        self.assertEqual(
            self.application._student_runtime_context()["functional"],
            "chorus",
        )

    def test_training_export_is_single_flight(self) -> None:
        self.application._training_export_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                self.application.export_training_data()
        finally:
            self.application._training_export_lock.release()

    def test_engine_start_waits_for_offline_worker_to_pause(self) -> None:
        release = threading.Event()
        worker = threading.Thread(
            target=lambda: release.wait(timeout=2.0), daemon=True
        )
        worker.start()
        self.application._research_worker_thread = worker
        try:
            with self.assertRaisesRegex(RuntimeError, "offline analysis"):
                self.application.start("monitor")
        finally:
            release.set()
            worker.join(timeout=1.0)

    def test_external_worker_lease_blocks_ui_analysis_training_and_live(
        self,
    ) -> None:
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "external-worker"},
        )
        claimed = self.application.memory.claim_analysis_job(
            worker_id="worker:external-test", worker_pid=os.getpid()
        )
        self.assertEqual(claimed["id"], job_id)

        status = self.application.research_status()
        self.assertTrue(status["worker"]["running"])
        self.assertTrue(status["worker"]["externally_managed"])
        self.assertFalse(status["worker"]["cancel_supported"])
        self.assertEqual(
            status["worker"]["progress"]["current_job_type"],
            EDMFORMER_JOB,
        )
        reopened = LumenApplication(
            rig_path=self.rig_path,
            memory_path=Path(self.temporary.name) / "memory.sqlite3",
            settings_path=Path(self.temporary.name) / "reopened.json",
        )
        try:
            reopened_status = reopened.research_status()
            self.assertTrue(reopened_status["worker"]["running"])
            self.assertTrue(
                reopened_status["worker"]["externally_managed"]
            )
            self.assertEqual(
                reopened_status["worker"]["recovered_jobs"], []
            )
        finally:
            reopened.close()

        with self.assertRaisesRegex(RuntimeError, "another Lumen process"):
            self.application.start("monitor")
        with patch.object(
            self.application, "export_training_data"
        ) as export:
            with self.assertRaisesRegex(
                RuntimeError, "another Lumen process"
            ):
                self.application.analyze_training_data()
            export.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "another Lumen process"):
            self.application.train_structure_student({"epochs": 1})

        # Completion by the external process is reflected from the durable
        # job row on the next status poll; no application restart is needed.
        externally_trained = StreamingStructureStudent()
        externally_trained.training_examples = 42
        externally_trained.save(self.application._student_model_path)
        self.application._student_model_path.with_name(
            self.application._student_model_path.stem + ".evaluation.json"
        ).write_text(
            json.dumps({
                "activated": True,
                "teacher_normalization_version": (
                    TEACHER_NORMALIZATION_VERSION
                ),
                "edmformer_preprocessing_version": (
                    EDMFORMER_PREPROCESSING_VERSION
                ),
                "activation_gate_version": (
                    STUDENT_ACTIVATION_GATE_VERSION
                ),
            }),
            encoding="utf-8",
        )
        self.application.memory.update_analysis_job(
            job_id, status="complete", result={"elapsed_s": 12.0}
        )
        completed = self.application.research_status()
        self.assertFalse(completed["worker"]["running"])
        self.assertFalse(completed["worker"]["externally_managed"])
        self.assertFalse(completed["worker"]["cancel_supported"])
        self.assertTrue(completed["training"]["model"]["active"])
        self.assertEqual(
            self.application._student_model.training_examples, 42
        )

    def test_obsolete_student_artifact_cannot_control_live(self) -> None:
        stale = StreamingStructureStudent()
        stale.save(self.application._student_model_path)
        self.application._student_model_path.with_name(
            self.application._student_model_path.stem + ".evaluation.json"
        ).write_text(
            json.dumps({
                "teacher_normalization_version": (
                    "lumen_normalized_structure_v1"
                )
            }),
            encoding="utf-8",
        )

        self.application._load_student_model()

        self.assertIsNone(self.application._student_model)
        self.assertEqual(self.application._student_model_state, "obsolete")
        self.assertIsNone(self.application._student_model_error)
        self.assertIn("obsolete teacher normalization", (
            self.application._student_model_notice or ""
        ))

    def test_obsolete_full_song_student_is_a_notice_not_load_error(self) -> None:
        stale = StreamingStructureStudent()
        stale.save(self.application._student_model_path)
        self.application._student_model_path.with_name(
            self.application._student_model_path.stem + ".evaluation.json"
        ).write_text(
            json.dumps({
                "activated": True,
                "teacher_normalization_version": (
                    TEACHER_NORMALIZATION_VERSION
                ),
                "edmformer_preprocessing_version": "short-context-v1",
            }),
            encoding="utf-8",
        )

        self.application._load_student_model()
        status = self.application.research_status()["training"]["model"]

        self.assertIsNone(self.application._student_model)
        self.assertEqual(status["runtime_state"], "obsolete")
        self.assertIsNone(status["runtime_error"])
        notice = status["runtime_notice"].lower()
        self.assertIn("previous active student", notice)
        self.assertNotIn("analyze and train again", notice)
        self.assertTrue(status["artifact_present"])

    def test_status_poll_recovers_external_worker_that_dies_after_startup(
        self,
    ) -> None:
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=SONGFORMER_JOB,
            payload={"recording_id": "worker-died-after-ui-open"},
        )
        claimed = self.application.memory.claim_analysis_job(
            worker_id="worker:dead-after-open",
            worker_pid=999_999_999,
        )
        self.assertEqual(claimed["id"], job_id)

        status = self.application.research_status()

        self.assertFalse(status["worker"]["running"])
        self.assertEqual(
            [row["job_id"] for row in status["worker"]["recovered_jobs"]],
            [job_id],
        )
        jobs = {
            row["id"]: row
            for row in self.application.memory.list_analysis_jobs()
        }
        self.assertEqual(jobs[job_id]["status"], "queued")

    def test_research_status_reports_runtime_model_failure_not_file_presence(
        self,
    ) -> None:
        self.application._student_model = None
        self.application._student_model_error = "invalid student artifact"
        with patch.object(
            self.application.research, "status", return_value={}
        ), patch(
            "lumen_engine.control.training_readiness",
            return_value={"model": {"active": True, "candidate": False}},
        ):
            status = self.application.research_status()

        model = status["training"]["model"]
        self.assertTrue(model["artifact_present"])
        self.assertFalse(model["active"])
        self.assertEqual(model["runtime_state"], "error")
        self.assertEqual(model["runtime_error"], "invalid student artifact")

    def test_analyze_is_a_clean_noop_when_every_recording_is_complete(self) -> None:
        with patch.object(
            self.application,
            "export_training_data",
            return_value={"path": "already-prepared"},
        ), patch.object(
            self.application,
            "research_status",
            return_value={"worker": {"running": False}},
        ):
            result = self.application.analyze_training_data()

        self.assertFalse(result["started"])
        self.assertIn("No new", result["message"])

    def test_analyze_reports_retained_ineligible_recordings(self) -> None:
        with patch.object(
            self.application,
            "_prepare_unindexed_research_captures",
            return_value={
                "sessions_prepared": 1,
                "recordings": 3,
                "jobs_queued": 0,
                "recordings_ineligible": 3,
                "recordings_partial": 2,
                "recordings_unknown": 1,
            },
        ), patch.object(
            self.application,
            "research_status",
            return_value={"worker": {"running": False}},
        ):
            result = self.application.analyze_training_data()

        self.assertFalse(result["started"])
        self.assertIn("Found 3 captured recording(s)", result["message"])
        self.assertIn("2 partial, 1 unidentified", result["message"])

    def test_controls_feedback_target_and_fixture_edit_are_operable(self) -> None:
        status = self.application.apply_preset("restrained")
        self.assertAlmostEqual(status["controls"]["motion"], 0.18)
        status = self.application.patch_controls({"blackout": True, "master": 0.4})
        self.assertTrue(status["controls"]["blackout"])
        self.assertAlmostEqual(status["controls"]["master"], 0.4)
        settings = self.application.patch_settings(
            {"audio_device": "hw:0,2", "spotify_client_id": "test-client-id-1234"}
        )
        self.assertEqual(settings["settings"]["audio_device"], "hw:0,2")
        self.assertEqual(
            settings["settings"]["spotify_client_id_masked"],
            "test-c…1234",
        )

        feedback = self.application.add_feedback(
            {"label": "liked_this", "value": 1, "note": "Good spatial restraint."}
        )
        self.assertGreater(feedback["feedback_id"], 0)
        self.assertEqual(
            self.application.memory.summary()["recent_feedback"][0]["note"],
            "Good spatial restraint.",
        )
        annotation = self.application.add_training_annotation(
            {
                "kind": "preferred_action",
                "label": "figure_eight",
                "scope": "group",
                "group_id": "movers",
            }
        )
        self.assertGreater(annotation["annotation_id"], 0)
        self.assertEqual(
            self.application.memory.training_summary()["annotations"], 1
        )

    def test_research_annotation_import_http_api_reports_ccmusic_gate(
        self,
    ) -> None:
        (
            self.application.research.sources / "ccmusic"
        ).mkdir(parents=True)
        server = LumenHTTPServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                "http://127.0.0.1:"
                f"{server.server_address[1]}"
                "/api/research/import-annotations",
                data=json.dumps({"components": ["ccmusic"]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                payload = json.load(response)
                self.assertEqual(response.status, 200)
            gate = payload["results"][0]
            self.assertEqual(
                gate["state"], "awaiting_authorized_metadata"
            )
            self.assertEqual(
                gate["reason_code"], "ccmusic_gated_metadata_not_ready"
            )
            self.assertFalse(gate["automatic_download_attempted"])
        finally:
            server.shutdown()
            ThreadingHTTPServer.server_close(server)
            thread.join(timeout=3)

    def test_lumen_link_http_status_and_controls_are_exposed(self) -> None:
        server = LumenHTTPServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                "http://127.0.0.1:"
                f"{server.server_address[1]}/api/link/status",
                timeout=3,
            ) as response:
                status = json.load(response)
                self.assertEqual(response.status, 200)
            self.assertEqual(status["schema"], "lumen.link.v1")
            self.assertEqual(
                status["pipeline"]["source"],
                "cached_verified_research_readiness",
            )
            self.assertEqual(
                len(status["pipeline"]["engine_capabilities"]), 3
            )
            with urlopen(
                "http://127.0.0.1:"
                f"{server.server_address[1]}/api/link/status?summary=1",
                timeout=3,
            ) as response:
                summary = json.load(response)
                self.assertEqual(response.status, 200)
            self.assertTrue(summary["summary"])
            self.assertEqual(summary["events"], [])
            self.assertLessEqual(len(summary["jobs"]), 1)
            with patch.object(
                self.application.lumen_link,
                "control",
                return_value={"paused": True},
            ) as control:
                request = Request(
                    "http://127.0.0.1:"
                    f"{server.server_address[1]}/api/link/pause",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    result = json.load(response)
                    self.assertEqual(response.status, 200)
                self.assertTrue(result["paused"])
                control.assert_called_once_with("pause")
            with patch.object(
                self.application.lumen_link,
                "control",
                return_value={"enabled": False},
            ) as control:
                request = Request(
                    "http://127.0.0.1:"
                    f"{server.server_address[1]}/api/link/disable",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    result = json.load(response)
                    self.assertEqual(response.status, 200)
                self.assertFalse(result["enabled"])
                control.assert_called_once_with("disable")
        finally:
            server.shutdown()
            ThreadingHTTPServer.server_close(server)
            thread.join(timeout=3)

    def test_remote_student_import_hot_loads_without_restart(self) -> None:
        loaded = threading.Event()
        refresh_scheduled = threading.Event()
        sentinel = object()

        def load_model():
            self.application._student_model = sentinel
            loaded.set()

        with patch.object(
            self.application,
            "_load_student_model",
            side_effect=load_model,
        ), patch.object(
            self.application,
            "_schedule_research_readiness_refresh",
            side_effect=lambda: refresh_scheduled.set() or True,
        ):
            self.application._on_lumen_link_import(
                {"id": "job:remote-student", "job_type": STUDENT_TRAIN_JOB},
                {"activated": True, "approved_axes": ["energy"]},
            )
            self.assertTrue(loaded.wait(1.0))
            self.assertTrue(refresh_scheduled.wait(1.0))
        self.assertIs(self.application._student_model, sentinel)
        self.assertIsNone(self.application._research_readiness_cache)
        self.assertTrue(any(
            "Imported Threadripper student.train result" in event["message"]
            for event in self.application.events
        ))

    def test_repeated_imports_share_one_deferred_readiness_audit(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def refresh():
            nonlocal calls
            calls += 1
            entered.set()
            release.wait(timeout=2.0)

        self.application._research_readiness_dirty_at = (
            time.monotonic() - 3.0
        )
        with patch.object(
            self.application,
            "_refresh_research_readiness_cache",
            side_effect=refresh,
        ):
            for _ in range(5):
                self.application._schedule_research_readiness_refresh()
            self.assertTrue(entered.wait(1.0))
            self.assertEqual(calls, 1)
            release.set()
            thread = self.application._research_readiness_thread
            if thread is not None:
                thread.join(timeout=2.0)
        self.assertEqual(calls, 1)

    def test_link_status_projects_cached_student_gate_without_a_new_audit(
        self,
    ) -> None:
        class RejectDeepCopy:
            def __deepcopy__(self, memo):
                del memo
                raise AssertionError("unrelated readiness data was copied")

        self.application._research_readiness_cache = {
            "schema": "lumen_operator_readiness_cache_v1",
            "created_unix_ms": 123456,
            "training": {
                "unused_large_payload": RejectDeepCopy(),
                "train_ready": True,
                "activation_blockers": ["one more independent test song"],
                "progress": 0.75,
                "teacher_jobs_complete": 6,
                "teacher_jobs_remaining": 2,
                "usable_examples": 400,
                "ontology": {
                    "axis_teachers": {
                        "energy": "EDMFormer",
                        "functional": "SongFormer",
                        "content": "SongFormer",
                        "boundary": "EDMFormer + SongFormer",
                    }
                },
                "model": {
                    "candidate": True,
                    "candidate_provenance_current": True,
                    "active": False,
                    "active_artifact_exists": False,
                    "evaluation": {
                        "unused_per_song_detail": RejectDeepCopy(),
                        "activated": True,
                        "held_out_split": "test",
                        "approved_axes": ["energy", "boundary"],
                        "inactive_axes": ["functional"],
                        "axis_gate_reasons": {
                            "functional": ["below held-out baseline"]
                        },
                        "split_group_counts": {"test": 2},
                    },
                },
            },
        }
        remote = {
            "schema": "lumen.link.v1",
            "capabilities": {
                "supported_job_types": [
                    "teacher.edmformer",
                    "teacher.songformer",
                    "student.train",
                ],
                "gated_job_types": {
                    "teacher.songformer": "result importer unavailable"
                },
            },
        }
        with patch.object(
            self.application.lumen_link, "status", return_value=remote
        ), patch.object(
            self.application, "research_status"
        ) as full_audit:
            status = self.application.lumen_link_status()

        full_audit.assert_not_called()
        engines = {
            item["job_type"]: item
            for item in status["pipeline"]["engine_capabilities"]
        }
        self.assertEqual(engines["teacher.edmformer"]["state"], "available")
        self.assertEqual(engines["teacher.songformer"]["state"], "gated")
        self.assertEqual(engines["student.train"]["state"], "available")
        student = status["pipeline"]["student"]
        self.assertEqual(student["validation"]["state"], "partial")
        self.assertEqual(
            student["validation"]["approved_axes"], ["energy", "boundary"]
        )
        self.assertEqual(student["activation"]["state"], "candidate_only")
        self.assertNotIn("unused_large_payload", status["pipeline"])

    def test_http_status_body_is_shared_across_concurrent_ui_reads(
        self,
    ) -> None:
        server = LumenHTTPServer(("127.0.0.1", 0), self.application)
        try:
            self.assertGreaterEqual(server.request_queue_size, 64)
            with patch.object(
                self.application,
                "snapshot",
                wraps=self.application.snapshot,
            ) as snapshot:
                first = server.status_body()
                second = server.status_body()
                self.assertEqual(first, second)
                self.assertEqual(snapshot.call_count, 1)

                time.sleep(0.10)
                server.status_body()
                self.assertEqual(snapshot.call_count, 2)

                without_history = server.status_body(
                    include_analysis_history=False
                )
                self.assertEqual(snapshot.call_count, 3)
                self.assertEqual(
                    json.loads(without_history)["analysis_history"], []
                )
        finally:
            ThreadingHTTPServer.server_close(server)

    def test_analyze_http_api_starts_and_reports_virtual_batch(self) -> None:
        job_id = self.application.memory.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "http-analyze-smoke"},
        )
        claimed = self.application.memory.claim_analysis_job(
            worker_id="crashed-http-worker", worker_pid=999_999_999
        )
        assert claimed is not None

        class FakeWorker:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.called = False

            def run_once(self, job_types):
                self.asserted_types = tuple(job_types)
                if self.called:
                    return None
                self.called = True
                return {
                    "job_id": "http-smoke",
                    "job_type": EDMFORMER_JOB,
                    "status": "complete",
                    "result": {"virtual": True},
                }

        server = LumenHTTPServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(
                self.application,
                "export_training_data",
                return_value={"path": "virtual-export"},
            ), patch("lumen_engine.control.OfflineResearchWorker", FakeWorker):
                request = Request(
                    "http://127.0.0.1:"
                    f"{server.server_address[1]}/api/research/analyze",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    payload = json.load(response)
                    self.assertEqual(response.status, 200)
                self.assertTrue(payload["started"])
                worker_thread = self.application._research_worker_thread
                assert worker_thread is not None
                worker_thread.join(timeout=2.0)

            with urlopen(
                "http://127.0.0.1:"
                f"{server.server_address[1]}/api/research",
                timeout=3,
            ) as response:
                status = json.load(response)
                self.assertEqual(response.status, 200)
            self.assertFalse(status["worker"]["running"])
            self.assertIn(
                job_id,
                [
                    item["job_id"]
                    for item in status["worker"]["recovered_jobs"]
                ],
            )
            self.assertEqual(
                status["worker"]["last_result"]["jobs"][0]["result"],
                {"virtual": True},
            )
        finally:
            server.shutdown()
            ThreadingHTTPServer.server_close(server)
            thread.join(timeout=3)

        solutions = self.application.solve_target(
            self.application.selected_target
        )
        self.assertEqual(len(solutions), 2)

        fixture_id = self.application.rig.fixtures[0].fixture_id
        bootstrap = self.application.patch_fixture(
            {
                "fixture_id": fixture_id,
                "name": "Edited in console",
                "position_m": [-1.1, -2.2, 2.8],
                "calibration": {
                    "pan_left_dmx": 128,
                    "home_pan_dmx": 92,
                    "pan_right_dmx": 81,
                    "tilt_high_dmx": 17,
                    "home_tilt_dmx": 72,
                    "tilt_low_dmx": 155,
                },
            }
        )
        saved = json.loads(self.rig_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["fixtures"][0]["name"], "Edited in console")
        self.assertEqual(
            bootstrap["rig"]["fixtures"][0]["position_m"],
            [-1.1, -2.2, 2.8],
        )
        self.assertEqual(
            saved["fixtures"][0]["calibration"]["tilt_high_dmx"], 17
        )
        self.assertEqual(
            bootstrap["rig"]["fixtures"][0]["calibration"]["home_tilt_dmx"],
            72,
        )

    def test_operator_note_is_rebuilt_and_fixture_feedback_stays_scoped(self) -> None:
        self.application.add_feedback(
            {
                "label": "operator_note",
                "value": 1,
                "note": "more movement, no strobe, cooler",
            }
        )
        overall = self.application._feedback_biases["overall"]
        self.assertGreater(overall["motion"], 0.0)
        self.assertLess(overall["strobe"], 0.0)
        self.assertLess(overall["strobe_enabled"], 0.0)
        self.assertLess(overall["palette"], 0.0)

        fixture_id = self.application.rig.fixtures[0].fixture_id
        other_id = self.application.rig.fixtures[1].fixture_id
        self.application.add_feedback(
            {
                "label": "increase_movement",
                "value": 1,
                "scope": "fixture",
                "fixture_id": fixture_id,
            }
        )
        self.assertIn(fixture_id, self.application._feedback_biases)
        self.assertNotIn(other_id, self.application._feedback_biases)

        memory_path = self.application.memory_path
        self.application.close()
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=memory_path,
            settings_path=Path(self.temporary.name) / "settings.json",
        )
        rebuilt = self.application._feedback_biases["overall"]
        self.assertGreater(rebuilt["motion"], 0.0)
        self.assertLess(rebuilt["strobe"], 0.0)
        self.assertLess(rebuilt["strobe_enabled"], 0.0)

    def test_feedback_rebuild_preserves_literal_motion_axes_across_restart(self) -> None:
        for label in ("faster", "increase_movement", "too_busy"):
            self.application.add_feedback({
                "label": label,
                "value": 1,
                "scope": "overall",
                "participant_id": f"listener-{label}",
                "client_event_id": f"event-{label}",
            })
        bias = self.application._feedback_biases["overall"]
        self.assertGreater(bias["motion_speed"], 0.0)
        self.assertGreater(bias["travel_size"], 0.0)
        self.assertLess(bias["activity_density"], 0.0)

        expected = {
            axis: bias[axis]
            for axis in (
                "motion_speed", "travel_size", "activity_density"
            )
        }
        memory_path = self.application.memory_path
        self.application.close()
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=memory_path,
            settings_path=Path(self.temporary.name) / "settings.json",
        )
        restored = self.application._feedback_biases["overall"]
        for axis, value in expected.items():
            self.assertAlmostEqual(restored[axis], value)

    def test_strobe_enable_and_rate_remain_independent(self) -> None:
        self.application.add_feedback({
            "label": "faster_strobe",
            "value": 1,
            "scope": "overall",
        })
        overall = self.application._feedback_biases["overall"]
        self.assertGreater(overall["strobe_rate"], 0.0)
        self.assertEqual(overall["strobe_enabled"], 0.0)
        self.assertEqual(overall["strobe"], 0.0)

        self.application.add_feedback({
            "label": "strobe",
            "value": 1,
            "scope": "group",
            "group_id": "movers",
        })
        mover = self.application._feedback_biases[
            self.application.rig.fixtures[0].fixture_id
        ]
        self.assertGreater(mover["strobe_enabled"], 0.0)
        self.assertEqual(mover["strobe_rate"], 0.0)

    def test_group_song_and_section_feedback_keys_keep_literal_axis(self) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:literal-scope",
            title="Literal Scope",
            artists=("Scope Artist",),
            is_playing=False,
        )
        self.application._remember_media_identity(media)
        self.application.observation = replace(
            self.application.observation, section="build"
        )
        self.application.add_feedback({
            "label": "faster",
            "value": 1,
            "scope": "group",
            "group_id": "movers",
        })
        assert self.application.song_id is not None
        song_id = self.application.song_id
        for fixture in self.application.rig.fixtures:
            section_key = (
                f"song:{self.application.song_id}:section:build:fixture:"
                f"{fixture.fixture_id}"
            )
            self.assertIn(section_key, self.application._feedback_biases)
            self.assertGreater(
                self.application._feedback_biases[section_key]["motion_speed"],
                0.0,
            )
            self.assertEqual(
                self.application._feedback_biases[section_key]["travel_size"],
                0.0,
            )
            self.assertNotIn(
                fixture.fixture_id, self.application._feedback_biases
            )
            self.assertNotIn(
                f"song:{song_id}:fixture:{fixture.fixture_id}",
                self.application._feedback_biases,
            )
            self.assertNotIn(
                f"artist:scope artist:fixture:{fixture.fixture_id}",
                self.application._feedback_biases,
            )
        memory_path = self.application.memory_path
        self.application.close()
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=memory_path,
            settings_path=Path(self.temporary.name) / "settings.json",
        )
        for fixture in self.application.rig.fixtures:
            section_key = (
                f"song:{song_id}:section:build:fixture:{fixture.fixture_id}"
            )
            self.assertGreater(
                self.application._feedback_biases[section_key]["motion_speed"],
                0.0,
            )

    def test_default_feedback_does_not_cross_song_or_section(self) -> None:
        first = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:first-feedback-context",
            title="First Feedback Context",
            is_playing=False,
        )
        self.application._remember_media_identity(first)
        self.application.observation = replace(
            self.application.observation, section="build"
        )
        self.application.add_feedback({
            "label": "faster",
            "value": 1,
            "scope": "overall",
        })
        assert self.application.song_id is not None
        first_song_id = self.application.song_id
        expected_key = f"song:{first_song_id}:section:build"
        self.assertIn(expected_key, self.application._feedback_biases)
        self.assertNotIn("overall", self.application._feedback_biases)
        self.assertNotIn(
            f"song:{first_song_id}", self.application._feedback_biases
        )

        runtime = PerformanceRuntime(
            self.application.rig.fixtures, VirtualDMXOutput()
        )
        fixture_id = self.application.rig.fixtures[0].fixture_id
        runtime.replace_feedback(self.application._feedback_biases)
        runtime.set_media_context(first_song_id, "build")
        self.assertGreater(
            runtime._characteristics_feedback_for(fixture_id).motion_speed,
            0.0,
        )
        runtime.set_media_context(first_song_id, "groove")
        self.assertEqual(
            runtime._characteristics_feedback_for(fixture_id).motion_speed,
            0.0,
        )
        second = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:second-feedback-context",
            title="Second Feedback Context",
            is_playing=False,
        )
        self.application._remember_media_identity(second)
        assert self.application.song_id is not None
        runtime.set_media_context(self.application.song_id, "build")
        self.assertEqual(
            runtime._characteristics_feedback_for(fixture_id).motion_speed,
            0.0,
        )

        self.application.add_feedback({
            "label": "slower",
            "value": 1,
            "scope": "overall",
            "lifetime": "global",
        })
        self.assertLess(
            self.application._feedback_biases["overall"]["motion_speed"],
            0.0,
        )

    def test_free_form_note_uses_literal_axes_and_routine_preferences(self) -> None:
        self.application.add_feedback({
            "label": "operator_note",
            "value": 0,
            "scope": "overall",
            "note": (
                "Faster strobe, more movement, too busy, timing off; "
                "make it brighter and warmer, lock to the beat; use figure eight"
            ),
        })
        bias = self.application._feedback_biases["overall"]
        self.assertGreater(bias["strobe_rate"], 0.0)
        self.assertEqual(bias["strobe_enabled"], 0.0)
        self.assertEqual(bias["motion_speed"], 0.0)
        self.assertGreater(bias["travel_size"], 0.0)
        self.assertLess(bias["activity_density"], 0.0)
        self.assertGreater(bias["brightness"], 0.0)
        self.assertGreater(bias["palette"], 0.0)
        self.assertGreater(bias["beat_sync"], 0.0)
        self.assertLess(bias["cue_timing"], 0.0)
        self.assertGreater(bias["routines"]["figure_eight"], 0.0)
        memory_path = self.application.memory_path
        self.application.close()
        self.application = LumenApplication(
            rig_path=self.rig_path,
            memory_path=memory_path,
            settings_path=Path(self.temporary.name) / "settings.json",
        )
        restored = self.application._feedback_biases["overall"]
        self.assertGreater(restored["strobe_rate"], 0.0)
        self.assertEqual(restored["strobe_enabled"], 0.0)
        self.assertGreater(restored["brightness"], 0.0)
        self.assertGreater(restored["palette"], 0.0)
        self.assertGreater(restored["beat_sync"], 0.0)
        self.assertLess(restored["cue_timing"], 0.0)
        self.assertGreater(restored["routines"]["figure_eight"], 0.0)

    def test_zero_value_ui_note_still_applies_parsed_semantics(self) -> None:
        self.application.add_feedback({
            "label": "operator_note",
            "value": 0,
            "scope": "overall",
            "note": "Please use more movement and make it brighter",
        })
        bias = self.application._feedback_biases["overall"]
        self.assertGreater(bias["motion"], 0.0)
        self.assertGreater(bias["intensity"], 0.0)

    def test_feedback_consensus_key_changes_with_performed_context(self) -> None:
        common = {
            "song_id": 1,
            "listening_session_id": "session",
            "created_unix_ms": 10_001,
            "label": "pick_it_up",
            "scope": "overall",
            "fixture_id": None,
            "section": "groove",
        }
        first = self.application._feedback_batch_event_id(
            **common, routine="figure_eight"
        )
        second = self.application._feedback_batch_event_id(
            **common, routine="fan_sweep"
        )
        self.assertNotEqual(first, second)

        lane_common = {
            **common,
            "routine": "fan_sweep",
            "lane": "center",
        }
        first_lease = self.application._feedback_batch_event_id(
            **lane_common,
            active_sequence_id="center-sequence-a",
            boundary_id="center-boundary-4",
        )
        second_lease = self.application._feedback_batch_event_id(
            **lane_common,
            active_sequence_id="center-sequence-b",
            boundary_id="center-boundary-5",
        )
        self.assertNotEqual(first_lease, second_lease)

    def test_center_feedback_persists_center_planner_context(self) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        while (
            self.application.snapshot()["decision"] is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        choreography = self.application.snapshot()["choreography"]
        center = choreography["lanes"]["center"]
        movers = choreography["lanes"]["movers"]
        result = self.application.add_feedback({
            "label": "more_like_this",
            "value": 1,
            "scope": "group",
            "group_id": "center",
        })
        row = next(
            item for item in self.application.memory.all_feedback()
            if item["id"] == result["feedback_id"]
        )
        self.assertEqual(set(row["lane_context"]["lanes"]), {"center"})
        stored = row["lane_context"]["lanes"]["center"]
        self.assertEqual(
            stored["active_sequence_id"], center["active_sequence_id"]
        )
        self.assertEqual(
            stored["boundary_id"], center["active_boundary_id"]
        )
        self.assertEqual(
            stored["routine"], center["active_step"]["routine"]
        )
        self.assertEqual(row["routine"], center["active_step"]["routine"])
        if (
            movers["active_step"]["routine"]
            != center["active_step"]["routine"]
        ):
            self.assertNotEqual(
                row["routine"], movers["active_step"]["routine"]
            )
        self.assertEqual(set(result["model_event_ids"]), {"center"})
        self.application.stop()

    def test_unimplemented_preferred_actions_are_rejected(self) -> None:
        for label in ("hold_position", "blackout_accent"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.application.add_training_annotation({
                    "kind": "preferred_action",
                    "label": label,
                    "scope": "overall",
                })

    def test_song_context_uses_canonical_techno_states_and_events(self) -> None:
        self.application._remember_media_identity(MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:structure-scope",
            title="Structure Scope",
            duration_ms=180_000,
            observed_position_ms=10_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        ))
        state = self.application.add_training_annotation({
            "kind": "musical_context",
            "label": "drop",
            # Legacy/mobile clients may still send the currently selected
            # fixture target. Song structure must ignore it.
            "scope": "group",
            "group_id": "movers",
        })
        event = self.application.add_training_annotation({
            "kind": "musical_context",
            "label": "drop_onset",
            "scope": "overall",
        })
        self.assertEqual(state["label"], "drop")
        self.assertEqual(event["label"], "drop_onset")
        stored = self.application.memory.musical_structure_annotations()
        self.assertTrue(stored)
        self.assertTrue(all(item["scope"] == "overall" for item in stored))
        self.assertTrue(all(item["fixture_id"] is None for item in stored))
        with self.assertRaisesRegex(ValueError, "unknown training annotation"):
            self.application.add_training_annotation({
                "kind": "musical_context",
                "label": "release",
                "scope": "overall",
            })

    def test_feedback_delete_reverses_sequence_learning_event(self) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        while (
            self.application.snapshot()["decision"] is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        result = self.application.add_feedback(
            {
                "label": "strobe",
                "value": 1,
                "scope": "overall",
            }
        )
        event_id = result["model_event_id"]
        lane_event_ids = result["model_event_ids"]
        state = self.application._choreography_model.state_dict()
        self.assertEqual(set(lane_event_ids), {"movers", "center"})
        self.assertTrue(all(
            lane_event_id.startswith(event_id + ":")
            for lane_event_id in lane_event_ids.values()
        ))
        self.assertIn(lane_event_ids["movers"], state["events"])
        self.assertIn(lane_event_ids["center"], state["events"])
        removed = self.application.delete_feedback(
            {"feedback_id": result["feedback_id"]}
        )
        self.assertTrue(removed["sequence_update_removed"])
        events = self.application._choreography_model.state_dict()["events"]
        self.assertNotIn(lane_event_ids["movers"], events)
        self.assertNotIn(lane_event_ids["center"], events)
        self.application.stop()

    def test_feedback_delete_forgets_legacy_feedback_id_event(self) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        while (
            self.application.snapshot()["decision"] is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        result = self.application.add_feedback({
            "label": "more_like_this",
            "value": 1,
            "scope": "group",
            "group_id": "center",
        })
        state = self.application._choreography_model.state_dict()
        current_event_id = result["model_event_ids"]["center"]
        legacy_event_id = f"feedback:{result['feedback_id']}"
        state["events"][legacy_event_id] = state["events"].pop(
            current_event_id
        )
        self.application._choreography_model = (
            SequencePreferenceModel.from_state_dict(state)
        )
        removed = self.application.delete_feedback({
            "feedback_id": result["feedback_id"]
        })
        self.assertTrue(removed["sequence_update_removed"])
        self.assertNotIn(
            legacy_event_id,
            self.application._choreography_model.state_dict()["events"],
        )
        self.application.stop()

    def test_repeated_directional_feedback_is_one_urgency_event_without_named_routine(
        self,
    ) -> None:
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        while (
            self.application.snapshot()["decision"] is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        first = self.application.add_feedback({
            "label": "pick_it_up",
            "value": 1,
            "scope": "overall",
            "participant_id": "listener-a",
            "client_event_id": "tap-1",
        })
        second = self.application.add_feedback({
            "label": "pick_it_up",
            "value": 1,
            "scope": "overall",
            "participant_id": "listener-a",
            "client_event_id": "tap-2",
        })
        self.assertEqual(first["model_event_id"], second["model_event_id"])
        self.assertEqual(second["feedback_occurrences"], 2)
        self.assertGreater(second["urgency"], first["urgency"])
        self.assertEqual(
            self.application._feedback_routine_effect(
                "pick_it_up", None
            ),
            {},
        )
        events = self.application._choreography_model.state_dict()["events"]
        matching = {
            key: value for key, value in events.items()
            if key.startswith(second["model_event_id"] + ":")
        }
        self.assertEqual(set(key.rsplit(":", 1)[-1] for key in matching), {
            "movers", "center",
        })
        for event in matching.values():
            example = event["example"]
            self.assertIsNone(example["preferred"])
            self.assertEqual(example["feedback"][0]["occurrences"], 2)
        self.application.delete_feedback({
            "feedback_id": second["feedback_id"],
            "participant_id": "listener-a",
        })
        events = self.application._choreography_model.state_dict()["events"]
        matching = [
            value for key, value in events.items()
            if key.startswith(second["model_event_id"] + ":")
        ]
        self.assertTrue(matching)
        self.assertTrue(all(
            value["example"]["feedback"][0]["occurrences"] == 1
            for value in matching
        ))
        self.application.stop()

    def test_structure_resolver_rejects_false_student_silence_and_falls_back_per_axis(
        self,
    ) -> None:
        raw = MusicalObservation(
            timestamp_s=10.0,
            loudness=0.45,
            onset_strength=0.2,
            low_energy=0.4,
            mid_energy=0.4,
            high_energy=0.2,
            section="groove",
            section_confidence=0.62,
        )
        metrics = AudioInputMetrics(
            timestamp_s=10.0,
            frame_count=2048,
            rms=0.12,
            dbfs=-18.4,
            peak=0.42,
            channel_rms=(0.12, 0.12),
            channel_peak=(0.42, 0.40),
            clipped_samples=0,
            waveform=(0.0,),
        )
        self.application._student_prediction = {
            "functional": "verse",
            "energy": "silence",
            "content": "instrumental",
            "confidence": {
                "functional": 0.8, "energy": 0.99, "content": 0.8,
            },
            "accepted_axes": {
                "functional": True, "energy": True,
                "content": True, "boundary": False,
            },
        }
        resolved = self.application._resolve_structure(raw, metrics)
        self.assertEqual(resolved.section, "groove")
        self.assertEqual(
            self.application._effective_structure["axes"]["energy"]["source"],
            "live_analyzer",
        )

        self.application._student_prediction["energy"] = "build"
        self.application._student_prediction["confidence"]["energy"] = 0.82
        self.application._cached_structure_prediction = {
            "axes": {
                "functional": {
                    "label": "chorus",
                    "confidence": 0.91,
                    "timeline_id": "timeline-functional",
                    "provenance": "songformer:test",
                },
                "energy": {"label": "unknown", "confidence": 0.0},
                "content": {"label": "unknown", "confidence": 0.0},
            },
            "boundary": {},
        }
        resolved = self.application._resolve_structure(raw, metrics)
        axes = self.application._effective_structure["axes"]
        self.assertEqual(resolved.section, "build")
        self.assertEqual(axes["functional"]["source"], "cached_offline_teacher")
        self.assertEqual(axes["energy"]["source"], "streaming_student")
        self.assertEqual(axes["functional"]["timeline_id"], "timeline-functional")

    def test_physical_silence_overrides_cached_and_student_structure(self) -> None:
        self.application._cached_structure_prediction = {
            "axes": {
                "energy": {"label": "drop", "confidence": 0.99},
            },
            "boundary": {"current_confidence": 0.9},
        }
        self.application._student_prediction = {
            "energy": "drop",
            "confidence": {"energy": 0.99},
            "accepted_axes": {"energy": True, "boundary": True},
            "boundary_probability": 0.9,
        }
        quiet = MusicalObservation(
            timestamp_s=20.0,
            loudness=0.0,
            onset_strength=0.0,
            low_energy=0.0,
            mid_energy=0.0,
            high_energy=0.0,
            section="groove",
            section_confidence=0.2,
        )
        self.application._resolve_structure(
            quiet, AudioInputMetrics.silence(timestamp_s=20.0)
        )
        quiet = replace(quiet, timestamp_s=20.6)
        resolved = self.application._resolve_structure(
            quiet, AudioInputMetrics.silence(timestamp_s=20.6)
        )
        resolution = self.application._effective_structure
        self.assertEqual(resolved.section, "silence")
        self.assertEqual(resolution["source"], "live_audio_silence")
        self.assertEqual(
            resolution["axes"]["boundary"]["source"],
            "live_audio_silence",
        )

        audible_metrics = replace(
            AudioInputMetrics.silence(timestamp_s=20.64),
            rms=0.10,
            dbfs=-20.0,
            peak=0.35,
        )
        audible = replace(
            quiet,
            timestamp_s=20.64,
            loudness=0.45,
            onset_strength=0.2,
        )
        # One packet cannot knock the physical silence gate open.
        resolved = self.application._resolve_structure(
            audible, audible_metrics
        )
        self.assertEqual(resolved.section, "silence")
        # Sustained signal can.
        resolved = self.application._resolve_structure(
            replace(audible, timestamp_s=20.78),
            replace(audible_metrics, timestamp_s=20.78),
        )
        self.assertNotEqual(resolved.section, "silence")

    def test_uncalibrated_teacher_default_does_not_override_student_energy(
        self,
    ) -> None:
        raw = MusicalObservation(
            timestamp_s=30.0,
            loudness=0.5,
            onset_strength=0.3,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            section="groove",
            section_confidence=0.65,
        )
        metrics = AudioInputMetrics(
            timestamp_s=30.0,
            frame_count=2048,
            rms=0.12,
            dbfs=-18.0,
            peak=0.4,
            channel_rms=(0.12, 0.12),
            channel_peak=(0.4, 0.4),
            clipped_samples=0,
            waveform=(0.0,),
        )
        self.application._cached_structure_prediction = {
            "axes": {
                "energy": {
                    "label": "drop",
                    # Legacy teacher default after memory reliability scaling.
                    "confidence": 0.72 * 0.72,
                }
            },
            "boundary": {},
        }
        self.application._student_prediction = {
            "energy": "build",
            "confidence": {"energy": 0.82},
            "accepted_axes": {"energy": True, "boundary": False},
        }

        resolved = self.application._resolve_structure(raw, metrics)

        self.assertEqual(resolved.section, "build")
        self.assertEqual(
            self.application._effective_structure["axes"]["energy"]["source"],
            "streaming_student",
        )

    def test_approved_unscored_teacher_is_authoritative_without_fake_probability(
        self,
    ) -> None:
        raw = MusicalObservation(
            timestamp_s=30.0,
            loudness=0.5,
            onset_strength=0.3,
            low_energy=0.5,
            mid_energy=0.4,
            high_energy=0.3,
            section="groove",
            section_confidence=0.65,
        )
        metrics = AudioInputMetrics(
            timestamp_s=30.0,
            frame_count=2048,
            rms=0.12,
            dbfs=-18.0,
            peak=0.4,
            channel_rms=(0.12, 0.12),
            channel_peak=(0.4, 0.4),
            clipped_samples=0,
            waveform=(0.0,),
        )
        self.application._cached_structure_prediction = {
            "axes": {
                "energy": {
                    "label": "build",
                    "confidence": 0.0,
                    "model_confidence": 0.0,
                    "operator_trust": 1.0,
                    "recall_authority": "operator_approved",
                    "timeline_id": "approved-zero",
                }
            },
            "boundary": {},
        }

        resolved = self.application._resolve_structure(raw, metrics)
        axis = self.application._effective_structure["axes"]["energy"]
        self.assertEqual(resolved.section, "build")
        self.assertEqual(resolved.section_confidence, 1.0)
        self.assertEqual(axis["source"], "operator_approved_timeline")
        self.assertEqual(axis["model_confidence"], 0.0)
        self.assertEqual(axis["operator_trust"], 1.0)
        self.assertEqual(axis["confidence"], 0.0)

    def test_every_performance_trace_frame_names_effective_structure_source(
        self,
    ) -> None:
        self.application._effective_structure = {
            "schema": "lumen_structure_resolution_v2",
            "source": "streaming_student",
            "section": "build",
            "confidence": 0.81,
            "axes": {
                "energy": {
                    "label": "build",
                    "confidence": 0.81,
                    "source": "streaming_student",
                    "accepted_reason": "test approved axis",
                    "provenance": "model:test",
                    "timeline_id": None,
                }
            },
            "beat_timing_authority": "audio_sample_clock",
        }
        self.application.start("demo")
        deadline = time.monotonic() + 2.0
        samples = []
        while time.monotonic() < deadline:
            time.sleep(0.05)
            self.application._trace_queue.join()
            samples = self.application.memory.latest_performance_session()
            if samples:
                break
        self.application.stop()
        self.assertTrue(samples)
        for sample in samples:
            payload = sample["payload"]
            self.assertEqual(payload["schema"], "lumen_performance_trace_v2")
            self.assertEqual(
                payload["structure_resolution"]["source"],
                "simulated_demo",
            )
            self.assertEqual(
                payload["structure_resolution"]["axes"]["energy"]["source"],
                "simulated_demo",
            )
            self.assertEqual(
                payload["resolved_observation"]["section"],
                payload["structure_resolution"]["section"],
            )
            self.assertEqual(
                {item["fixture_id"] for item in payload["fixture_dmx"]},
                {
                    fixture.fixture_id
                    for fixture in (
                        *self.application.rig.fixtures,
                        *self.application.rig.auxiliary_fixtures,
                    )
                },
            )
            self.assertTrue(all(
                item["channels"] for item in payload["fixture_dmx"]
            ))

    def test_operator_structure_correction_is_visible_and_recalled_on_replay(
        self,
    ) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:operator-correction",
            title="Operator Correction",
            duration_ms=60_000,
            observed_position_ms=35_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        self.application._remember_media_identity(media)
        recording_id = self.application.memory.remember_recording_version(
            provider=media.provider,
            provider_item_id=media.provider_item_id,
            song_id=self.application.song_id,
            duration_ms=media.duration_ms,
        )
        run_id = self.application.memory.begin_teacher_run(
            teacher_name="EDMFormer",
            teacher_version="test",
            device="cpu",
            preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
            recording_id=recording_id,
        )
        base_id = self.application.memory.save_structure_timeline(
            provenance="edmformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.8,
            recording_id=recording_id,
            song_id=self.application.song_id,
            teacher_run_id=run_id,
            segments=[
                {
                    "start_ms": 0,
                    "end_ms": 30_000,
                    "energy_label": "groove",
                    "raw_label": "Drop",
                },
                {
                    "start_ms": 30_000,
                    "end_ms": 60_000,
                    "energy_label": "drop",
                    "raw_label": "Breakdown",
                },
            ],
        )
        self.application.memory.finish_teacher_run(run_id, status="complete")

        before = self.application.song_teaching_snapshot(force=True)
        self.assertEqual(before["recording_id"], recording_id)
        self.assertEqual(before["structure_timelines"][0]["id"], base_id)
        self.assertEqual(
            before["structure_timelines"][0]["segments"][1]["raw_label"],
            "Breakdown",
        )
        saved = self.application.correct_structure_timeline({
            "base_timeline_id": base_id,
            "participant_id": "console",
            "segments": [
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
                },
            ],
        })
        self.assertNotEqual(saved["timeline_id"], base_id)
        self.assertEqual(
            saved["cached_structure"]["axes"]["energy"]["label"],
            "breakdown",
        )

        # Simulate a later play: clear the in-memory answer and reconstruct it
        # exclusively from the stable Spotify recording identity.
        self.application._cached_structure_prediction = None
        self.application._poll_memory_context_once()
        replay = self.application._cached_structure_prediction
        assert replay is not None
        self.assertEqual(replay["recording"]["id"], recording_id)
        self.assertEqual(replay["axes"]["energy"]["label"], "breakdown")
        self.assertEqual(
            replay["axes"]["energy"]["timeline_id"], saved["timeline_id"]
        )
        after = self.application.song_teaching_snapshot(force=True)
        self.assertEqual(len(after["structure_timelines"]), 2)
        original = next(
            item for item in after["structure_timelines"] if item["id"] == base_id
        )
        self.assertEqual(original["segments"][1]["energy_label"], "drop")

    def test_composite_review_supersedes_both_teachers_and_wins_every_axis(
        self,
    ) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:composite-review",
            title="Composite Review",
            duration_ms=60_000,
            observed_position_ms=45_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        self.application._remember_media_identity(media)
        recording_id = self.application.memory.remember_recording_version(
            provider=media.provider,
            provider_item_id=media.provider_item_id,
            song_id=self.application.song_id,
            duration_ms=media.duration_ms,
        )
        edm_run = self.application.memory.begin_teacher_run(
            teacher_name="EDMFormer",
            teacher_version="test",
            device="cpu",
            preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
            recording_id=recording_id,
        )
        edm_timeline = self.application.memory.save_structure_timeline(
            provenance="edmformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.8,
            recording_id=recording_id,
            song_id=self.application.song_id,
            teacher_run_id=edm_run,
            segments=[{
                "start_ms": 0,
                "end_ms": 60_000,
                "energy_label": "groove",
            }],
        )
        self.application.memory.finish_teacher_run(edm_run, status="complete")
        song_run = self.application.memory.begin_teacher_run(
            teacher_name="SongFormer",
            teacher_version="test",
            device="cpu",
            preprocessing_version=(
                "songformer_official_features_cpu_windowed_v1:60s:"
                f"{TEACHER_NORMALIZATION_VERSION}"
            ),
            recording_id=recording_id,
        )
        song_timeline = self.application.memory.save_structure_timeline(
            provenance="songformer_teacher",
            timeline_version=TEACHER_NORMALIZATION_VERSION,
            confidence=0.7,
            recording_id=recording_id,
            song_id=self.application.song_id,
            teacher_run_id=song_run,
            segments=[{
                "start_ms": 0,
                "end_ms": 60_000,
                "functional_label": "chorus",
                "content_label": "vocal",
            }],
        )
        self.application.memory.finish_teacher_run(song_run, status="complete")

        saved = self.application.correct_structure_timeline({
            "base_timeline_id": edm_timeline,
            "recording_id": recording_id,
            "composite_review": True,
            "complete_review_timeline_ids": [edm_timeline, song_timeline],
            "participant_id": "console",
            "segments": [
                {
                    "segment_index": 0,
                    "start_ms": 0,
                    "end_ms": 30_000,
                    "functional_label": "verse",
                    "energy_label": "groove",
                    "content_label": "vocal",
                    "axis_sources": {
                        "functional": "SongFormer",
                        "energy": "EDMFormer",
                        "content": "SongFormer",
                    },
                },
                {
                    "segment_index": 1,
                    "start_ms": 30_000,
                    "end_ms": 60_000,
                    "functional_label": "chorus",
                    "energy_label": "drop",
                    "content_label": "vocal",
                },
            ],
        })

        correction = self.application.memory.structure_timeline(
            saved["timeline_id"]
        )
        assert correction is not None
        self.assertEqual(
            correction["timeline_version"],
            "lumen_operator_composite_v1",
        )
        self.assertEqual(
            correction["metadata"]["source_timeline_ids"],
            [edm_timeline, song_timeline],
        )
        self.assertEqual(
            correction["segments"][0]["provenance"]["axis_sources"],
            {
                "functional": "SongFormer",
                "energy": "EDMFormer",
                "content": "SongFormer",
            },
        )
        for timeline_id in (edm_timeline, song_timeline):
            review = self.application.memory.structure_timeline_review(
                timeline_id
            )
            assert review is not None
            self.assertEqual(review["status"], "superseded")
        library = self.application.structure_training_library(recording_id)
        self.assertEqual(library["needs_review"], 0)
        self.assertEqual(library["selected_recording"]["review_status"], "corrected")
        recalled = self.application.memory.cached_structure_at(
            recording_id=recording_id,
            playback_position_ms=45_000,
        )
        assert recalled is not None
        self.assertEqual(recalled["axes"]["functional"]["label"], "chorus")
        self.assertEqual(recalled["axes"]["energy"]["label"], "drop")
        self.assertEqual(
            recalled["axes"]["energy"]["timeline_id"],
            saved["timeline_id"],
        )
        original = self.application.memory.structure_timeline(edm_timeline)
        assert original is not None
        self.assertEqual(original["segments"][0]["energy_label"], "groove")
        with self.assertRaisesRegex(ValueError, "approved, rejected, or unreviewed"):
            self.application.review_structure_timeline({
                "timeline_id": edm_timeline,
                "recording_id": recording_id,
                "status": "superseded",
            })
        with self.assertRaisesRegex(ValueError, "revise the combined timeline"):
            self.application.review_structure_timeline({
                "timeline_id": edm_timeline,
                "recording_id": recording_id,
                "status": "unreviewed",
            })

        # A browser can retain both its raw base and source IDs while the first
        # save marks those teachers superseded. The stale request is rebased
        # onto the exact composite which completed those sources; it must not
        # append duplicate superseded reviews.
        revised = self.application.correct_structure_timeline({
            "base_timeline_id": edm_timeline,
            "recording_id": recording_id,
            "composite_review": True,
            "complete_review_timeline_ids": [edm_timeline, song_timeline],
            "participant_id": "console",
            "segments": [
                {
                    "segment_index": 0,
                    "start_ms": 0,
                    "end_ms": 30_000,
                    "functional_label": "verse",
                    "energy_label": "groove",
                    "content_label": "vocal",
                },
                {
                    "segment_index": 1,
                    "start_ms": 30_000,
                    "end_ms": 60_000,
                    "functional_label": "chorus",
                    "energy_label": "drop",
                    "content_label": "vocal",
                },
            ],
        })
        revised_timeline = self.application.memory.structure_timeline(
            revised["timeline_id"]
        )
        assert revised_timeline is not None
        self.assertEqual(revised["base_timeline_id"], saved["timeline_id"])
        self.assertEqual(
            revised_timeline["metadata"]["corrects_timeline_id"],
            saved["timeline_id"],
        )
        self.assertEqual(
            revised_timeline["metadata"]["source_timeline_ids"],
            [edm_timeline, song_timeline],
        )
        for timeline_id in (edm_timeline, song_timeline):
            review = self.application.memory.structure_timeline_review(
                timeline_id
            )
            assert review is not None
            self.assertIn(saved["timeline_id"], str(review["note"]))
            self.assertNotIn(revised["timeline_id"], str(review["note"]))

    def test_song_sequence_round_trips_canonical_cue_fields(self) -> None:
        media = MediaIdentity(
            provider="spotify",
            provider_item_id="spotify:track:cue-fields",
            title="Cue Fields",
            duration_ms=120_000,
            observed_position_ms=20_000,
            observed_at_unix_ms=round(time.time() * 1000),
            is_playing=False,
        )
        self.application._remember_media_identity(media)
        self.application.memory.remember_recording_version(
            provider=media.provider,
            provider_item_id=media.provider_item_id,
            song_id=self.application.song_id,
            duration_ms=media.duration_ms,
        )
        saved = self.application.save_choreography_proposal({
            "name": "Canonical cue",
            "scope": "movers",
            "place": True,
            "steps": [{
                "routine": "figure_eight",
                "duration_beats": 8,
                "motion_speed": 0.22,
                "travel_size": 0.83,
                "activity_density": 0.61,
                "brightness": 0.72,
                "palette": "midnight_teal",
                "strobe_enabled": True,
                "strobe_rate": 0.31,
                "beat_sync": 0.44,
                "cue_timing": 0.91,
            }],
        })
        stored = self.application.memory.choreography_sequence(
            saved["sequence_id"]
        )
        assert stored is not None
        parameters = stored["steps"][0]["parameters"]
        self.assertEqual(parameters["motion_speed"], 0.22)
        self.assertEqual(parameters["travel_size"], 0.83)
        self.assertEqual(parameters["activity_density"], 0.61)
        self.assertEqual(parameters["brightness"], 0.72)
        self.assertEqual(parameters["cue_timing"], 0.91)
        self.assertEqual(stored["steps"][0]["strobe"], {
            "enabled": True, "rate": 0.31,
        })

        self.application._poll_memory_context_once()
        self.assertEqual(len(self.application._prepared_recalled_choreography), 1)
        recalled = self.application._prepared_recalled_choreography[0].steps[0]
        self.assertEqual(recalled.motion_speed, 0.22)
        self.assertEqual(recalled.travel_size, 0.83)
        self.assertEqual(recalled.activity_density, 0.61)
        self.assertEqual(recalled.brightness, 0.72)
        self.assertTrue(recalled.strobe_enabled)
        self.assertEqual(recalled.strobe_rate, 0.31)
        self.assertEqual(recalled.beat_sync, 0.44)
        self.assertEqual(recalled.cue_timing, 0.91)

    @patch("lumen_engine.control.subprocess.run")
    @patch("lumen_engine.control.shutil.which", return_value="/usr/bin/amixer")
    def test_default_input_prepares_dedicated_line_mixer(
        self,
        _which: object,
        run: object,
    ) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""
        self.application._prepare_dedicated_line_input()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "amixer",
                    "-q",
                    "-c",
                    "0",
                    "sset",
                    "Input Source",
                    "Line",
                ],
                ["amixer", "-q", "-c", "0", "sset", "Capture", "0dB"],
            ],
        )
        self.assertIn(
            "0 dB",
            self.application.snapshot()["events"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()
