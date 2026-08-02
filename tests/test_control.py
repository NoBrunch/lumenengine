from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest.mock import patch

from lumen_engine.control import (
    LumenApplication,
    LumenHTTPServer,
    OperatorControls,
    OperatorExpressionEngine,
    RehearsalControls,
    _rehearsal_observation,
)
from lumen_engine.audio import AudioInputMetrics
from lumen_engine.expression import ExpressionPolicy
from lumen_engine.models import MusicalObservation
from lumen_engine.student import LABELS, StreamingStructureStudent
from lumen_engine.offline import EDMFORMER_JOB, SONGFORMER_JOB


class ControlApplicationTests(unittest.TestCase):
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
                {
                    "job_id": "two",
                    "job_type": SONGFORMER_JOB,
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
        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["failed"], 0)

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

    def test_feedback_persistence_never_holds_audio_publication_lock(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        original = self.application.memory.add_feedback

        def delayed(feedback):
            entered.set()
            release.wait(timeout=2.0)
            return original(feedback)

        with patch.object(
            self.application.memory, "add_feedback", side_effect=delayed
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

    def test_rehearsal_controls_validate_and_clamp(self) -> None:
        controls = RehearsalControls()
        controls.patch({
            "routine": "fan_sweep", "scope": "center", "output": "virtual",
            "bpm": 900, "size": -2, "intensity": 2, "strobe": 0.4,
        })
        self.assertEqual(controls.routine, "fan_sweep")
        self.assertEqual(controls.scope, "center")
        self.assertEqual(controls.bpm, 240.0)
        self.assertEqual(controls.size, 0.0)
        self.assertEqual(controls.intensity, 1.0)
        with self.assertRaises(ValueError):
            controls.patch({"routine": "not-a-routine"})

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
        self.assertTrue(self.application.motion_path.is_file())
        restored = self.application._load_motion_tunings()
        self.assertEqual(restored["figure_eight"].pan_size, 0.88)
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

    def test_weak_student_axes_are_visible_but_cannot_drive_choreography(self) -> None:
        self.application._student_prediction = {
            "functional": "chorus",
            "energy": "release",
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
        model.head_bias["energy"][LABELS["energy"].index("sustained")] = 8.0
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
        event_id = f"feedback:{result['feedback_id']}"
        state = self.application._choreography_model.state_dict()
        self.assertIn(event_id, state["events"])
        removed = self.application.delete_feedback(
            {"feedback_id": result["feedback_id"]}
        )
        self.assertTrue(removed["sequence_update_removed"])
        self.assertNotIn(
            event_id,
            self.application._choreography_model.state_dict()["events"],
        )
        self.application.stop()

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
