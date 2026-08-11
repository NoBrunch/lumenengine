from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import shutil
import wave
from urllib.request import urlopen

from lumen_engine.link import (
    LINK_SCHEMA,
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    LinkAuthenticator,
    LinkAuthenticationError,
    LinkClient,
    LinkConfiguration,
    LinkNodeExecutor,
    LinkNodeRuntime,
    LinkNodeServer,
    LinkProtocolError,
    LinkSpool,
    LumenLinkCoordinator,
    _job_asset_contract,
)
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SONGFORMER_PREPROCESSING_PREFIX,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.offline import (
    EDMFORMER_JOB,
    SONGFORMER_JOB,
    STUDENT_ACTIVATION_GATE_VERSION,
    STUDENT_AUDIO_FEATURE_VERSION,
    STUDENT_TRAIN_JOB,
    TEACHER_FUSION_VERSION,
)
from lumen_engine.student import StreamingStructureStudent


SECRET = b"0123456789abcdef0123456789abcdef"
TEST_CONTRACT = {
    "code_revision": "code-test",
    "code_clean": True,
    "teacher_revision": "teacher-test",
    "teacher_clean": True,
    "musicfm_source_revision": "musicfm-test",
    "musicfm_source_clean": True,
    "model_sha256": "a" * 64,
    "musicfm_stats_sha256": "b" * 64,
    "musicfm_model_sha256": "c" * 64,
    "muq_assets_sha256": "d" * 64,
    "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
    "edmformer_preprocessing_version": EDMFORMER_PREPROCESSING_VERSION,
}
SONG_TEST_CONTRACT = {
    "code_revision": "code-test",
    "code_clean": True,
    "songformer_revision": "songformer-test",
    "songformer_clean": True,
    "musicfm_source_revision": "musicfm-test",
    "musicfm_source_clean": True,
    "songformer_head_sha256": "e" * 64,
    "musicfm_stats_sha256": "b" * 64,
    "musicfm_model_sha256": "c" * 64,
    "muq_assets_sha256": "d" * 64,
    "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
    "songformer_preprocessing_version": (
        f"{SONGFORMER_PREPROCESSING_PREFIX}60s:"
        f"{TEACHER_NORMALIZATION_VERSION}"
    ),
}
STUDENT_TEST_CONTRACT = {
    "code_revision": "code-test",
    "code_clean": True,
    "student_format_version": StreamingStructureStudent.format_version,
    "student_audio_feature_version": STUDENT_AUDIO_FEATURE_VERSION,
    "student_activation_gate_version": STUDENT_ACTIVATION_GATE_VERSION,
    "teacher_fusion_version": TEACHER_FUSION_VERSION,
    "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
}


class FakeExecutor:
    def capabilities(self):
        return {
            "protocol_schema": LINK_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            "job_contracts": {
                EDMFORMER_JOB: TEST_CONTRACT,
                SONGFORMER_JOB: SONG_TEST_CONTRACT,
                STUDENT_TRAIN_JOB: STUDENT_TEST_CONTRACT,
            },
            **TEST_CONTRACT,
            "supported_job_types": [
                EDMFORMER_JOB,
                SONGFORMER_JOB,
                STUDENT_TRAIN_JOB,
            ],
            "gated_job_types": {},
            "max_threads": 24,
            "max_memory_bytes": 96 * 1024**3,
            "gpu": False,
            "live_timing": False,
            "dmx": False,
        }

    def execute(self, state, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"elapsed_s": 0.005, "rss_bytes": 1024})
        manifest = state["manifest"]
        audio = manifest["objects"][0]
        contract = self.capabilities()["job_contracts"][
            manifest["job_type"]
        ]
        return {
            "schema": RESULT_SCHEMA,
            "job_id": manifest["job_id"],
            "job_type": manifest["job_type"],
            "manifest_sha256": state["manifest_sha256"],
            "input_sha256": audio["sha256"],
            "duration_ms": manifest["identity"]["duration_ms"],
            **contract,
            "segments": [{"start": 0, "end": 1, "label": "drop"}],
            "resources": {"elapsed_s": 0.01},
        }


def manifest(
    job_id: str,
    digest: str,
    byte_count: int,
    *,
    job_type: str = EDMFORMER_JOB,
):
    contract = (
        TEST_CONTRACT
        if job_type == EDMFORMER_JOB
        else SONG_TEST_CONTRACT
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "job_id": job_id,
        "job_type": job_type,
        "identity": {
            "recording_id": "recording:test",
            "capture_session_id": "session:test",
            "song_id": 1,
            "duration_ms": 1_000,
        },
        "objects": [
            {
                "role": "audio",
                "sha256": digest,
                "bytes": byte_count,
                "format": "wav-pcm",
            }
        ],
        "contract": {
            **contract,
            "result_schema": RESULT_SCHEMA,
        },
        "resources": {"threads": 99},
        "created_unix_ms": 1,
    }


class LinkTests(unittest.TestCase):
    def test_student_asset_contract_does_not_require_manifest_objects(self):
        """Console compatibility scans must be independent of job manifests."""
        with tempfile.TemporaryDirectory() as temporary:
            contract = _job_asset_contract(
                STUDENT_TRAIN_JOB,
                Path(temporary) / "research",
                Path(temporary) / "project",
            )
        self.assertEqual(
            contract["student_format_version"],
            StreamingStructureStudent.format_version,
        )

    def test_teacher_manifest_accepts_pcm_recording_identity(self):
        """Queued recording metadata hashes PCM, while Link sends full WAV."""
        audio = self.root / "pcm-identity.wav"
        pcm = (b"\x01\x02\x03\x04" * 2048)
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(pcm)
        pcm_digest = hashlib.sha256(pcm).hexdigest()
        full_digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        coordinator = LumenLinkCoordinator(
            SongMemoryStore(self.root / "pcm-identity.sqlite3"),
            research_root=self.root / "research",
            state_root=self.root / "pcm-identity-state",
            config_path=self.root / "pcm-identity-state" / "config.json",
        )
        job = {
            "id": "job:pcm-identity",
            "job_type": EDMFORMER_JOB,
            "created_unix_ms": 1,
            "payload": {
                "audio_path": str(audio),
                "content_sha256": pcm_digest,
                "duration_ms": 1_000,
                "recording_id": "recording:pcm-identity",
            },
        }
        with patch.object(coordinator, "_local_contract", return_value={}):
            value = coordinator._manifest(job)
        self.assertEqual(value["objects"][0]["sha256"], full_digest)
        self.assertNotEqual(pcm_digest, full_digest)
        coordinator.close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spool = LinkSpool(self.root / "spool")
        self.runtime = LinkNodeRuntime(self.spool, FakeExecutor())
        self.server = LinkNodeServer(
            ("127.0.0.1", 0),
            spool=self.spool,
            authenticator=LinkAuthenticator(SECRET),
            runtime=self.runtime,
        )
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.runtime.start()
        self.server_thread.start()
        host, port = self.server.server_address
        self.client = LinkClient(f"http://{host}:{port}", SECRET)

    def tearDown(self):
        self.server.shutdown()
        self.runtime.stop()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temporary.cleanup()

    def test_health_is_mutually_authenticated_and_reports_gates(self):
        health = self.client.health()
        self.assertTrue(health["authenticated"])
        self.assertEqual(
            health["capabilities"]["supported_job_types"],
            [EDMFORMER_JOB, SONGFORMER_JOB, STUDENT_TRAIN_JOB],
        )
        self.assertNotIn("student.train", health["capabilities"]["gated_job_types"])
        with self.assertRaises(LinkProtocolError):
            LinkClient(self.client.endpoint, b"x" * 32).health()
        authenticator = LinkAuthenticator(SECRET)
        headers = authenticator.headers("GET", "/v1/health", hashlib.sha256(b"").hexdigest())
        verifier = LinkAuthenticator(SECRET)
        verifier.verify(
            "GET", "/v1/health", headers, hashlib.sha256(b"").hexdigest()
        )
        with self.assertRaises(LinkAuthenticationError):
            verifier.verify(
                "GET",
                "/v1/health",
                headers,
                hashlib.sha256(b"").hexdigest(),
            )

    def test_authenticated_health_marks_lumen_contact_on_dashboard(self):
        before = self.runtime.dashboard_status()
        self.assertEqual(before["connection"]["state"], "waiting")
        self.client.health()
        after = self.runtime.dashboard_status()
        self.assertEqual(after["connection"]["state"], "connected")
        self.assertLess(after["connection"]["last_contact_age_s"], 1.0)

    def test_threadripper_dashboard_is_read_only_and_identity_free(self):
        with urlopen(self.client.endpoint + "/dashboard", timeout=2) as response:
            html = response.read().decode()
        with urlopen(self.client.endpoint + "/dashboard/status", timeout=2) as response:
            status = json.loads(response.read())
        self.assertIn("Lumen Link · Threadripper", html)
        self.assertIn("active_slots", status)
        self.assertIn("maximum_parallel_jobs", status)
        self.assertIn("connection", status)
        self.assertNotIn("manifest", json.dumps(status))
        self.assertNotIn("recording_id", json.dumps(status))
        self.assertIn("cpu_usage_percent", status["node"])

    def test_threadripper_runs_six_teacher_jobs_concurrently(self):
        release = threading.Event()
        lock = threading.Lock()

        class BlockingExecutor(FakeExecutor):
            def __init__(self):
                self.active = 0
                self.peak = 0

            def execute(self, state, progress_callback=None):
                with lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    release.wait(timeout=3.0)
                    return super().execute(state, progress_callback)
                finally:
                    with lock:
                        self.active -= 1

        executor = BlockingExecutor()
        self.runtime.executor = executor
        source = self.root / "parallel.wav"
        source.write_bytes(b"RIFF-parallel-teacher-input")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        try:
            for index in range(6):
                self.client.submit(
                    manifest(f"job:parallel:{index}", digest, source.stat().st_size)
                )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and executor.peak < 6:
                time.sleep(0.02)
            self.assertEqual(executor.peak, 6)
            health = self.client.health()
            self.assertEqual(health["maximum_parallel_jobs"], 6)
            self.assertEqual(health["active_slots"], 6)
        finally:
            release.set()

    def test_chunk_upload_resumes_and_rejects_corruption(self):
        content = b"Lumen immutable recording" * 200
        source = self.root / "audio.wav"
        source.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        first = content[:1000]
        accepted = self.client.request(
            "PUT",
            f"/v1/objects/{digest}",
            first,
            {"Content-Range": f"bytes 0-999/{len(content)}"},
        )
        self.assertEqual(accepted["bytes"], 1000)
        completed = self.client.upload(source, digest)
        self.assertEqual(completed["resumed_from"], 1000)
        self.assertEqual(self.spool.object_path(digest).read_bytes(), content)

        bad_digest = hashlib.sha256(b"different").hexdigest()
        with self.assertRaises(LinkProtocolError):
            self.client.request(
                "PUT",
                f"/v1/objects/{bad_digest}",
                b"wrong",
                {"Content-Range": "bytes 0-4/5"},
            )
        self.assertFalse(self.spool.object_path(bad_digest).exists())

        produced = self.root / "candidate.npz"
        produced.write_bytes(b"immutable-result")
        descriptor = self.spool.publish_file(produced)
        downloaded = self.client.download(
            descriptor["sha256"],
            descriptor["bytes"],
            self.root / "downloaded.npz",
        )
        self.assertEqual(downloaded.read_bytes(), produced.read_bytes())
        resumed_target = self.root / "resumed.npz"
        resumed_partial = resumed_target.with_suffix(".npz.partial")
        resumed_partial.write_bytes(produced.read_bytes()[:5])
        resumed = self.client.download(
            descriptor["sha256"],
            descriptor["bytes"],
            resumed_target,
        )
        self.assertEqual(resumed.read_bytes(), produced.read_bytes())
        with self.assertRaises(LinkProtocolError):
            self.client.download(
                descriptor["sha256"],
                descriptor["bytes"] + 1,
                self.root / "bad-download.npz",
            )

        large = self.root / "large.wav"
        large.write_bytes(b"x" * (9 * 1024 * 1024))
        large_progress = []
        self.client.upload(
            large,
            progress_callback=lambda current, total: large_progress.append(
                (current, total)
            ),
        )
        self.assertGreaterEqual(len(large_progress), 2)
        self.assertLess(large_progress[0][0], large_progress[-1][0])
        self.assertEqual(large_progress[-1], (large.stat().st_size,) * 2)

    def test_submit_is_idempotent_and_result_survives_polling(self):
        source = self.root / "audio.wav"
        source.write_bytes(b"RIFF-test")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        value = manifest("job:test", digest, source.stat().st_size)
        first = self.client.submit(value)
        second = self.client.submit(value)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        deadline = time.time() + 3
        while time.time() < deadline:
            state = self.client.request("GET", "/v1/jobs/job:test")
            if state["status"] == "complete":
                break
            time.sleep(0.02)
        self.assertEqual(state["status"], "complete")
        result = self.client.request("GET", "/v1/jobs/job:test/result")
        self.assertEqual(result["input_sha256"], digest)
        self.assertEqual(result["code_revision"], "code-test")

    def test_songformer_uses_its_own_contract_end_to_end(self):
        source = self.root / "song.wav"
        source.write_bytes(b"RIFF-song")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        value = manifest(
            "job:song", digest, source.stat().st_size,
            job_type=SONGFORMER_JOB,
        )
        self.client.submit(value)
        deadline = time.time() + 3
        while time.time() < deadline:
            state = self.client.request("GET", "/v1/jobs/job:song")
            if state["status"] == "complete":
                break
            time.sleep(0.02)
        self.assertEqual(state["status"], "complete")
        result = self.client.request("GET", "/v1/jobs/job:song/result")
        self.assertEqual(result["job_type"], SONGFORMER_JOB)
        self.assertEqual(
            result["songformer_revision"], "songformer-test"
        )
        self.assertNotIn("teacher_revision", result)

    def test_node_restart_requeues_running_job(self):
        self.runtime.stop()
        source = self.root / "restart.wav"
        source.write_bytes(b"restart")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        state = self.spool.submit(
            manifest("job:restart", digest, source.stat().st_size)
        )
        self.spool.update_job(
            state["job_id"], status="running", stage="inference"
        )
        self.assertEqual(self.spool.recover_running(), 1)
        recovered = self.spool.job("job:restart")
        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["stage"], "recovered")

    def test_queued_job_can_be_canceled_before_execution(self):
        self.runtime.stop()
        source = self.root / "cancel.wav"
        source.write_bytes(b"cancel")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        self.client.submit(
            manifest("job:cancel", digest, source.stat().st_size)
        )
        canceled = self.client.request(
            "POST", "/v1/jobs/job:cancel/cancel", b"{}"
        )
        self.assertEqual(canceled["status"], "canceled")
        self.assertEqual(
            self.client.request("GET", "/v1/jobs/job:cancel")["stage"],
            "canceled",
        )

    def test_manifest_cannot_request_arbitrary_job_or_mismatched_code(self):
        source = self.root / "fixed.wav"
        source.write_bytes(b"fixed")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.client.upload(source, digest)
        value = manifest("job:fixed", digest, source.stat().st_size)
        value["job_type"] = "shell.command"
        value["command"] = ["touch", "/tmp/not-allowed"]
        with self.assertRaises(LinkProtocolError):
            self.client.submit(value)
        value = manifest("job:version", digest, source.stat().st_size)
        value["contract"]["code_revision"] = "wrong"
        with self.assertRaises(LinkProtocolError):
            self.client.submit(value)

    def test_executor_rejects_contract_cached_before_source_change(self):
        executor = LinkNodeExecutor(
            self.spool,
            research_root=self.root / "research",
            project_root=self.root / "project",
        )
        executor._capabilities_cache = {
            "supported_job_types": [STUDENT_TRAIN_JOB],
            "job_contracts": {
                STUDENT_TRAIN_JOB: dict(STUDENT_TEST_CONTRACT)
            },
            "gated_job_types": {},
        }
        executor._capability_signatures = {
            STUDENT_TRAIN_JOB: {"runner": [1, 1]}
        }
        changed_contract = {
            **STUDENT_TEST_CONTRACT,
            "code_clean": False,
        }
        with patch(
            "lumen_engine.link._job_asset_signature",
            return_value={"runner": [2, 2]},
        ), patch(
            "lumen_engine.link._job_asset_contract",
            return_value=changed_contract,
        ):
            with self.assertRaisesRegex(RuntimeError, "committed"):
                executor.validate_contract(
                    STUDENT_TRAIN_JOB, STUDENT_TEST_CONTRACT
                )

    def test_edmformer_executor_clamps_threads_to_runner_limit(self):
        research = self.root / "thread-limit-research"
        project = self.root / "thread-limit-project"
        checkpoint = research / "sources" / "edm98" / "data" / "checkpoints"
        for path in (
            research / "environments" / "edmformer" / "bin" / "python",
            project / "scripts" / "edmformer-cpu-runner.py",
            checkpoint / "model.pt",
            checkpoint / "msd_stats.json",
            checkpoint / "pretrained_msd.pt",
            research / "sources" / "edm98" / "configs" / "edmformer.yaml",
            research / "sources" / "musicfm" / "model" / "musicfm_25hz.py",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        audio_content = b"RIFF-thread-limit"
        digest = hashlib.sha256(audio_content).hexdigest()
        audio = self.spool.object_path(digest)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(audio_content)
        executor = LinkNodeExecutor(
            self.spool,
            research_root=research,
            project_root=project,
            max_threads=24,
        )
        executor._capabilities_cache = {
            "job_contracts": {EDMFORMER_JOB: dict(TEST_CONTRACT)}
        }
        state = {
            "manifest_sha256": "manifest",
            "manifest": {
                "job_id": "job:thread-limit",
                "job_type": EDMFORMER_JOB,
                "identity": {"duration_ms": 1_000},
                "objects": [{"role": "audio", "sha256": digest}],
                "resources": {"threads": 24},
            },
        }
        commands = []

        class CompletedProcess:
            returncode = 0

            def communicate(self, timeout=None):
                del timeout
                return "", ""

        def start_process(command, **kwargs):
            del kwargs
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps([{"label": "drop", "start": 0, "end": 1}]),
                encoding="utf-8",
            )
            return CompletedProcess()

        with patch("lumen_engine.link.subprocess.Popen", side_effect=start_process):
            result = executor._execute_teacher(state, None)
        command = commands[0]
        self.assertEqual(command[command.index("--threads") + 1], "8")
        self.assertEqual(result["resources"]["threads"], 8)

    def test_completed_remote_result_waits_for_live_to_stop_before_import(self):
        store = SongMemoryStore(self.root / "memory.sqlite3")
        audio = self.root / "deferred.wav"
        audio.write_bytes(b"deferred")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "content_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_ms": 1_000,
                "recording_id": "recording:deferred",
                "execution_target": "threadripper",
            },
        )
        job = store.claim_analysis_job_by_id(
            job_id,
            worker_id="lumen-link:test",
            worker_pid=123,
        )
        assert job is not None
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "research",
            state_root=self.root / "link-state",
            config_path=self.root / "link-state" / "config.json",
            can_import=lambda: False,
        )
        coordinator.active = {
            "job": job,
            "manifest": {"objects": [{"sha256": "0" * 64}]},
            "stage": "remote",
            "progress": 0.5,
        }

        class CompletedClient:
            def request(self, method, path):
                del method, path
                return {"status": "complete", "stage": "complete", "progress": 1.0}

        with patch.object(coordinator, "_client", return_value=CompletedClient()):
            coordinator._advance()
        self.assertEqual(coordinator.active["stage"], "awaiting_local_import")
        canonical = next(
            item for item in store.list_analysis_jobs() if item["id"] == job_id
        )
        self.assertEqual(canonical["status"], "running")

    def test_live_remote_poll_is_memory_only_and_heavy_work_is_parked(self):
        store = SongMemoryStore(self.root / "live-poll.sqlite3")
        audio = self.root / "live-poll.wav"
        audio.write_bytes(b"remote")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "duration_ms": 1_000,
                "recording_id": "recording:live-poll",
                "execution_target": "threadripper",
            },
        )
        job = store.claim_analysis_job_by_id(
            job_id, worker_id="lumen-link:live", worker_pid=1
        )
        assert job is not None
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "live-poll-research",
            state_root=self.root / "live-poll-state",
            config_path=self.root / "live-poll-state" / "config.json",
            can_import=lambda: False,
        )
        coordinator.active = {
            "job": job,
            "manifest": {"objects": [{"sha256": "0" * 64}]},
            "stage": "remote",
            "progress": None,
        }

        class RunningClient:
            def request(self, method, path):
                del method, path
                return {
                    "status": "running",
                    "stage": "inference",
                    "progress": None,
                }

        with patch.object(coordinator, "_client", return_value=RunningClient()), patch.object(
            store, "heartbeat_analysis_job"
        ) as heartbeat, patch.object(coordinator, "_persist") as persist, patch.object(
            store, "list_analysis_jobs", wraps=store.list_analysis_jobs
        ) as scans:
            coordinator._advance()
            self.assertEqual(coordinator.route_queued_jobs(), 0)
            coordinator.status()
        heartbeat.assert_not_called()
        persist.assert_not_called()
        scans.assert_not_called()
        self.assertEqual(coordinator.active["stage"], "inference")

    def test_live_transition_waits_for_link_io_checkpoint(self):
        coordinator = LumenLinkCoordinator(
            SongMemoryStore(self.root / "guard.sqlite3"),
            research_root=self.root / "guard-research",
            state_root=self.root / "guard-state",
            config_path=self.root / "guard-state" / "config.json",
        )
        entered = threading.Event()
        release = threading.Event()
        live_entered = threading.Event()

        def heavy_work():
            with coordinator._standby_guard():
                entered.set()
                release.wait(2.0)

        worker = threading.Thread(target=heavy_work)
        worker.start()
        self.assertTrue(entered.wait(1.0))

        def start_live():
            with coordinator.live_transition_guard():
                live_entered.set()

        starter = threading.Thread(target=start_live)
        starter.start()
        self.assertFalse(live_entered.wait(0.05))
        release.set()
        self.assertTrue(live_entered.wait(1.0))
        worker.join(1.0)
        starter.join(1.0)

    def test_completed_job_reports_returning_then_importing(self):
        store = SongMemoryStore(self.root / "return-stages.sqlite3")
        audio = self.root / "return.wav"
        audio.write_bytes(b"return")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "content_sha256": hashlib.sha256(
                    audio.read_bytes()
                ).hexdigest(),
                "duration_ms": 1_000,
                "recording_id": "recording:return",
                "execution_target": "threadripper",
            },
        )
        job = store.claim_analysis_job_by_id(
            job_id, worker_id="lumen-link:stage", worker_pid=1
        )
        assert job is not None
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "return-research",
            state_root=self.root / "return-state",
            config_path=self.root / "return-state" / "config.json",
        )
        coordinator.active = {
            "job": job,
            "manifest": {"objects": [{"sha256": "0" * 64}]},
            "stage": "remote",
            "progress": None,
        }

        class CompletedClient:
            def request(self, method, path):
                del method
                if path.endswith("/result"):
                    return {"result": True}
                return {
                    "status": "complete",
                    "stage": "complete",
                    "progress": 1.0,
                }

        stages = []
        original_persist = coordinator._persist

        def capture_persist():
            if coordinator.active:
                stages.append(coordinator.active.get("stage"))
            original_persist()

        with patch.object(coordinator, "_client", return_value=CompletedClient()), patch.object(
            coordinator, "_persist", side_effect=capture_persist
        ), patch.object(
            coordinator, "_import_teacher", return_value={"imported": True}
        ):
            coordinator._advance()
        self.assertIn("returning", stages)
        self.assertIn("importing", stages)
        self.assertLess(stages.index("returning"), stages.index("importing"))
        status = coordinator.status()
        self.assertEqual(status["queue"]["locally_imported"], 1)
        self.assertEqual(status["recent_imports"][0]["job_id"], job_id)
        imported_job = next(
            item for item in status["jobs"] if item.get("job_id") == job_id
        )
        self.assertTrue(imported_job["locally_imported"])
        self.assertEqual(imported_job["local_import_state"], "imported")

    def test_coordinator_reclaims_its_persisted_job_after_restart(self):
        store = SongMemoryStore(self.root / "restart-memory.sqlite3")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"execution_target": "threadripper"},
        )
        job = next(item for item in store.list_analysis_jobs() if item["id"] == job_id)
        state_root = self.root / "coordinator-restart"
        state_root.mkdir(parents=True)
        (state_root / "coordinator.json").write_text(
            json.dumps(
                {
                    "schema": LINK_SCHEMA,
                    "active": {
                        "job": job,
                        "manifest": {"schema": MANIFEST_SCHEMA},
                        "stage": "remote",
                    },
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "research-restart",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        coordinator._reconcile_active()
        self.assertIsNotNone(coordinator.active)
        self.assertEqual(coordinator.active["job"]["status"], "running")
        self.assertTrue(
            coordinator.active["job"]["worker_id"].startswith("lumen-link:")
        )

    def test_enabled_coordinator_routes_automatic_edmformer_off_local_worker(self):
        store = SongMemoryStore(self.root / "route-memory.sqlite3")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB, payload={"recording_id": "route"}
        )
        state_root = self.root / "route-state"
        state_root.mkdir(parents=True)
        (state_root / "secret").write_bytes(SECRET)
        (state_root / "config.json").write_text(
            json.dumps(
                {
                    "endpoint": "http://127.0.0.1:1",
                    "secret_file": "secret",
                    "enabled": True,
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "route-research",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        coordinator.remote_status = {
            "authenticated": True,
            "capabilities": {
                **TEST_CONTRACT,
                "supported_job_types": [EDMFORMER_JOB],
            },
        }
        coordinator._local_contract_cache = dict(TEST_CONTRACT)
        self.assertEqual(coordinator.route_queued_jobs(), 1)
        routed = next(
            item for item in store.list_analysis_jobs() if item["id"] == job_id
        )
        self.assertEqual(routed["payload"]["execution_target"], "threadripper")
        self.assertIsNone(
            store.claim_analysis_job(
                (EDMFORMER_JOB,),
                worker_id="local",
                worker_pid=1,
                execution_targets=("automatic", "local"),
            )
        )

    def test_coordinator_does_not_route_before_authenticated_compatible_health(self):
        store = SongMemoryStore(self.root / "no-health.sqlite3")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB, payload={"recording_id": "wait"}
        )
        state_root = self.root / "no-health"
        state_root.mkdir(parents=True)
        (state_root / "secret").write_bytes(SECRET)
        (state_root / "config.json").write_text(
            json.dumps(
                {
                    "endpoint": "http://127.0.0.1:1",
                    "secret_file": "secret",
                    "enabled": True,
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "no-health-research",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        self.assertEqual(coordinator.route_queued_jobs(), 0)
        job = next(item for item in store.list_analysis_jobs() if item["id"] == job_id)
        self.assertNotIn("execution_target", job["payload"])

    def test_incompatible_worker_is_named_and_enable_returns_update_instruction(self):
        store = SongMemoryStore(self.root / "revision-mismatch.sqlite3")
        state_root = self.root / "revision-mismatch"
        state_root.mkdir(parents=True)
        (state_root / "secret").write_bytes(SECRET)
        (state_root / "config.json").write_text(
            json.dumps(
                {
                    "endpoint": "http://127.0.0.1:1",
                    "secret_file": "secret",
                    "enabled": False,
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "revision-mismatch-research",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        local_contracts = {
            EDMFORMER_JOB: {**TEST_CONTRACT, "code_revision": "local-revision"},
            SONGFORMER_JOB: {**SONG_TEST_CONTRACT, "code_revision": "local-revision"},
            STUDENT_TRAIN_JOB: {**STUDENT_TEST_CONTRACT, "code_revision": "local-revision"},
        }
        remote_contracts = {
            job_type: {**contract, "code_revision": "remote-revision"}
            for job_type, contract in local_contracts.items()
        }
        coordinator._local_contract_cache = local_contracts
        coordinator.remote_status = {
            "authenticated": True,
            "capabilities": {
                "supported_job_types": list(remote_contracts),
                "job_contracts": remote_contracts,
            },
        }

        status = coordinator.status()
        self.assertEqual(status["connection"]["state"], "incompatible")
        self.assertIn("remote-", status["connection"]["detail"])
        self.assertIn("local-r", status["setup"]["next_action"])
        self.assertIn("git pull --ff-only", status["setup"]["commands"])

        with patch.object(coordinator, "_poll_health"):
            with self.assertRaisesRegex(RuntimeError, "Update and restart"):
                coordinator.control("enable")
        self.assertFalse(
            json.loads((state_root / "config.json").read_text())["enabled"]
        )

    def test_coordinator_routes_mixed_teachers_one_at_a_time(self):
        store = SongMemoryStore(self.root / "mixed-route.sqlite3")
        edm_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "edm"},
            priority=20,
        )
        song_id = store.enqueue_analysis_job(
            job_type=SONGFORMER_JOB,
            payload={"recording_id": "song"},
            priority=10,
        )
        state_root = self.root / "mixed-route"
        state_root.mkdir(parents=True)
        (state_root / "secret").write_bytes(SECRET)
        (state_root / "config.json").write_text(
            json.dumps(
                {
                    "endpoint": "http://127.0.0.1:1",
                    "secret_file": "secret",
                    "enabled": True,
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "mixed-research",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        coordinator.remote_status = {
            "authenticated": True,
            "capabilities": {
                "supported_job_types": [EDMFORMER_JOB, SONGFORMER_JOB],
                "job_contracts": {
                    EDMFORMER_JOB: TEST_CONTRACT,
                    SONGFORMER_JOB: SONG_TEST_CONTRACT,
                },
            },
        }
        coordinator._local_contract_cache = {
            EDMFORMER_JOB: dict(TEST_CONTRACT),
            SONGFORMER_JOB: dict(SONG_TEST_CONTRACT),
        }
        self.assertEqual(coordinator.route_queued_jobs(), 1)
        jobs = {item["id"]: item for item in store.list_analysis_jobs()}
        self.assertEqual(
            jobs[edm_id]["payload"]["execution_target"], "threadripper"
        )
        self.assertNotIn("execution_target", jobs[song_id]["payload"])
        self.assertEqual(coordinator.route_queued_jobs(), 0)
        store.update_analysis_job(edm_id, status="complete")
        self.assertEqual(coordinator.route_queued_jobs(), 1)
        jobs = {item["id"]: item for item in store.list_analysis_jobs()}
        self.assertEqual(
            jobs[song_id]["payload"]["execution_target"], "threadripper"
        )

    def test_student_manifest_contains_every_coherent_recording_wav(self):
        store = SongMemoryStore(self.root / "student-manifest.sqlite3")
        recording_ids = ("recording:one", "recording:two")
        for index, recording_id in enumerate(recording_ids):
            audio = self.root / f"student-{index}.wav"
            audio.write_bytes(f"RIFF-{index}".encode())
            store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={
                    "recording_id": recording_id,
                    "audio_path": str(audio),
                    "content_sha256": hashlib.sha256(
                        audio.read_bytes()
                    ).hexdigest(),
                },
            )
        examples = self.root / "student.jsonl"
        examples.write_text(
            "".join(
                json.dumps(
                    {
                        "recording_id": recording_id,
                        "split_group_id": recording_id,
                        "split": "train",
                    }
                )
                + "\n"
                for recording_id in recording_ids
            ),
            encoding="utf-8",
        )
        job_id = store.enqueue_analysis_job(
            job_type=STUDENT_TRAIN_JOB,
            payload={
                "examples_path": str(examples),
                "examples_sha256": hashlib.sha256(
                    examples.read_bytes()
                ).hexdigest(),
                "output_path": str(self.root / "student.npz"),
                "epochs": 1,
            },
        )
        job = next(
            item for item in store.list_analysis_jobs() if item["id"] == job_id
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "student-research",
            state_root=self.root / "student-state",
            config_path=self.root / "student-state" / "config.json",
        )
        coordinator._local_contract_cache = {
            STUDENT_TRAIN_JOB: dict(STUDENT_TEST_CONTRACT)
        }
        coordinator._local_contract_signatures[STUDENT_TRAIN_JOB] = {
            "test": True
        }
        with patch(
            "lumen_engine.link._job_asset_signature",
            return_value={"test": True},
        ):
            value = coordinator._student_manifest(job)
            coordinator._refresh_job_snapshot(
                force=True, precompute_students=True
            )
        audio_objects = [
            item
            for item in value["objects"]
            if item["role"] == "recording_audio"
        ]
        self.assertEqual(
            {item["recording_id"] for item in audio_objects},
            set(recording_ids),
        )
        self.assertEqual(len(value["objects"]), 3)
        expected_unique_bytes = examples.stat().st_size + sum(
            (self.root / f"student-{index}.wav").stat().st_size
            for index in range(2)
        )
        self.assertEqual(
            coordinator.status()["queue"]["bytes_pending"],
            expected_unique_bytes,
        )

    def test_stale_student_snapshot_fails_only_its_job(self):
        store = SongMemoryStore(self.root / "stale-student.sqlite3")
        examples = self.root / "mutable-student.jsonl"
        examples.write_text('{"recording_id":"one"}\n', encoding="utf-8")
        job_id = store.enqueue_analysis_job(
            job_type=STUDENT_TRAIN_JOB,
            payload={
                "examples_path": str(examples),
                "examples_sha256": "0" * 64,
            },
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "stale-student-research",
            state_root=self.root / "stale-student-state",
            config_path=self.root / "stale-student-state" / "config.json",
        )
        coordinator._refresh_job_snapshot(
            force=True, precompute_students=True
        )
        job = next(
            item for item in store.list_analysis_jobs()
            if item["id"] == job_id
        )
        self.assertEqual(job["status"], "failed")
        self.assertIn("changed after queueing", job["error"])
        self.assertIsNone(coordinator.last_error)

    def test_manifest_conflict_rekeys_transport_job_after_upgrade(self):
        coordinator = LumenLinkCoordinator(
            SongMemoryStore(self.root / "rekey.sqlite3"),
            research_root=self.root / "rekey-research",
            state_root=self.root / "rekey-state",
            config_path=self.root / "rekey-state" / "config.json",
        )

        class ConflictOnceClient:
            def __init__(self):
                self.manifests = []

            def submit(self, value):
                self.manifests.append(dict(value))
                if len(self.manifests) == 1:
                    raise LinkProtocolError(
                        "compute node returned HTTP 400: job ID already has "
                        "a different manifest"
                    )
                return {"status": "queued"}

        client = ConflictOnceClient()
        job = {"id": "job:canonical"}
        original = {
            "schema": MANIFEST_SCHEMA,
            "job_id": job["id"],
            "job_type": EDMFORMER_JOB,
        }
        submitted, remote = coordinator._submit_manifest(
            client, job, original
        )
        self.assertEqual(len(client.manifests), 2)
        self.assertEqual(submitted, client.manifests[-1])
        self.assertTrue(
            submitted["job_id"].startswith("job:canonical.manifest-")
        )
        self.assertEqual(original["job_id"], "job:canonical")
        self.assertEqual(remote["status"], "queued")

    def test_prefill_skips_completed_results_and_fills_real_slots(self):
        coordinator = LumenLinkCoordinator(
            SongMemoryStore(self.root / "prefill.sqlite3"),
            research_root=self.root / "prefill-research",
            state_root=self.root / "prefill-state",
            config_path=self.root / "prefill-state" / "config.json",
        )
        coordinator.remote_status = {
            "capabilities": {"maximum_parallel_jobs": 2},
            "jobs": [],
        }
        coordinator._job_snapshot = [
            {
                "id": f"job:prefill:{index}",
                "job_type": EDMFORMER_JOB,
                "status": "queued",
                "priority": 0,
                "created_unix_ms": index,
                "payload": {"execution_target": "threadripper"},
            }
            for index in range(3)
        ]

        class PrefillClient:
            def __init__(self):
                self.submitted = []

            def submit(self, manifest):
                self.submitted.append(dict(manifest))
                return {
                    "status": (
                        "complete" if len(self.submitted) == 1 else "queued"
                    )
                }

        client = PrefillClient()
        with patch.object(
            coordinator, "_remote_is_compatible", return_value=True
        ), patch.object(
            coordinator,
            "_manifest",
            side_effect=lambda job, jobs=None: {
                "job_id": job["id"],
                "objects": [],
            },
        ), patch.object(
            coordinator, "_object_sources", return_value=[]
        ), patch.object(
            coordinator, "_client", return_value=client
        ):
            self.assertEqual(coordinator._prefill_remote_queue(), 2)
            self.assertEqual(len(client.submitted), 3)
            self.assertEqual(coordinator._prefill_remote_queue(), 0)
            self.assertEqual(len(client.submitted), 3)

    def test_routing_maintains_standby_buffer_during_active_import(self):
        store = SongMemoryStore(self.root / "route-buffer.sqlite3")
        job_ids = [
            store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={"execution_target": "automatic"},
            )
            for _ in range(3)
        ]
        store.set_analysis_job_execution_target(
            job_ids[0], execution_target="threadripper"
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "route-buffer-research",
            state_root=self.root / "route-buffer-state",
            config_path=self.root / "route-buffer-state" / "config.json",
        )
        coordinator.configuration = LinkConfiguration(
            "http://127.0.0.1:1", SECRET, enabled=True
        )
        coordinator.remote_status = {
            "capabilities": {"maximum_parallel_jobs": 2}
        }
        coordinator.active = {"job": {"id": "job:being-imported"}}
        coordinator._submitted_local_job_ids.add(job_ids[0])
        with patch.object(
            coordinator, "_remote_is_compatible", return_value=True
        ):
            self.assertEqual(coordinator.route_queued_jobs(), 2)
        jobs = {job["id"]: job for job in store.list_analysis_jobs()}
        self.assertTrue(
            all(
                jobs[job_id]["payload"]["execution_target"]
                == "threadripper"
                for job_id in job_ids
            )
        )

    def test_refresh_quarantines_legacy_partial_teacher_without_blocking_queue(
        self,
    ):
        store = SongMemoryStore(self.root / "partial-link.sqlite3")
        audio = self.root / "partial-link.wav"
        audio.write_bytes(b"partial")
        bad_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "recording_id": "recording:partial",
                "execution_target": "threadripper",
            },
        )
        good_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "recording_id": "recording:complete",
                "structure_supervision": {"eligible": True},
                "execution_target": "automatic",
            },
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "partial-link-research",
            state_root=self.root / "partial-link-state",
            config_path=self.root / "partial-link-state" / "config.json",
        )

        snapshot = {
            job["id"]: job
            for job in coordinator._refresh_job_snapshot(force=True)
        }

        self.assertEqual(snapshot[bad_id]["status"], "failed")
        self.assertIn("whole-song supervision", snapshot[bad_id]["error"])
        self.assertEqual(snapshot[good_id]["status"], "queued")
        self.assertTrue(any(
            event["kind"] == "warning" and "quarantined 1" in event["message"]
            for event in coordinator.events
        ))

    def test_workload_completed_is_durable_total_not_recent_receipt_window(
        self,
    ):
        store = SongMemoryStore(self.root / "link-totals.sqlite3")
        for index in range(25):
            job_id = store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload={"execution_target": "threadripper", "index": index},
            )
            store.update_analysis_job(
                job_id,
                status="complete",
                result={"execution_target": "threadripper"},
            )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "link-totals-research",
            state_root=self.root / "link-totals-state",
            config_path=self.root / "link-totals-state" / "config.json",
        )
        coordinator._refresh_job_snapshot(force=True)

        status = coordinator.status()

        self.assertEqual(status["queue"]["completed"], 25)
        self.assertEqual(status["queue"]["locally_imported"], 25)
        self.assertEqual(status["queue"]["recent_imports"], 0)
        self.assertEqual(status["queue"]["link"]["complete"], 25)

    def test_deterministic_local_import_rejection_fails_once_and_advances(self):
        store = SongMemoryStore(self.root / "import-reject.sqlite3")
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"execution_target": "threadripper"},
        )
        job = store.claim_analysis_job_by_id(
            job_id, worker_id="lumen-link:test", worker_pid=1
        )
        assert job is not None
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "import-reject-research",
            state_root=self.root / "import-reject-state",
            config_path=self.root / "import-reject-state" / "config.json",
        )
        coordinator.active = {
            "job": job,
            "manifest": {"job_id": job_id, "objects": []},
            "stage": "remote",
        }

        class CompletedClient:
            def request(self, method, path):
                del method
                if path.endswith("/result"):
                    return {"result": True}
                return {"status": "complete", "stage": "complete"}

        with patch.object(
            coordinator, "_client", return_value=CompletedClient()
        ), patch.object(
            coordinator,
            "_import_teacher",
            side_effect=ValueError("partial recording"),
        ):
            coordinator._advance()

        stored = {
            item["id"]: item for item in store.list_analysis_jobs()
        }[job_id]
        self.assertEqual(stored["status"], "failed")
        self.assertIn("local import rejected", stored["error"])
        self.assertIsNone(coordinator.active)
        self.assertTrue(any(
            "local import rejected" in event["message"]
            for event in coordinator.events
        ))

    def test_student_executor_runs_fixed_child_and_publishes_artifacts(self):
        """Exercise the real Link runner boundary without a model mock."""
        spool = LinkSpool(self.root / "student-executor-spool")
        recordings = []
        rows = []
        for split in ("train", "test"):
            recording_id = f"recording:{split}"
            audio = self.root / f"student-executor-{split}.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8_000)
                output.writeframes(b"\0\0" * 8_000)
            descriptor = spool.publish_file(audio)
            recordings.append(
                {
                    "role": "recording_audio",
                    "recording_id": recording_id,
                    "sha256": descriptor["sha256"],
                    "bytes": descriptor["bytes"],
                    "format": "wav-pcm",
                }
            )
            for index in range(10):
                rows.append(
                    {
                        "recording_id": recording_id,
                        "recording_offset_ms": index * 100,
                        "audio_frame_index": index,
                        "split_group_id": f"group:{split}",
                        "split": split,
                        "features": [0.0] * 15,
                        "functional": "unknown",
                        "energy": "groove" if index < 5 else "build",
                        "content": "unknown",
                        "boundary": 0,
                    }
                )
        examples = self.root / "student-executor.jsonl"
        examples.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        example_descriptor = spool.publish_file(examples)
        job_manifest = {
            "schema": MANIFEST_SCHEMA,
            "job_id": "job:student-executor",
            "job_type": STUDENT_TRAIN_JOB,
            "identity": {
                "recording_ids": ["recording:test", "recording:train"]
            },
            "objects": [
                {
                    "role": "student_examples",
                    "sha256": example_descriptor["sha256"],
                    "bytes": example_descriptor["bytes"],
                    "format": "jsonl",
                },
                *recordings,
            ],
            "contract": {
                **STUDENT_TEST_CONTRACT,
                "result_schema": RESULT_SCHEMA,
            },
            "training": {
                "epochs": 1,
                "hidden_size": 8,
                "applicable_axes": ["energy"],
            },
            "resources": {"threads": 1},
            "created_unix_ms": 1,
        }
        state = {
            "manifest": job_manifest,
            "manifest_sha256": hashlib.sha256(
                json.dumps(
                    job_manifest, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        executor = LinkNodeExecutor(
            spool,
            research_root=self.root / "student-executor-research",
            project_root=Path(__file__).resolve().parents[1],
            max_threads=1,
            max_memory_gib=2,
        )
        stages = []
        with patch.object(
            executor,
            "capabilities",
            return_value={
                "job_contracts": {
                    STUDENT_TRAIN_JOB: STUDENT_TEST_CONTRACT
                }
            },
        ):
            result = executor._execute_student(
                state,
                lambda progress: stages.append(progress.get("stage")),
            )
        self.assertEqual(result["job_type"], STUDENT_TRAIN_JOB)
        self.assertEqual(
            set(result["artifacts"]),
            {"candidate_model", "evaluation", "prepared_examples"},
        )
        for descriptor in result["artifacts"].values():
            artifact = spool.object_path(descriptor["sha256"])
            self.assertTrue(artifact.is_file())
            self.assertEqual(artifact.stat().st_size, descriptor["bytes"])
        self.assertIn("student_feature_preparation", stages)
        self.assertIn("student_training", stages)
        self.assertIn("student_validation", stages)

    def test_student_import_revalidates_activates_and_is_idempotent(self):
        store = SongMemoryStore(self.root / "student-import.sqlite3")
        examples = self.root / "original.jsonl"
        original_row = {
            "recording_id": "recording:student",
            "recording_offset_ms": 0,
            "split_group_id": "group:student",
            "split": "test",
            "features": [0.0] * 15,
            "functional": "intro",
            "energy": "groove",
            "content": "instrumental",
            "boundary": 0,
        }
        examples.write_text(json.dumps(original_row) + "\n", encoding="utf-8")
        output = self.root / "models" / "student.npz"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"previous-model")
        output.with_name("student.evaluation.json").write_text(
            "{}\n", encoding="utf-8"
        )
        job = {
            "id": "job:student-import",
            "job_type": STUDENT_TRAIN_JOB,
            "payload": {
                "examples_path": str(examples),
                "output_path": str(output),
                "applicable_axes": ["energy"],
            },
        }
        example_digest = hashlib.sha256(examples.read_bytes()).hexdigest()
        job_manifest = {
            "schema": MANIFEST_SCHEMA,
            "job_id": job["id"],
            "job_type": STUDENT_TRAIN_JOB,
            "identity": {"recording_ids": ["recording:student"]},
            "objects": [
                {
                    "role": "student_examples",
                    "sha256": example_digest,
                    "bytes": examples.stat().st_size,
                    "format": "jsonl",
                }
            ],
            "contract": {
                **STUDENT_TEST_CONTRACT,
                "result_schema": RESULT_SCHEMA,
            },
        }
        candidate = self.root / "remote-candidate.npz"
        candidate.write_bytes(b"verified-candidate")
        prepared = self.root / "remote-prepared.jsonl"
        prepared_row = {
            **original_row,
            "features": [0.25] * 15,
            "feature_preprocessing_version": STUDENT_AUDIO_FEATURE_VERSION,
        }
        prepared.write_text(json.dumps(prepared_row) + "\n", encoding="utf-8")
        local_gate = {
            "activated": True,
            "approved_axes": ["energy"],
            "inactive_axes": [],
            "not_applicable_axes": ["boundary", "content", "functional"],
            "held_out_split": "test",
            "evaluation": {"test": {"energy": {"accuracy": 1.0}}},
            "axis_gate_reasons": {},
            "gate_reasons": [],
            "test_population_reliable": True,
            "split_counts": {"train": 0, "validation": 0, "test": 1},
            "split_group_counts": {"train": 0, "validation": 0, "test": 5},
            "label_balance": {},
        }
        remote_report = {
            "activated": True,
            "approved_axes": ["energy"],
            "evaluation": local_gate["evaluation"],
        }
        evaluation = self.root / "remote-evaluation.json"
        evaluation.write_text(
            json.dumps(remote_report, sort_keys=True), encoding="utf-8"
        )

        def descriptor(path):
            return {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }

        source_by_digest = {
            descriptor(candidate)["sha256"]: candidate,
            descriptor(prepared)["sha256"]: prepared,
            descriptor(evaluation)["sha256"]: evaluation,
        }

        class ArtifactClient:
            def __init__(self):
                self.downloads = 0

            def download(self, digest, byte_count, target):
                self.downloads += 1
                source = source_by_digest[digest]
                if source.stat().st_size != byte_count:
                    raise AssertionError("test descriptor mismatch")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                return target

        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job["id"],
            "job_type": STUDENT_TRAIN_JOB,
            "manifest_sha256": hashlib.sha256(
                json.dumps(
                    job_manifest, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "input_sha256": example_digest,
            **STUDENT_TEST_CONTRACT,
            "artifacts": {
                "candidate_model": {
                    **descriptor(candidate),
                    "format": "numpy-npz",
                },
                "evaluation": {
                    **descriptor(evaluation),
                    "format": "json",
                },
                "prepared_examples": {
                    **descriptor(prepared),
                    "format": "jsonl",
                },
            },
            "report": remote_report,
            "resources": {"elapsed_s": 1.0},
        }
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "student-import-research",
            state_root=self.root / "student-import-state",
            config_path=self.root / "student-import-state" / "config.json",
        )
        coordinator.active = {"stage": "importing", "progress": None}

        client = ArtifactClient()
        rejected_gate = {
            **local_gate,
            "activated": False,
            "approved_axes": [],
            "inactive_axes": ["energy"],
        }

        with patch.object(
            coordinator,
            "_validate_student_candidate_isolated",
            return_value=rejected_gate,
        ):
            with self.assertRaisesRegex(
                LinkProtocolError,
                "does not reproduce",
            ):
                coordinator._import_student(
                    client, job, job_manifest, result
                )
        with patch.object(
            coordinator,
            "_validate_student_candidate_isolated",
            return_value=local_gate,
        ):
            imported = coordinator._import_student(
                client, job, job_manifest, result
            )
            reused = coordinator._import_student(
                client, job, job_manifest, result
            )
        self.assertTrue(imported["activated"])
        self.assertTrue(imported["local_revalidated"])
        self.assertEqual(
            coordinator.active["stage"], "student_activation_commit"
        )
        self.assertEqual(coordinator.active["progress"], 0.90)
        self.assertTrue(reused["import_reused"])
        self.assertEqual(client.downloads, 6)
        self.assertEqual(output.read_bytes(), candidate.read_bytes())
        self.assertEqual(
            output.with_name("student.previous.npz").read_bytes(),
            b"previous-model",
        )

        rejected_output = self.root / "models" / "rejected.npz"
        rejected_output.write_bytes(b"still-active")
        rejected_job = {
            **job,
            "id": "job:student-rejected",
            "payload": {**job["payload"], "output_path": str(rejected_output)},
        }
        rejected_manifest = {
            **job_manifest,
            "job_id": rejected_job["id"],
        }
        rejected_report = {
            "activated": False,
            "approved_axes": [],
            "evaluation": local_gate["evaluation"],
        }
        rejected_evaluation = self.root / "remote-rejected-evaluation.json"
        rejected_evaluation.write_text(
            json.dumps(rejected_report, sort_keys=True), encoding="utf-8"
        )
        source_by_digest[descriptor(rejected_evaluation)["sha256"]] = (
            rejected_evaluation
        )
        rejected_result = {
            **result,
            "job_id": rejected_job["id"],
            "manifest_sha256": hashlib.sha256(
                json.dumps(
                    rejected_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "report": rejected_report,
            "artifacts": {
                **result["artifacts"],
                "evaluation": {
                    **descriptor(rejected_evaluation),
                    "format": "json",
                },
            },
        }
        with patch.object(
            coordinator,
            "_validate_student_candidate_isolated",
            return_value=rejected_gate,
        ):
            rejected = coordinator._import_student(
                client,
                rejected_job,
                rejected_manifest,
                rejected_result,
            )
        self.assertFalse(rejected["activated"])
        self.assertEqual(rejected_output.read_bytes(), b"still-active")
        self.assertTrue(Path(rejected["candidate_model_path"]).is_file())

        # A missing rejected/accepted candidate invalidates the receipt and
        # forces artifact recovery rather than claiming a phantom success.
        Path(imported["candidate_model_path"]).unlink()
        with patch.object(
            coordinator,
            "_validate_student_candidate_isolated",
            return_value=local_gate,
        ):
            coordinator._import_student(client, job, job_manifest, result)
        self.assertEqual(client.downloads, 12)
        self.assertEqual(
            output.with_name("student.previous.npz").read_bytes(),
            b"previous-model",
        )

    def test_student_local_validation_runs_in_disposable_process(self):
        store = SongMemoryStore(self.root / "isolated-validation.sqlite3")
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "isolated-validation-research",
            state_root=self.root / "isolated-validation-state",
            config_path=self.root / "isolated-validation-state/config.json",
        )
        candidate = self.root / "isolated-validation-candidate.npz"
        model = StreamingStructureStudent(hidden_size=4)
        model.approved_axes = set()
        model.save(candidate)
        original = self.root / "isolated-validation-original.jsonl"
        prepared = self.root / "isolated-validation-prepared.jsonl"
        row = {
            "recording_id": "recording:isolated-validation",
            "recording_offset_ms": 0,
            "split_group_id": "group:isolated-validation",
            "split": "train",
            "features": [0.0] * 15,
            "functional": "unknown",
            "energy": "groove",
            "content": "unknown",
            "boundary": 0,
        }
        original.write_text(json.dumps(row) + "\n", encoding="utf-8")
        prepared.write_text(
            json.dumps({
                **row,
                "features": [0.25] * 15,
                "feature_preprocessing_version": (
                    STUDENT_AUDIO_FEATURE_VERSION
                ),
            }) + "\n",
            encoding="utf-8",
        )
        work = self.root / "isolated-validation-work"
        work.mkdir()

        local_gate = coordinator._validate_student_candidate_isolated(
            candidate_path=candidate,
            original_path=original,
            prepared_path=prepared,
            payload={"applicable_axes": ["energy"]},
            student_audio_feature_version=STUDENT_AUDIO_FEATURE_VERSION,
            work_root=work,
        )

        self.assertEqual(local_gate["approved_axes"], [])
        self.assertEqual(local_gate["split_counts"]["train"], 1)
        self.assertTrue((work / "local-validation.result.json").is_file())

    def test_disable_restores_queued_jobs_and_restart_still_drains_active(self):
        store = SongMemoryStore(self.root / "disable.sqlite3")
        queued_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB, payload={"recording_id": "queued"}
        )
        store.set_analysis_job_execution_target(
            queued_id, execution_target="threadripper"
        )
        active_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={"recording_id": "active", "execution_target": "threadripper"},
        )
        active_job = store.claim_analysis_job_by_id(
            active_id,
            worker_id="lumen-link:old",
            worker_pid=999_999,
        )
        assert active_job is not None
        state_root = self.root / "disable-state"
        state_root.mkdir(parents=True)
        (state_root / "secret").write_bytes(SECRET)
        (state_root / "config.json").write_text(
            json.dumps(
                {
                    "endpoint": "http://127.0.0.1:1",
                    "secret_file": "secret",
                    "enabled": True,
                }
            ),
            encoding="utf-8",
        )
        (state_root / "coordinator.json").write_text(
            json.dumps(
                {
                    "schema": LINK_SCHEMA,
                    "active": {
                        "job": active_job,
                        "manifest": {"schema": MANIFEST_SCHEMA},
                        "stage": "remote",
                    },
                }
            ),
            encoding="utf-8",
        )
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "disable-research",
            state_root=state_root,
            config_path=state_root / "config.json",
        )
        coordinator.control("disable")
        queued = next(
            item for item in store.list_analysis_jobs() if item["id"] == queued_id
        )
        self.assertEqual(queued["payload"]["execution_target"], "automatic")
        self.assertIsNotNone(coordinator.active)
        coordinator.start()
        self.assertIsNotNone(coordinator.thread)
        self.assertTrue(coordinator.thread.is_alive())
        coordinator.close()

    def test_remote_import_resumes_after_timeline_commit_without_duplicates(self):
        store = SongMemoryStore(self.root / "import-receipt.sqlite3")
        recording_id = store.remember_recording_version(
            provider="test",
            provider_item_id="receipt",
            duration_ms=1_000,
            metadata={},
        )
        audio = self.root / "receipt.wav"
        audio.write_bytes(b"receipt")
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        job_id = store.enqueue_analysis_job(
            job_type=EDMFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "content_sha256": digest,
                "duration_ms": 1_000,
                "recording_id": recording_id,
                "execution_target": "threadripper",
            },
        )
        job = store.claim_analysis_job_by_id(
            job_id,
            worker_id="lumen-link:receipt",
            worker_pid=1,
        )
        assert job is not None
        job_manifest = manifest(job_id, digest, audio.stat().st_size)
        job_manifest["identity"]["recording_id"] = recording_id
        manifest_hash = hashlib.sha256(
            json.dumps(
                job_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job_id,
            "job_type": EDMFORMER_JOB,
            "manifest_sha256": manifest_hash,
            "input_sha256": digest,
            "duration_ms": 1_000,
            **TEST_CONTRACT,
            "segments": [{"start": 0.0, "end": 1.0, "label": "Drop"}],
            "resources": {"elapsed_s": 1.0},
        }
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "import-research",
            state_root=self.root / "import-state",
            config_path=self.root / "import-state" / "config.json",
        )
        with patch(
            "lumen_engine.offline.build_student_examples",
            side_effect=RuntimeError("simulated crash after timeline commit"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                coordinator._import_edmformer(job, job_manifest, result)
        self.assertEqual(len(store.list_teacher_runs()), 1)
        with patch(
            "lumen_engine.offline.build_student_examples",
            return_value={"path": "examples.jsonl", "examples": 10},
        ):
            resumed = coordinator._import_edmformer(
                job, job_manifest, result
            )
            reused = coordinator._import_edmformer(
                job, job_manifest, result
            )
        self.assertTrue(resumed["import_resumed"])
        self.assertTrue(reused["import_reused"])
        runs = store.list_teacher_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "complete")
        self.assertIsNotNone(
            store.structure_timeline_for_teacher_run(str(runs[0]["id"]))
        )

    def test_teacher_import_rejects_cross_teacher_result_envelope(self):
        store = SongMemoryStore(self.root / "cross-teacher.sqlite3")
        audio = self.root / "cross.wav"
        audio.write_bytes(b"cross")
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        job_id = store.enqueue_analysis_job(
            job_type=SONGFORMER_JOB,
            payload={
                "audio_path": str(audio),
                "content_sha256": digest,
                "duration_ms": 1_000,
                "recording_id": "recording:cross",
            },
        )
        job = next(
            item for item in store.list_analysis_jobs() if item["id"] == job_id
        )
        job_manifest = manifest(
            job_id,
            digest,
            audio.stat().st_size,
            job_type=SONGFORMER_JOB,
        )
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job_id,
            "job_type": EDMFORMER_JOB,
        }
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "cross-research",
            state_root=self.root / "cross-state",
            config_path=self.root / "cross-state" / "config.json",
        )
        with self.assertRaises(LinkProtocolError):
            coordinator._import_teacher(
                job, job_manifest, result, teacher="SongFormer"
            )

    def test_status_is_cached_across_desktop_and_phone_polls(self):
        store = SongMemoryStore(self.root / "status-cache.sqlite3")
        coordinator = LumenLinkCoordinator(
            store,
            research_root=self.root / "status-research",
            state_root=self.root / "status-state",
            config_path=self.root / "status-state" / "config.json",
        )
        with patch.object(
            store,
            "list_analysis_jobs",
            wraps=store.list_analysis_jobs,
        ) as jobs:
            first = coordinator.status()
            second = coordinator.status()
        self.assertEqual(first, second)
        # UI polling consumes the standby-prepared snapshot; it never starts
        # a database scan of its own, including while Live is active.
        self.assertEqual(jobs.call_count, 0)
        self.assertEqual(first["capabilities"]["supported_job_types"], [])


if __name__ == "__main__":
    unittest.main()
