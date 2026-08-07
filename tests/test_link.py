from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from lumen_engine.link import (
    LINK_SCHEMA,
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    LinkAuthenticator,
    LinkAuthenticationError,
    LinkClient,
    LinkNodeRuntime,
    LinkNodeServer,
    LinkProtocolError,
    LinkSpool,
    LumenLinkCoordinator,
)
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)
from lumen_engine.offline import EDMFORMER_JOB


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


class FakeExecutor:
    def capabilities(self):
        return {
            "protocol_schema": LINK_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            **TEST_CONTRACT,
            "supported_job_types": [EDMFORMER_JOB],
            "gated_job_types": {
                "teacher.songformer": "not implemented",
                "student.train": "not implemented",
            },
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
        return {
            "schema": RESULT_SCHEMA,
            "job_id": manifest["job_id"],
            "job_type": manifest["job_type"],
            "manifest_sha256": state["manifest_sha256"],
            "input_sha256": audio["sha256"],
            "duration_ms": manifest["identity"]["duration_ms"],
            **TEST_CONTRACT,
            "segments": [{"start": 0, "end": 1, "label": "drop"}],
            "resources": {"elapsed_s": 0.01},
        }


def manifest(job_id: str, digest: str, byte_count: int):
    return {
        "schema": MANIFEST_SCHEMA,
        "job_id": job_id,
        "job_type": EDMFORMER_JOB,
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
            **TEST_CONTRACT,
            "result_schema": RESULT_SCHEMA,
        },
        "resources": {"threads": 99},
        "created_unix_ms": 1,
    }


class LinkTests(unittest.TestCase):
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
            [EDMFORMER_JOB],
        )
        self.assertIn("student.train", health["capabilities"]["gated_job_types"])
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
        self.assertEqual(jobs.call_count, 1)
        self.assertEqual(first["capabilities"]["supported_job_types"], [])


if __name__ == "__main__":
    unittest.main()
