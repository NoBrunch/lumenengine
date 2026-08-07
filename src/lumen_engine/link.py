"""Lumen Link: authenticated, durable offline compute over a private LAN.

The live computer owns SQLite and imports every result.  A compute node sees
only immutable objects and versioned manifests; it cannot drive audio or DMX.
The wire implementation intentionally uses the standard library so the link
can be diagnosed before either research environment is installed in WSL.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
)


LINK_SCHEMA = "lumen.link.v1"
MANIFEST_SCHEMA = "lumen.link.job.v1"
RESULT_SCHEMA = "lumen.link.result.v1"
DEFAULT_LINK_ENDPOINT = "http://192.168.50.1:8765"
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CONTROL_BYTES = 2 * 1024 * 1024
EDMFORMER_JOB = "teacher.edmformer"
SUPPORTED_JOB_TYPES = (EDMFORMER_JOB,)
EDM_CONTRACT_FIELDS = (
    "code_revision",
    "code_clean",
    "teacher_revision",
    "teacher_clean",
    "musicfm_source_revision",
    "musicfm_source_clean",
    "model_sha256",
    "musicfm_stats_sha256",
    "musicfm_model_sha256",
    "muq_assets_sha256",
    "teacher_normalization_version",
    "edmformer_preprocessing_version",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_clean(path: Path) -> bool:
    try:
        return not subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False


def _hash_tree(path: Path) -> str | None:
    """Hash resolved file content and relative names in a model asset tree."""
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        return None
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(item.resolve()).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _edmformer_asset_contract(
    research_root: Path, project_root: Path
) -> dict[str, Any]:
    source = research_root / "sources" / "edm98"
    checkpoint = source / "data" / "checkpoints"
    musicfm_source = research_root / "sources" / "musicfm"
    muq_cache = (
        research_root
        / "cache"
        / "huggingface"
        / "models--OpenMuQ--MuQ-large-msd-iter"
    )
    return {
        "code_revision": _git_revision(project_root),
        "code_clean": _git_clean(project_root),
        "teacher_revision": _git_revision(source),
        "teacher_clean": _git_clean(source),
        "musicfm_source_revision": _git_revision(musicfm_source),
        "musicfm_source_clean": _git_clean(musicfm_source),
        "model_sha256": (
            _hash_file(checkpoint / "model.pt")
            if (checkpoint / "model.pt").is_file()
            else None
        ),
        "musicfm_stats_sha256": (
            _hash_file(checkpoint / "msd_stats.json")
            if (checkpoint / "msd_stats.json").is_file()
            else None
        ),
        "musicfm_model_sha256": (
            _hash_file(checkpoint / "pretrained_msd.pt")
            if (checkpoint / "pretrained_msd.pt").is_file()
            else None
        ),
        "muq_assets_sha256": _hash_tree(muq_cache / "snapshots"),
        "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
        "edmformer_preprocessing_version": EDMFORMER_PREPROCESSING_VERSION,
    }


def _edmformer_asset_signature(
    research_root: Path, project_root: Path
) -> dict[str, Any]:
    source = research_root / "sources" / "edm98"
    checkpoint = source / "data" / "checkpoints"
    musicfm = research_root / "sources" / "musicfm"
    muq = (
        research_root
        / "cache"
        / "huggingface"
        / "models--OpenMuQ--MuQ-large-msd-iter"
        / "snapshots"
    )

    def file_signature(path: Path) -> list[int] | None:
        try:
            stat = path.resolve().stat()
            return [int(stat.st_size), int(stat.st_mtime_ns)]
        except OSError:
            return None

    return {
        "code_revision": _git_revision(project_root),
        "code_clean": _git_clean(project_root),
        "teacher_revision": _git_revision(source),
        "teacher_clean": _git_clean(source),
        "musicfm_revision": _git_revision(musicfm),
        "musicfm_clean": _git_clean(musicfm),
        "files": {
            name: file_signature(path)
            for name, path in {
                "model": checkpoint / "model.pt",
                "stats": checkpoint / "msd_stats.json",
                "musicfm": checkpoint / "pretrained_msd.pt",
                "config": source / "configs" / "edmformer.yaml",
                "runner": project_root / "scripts" / "edmformer-cpu-runner.py",
            }.items()
        },
        "muq": {
            str(path.relative_to(muq)): file_signature(path)
            for path in sorted(muq.rglob("*"))
            if path.is_file()
        } if muq.is_dir() else {},
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(_json_bytes(value) + b"\n")
    os.chmod(partial, 0o600)
    partial.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


class LinkAuthenticationError(RuntimeError):
    """A request did not prove possession of the shared link secret."""


class LinkProtocolError(RuntimeError):
    """A remote response violates the versioned Link contract."""


@dataclass(frozen=True, slots=True)
class LinkConfiguration:
    endpoint: str
    secret: bytes
    enabled: bool = False
    paused: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "LinkConfiguration":
        config_path = Path(path)
        if not config_path.is_file():
            return cls(DEFAULT_LINK_ENDPOINT, b"")
        value = _read_json(config_path)
        secret_path = Path(str(value.get("secret_file") or ""))
        if not secret_path.is_absolute():
            secret_path = (config_path.parent / secret_path).resolve()
        try:
            secret = secret_path.read_text(encoding="utf-8").strip().encode()
        except OSError:
            secret = b""
        return cls(
            endpoint=str(value.get("endpoint") or DEFAULT_LINK_ENDPOINT).rstrip(
                "/"
            ),
            secret=secret,
            enabled=bool(value.get("enabled", False)),
            paused=bool(value.get("paused", False)),
        )


class LinkAuthenticator:
    """Sign HTTP requests with timestamped, nonce-bound HMAC-SHA256."""

    def __init__(self, secret: bytes, *, maximum_skew_s: int = 90) -> None:
        if len(secret) < 32:
            raise ValueError("Lumen Link secret must contain at least 32 bytes")
        self.secret = secret
        self.maximum_skew_s = max(10, int(maximum_skew_s))
        self._nonces: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _message(
        method: str,
        path: str,
        timestamp: str,
        nonce: str,
        content_sha256: str,
    ) -> bytes:
        return "\n".join(
            (method.upper(), path, timestamp, nonce, content_sha256)
        ).encode("utf-8")

    def headers(
        self, method: str, path: str, content_sha256: str
    ) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = hmac.new(
            self.secret,
            self._message(method, path, timestamp, nonce, content_sha256),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-Lumen-Timestamp": timestamp,
            "X-Lumen-Nonce": nonce,
            "X-Lumen-Content-SHA256": content_sha256,
            "X-Lumen-Signature": signature,
        }

    def verify(
        self,
        method: str,
        path: str,
        headers: Any,
        content_sha256: str,
    ) -> None:
        timestamp = str(headers.get("X-Lumen-Timestamp") or "")
        nonce = str(headers.get("X-Lumen-Nonce") or "")
        claimed_hash = str(headers.get("X-Lumen-Content-SHA256") or "")
        supplied = str(headers.get("X-Lumen-Signature") or "")
        try:
            timestamp_int = int(timestamp)
        except ValueError as error:
            raise LinkAuthenticationError("invalid authentication timestamp") from error
        now = int(time.time())
        if abs(now - timestamp_int) > self.maximum_skew_s:
            raise LinkAuthenticationError("authentication timestamp is stale")
        if len(nonce) != 32 or claimed_hash != content_sha256:
            raise LinkAuthenticationError("invalid authentication proof")
        expected = hmac.new(
            self.secret,
            self._message(method, path, timestamp, nonce, claimed_hash),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, supplied):
            raise LinkAuthenticationError("invalid authentication signature")
        with self._lock:
            cutoff = now - self.maximum_skew_s
            self._nonces = {
                key: seen for key, seen in self._nonces.items() if seen >= cutoff
            }
            if nonce in self._nonces:
                raise LinkAuthenticationError("authentication nonce was replayed")
            self._nonces[nonce] = now


class LinkSpool:
    """Content-addressed objects and atomic durable job state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.uploads = self.root / "uploads"
        self.jobs = self.root / "jobs"
        self.results = self.root / "results"
        for path in (self.objects, self.uploads, self.jobs, self.results):
            path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    @staticmethod
    def validate_digest(digest: str) -> str:
        normalized = str(digest).lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("object digest must be a SHA-256 hex string")
        return normalized

    def object_path(self, digest: str) -> Path:
        digest = self.validate_digest(digest)
        return self.objects / digest[:2] / digest

    def upload_path(self, digest: str) -> Path:
        return self.uploads / (self.validate_digest(digest) + ".partial")

    def object_state(self, digest: str) -> dict[str, Any]:
        final = self.object_path(digest)
        partial = self.upload_path(digest)
        if final.is_file():
            return {"complete": True, "bytes": final.stat().st_size}
        return {
            "complete": False,
            "bytes": partial.stat().st_size if partial.is_file() else 0,
        }

    def append_object(
        self,
        digest: str,
        stream: BinaryIO,
        *,
        offset: int,
        length: int,
        total: int,
        expected_chunk_sha256: str,
    ) -> dict[str, Any]:
        digest = self.validate_digest(digest)
        if offset < 0 or length < 0 or total <= 0 or offset + length > total:
            raise ValueError("invalid object byte range")
        final = self.object_path(digest)
        partial = self.upload_path(digest)
        with self.lock:
            if final.is_file():
                if final.stat().st_size != total:
                    raise ValueError("existing object size does not match upload")
                return {"complete": True, "bytes": total}
            current = partial.stat().st_size if partial.is_file() else 0
            if offset != current:
                raise ValueError(f"upload offset mismatch; expected {current}")
            partial.parent.mkdir(parents=True, exist_ok=True)
            remaining = length
            chunk_hash = hashlib.sha256()
            with partial.open("ab") as output:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("object request ended before declared length")
                    output.write(chunk)
                    chunk_hash.update(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if not hmac.compare_digest(
                chunk_hash.hexdigest(), expected_chunk_sha256
            ):
                with partial.open("r+b") as output:
                    output.truncate(offset)
                raise ValueError("uploaded chunk checksum does not match")
            received = partial.stat().st_size
            if received == total:
                if _hash_file(partial) != digest:
                    partial.unlink(missing_ok=True)
                    raise ValueError("completed object checksum does not match")
                final.parent.mkdir(parents=True, exist_ok=True)
                partial.replace(final)
                return {"complete": True, "bytes": received}
            return {"complete": False, "bytes": received}

    def job_path(self, job_id: str) -> Path:
        identity = str(job_id)
        if (
            not identity
            or len(identity) > 200
            or any(
                char
                not in "-_.:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                for char in identity
            )
        ):
            raise ValueError("invalid job ID")
        # Hash the filename so IDs containing ':' cannot collide with IDs
        # containing '-'. The original identity remains inside signed JSON.
        safe = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.jobs / f"{safe}.json"

    def submit(self, manifest: dict[str, Any]) -> dict[str, Any]:
        _validate_manifest(manifest)
        job_id = str(manifest["job_id"])
        manifest_hash = _sha256_bytes(_json_bytes(manifest))
        path = self.job_path(job_id)
        now_ms = int(time.time() * 1000)
        with self.lock:
            if path.is_file():
                existing = _read_json(path)
                if existing.get("manifest_sha256") != manifest_hash:
                    raise ValueError("job ID already has a different manifest")
                return existing
            for item in manifest["objects"]:
                state = self.object_state(str(item["sha256"]))
                if not state["complete"] or state["bytes"] != int(item["bytes"]):
                    raise ValueError(f"required object {item['sha256']} is incomplete")
            state = {
                "schema": LINK_SCHEMA,
                "job_id": job_id,
                "job_type": manifest["job_type"],
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "manifest": manifest,
                "manifest_sha256": manifest_hash,
                "created_unix_ms": now_ms,
                "updated_unix_ms": now_ms,
                "error": None,
                "resources": {},
            }
            _atomic_json(path, state)
            return state

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self.lock:
            path = self.job_path(job_id)
            state = _read_json(path)
            state.update(updates)
            state["updated_unix_ms"] = int(time.time() * 1000)
            _atomic_json(path, state)
            return state

    def job(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_path(job_id)
        return _read_json(path) if path.is_file() else None

    def list_jobs(self) -> list[dict[str, Any]]:
        result = []
        for path in self.jobs.glob("*.json"):
            try:
                result.append(_read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(
            result,
            key=lambda item: int(item.get("created_unix_ms") or 0),
            reverse=True,
        )

    def next_queued(self) -> dict[str, Any] | None:
        queued = [job for job in self.list_jobs() if job.get("status") == "queued"]
        return queued[-1] if queued else None

    def recover_running(self) -> int:
        """Requeue jobs interrupted by compute-node or WSL shutdown."""
        recovered = 0
        for job in self.list_jobs():
            if job.get("status") != "running":
                continue
            self.update_job(
                str(job["job_id"]),
                status="queued",
                stage="recovered",
                error="Compute node restarted; the immutable job was requeued",
            )
            recovered += 1
        return recovered

    def result_path(self, job_id: str) -> Path:
        return self.results / (self.job_path(job_id).stem + ".json")

    def save_result(self, job_id: str, value: dict[str, Any]) -> None:
        _atomic_json(self.result_path(job_id), value)

    def result(self, job_id: str) -> dict[str, Any] | None:
        path = self.result_path(job_id)
        return _read_json(path) if path.is_file() else None


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported Lumen Link manifest schema")
    if manifest.get("job_type") not in SUPPORTED_JOB_TYPES:
        raise ValueError(
            f"job type {manifest.get('job_type')!r} is not supported by this node"
        )
    if not str(manifest.get("job_id") or ""):
        raise ValueError("manifest has no job ID")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("manifest contains no immutable input objects")
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("manifest object entry is invalid")
        LinkSpool.validate_digest(str(item.get("sha256") or ""))
        if int(item.get("bytes") or 0) <= 0:
            raise ValueError("manifest object has no byte length")
    contract = manifest.get("contract") or {}
    if contract.get("teacher_normalization_version") != TEACHER_NORMALIZATION_VERSION:
        raise ValueError("teacher normalization contract does not match node")
    if contract.get("edmformer_preprocessing_version") != EDMFORMER_PREPROCESSING_VERSION:
        raise ValueError("EDMFormer preprocessing contract does not match node")


def _node_resources() -> dict[str, Any]:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    disk = shutil.disk_usage(Path.cwd())
    try:
        load = os.getloadavg()
    except OSError:
        load = (0.0, 0.0, 0.0)
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "load_1m": load[0],
        "load_5m": load[1],
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": None,
    }


class LinkNodeExecutor:
    """Execute pure compute jobs without access to Lumen's database."""

    def __init__(
        self,
        spool: LinkSpool,
        *,
        research_root: Path,
        project_root: Path,
        max_threads: int = 24,
        max_memory_gib: float = 96.0,
    ) -> None:
        self.spool = spool
        self.research_root = research_root.resolve()
        self.project_root = project_root.resolve()
        self.max_threads = max(1, int(max_threads))
        self.max_memory_bytes = int(max(1.0, float(max_memory_gib)) * 1024**3)
        self._capabilities_cache: dict[str, Any] | None = None

    def capabilities(self) -> dict[str, Any]:
        if self._capabilities_cache is not None:
            return dict(self._capabilities_cache)
        asset_contract = _edmformer_asset_contract(
            self.research_root, self.project_root
        )
        checkpoint_root = (
            self.research_root / "sources" / "edm98" / "data" / "checkpoints"
        )
        required = (
            self.research_root
            / "environments"
            / "edmformer"
            / "bin"
            / "python",
            self.project_root / "scripts" / "edmformer-cpu-runner.py",
            checkpoint_root / "model.pt",
            checkpoint_root / "msd_stats.json",
            checkpoint_root / "pretrained_msd.pt",
            self.research_root
            / "sources"
            / "edm98"
            / "configs"
            / "edmformer.yaml",
            self.research_root
            / "sources"
            / "musicfm"
            / "model"
            / "musicfm_25hz.py",
        )
        provisioned = all(path.exists() for path in required) and all(
            asset_contract.get(name)
            for name in (
                "code_revision",
                "teacher_revision",
                "musicfm_source_revision",
                "model_sha256",
                "musicfm_stats_sha256",
                "musicfm_model_sha256",
                "muq_assets_sha256",
            )
        )
        deterministic = all(
            asset_contract.get(name)
            for name in ("code_clean", "teacher_clean", "musicfm_source_clean")
        )
        gated_jobs = {
            "teacher.songformer": (
                "remote immutable-result importer is not implemented"
            ),
            "student.train": (
                "remote immutable-model importer is not implemented"
            ),
        }
        if not provisioned:
            gated_jobs[EDMFORMER_JOB] = (
                "EDMFormer environment or model assets are not provisioned"
            )
        elif not deterministic:
            gated_jobs[EDMFORMER_JOB] = (
                "Lumen, EDMFormer, or MusicFM has uncommitted source changes"
            )
        self._capabilities_cache = {
            "protocol_schema": LINK_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            **asset_contract,
            "supported_job_types": (
                list(SUPPORTED_JOB_TYPES)
                if provisioned and deterministic
                else []
            ),
            "gated_job_types": gated_jobs,
            "max_threads": self.max_threads,
            "max_memory_bytes": self.max_memory_bytes,
            "gpu": False,
            "live_timing": False,
            "dmx": False,
        }
        return dict(self._capabilities_cache)

    def execute(
        self,
        state: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        manifest = state["manifest"]
        if manifest["job_type"] != EDMFORMER_JOB:
            raise ValueError("compute node currently supports EDMFormer only")
        audio = next(item for item in manifest["objects"] if item["role"] == "audio")
        audio_path = self.spool.object_path(audio["sha256"])
        source = self.research_root / "sources" / "edm98"
        checkpoint = source / "data" / "checkpoints"
        musicfm_source = self.research_root / "sources" / "musicfm"
        output = self.spool.results / (self.spool.job_path(manifest["job_id"]).stem + ".raw.json")
        command = [
            str(self.research_root / "environments" / "edmformer" / "bin" / "python"),
            str(self.project_root / "scripts" / "edmformer-cpu-runner.py"),
            str(audio_path),
            "--checkpoint", str(checkpoint / "model.pt"),
            "--config", str(source / "configs" / "edmformer.yaml"),
            "--musicfm-stat", str(checkpoint / "msd_stats.json"),
            "--musicfm-model", str(checkpoint / "pretrained_msd.pt"),
            "--musicfm-source", str(musicfm_source),
            "--hf-cache-dir", str(self.research_root / "cache" / "huggingface"),
            "--threads", str(max(1, min(self.max_threads, int(manifest.get("resources", {}).get("threads") or self.max_threads)))),
            "--output", str(output),
        ]
        missing = [path for path in (Path(command[0]), Path(command[1]), Path(command[4]), Path(command[6]), Path(command[8]), Path(command[10]), Path(command[12])) if not path.exists()]
        if missing:
            raise RuntimeError("EDMFormer compute node is not provisioned: " + ", ".join(map(str, missing)))
        started = time.monotonic()
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": command[-3],
                "MKL_NUM_THREADS": command[-3],
                "OPENBLAS_NUM_THREADS": command[-3],
                "NUMEXPR_NUM_THREADS": command[-3],
            }
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        peak_rss = 0
        while True:
            try:
                stdout, stderr = process.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                try:
                    for line in Path(
                        f"/proc/{process.pid}/status"
                    ).read_text(encoding="utf-8").splitlines():
                        if line.startswith("VmRSS:"):
                            peak_rss = max(
                                peak_rss,
                                int(line.split()[1]) * 1024,
                            )
                            break
                except (OSError, ValueError, IndexError):
                    pass
                if progress_callback is not None:
                    progress_callback(
                        {
                            "elapsed_s": time.monotonic() - started,
                            "rss_bytes": peak_rss,
                            "peak_rss_bytes": peak_rss,
                            "memory_limit_bytes": self.max_memory_bytes,
                            "threads": int(command[-3]),
                        }
                    )
                if peak_rss > self.max_memory_bytes:
                    process.terminate()
                    try:
                        process.communicate(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise RuntimeError(
                        "EDMFormer exceeded the compute-node memory limit"
                    )
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or f"EDMFormer exited {process.returncode}")[-4000:])
        segments = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(segments, list) or not segments:
            raise RuntimeError("EDMFormer returned no timeline segments")
        capabilities = self.capabilities()
        return {
            "schema": RESULT_SCHEMA,
            "job_id": manifest["job_id"],
            "job_type": manifest["job_type"],
            "manifest_sha256": state["manifest_sha256"],
            "input_sha256": audio["sha256"],
            "duration_ms": manifest["identity"]["duration_ms"],
            **{
                name: capabilities.get(name)
                for name in EDM_CONTRACT_FIELDS
            },
            "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
            "edmformer_preprocessing_version": EDMFORMER_PREPROCESSING_VERSION,
            "segments": segments,
            "resources": {
                "elapsed_s": time.monotonic() - started,
                "threads": int(command[-3]),
                "peak_rss_bytes": peak_rss,
                "memory_limit_bytes": self.max_memory_bytes,
                "returncode": process.returncode,
            },
        }


class LinkNodeRuntime:
    def __init__(self, spool: LinkSpool, executor: LinkNodeExecutor) -> None:
        self.spool = spool
        self.executor = executor
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.spool.recover_running()
        self.thread = threading.Thread(target=self._run, name="lumen-link-node", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.wait(0.5):
            state = self.spool.next_queued()
            if state is None:
                continue
            job_id = str(state["job_id"])
            self.spool.update_job(
                job_id,
                status="running",
                stage="inference",
                progress=None,
                progress_kind="indeterminate",
            )
            try:
                result = self.executor.execute(
                    state,
                    progress_callback=lambda resources: self.spool.update_job(
                        job_id,
                        status="running",
                        stage="inference",
                        progress=None,
                        progress_kind="indeterminate",
                        resources=resources,
                    ),
                )
                self.spool.save_result(job_id, result)
                self.spool.update_job(job_id, status="complete", stage="complete", progress=1.0, resources=result.get("resources", {}))
            except Exception as error:
                self.spool.update_job(job_id, status="failed", stage="failed", error=str(error))

    def health(self) -> dict[str, Any]:
        jobs = self.spool.list_jobs()
        counts = {state: sum(job.get("status") == state for job in jobs) for state in ("queued", "running", "complete", "failed", "canceled")}
        return {
            "schema": LINK_SCHEMA,
            "service": "Lumen Link compute node",
            "authenticated": True,
            "uptime_s": max(0.0, time.time() - self.started_at),
            "capabilities": {
                **self.executor.capabilities(),
            },
            "node": _node_resources(),
            "queue": counts,
            "jobs": jobs[:50],
        }


class LinkNodeHandler(BaseHTTPRequestHandler):
    server: "LinkNodeServer"

    def do_HEAD(self) -> None:  # noqa: N802
        try:
            self._authenticate(_sha256_bytes(b""))
            if not self.path.startswith("/v1/objects/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            state = self.server.spool.object_state(self.path.rsplit("/", 1)[-1])
            self.send_response(HTTPStatus.OK if state["complete"] else HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Upload-Offset", str(state["bytes"]))
            self.send_header("Content-Length", "0")
            for key, value in self.server.authenticator.headers(
                "RESPONSE", self.path, _sha256_bytes(b"")
            ).items():
                self.send_header(key, value)
            self.end_headers()
        except Exception as error:
            self._error(error)

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._authenticate(_sha256_bytes(b""))
            if self.path == "/v1/health":
                self._json(HTTPStatus.OK, self.server.runtime.health())
                return
            if self.path.startswith("/v1/jobs/"):
                job_id = self.path.split("/")[3]
                if self.path.endswith("/result"):
                    result = self.server.spool.result(job_id)
                    if result is None:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "result is not ready"})
                    else:
                        self._json(HTTPStatus.OK, result)
                    return
                job = self.server.spool.job(job_id)
                self._json(HTTPStatus.OK if job else HTTPStatus.NOT_FOUND, job or {"error": "unknown job"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        except Exception as error:
            self._error(error)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            if not self.path.startswith("/v1/objects/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            content_hash = str(self.headers.get("X-Lumen-Content-SHA256") or "")
            self._authenticate(content_hash)
            range_value = str(self.headers.get("Content-Range") or "")
            if not range_value.startswith("bytes ") or "/" not in range_value:
                raise ValueError("object PUT requires Content-Range")
            span, total_text = range_value[6:].split("/", 1)
            start_text, end_text = span.split("-", 1)
            start, end, total = int(start_text), int(end_text), int(total_text)
            if end - start + 1 != length:
                raise ValueError("Content-Range does not match Content-Length")
            # Authentication covers this chunk; the URL covers the complete
            # object digest, which is verified before atomic publication.
            state = self.server.spool.append_object(
                self.path.rsplit("/", 1)[-1],
                self.rfile,
                offset=start,
                length=length,
                total=total,
                expected_chunk_sha256=content_hash,
            )
            self._json(HTTPStatus.CREATED if state["complete"] else HTTPStatus.ACCEPTED, state)
        except Exception as error:
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_CONTROL_BYTES:
                raise ValueError("control request is too large")
            body = self.rfile.read(length)
            self._authenticate(_sha256_bytes(body))
            value = json.loads(body or b"{}")
            if self.path == "/v1/jobs":
                contract = value.get("contract") or {}
                capabilities = self.server.runtime.executor.capabilities()
                for name in EDM_CONTRACT_FIELDS:
                    if contract.get(name) != capabilities.get(name):
                        raise ValueError(
                            f"compute-node {name} does not match manifest"
                        )
                state = self.server.spool.submit(value)
                self._json(HTTPStatus.ACCEPTED, state)
                return
            if self.path.startswith("/v1/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.split("/")[3]
                job = self.server.spool.job(job_id)
                if not job:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown job"})
                elif job.get("status") == "running":
                    self._json(HTTPStatus.CONFLICT, {"error": "running cancellation is not yet supported"})
                else:
                    self._json(HTTPStatus.OK, self.server.spool.update_job(job_id, status="canceled", stage="canceled"))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        except Exception as error:
            self._error(error)

    def _authenticate(self, body_hash: str) -> None:
        self.server.authenticator.verify(self.command, self.path, self.headers, body_hash)

    def _json(self, status: HTTPStatus, value: Any) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, header_value in self.server.authenticator.headers(
            "RESPONSE", self.path, _sha256_bytes(body)
        ).items():
            self.send_header(key, header_value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, error: Exception) -> None:
        status = HTTPStatus.UNAUTHORIZED if isinstance(error, LinkAuthenticationError) else HTTPStatus.BAD_REQUEST
        self._json(status, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return


class LinkNodeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, spool: LinkSpool, authenticator: LinkAuthenticator, runtime: LinkNodeRuntime) -> None:
        self.spool = spool
        self.authenticator = authenticator
        self.runtime = runtime
        super().__init__(address, LinkNodeHandler)


class LinkClient:
    def __init__(self, endpoint: str, secret: bytes, *, timeout_s: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.authenticator = LinkAuthenticator(secret)
        self.timeout_s = timeout_s

    def request(self, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None) -> dict[str, Any]:
        request_headers = self.authenticator.headers(method, path, _sha256_bytes(body))
        request_headers.update(headers or {})
        request = Request(self.endpoint + path, data=body if method not in {"GET", "HEAD"} else None, method=method, headers=request_headers)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read()
                self.authenticator.verify(
                    "RESPONSE",
                    path,
                    response.headers,
                    _sha256_bytes(payload),
                )
                if method == "HEAD":
                    return {"status": response.status, "offset": int(response.headers.get("Upload-Offset", "0"))}
        except HTTPError as error:
            if method == "HEAD" and error.code == HTTPStatus.PERMANENT_REDIRECT:
                self.authenticator.verify(
                    "RESPONSE",
                    path,
                    error.headers,
                    _sha256_bytes(b""),
                )
                result = {"status": error.code, "offset": int(error.headers.get("Upload-Offset", "0"))}
                error.close()
                return result
            error_body = error.read()
            message = error_body.decode("utf-8", "replace")
            try:
                self.authenticator.verify(
                    "RESPONSE",
                    path,
                    error.headers,
                    _sha256_bytes(error_body),
                )
            except LinkAuthenticationError as authentication_error:
                error.close()
                raise LinkProtocolError(
                    "compute-node response authentication failed"
                ) from authentication_error
            error.close()
            raise LinkProtocolError(f"compute node returned HTTP {error.code}: {message}") from error
        except URLError as error:
            raise LinkProtocolError(f"compute node is unavailable: {error.reason}") from error
        value = json.loads(payload or b"{}")
        if not isinstance(value, dict):
            raise LinkProtocolError("compute node returned a non-object response")
        return value

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/v1/health")

    def upload(self, path: Path, digest: str | None = None) -> dict[str, Any]:
        digest = digest or _hash_file(path)
        total = path.stat().st_size
        remote = self.request("HEAD", f"/v1/objects/{digest}")
        offset = int(remote["offset"])
        if offset > total:
            raise LinkProtocolError(
                "remote partial object is larger than the local object"
            )
        if remote["status"] == HTTPStatus.OK:
            if offset != total:
                raise LinkProtocolError("remote immutable object has the wrong size")
            return {"complete": True, "bytes": total, "resumed_from": offset}
        with path.open("rb") as source:
            source.seek(offset)
            while offset < total:
                body = source.read(min(DEFAULT_CHUNK_BYTES, total - offset))
                end = offset + len(body) - 1
                result = self.request("PUT", f"/v1/objects/{digest}", body, {"Content-Range": f"bytes {offset}-{end}/{total}"})
                offset = int(result["bytes"])
        return {"complete": True, "bytes": total, "resumed_from": remote["offset"]}

    def submit(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/jobs", _json_bytes(manifest), {"Content-Type": "application/json"})


class LumenLinkCoordinator:
    """Own remote leases and import verified results into canonical memory."""

    def __init__(
        self,
        store: SongMemoryStore,
        *,
        research_root: Path,
        state_root: Path,
        config_path: Path,
        can_import: Callable[[], bool] | None = None,
    ) -> None:
        self.store = store
        self.research_root = research_root.resolve()
        self.state_root = state_root.resolve()
        self.config_path = config_path.resolve()
        self.state_path = self.state_root / "coordinator.json"
        self.contract_cache_path = self.state_root / "asset-contract.json"
        self.lock = threading.RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.remote_status: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_seen: float | None = None
        self.latency_ms: float | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active: dict[str, Any] | None = None
        try:
            persisted = _read_json(self.state_path)
            if isinstance(persisted.get("active"), dict):
                self.active = persisted["active"]
        except (OSError, ValueError, json.JSONDecodeError):
            self.active = None
        self.can_import = can_import or (lambda: True)
        self.configuration = LinkConfiguration.load(self.config_path)
        self._local_contract_cache: dict[str, Any] | None = None
        self._status_cache: dict[str, Any] | None = None
        self._status_cached_at = 0.0

    def start(self) -> None:
        if not self.configuration.secret:
            return
        if not self.configuration.enabled and self.active is None:
            return
        if self.thread and self.thread.is_alive():
            return
        self._reconcile_active()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="lumen-link-coordinator", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def control(self, action: str) -> dict[str, Any]:
        action = str(action).casefold()
        if action == "test":
            if not self.can_import():
                raise RuntimeError(
                    "Stop Live before verifying multi-gigabyte model assets"
                )
            self._poll_health()
            self._remote_is_compatible(allow_contract_scan=True)
        elif action in {"enable", "pause", "resume", "disable"}:
            value = _read_json(self.config_path) if self.config_path.is_file() else {}
            value.setdefault("endpoint", DEFAULT_LINK_ENDPOINT)
            value.setdefault("secret_file", "secret")
            if action == "enable":
                value["enabled"] = True
                value["paused"] = False
            elif action == "disable":
                value["enabled"] = False
                value["paused"] = False
            else:
                value["paused"] = action == "pause"
            _atomic_json(self.config_path, value)
            self.configuration = LinkConfiguration.load(self.config_path)
            self._event("control", action)
            if action == "disable":
                self._restore_queued_jobs()
            else:
                self.start()
        else:
            raise ValueError("unknown Lumen Link action")
        return self.status()

    def _client(self) -> LinkClient:
        if not self.configuration.secret:
            raise RuntimeError("Lumen Link secret is not configured")
        return LinkClient(self.configuration.endpoint, self.configuration.secret)

    def _poll_health(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            value = self._client().health()
            self.latency_ms = (time.monotonic() - started) * 1000
            self.remote_status = value
            self.last_seen = time.time()
            self.last_error = None
            self._invalidate_status()
            return value
        except Exception as error:
            self.last_error = str(error)
            self._invalidate_status()
            raise

    def _loop(self) -> None:
        while not self.stop_event.wait(2.0):
            self.configuration = LinkConfiguration.load(self.config_path)
            if (
                self.active is None
                and (
                    not self.configuration.enabled
                    or self.configuration.paused
                )
            ):
                continue
            try:
                self._poll_health()
                if self.active is None:
                    if self.can_import():
                        self._remote_is_compatible(
                            allow_contract_scan=True
                        )
                    self.route_queued_jobs()
                self._advance()
            except Exception as error:
                self.last_error = str(error)

    def route_queued_jobs(self) -> int:
        """Route one canary only after authenticated contract agreement."""
        if (
            not self.configuration.enabled
            or self.configuration.paused
            or not self._remote_is_compatible()
            or self.active is not None
        ):
            return 0
        jobs = self.store.list_analysis_jobs(limit=100_000)
        if any(
            job.get("status") == "queued"
            and (job.get("payload") or {}).get("execution_target")
            == "threadripper"
            for job in jobs
        ):
            return 0
        for job in reversed(jobs):
            target = str(
                (job.get("payload") or {}).get("execution_target")
                or "automatic"
            )
            if (
                job.get("status") == "queued"
                and job.get("job_type") in SUPPORTED_JOB_TYPES
                and target == "automatic"
                and self.store.set_analysis_job_execution_target(
                    str(job["id"]), execution_target="threadripper"
                )
            ):
                self._invalidate_status()
                return 1
        return 0

    def _restore_queued_jobs(self) -> int:
        restored = 0
        for job in self.store.list_analysis_jobs(limit=100_000):
            if (
                job.get("status") == "queued"
                and (job.get("payload") or {}).get("execution_target")
                == "threadripper"
                and self.store.set_analysis_job_execution_target(
                    str(job["id"]), execution_target="automatic"
                )
            ):
                restored += 1
        self._invalidate_status()
        return restored

    def _local_contract(self) -> dict[str, Any]:
        if self._local_contract_cache is None:
            project_root = Path(__file__).resolve().parents[2]
            signature = _edmformer_asset_signature(
                self.research_root, project_root
            )
            cached: dict[str, Any] | None = None
            try:
                receipt = _read_json(self.contract_cache_path)
                if receipt.get("signature") == signature and isinstance(
                    receipt.get("contract"), dict
                ):
                    cached = dict(receipt["contract"])
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            self._local_contract_cache = cached or _edmformer_asset_contract(
                self.research_root, project_root
            )
            if cached is None:
                _atomic_json(
                    self.contract_cache_path,
                    {
                        "schema": "lumen.link.asset-contract.v1",
                        "signature": signature,
                        "contract": self._local_contract_cache,
                        "created_unix_ms": int(time.time() * 1000),
                    },
                )
        return dict(self._local_contract_cache)

    def _remote_is_compatible(self, *, allow_contract_scan: bool = False) -> bool:
        if self.last_error or not self.remote_status:
            return False
        if not self.remote_status.get("authenticated"):
            return False
        capabilities = self.remote_status.get("capabilities") or {}
        if EDMFORMER_JOB not in capabilities.get("supported_job_types", []):
            return False
        local = self._local_contract_cache
        if local is None:
            if not allow_contract_scan:
                return False
            local = self._local_contract()
        return all(
            local.get(name) == capabilities.get(name)
            for name in EDM_CONTRACT_FIELDS
        )

    def ready_for_offload(self) -> bool:
        return bool(
            self.configuration.enabled
            and not self.configuration.paused
            and self._remote_is_compatible()
        )

    def _reconcile_active(self) -> None:
        if self.active is None:
            return
        job_id = str((self.active.get("job") or {}).get("id") or "")
        jobs = {
            str(job["id"]): job
            for job in self.store.list_analysis_jobs(limit=100_000)
        }
        canonical = jobs.get(job_id)
        if canonical is None or canonical.get("status") in {
            "complete",
            "failed",
            "canceled",
        }:
            self.active = None
            self._persist()
            return
        worker_id = f"lumen-link:{platform.node()}"
        if canonical.get("status") == "queued":
            canonical = self.store.claim_analysis_job_by_id(
                job_id,
                worker_id=worker_id,
                worker_pid=os.getpid(),
            )
        if canonical is None:
            self.active = None
            self._persist()
            return
        self.active["job"] = canonical
        self._persist()

    def _advance(self) -> None:
        client = self._client()
        if self.active is None:
            capabilities = (self.remote_status or {}).get("capabilities") or {}
            if EDMFORMER_JOB not in capabilities.get(
                "supported_job_types", []
            ):
                return
            job = self.store.claim_analysis_job(
                (EDMFORMER_JOB,),
                worker_id=f"lumen-link:{platform.node()}",
                worker_pid=os.getpid(),
                execution_targets=("threadripper",),
            )
            if job is None:
                return
            manifest = self._manifest(job)
            try:
                for name in EDM_CONTRACT_FIELDS:
                    if manifest["contract"].get(name) != capabilities.get(name):
                        raise RuntimeError(
                            f"Threadripper {name} does not match Lumen"
                        )
            except Exception as error:
                self.store.update_analysis_job(
                    job["id"], status="queued", error=str(error)
                )
                raise
            self.active = {"job": job, "manifest": manifest, "stage": "transfer", "progress": 0.0}
            self._persist()
            self._event("job", f"claimed {job['id']}")
        job = self.active["job"]
        manifest = self.active["manifest"]
        audio_path = Path(str(job["payload"]["audio_path"])).resolve()
        if self.active["stage"] == "transfer":
            transfer = client.upload(
                audio_path, manifest["objects"][0]["sha256"]
            )
            self.active["transferred_bytes"] = int(
                transfer.get("bytes") or 0
            )
            client.submit(manifest)
            self.active.update(stage="remote", progress=0.05)
            self._persist()
        remote = client.request("GET", f"/v1/jobs/{job['id']}")
        self.active["remote"] = remote
        self.active["stage"] = str(remote.get("stage") or remote.get("status"))
        remote_progress = remote.get("progress")
        self.active["progress"] = (
            float(remote_progress)
            if remote_progress is not None
            else None
        )
        self.store.heartbeat_analysis_job(job["id"], worker_id=str(job["worker_id"]), progress={"execution_target": "threadripper", "stage": self.active["stage"], "progress": self.active["progress"]})
        self._persist()
        if remote.get("status") == "failed":
            self.store.update_analysis_job(job["id"], status="failed", error=f"Threadripper: {remote.get('error')}")
            self._event("error", f"remote job {job['id']} failed")
            self.active = None
            self._persist()
        elif remote.get("status") == "complete":
            if not self.can_import():
                self.active["stage"] = "awaiting_local_import"
                self.active["progress"] = 1.0
                self._persist()
                return
            result = client.request("GET", f"/v1/jobs/{job['id']}/result")
            imported = self._import_edmformer(job, manifest, result)
            self.store.update_analysis_job(job["id"], status="complete", result=imported)
            self._event("complete", f"imported {job['id']}")
            self.active = None
            self._persist()

    def _manifest(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        audio_path = Path(str(payload["audio_path"])).resolve()
        digest = str(payload.get("content_sha256") or _hash_file(audio_path))
        if _hash_file(audio_path) != digest:
            raise ValueError("queued recording checksum does not match its WAV")
        contract = self._local_contract()
        return {
            "schema": MANIFEST_SCHEMA,
            "job_id": job["id"],
            "job_type": job["job_type"],
            "identity": {
                "recording_id": payload.get("recording_id"),
                "capture_session_id": payload.get("capture_session_id"),
                "song_id": payload.get("song_id"),
                "duration_ms": int(payload["duration_ms"]),
            },
            "objects": [{"role": "audio", "sha256": digest, "bytes": audio_path.stat().st_size, "format": "wav-pcm"}],
            "contract": {
                **contract,
                "result_schema": RESULT_SCHEMA,
            },
            "resources": {"threads": 24},
            # This must survive transfer retry and both-machine restart so
            # submitting the same canonical job remains byte-for-byte
            # idempotent.
            "created_unix_ms": int(job["created_unix_ms"]),
        }

    def _import_edmformer(self, job: dict[str, Any], manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        # Keep health checks and pairing dependency-free. Heavy audio/model
        # modules are needed only for the standby-time canonical import.
        from lumen_engine.offline import (
            _mean_confidence,
            _normalize_teacher_segments,
            _validate_teacher_coverage,
            build_student_examples,
        )

        if result.get("schema") != RESULT_SCHEMA or result.get("job_id") != job["id"]:
            raise LinkProtocolError("remote result identity does not match local job")
        if result.get("manifest_sha256") != _sha256_bytes(_json_bytes(manifest)):
            raise LinkProtocolError("remote result manifest checksum does not match")
        audio = manifest["objects"][0]
        if result.get("input_sha256") != audio["sha256"]:
            raise LinkProtocolError("remote result audio checksum does not match")
        for name in EDM_CONTRACT_FIELDS:
            if result.get(name) != manifest["contract"].get(name):
                raise LinkProtocolError(
                    f"remote result {name} does not match its manifest"
                )
        payload = job["payload"]
        segments = _normalize_teacher_segments(result.get("segments") or [], source="EDMFormer", source_version=str(result.get("teacher_revision") or "unknown"))
        _validate_teacher_coverage(segments, source="EDMFormer", duration_ms=int(payload["duration_ms"]))
        manifest_sha256 = str(result["manifest_sha256"])

        # Import is a transactionally recoverable receipt keyed by the
        # canonical analysis job and immutable manifest. A crash after saving
        # the timeline or completing the teacher run must resume that same run
        # instead of manufacturing duplicate authority.
        for existing in self.store.list_teacher_runs():
            if (
                str(existing.get("analysis_job_id") or "") != str(job["id"])
                or str(existing.get("teacher_name") or "").casefold()
                != "edmformer"
            ):
                continue
            timeline = self.store.structure_timeline_for_teacher_run(
                str(existing["id"])
            )
            metadata = (timeline or {}).get("metadata") or {}
            metrics = dict(existing.get("metrics") or {})
            receipt_hash = str(
                metadata.get("link_manifest_sha256")
                or metrics.get("link_manifest_sha256")
                or ""
            )
            if timeline is not None and receipt_hash == manifest_sha256:
                if existing.get("status") == "complete":
                    return {
                        **metrics,
                        "import_reused": True,
                        "link_manifest_sha256": manifest_sha256,
                    }
                examples = build_student_examples(
                    self.store,
                    research_root=self.research_root,
                    recording_id=str(payload["recording_id"]),
                    timeline_id=str(timeline["id"]),
                )
                resumed_metrics = {
                    **metrics,
                    "elapsed_s": result.get("resources", {}).get(
                        "elapsed_s"
                    ),
                    "segments": len(timeline.get("segments") or segments),
                    "timeline_id": timeline["id"],
                    "student_examples": examples,
                    "execution_target": "threadripper",
                    "link_manifest_sha256": manifest_sha256,
                    "teacher_normalization_version": (
                        TEACHER_NORMALIZATION_VERSION
                    ),
                    "edmformer_preprocessing_version": (
                        EDMFORMER_PREPROCESSING_VERSION
                    ),
                    "resources": result.get("resources", {}),
                    "import_resumed": True,
                }
                self.store.finish_teacher_run(
                    str(existing["id"]),
                    status="complete",
                    metrics=resumed_metrics,
                )
                return resumed_metrics
            if existing.get("status") != "complete":
                self.store.finish_teacher_run(
                    str(existing["id"]),
                    status="failed",
                    error=(
                        "Superseded incomplete Lumen Link import without a "
                        "matching immutable receipt"
                    ),
                )
        run_id = self.store.begin_teacher_run(teacher_name="EDMFormer", teacher_version=str(result.get("teacher_revision") or "unknown"), device="threadripper-cpu", preprocessing_version=EDMFORMER_PREPROCESSING_VERSION, recording_id=payload.get("recording_id"), capture_session_id=payload.get("capture_session_id"), analysis_job_id=str(job["id"]))
        try:
            timeline_id = self.store.save_structure_timeline(recording_id=payload.get("recording_id"), song_id=payload.get("song_id"), capture_session_id=payload.get("capture_session_id"), teacher_run_id=run_id, provenance="edmformer_teacher", timeline_version=TEACHER_NORMALIZATION_VERSION, confidence=_mean_confidence(segments), segments=segments, metadata={"audio_path": str(payload["audio_path"]), "content_sha256": audio["sha256"], "execution_target": "threadripper", "link_manifest_sha256": manifest_sha256, "structure_supervision": payload.get("structure_supervision")})
            examples = build_student_examples(self.store, research_root=self.research_root, recording_id=str(payload["recording_id"]), timeline_id=timeline_id)
            metrics = {"elapsed_s": result.get("resources", {}).get("elapsed_s"), "segments": len(segments), "timeline_id": timeline_id, "student_examples": examples, "execution_target": "threadripper", "link_manifest_sha256": manifest_sha256, "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION, "edmformer_preprocessing_version": EDMFORMER_PREPROCESSING_VERSION, "resources": result.get("resources", {})}
            self.store.finish_teacher_run(run_id, status="complete", metrics=metrics)
            return metrics
        except Exception as error:
            self.store.finish_teacher_run(run_id, status="failed", error=str(error))
            raise

    def _persist(self) -> None:
        _atomic_json(self.state_path, {"schema": LINK_SCHEMA, "active": self.active, "updated_unix_ms": int(time.time() * 1000)})
        self._invalidate_status()

    def _event(self, kind: str, message: str) -> None:
        self.events.appendleft({"kind": kind, "message": message, "unix_ms": int(time.time() * 1000)})
        self._invalidate_status()

    def _invalidate_status(self) -> None:
        with self.lock:
            self._status_cache = None
            self._status_cached_at = 0.0

    def status(self) -> dict[str, Any]:
        with self.lock:
            if (
                self._status_cache is not None
                and time.monotonic() - self._status_cached_at < 2.0
            ):
                return deepcopy(self._status_cache)
        configuration = self.configuration
        jobs = [self.active] if self.active else []
        remote_jobs = (self.remote_status or {}).get("jobs") or []
        known_ids = {
            str(
                (item.get("job") or item).get("job_id")
                or (item.get("job") or item).get("id")
                or ""
            )
            for item in jobs
        }
        jobs.extend(
            item
            for item in remote_jobs
            if str(item.get("job_id") or "") not in known_ids
        )
        local_jobs = self.store.list_analysis_jobs(limit=1000)
        queued = [job for job in local_jobs if job["status"] == "queued" and job["job_type"] == EDMFORMER_JOB]
        queue_bytes = sum(Path(str(job["payload"].get("audio_path") or "")).stat().st_size for job in queued if Path(str(job["payload"].get("audio_path") or "")).is_file())
        if not configuration.secret:
            setup_state = "needs_secret"
            next_action = (
                "Run the Lumen Link setup script on both computers."
            )
        elif self.last_error:
            setup_state = "connection_error"
            next_action = "Check the direct cable, WSL service, and firewall."
        elif self.remote_status and not self._remote_is_compatible():
            setup_state = "incompatible"
            next_action = (
                "Update both computers to the same clean code and model assets."
            )
        elif not configuration.enabled:
            setup_state = "ready_to_enable"
            next_action = "The verified link is ready to enable."
        elif configuration.paused:
            setup_state = "paused"
            next_action = "Resume when the Threadripper is available."
        elif not self.remote_status:
            setup_state = "awaiting_connection"
            next_action = "Test the authenticated Threadripper connection."
        else:
            setup_state = "ready"
            next_action = "Queued EDMFormer jobs will run on the Threadripper."
        result = {
            "schema": LINK_SCHEMA,
            "configured": bool(configuration.secret),
            "enabled": configuration.enabled,
            "paused": configuration.paused,
            "connection": {"state": "ready" if self._remote_is_compatible() else ("error" if self.last_error else "offline"), "authenticated": bool(self.remote_status and not self.last_error), "latency_ms": self.latency_ms, "last_seen_unix_ms": int(self.last_seen * 1000) if self.last_seen else None, "error": self.last_error, "endpoint": configuration.endpoint, "address": configuration.endpoint.removeprefix("http://").removeprefix("https://")},
            "local_node": {**_node_resources(), "address": "192.168.50.2"},
            "remote_node": {**((self.remote_status or {}).get("node") or {}), "address": configuration.endpoint.removeprefix("http://").removeprefix("https://")},
            "capabilities": (self.remote_status or {}).get("capabilities", {"supported_job_types": [], "gated_job_types": {EDMFORMER_JOB: "awaiting authenticated compatible health", "teacher.songformer": "not implemented", "student.train": "not implemented"}}),
            "queue": {"queued": len(queued), "bytes_pending": queue_bytes, "bytes_transferred": int((self.active or {}).get("transferred_bytes") or 0), "remote": (self.remote_status or {}).get("queue", {})},
            "jobs": jobs,
            "events": list(self.events),
            "setup": {
                "state": setup_state,
                "compatible": self._remote_is_compatible(),
                "next_action": next_action,
                "endpoint": configuration.endpoint,
                "config_path": str(self.config_path),
                "direct_link": (
                    "Threadripper 192.168.50.1 ↔ Lumen 192.168.50.2"
                ),
                "port": 8765,
                "commands": [
                    "./scripts/lumen-link status",
                    "./scripts/lumen-link test",
                    (
                        "git status --short"
                        if setup_state == "incompatible"
                        else "./scripts/lumen-link setup"
                    ),
                ],
            },
        }
        with self.lock:
            self._status_cache = deepcopy(result)
            self._status_cached_at = time.monotonic()
        return result


def serve_link_node(
    *,
    host: str,
    port: int,
    secret: bytes,
    spool_root: Path,
    research_root: Path,
    project_root: Path,
    max_threads: int = 24,
    max_memory_gib: float = 96.0,
) -> None:
    spool = LinkSpool(spool_root)
    runtime = LinkNodeRuntime(
        spool,
        LinkNodeExecutor(
            spool,
            research_root=research_root,
            project_root=project_root,
            max_threads=max_threads,
            max_memory_gib=max_memory_gib,
        ),
    )
    server = LinkNodeServer((host, port), spool=spool, authenticator=LinkAuthenticator(secret), runtime=runtime)
    runtime.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        runtime.stop()
        server.server_close()
