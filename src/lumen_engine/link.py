"""Lumen Link: authenticated, durable offline compute over a private LAN.

The live computer owns SQLite and imports every result.  A compute node sees
only immutable objects and versioned manifests; it cannot drive audio or DMX.
The wire implementation intentionally uses the standard library so the link
can be diagnosed before either research environment is installed in WSL.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
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
import sys
import threading
import time
from typing import Any, BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave

from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SONGFORMER_PREPROCESSING_PREFIX,
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
SONGFORMER_JOB = "teacher.songformer"
STUDENT_TRAIN_JOB = "student.train"
SONGFORMER_WINDOW_SECONDS = 60
EDMFORMER_MAX_THREADS = 8
DEFAULT_PARALLEL_JOBS = 6
COORDINATOR_POLL_SECONDS = 0.5
JOB_SNAPSHOT_SECONDS = 2.0
SONGFORMER_PREPROCESSING_VERSION = (
    f"{SONGFORMER_PREPROCESSING_PREFIX}{SONGFORMER_WINDOW_SECONDS}s:"
    f"{TEACHER_NORMALIZATION_VERSION}"
)
SUPPORTED_JOB_TYPES = (
    EDMFORMER_JOB,
    SONGFORMER_JOB,
    STUDENT_TRAIN_JOB,
)
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
SONG_CONTRACT_FIELDS = (
    "code_revision",
    "code_clean",
    "songformer_revision",
    "songformer_clean",
    "musicfm_source_revision",
    "musicfm_source_clean",
    "songformer_head_sha256",
    "musicfm_stats_sha256",
    "musicfm_model_sha256",
    "muq_assets_sha256",
    "teacher_normalization_version",
    "songformer_preprocessing_version",
)
STUDENT_CONTRACT_FIELDS = (
    "code_revision",
    "code_clean",
    "student_format_version",
    "student_audio_feature_version",
    "student_activation_gate_version",
    "teacher_fusion_version",
    "teacher_normalization_version",
)
JOB_CONTRACT_FIELDS = {
    EDMFORMER_JOB: EDM_CONTRACT_FIELDS,
    SONGFORMER_JOB: SONG_CONTRACT_FIELDS,
    STUDENT_TRAIN_JOB: STUDENT_CONTRACT_FIELDS,
}


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


def _hash_wav_pcm(path: Path) -> str | None:
    """Hash the PCM payload used by Lumen recording identities."""

    try:
        digest = hashlib.sha256()
        with wave.open(str(path), "rb") as source:
            while chunk := source.readframes(262_144):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, EOFError, wave.Error):
        return None


def _audio_object_digest(path: Path, declared: str | None) -> str:
    """Validate a queued audio identity and return the full-WAV digest.

    Recording metadata uses a PCM-content digest, whereas Link object
    transport hashes the complete WAV file. Existing queued jobs therefore
    need both forms accepted while the transferred object remains immutable.
    """

    wav_digest = _hash_file(path)
    if not declared or declared == wav_digest:
        return wav_digest
    if declared != _hash_wav_pcm(path):
        raise ValueError("queued recording checksum does not match its WAV")
    return wav_digest


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


def _git_clean_except(path: Path, allowed_prefixes: tuple[str, ...]) -> bool:
    """Accept provisioned model material only when separately checksummed."""
    try:
        rows = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    for row in rows:
        relative = row[3:].split(" -> ")[-1].strip()
        if not any(
            relative == prefix.rstrip("/")
            or relative.startswith(prefix)
            for prefix in allowed_prefixes
        ):
            return False
    return True


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
        "teacher_clean": _git_clean_except(
            source,
            (
                "data/checkpoints/model.pt",
                "data/checkpoints/pretrained_msd.pt",
                "data/checkpoints/msd_stats.json",
            ),
        ),
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
        "teacher_clean": _git_clean_except(
            source,
            (
                "data/checkpoints/model.pt",
                "data/checkpoints/pretrained_msd.pt",
                "data/checkpoints/msd_stats.json",
            ),
        ),
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


def _songformer_asset_contract(
    research_root: Path, project_root: Path
) -> dict[str, Any]:
    source = research_root / "sources" / "songformer"
    edm_checkpoint = (
        research_root / "sources" / "edm98" / "data" / "checkpoints"
    )
    musicfm_source = research_root / "sources" / "musicfm"
    muq_cache = (
        research_root
        / "cache"
        / "huggingface"
        / "models--OpenMuQ--MuQ-large-msd-iter"
    )
    head = research_root / "models" / "songformer" / "SongFormer.safetensors"
    return {
        "code_revision": _git_revision(project_root),
        "code_clean": _git_clean(project_root),
        "songformer_revision": _git_revision(source),
        "songformer_clean": _git_clean_except(
            source, ("src/SongFormer/ckpts/",)
        ),
        "musicfm_source_revision": _git_revision(musicfm_source),
        "musicfm_source_clean": _git_clean(musicfm_source),
        "songformer_head_sha256": (
            _hash_file(head) if head.is_file() else None
        ),
        "musicfm_stats_sha256": (
            _hash_file(edm_checkpoint / "msd_stats.json")
            if (edm_checkpoint / "msd_stats.json").is_file()
            else None
        ),
        "musicfm_model_sha256": (
            _hash_file(edm_checkpoint / "pretrained_msd.pt")
            if (edm_checkpoint / "pretrained_msd.pt").is_file()
            else None
        ),
        "muq_assets_sha256": _hash_tree(muq_cache / "snapshots"),
        "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
        "songformer_preprocessing_version": (
            SONGFORMER_PREPROCESSING_VERSION
        ),
    }


def _songformer_asset_signature(
    research_root: Path, project_root: Path
) -> dict[str, Any]:
    source = research_root / "sources" / "songformer"
    edm_checkpoint = (
        research_root / "sources" / "edm98" / "data" / "checkpoints"
    )
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
        "songformer_revision": _git_revision(source),
        "songformer_clean": _git_clean_except(
            source, ("src/SongFormer/ckpts/",)
        ),
        "musicfm_revision": _git_revision(musicfm),
        "musicfm_clean": _git_clean(musicfm),
        "files": {
            name: file_signature(path)
            for name, path in {
                "head": research_root
                / "models"
                / "songformer"
                / "SongFormer.safetensors",
                "stats": edm_checkpoint / "msd_stats.json",
                "musicfm": edm_checkpoint / "pretrained_msd.pt",
                "config": source
                / "src"
                / "SongFormer"
                / "configs"
                / "SongFormer.yaml",
                "runner": project_root / "scripts" / "songformer-cpu-runner.py",
            }.items()
        },
        "muq": {
            str(path.relative_to(muq)): file_signature(path)
            for path in sorted(muq.rglob("*"))
            if path.is_file()
        } if muq.is_dir() else {},
    }


def _student_asset_contract(project_root: Path) -> dict[str, Any]:
    """Return the exact scientific/code contract for remote student work."""
    from lumen_engine.offline import (
        STUDENT_ACTIVATION_GATE_VERSION,
        STUDENT_AUDIO_FEATURE_VERSION,
        TEACHER_FUSION_VERSION,
    )
    from lumen_engine.student import StreamingStructureStudent

    return {
        "code_revision": _git_revision(project_root),
        "code_clean": _git_clean(project_root),
        "student_format_version": StreamingStructureStudent.format_version,
        "student_audio_feature_version": STUDENT_AUDIO_FEATURE_VERSION,
        "student_activation_gate_version": STUDENT_ACTIVATION_GATE_VERSION,
        "teacher_fusion_version": TEACHER_FUSION_VERSION,
        "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
    }


def _job_asset_contract(
    job_type: str, research_root: Path, project_root: Path
) -> dict[str, Any]:
    if job_type == EDMFORMER_JOB:
        return _edmformer_asset_contract(research_root, project_root)
    if job_type == SONGFORMER_JOB:
        return _songformer_asset_contract(research_root, project_root)
    if job_type == STUDENT_TRAIN_JOB:
        # Asset contracts describe the code/model versions required by a job.
        # Manifest object-role validation happens when the manifest is built or
        # validated, where the manifest's ``objects`` list is actually in
        # scope.  Keeping that validation here made the console compatibility
        # scan raise ``NameError: objects is not defined`` for student jobs.
        return _student_asset_contract(project_root)
    raise ValueError(f"unsupported Lumen Link job type {job_type!r}")


def _job_asset_signature(
    job_type: str, research_root: Path, project_root: Path
) -> dict[str, Any]:
    if job_type == EDMFORMER_JOB:
        return _edmformer_asset_signature(research_root, project_root)
    if job_type == SONGFORMER_JOB:
        return _songformer_asset_signature(research_root, project_root)
    if job_type == STUDENT_TRAIN_JOB:
        return _student_asset_contract(project_root)
    raise ValueError(f"unsupported Lumen Link job type {job_type!r}")


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


class LinkStandbyRequired(RuntimeError):
    """Heavy local Link work paused because the engine entered Live."""


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
        # Discover the durable spool once at process start.  Repeated status and
        # worker polls then use this bounded index instead of rescanning a large
        # directory every two seconds.
        self._known_job_paths: deque[Path] = deque(
            self.jobs.glob("*.json"), maxlen=10_000
        )

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

    def publish_file(self, source: Path) -> dict[str, Any]:
        """Atomically publish a node-produced immutable result artifact."""
        source = source.resolve()
        digest = _hash_file(source)
        target = self.object_path(digest)
        with self.lock:
            if target.is_file():
                if target.stat().st_size != source.stat().st_size:
                    raise ValueError("immutable artifact digest collision")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(".publish")
                shutil.copyfile(source, partial)
                if _hash_file(partial) != digest:
                    partial.unlink(missing_ok=True)
                    raise ValueError("published artifact checksum changed")
                partial.replace(target)
        return {"sha256": digest, "bytes": target.stat().st_size}

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
            if path not in self._known_job_paths:
                self._known_job_paths.append(path)
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

    def list_jobs(self, *, limit: int = 1_000) -> list[dict[str, Any]]:
        result = []
        paths = list(self._known_job_paths)[-max(1, int(limit)) :]
        for path in paths:
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
        queued = [
            job
            for job in self.list_jobs(limit=1_000)
            if job.get("status") == "queued"
        ]
        return queued[-1] if queued else None

    def recover_running(self) -> int:
        """Requeue jobs interrupted by compute-node or WSL shutdown."""
        recovered = 0
        for job in self.list_jobs(limit=1_000):
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
    job_type = str(manifest["job_type"])
    identity = manifest.get("identity") or {}
    if job_type in {EDMFORMER_JOB, SONGFORMER_JOB}:
        audio_objects = [
            item for item in objects if item.get("role") == "audio"
        ]
        if len(audio_objects) != 1 or len(objects) != 1:
            raise ValueError(
                "teacher manifest requires exactly one immutable audio object"
            )
        if audio_objects[0].get("format") != "wav-pcm":
            raise ValueError("teacher audio object must be PCM WAV")
        if not str(identity.get("recording_id") or ""):
            raise ValueError("teacher manifest has no recording identity")
        if int(identity.get("duration_ms") or 0) <= 0:
            raise ValueError("teacher manifest duration must be positive")
    contract = manifest.get("contract") or {}
    if contract.get("teacher_normalization_version") != TEACHER_NORMALIZATION_VERSION:
        raise ValueError("teacher normalization contract does not match node")
    if (
        job_type == EDMFORMER_JOB
        and contract.get("edmformer_preprocessing_version")
        != EDMFORMER_PREPROCESSING_VERSION
    ):
        raise ValueError("EDMFormer preprocessing contract does not match node")
    if (
        job_type == SONGFORMER_JOB
        and contract.get("songformer_preprocessing_version")
        != SONGFORMER_PREPROCESSING_VERSION
    ):
        raise ValueError("SongFormer preprocessing contract does not match node")
    if job_type == STUDENT_TRAIN_JOB:
        example_objects = [
            item for item in objects if item.get("role") == "student_examples"
        ]
        if (
            len(example_objects) != 1
            or example_objects[0].get("format") != "jsonl"
        ):
            raise ValueError("student training manifest has no examples object")
        audio_rows = [
            item for item in objects if item.get("role") == "recording_audio"
        ]
        if len(objects) != 1 + len(audio_rows):
            raise ValueError("student manifest object set is ambiguous")
        recording_values = [
            str(item.get("recording_id") or "")
            for item in audio_rows
        ]
        if (
            not recording_values
            or any(not value for value in recording_values)
            or len(recording_values) != len(set(recording_values))
            or any(item.get("format") != "wav-pcm" for item in audio_rows)
        ):
            raise ValueError("student recording audio objects are ambiguous")
        recording_ids = set(recording_values)
        declared = {
            str(value)
            for value in (manifest.get("identity") or {}).get(
                "recording_ids", []
            )
        }
        if not declared or recording_ids != declared:
            raise ValueError(
                "student training audio objects do not match its recording set"
            )


_cpu_sample_lock = threading.Lock()
_cpu_sample: tuple[int, int] | None = None


def _cpu_usage_percent() -> float | None:
    """Return CPU use since the previous telemetry sample from /proc/stat."""

    global _cpu_sample
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
    except (OSError, ValueError, IndexError):
        return None
    with _cpu_sample_lock:
        previous = _cpu_sample
        _cpu_sample = (total, idle)
    if previous is None:
        return None
    total_delta = total - previous[0]
    idle_delta = idle - previous[1]
    if total_delta <= 0:
        return None
    return max(
        0.0,
        min(100.0, 100.0 * (total_delta - idle_delta) / total_delta),
    )


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
        "cpu_usage_percent": _cpu_usage_percent(),
        "load_1m": load[0],
        "load_5m": load[1],
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": None,
    }


def _student_gate_assessment(
    model: Any,
    examples: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Independently reproduce the canonical held-out activation decision."""
    from lumen_engine.offline import (
        COMBINED_STUDENT_AXES,
        MIN_ACTIVATION_TEST_GROUPS,
        MIN_BALANCED_ACCURACY_MARGIN,
        MIN_CLASSIFIER_BASELINE_MARGIN,
        _student_example_statistics,
    )
    from lumen_engine.student import LABELS

    statistics = _student_example_statistics(
        examples, require_group_identity=True
    )
    evaluation = {
        split: model.evaluate(
            [
                row
                for row in examples
                if row.get("split", "train") == split
            ]
        )
        for split in ("train", "validation", "test")
    }
    held_out_name = (
        "test" if evaluation["test"]["energy"]["examples"] else "validation"
    )
    held_out = evaluation[held_out_name]
    all_axes = {*LABELS.keys(), "boundary"}
    applicable_axes = {
        str(axis)
        for axis in (
            payload.get("applicable_axes") or COMBINED_STUDENT_AXES
        )
        if str(axis) in all_axes
    }
    axis_reasons = {axis: [] for axis in all_axes}
    test_groups = int(statistics["split_group_counts"].get("test") or 0)
    reliable = test_groups >= MIN_ACTIVATION_TEST_GROUPS
    gate_reasons: list[str] = []
    if not reliable:
        gate_reasons.append(
            "held-out test contains "
            f"{test_groups} independent songs; at least "
            f"{MIN_ACTIVATION_TEST_GROUPS} are required"
        )
    energy = held_out["energy"]
    if "energy" in applicable_axes:
        if int(energy.get("examples") or 0) < 10:
            axis_reasons["energy"].append(
                "held-out set has fewer than 10 energy frames"
            )
        elif float(energy.get("majority_baseline") or 0.0) >= 0.999:
            axis_reasons["energy"].append(
                "held-out energy set does not contain multiple classes"
            )
        elif float(energy.get("accuracy") or 0.0) < max(
            0.35,
            float(energy.get("majority_baseline") or 0.0)
            + MIN_CLASSIFIER_BASELINE_MARGIN,
        ):
            axis_reasons["energy"].append(
                "held-out energy accuracy did not meet its baseline gate"
            )
        elif float(energy.get("balanced_accuracy") or 0.0) < max(
            0.25,
            float(energy.get("balanced_baseline") or 0.0)
            + MIN_BALANCED_ACCURACY_MARGIN,
        ):
            axis_reasons["energy"].append(
                "held-out energy balanced accuracy did not meet its per-class gate"
            )
    for axis, minimum, baseline_floor in (
        ("functional", 10, 0.25),
        ("content", 10, 0.35),
    ):
        metrics = held_out[axis]
        if axis not in applicable_axes:
            continue
        if int(metrics.get("examples") or 0) < minimum:
            axis_reasons[axis].append(
                f"held-out set has fewer than {minimum} {axis} frames"
            )
        elif float(metrics.get("majority_baseline") or 0.0) >= 0.999:
            axis_reasons[axis].append(
                f"held-out {axis} set does not contain multiple classes"
            )
        elif float(metrics.get("accuracy") or 0.0) < max(
            baseline_floor,
            float(metrics.get("majority_baseline") or 0.0)
            + MIN_CLASSIFIER_BASELINE_MARGIN,
        ):
            axis_reasons[axis].append(
                f"held-out {axis} accuracy did not meet its baseline gate"
            )
    boundary = held_out["boundary"]
    if "boundary" in applicable_axes:
        positives = int(boundary.get("event_tp") or 0) + int(
            boundary.get("event_fn") or 0
        )
        if int(boundary.get("examples") or 0) < 10:
            axis_reasons["boundary"].append(
                "held-out set has fewer than 10 boundary frames"
            )
        elif positives < 5:
            axis_reasons["boundary"].append(
                "held-out set has fewer than 5 positive boundaries"
            )
        elif (
            float(boundary.get("event_f1") or 0.0) < 0.20
            or float(boundary.get("event_precision") or 0.0) < 0.12
        ):
            axis_reasons["boundary"].append(
                "held-out boundary events did not meet their tolerant precision/F1 gate"
            )
    approved = {
        axis for axis in applicable_axes if not axis_reasons[axis]
    }
    if not reliable:
        approved.clear()
    if not approved and not gate_reasons:
        gate_reasons.append("no student axis passed its held-out activation gate")
    return {
        "activated": bool(approved),
        "approved_axes": sorted(approved),
        "inactive_axes": sorted(applicable_axes - approved),
        "not_applicable_axes": sorted(all_axes - applicable_axes),
        "held_out_split": held_out_name,
        "evaluation": evaluation,
        "axis_gate_reasons": axis_reasons,
        "gate_reasons": gate_reasons,
        "test_population_reliable": reliable,
        "split_counts": statistics["split_counts"],
        "split_group_counts": statistics["split_group_counts"],
        "label_balance": statistics["label_balance"],
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
        self._capability_signatures: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> dict[str, Any]:
        if self._capabilities_cache is not None:
            return dict(self._capabilities_cache)
        contracts = {
            job_type: _job_asset_contract(
                job_type, self.research_root, self.project_root
            )
            for job_type in SUPPORTED_JOB_TYPES
        }
        self._capability_signatures = {
            job_type: _job_asset_signature(
                job_type, self.research_root, self.project_root
            )
            for job_type in SUPPORTED_JOB_TYPES
        }
        checkpoint_root = (
            self.research_root / "sources" / "edm98" / "data" / "checkpoints"
        )
        edm_required = (
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
        song_required = (
            self.research_root
            / "environments"
            / "songformer"
            / "bin"
            / "python",
            self.project_root / "scripts" / "songformer-cpu-runner.py",
            self.research_root
            / "models"
            / "songformer"
            / "SongFormer.safetensors",
            self.research_root
            / "sources"
            / "songformer"
            / "src"
            / "SongFormer"
            / "configs"
            / "SongFormer.yaml",
            checkpoint_root / "msd_stats.json",
            checkpoint_root / "pretrained_msd.pt",
            self.research_root
            / "sources"
            / "musicfm"
            / "model"
            / "musicfm_25hz.py",
        )
        edm = contracts[EDMFORMER_JOB]
        song = contracts[SONGFORMER_JOB]
        student = contracts[STUDENT_TRAIN_JOB]
        edm_provisioned = all(path.exists() for path in edm_required) and all(
            edm.get(name)
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
        edm_deterministic = all(
            edm.get(name)
            for name in ("code_clean", "teacher_clean", "musicfm_source_clean")
        )
        song_provisioned = all(path.exists() for path in song_required) and all(
            song.get(name)
            for name in (
                "code_revision",
                "songformer_revision",
                "musicfm_source_revision",
                "songformer_head_sha256",
                "musicfm_stats_sha256",
                "musicfm_model_sha256",
                "muq_assets_sha256",
            )
        )
        song_deterministic = all(
            song.get(name)
            for name in (
                "code_clean",
                "songformer_clean",
                "musicfm_source_clean",
            )
        )
        student_ready = bool(student.get("code_revision") and student.get("code_clean"))
        supported: list[str] = []
        gated_jobs: dict[str, str] = {}
        if edm_provisioned and edm_deterministic:
            supported.append(EDMFORMER_JOB)
        elif not edm_provisioned:
            gated_jobs[EDMFORMER_JOB] = (
                "EDMFormer environment or model assets are not provisioned"
            )
        else:
            gated_jobs[EDMFORMER_JOB] = (
                "Lumen, EDMFormer, or MusicFM has uncommitted source changes"
            )
        if song_provisioned and song_deterministic:
            supported.append(SONGFORMER_JOB)
        elif not song_provisioned:
            gated_jobs[SONGFORMER_JOB] = (
                "SongFormer environment or model assets are not provisioned"
            )
        else:
            gated_jobs[SONGFORMER_JOB] = (
                "Lumen, SongFormer, or MusicFM has uncommitted source changes"
            )
        if student_ready:
            supported.append(STUDENT_TRAIN_JOB)
        else:
            gated_jobs[STUDENT_TRAIN_JOB] = (
                "Lumen code must be committed and identical on both computers"
            )
        self._capabilities_cache = {
            "protocol_schema": LINK_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            "job_contracts": contracts,
            # Retain the original flat EDM fields for older dashboard builds.
            **edm,
            "supported_job_types": supported,
            "gated_job_types": gated_jobs,
            "max_threads": self.max_threads,
            "max_memory_bytes": self.max_memory_bytes,
            "gpu": False,
            "live_timing": False,
            "dmx": False,
        }
        return dict(self._capabilities_cache)

    def validate_contract(
        self, job_type: str, supplied: dict[str, Any]
    ) -> None:
        """Reject a stale health contract if code/assets changed afterward."""
        if self._capabilities_cache is None:
            self.capabilities()
        current_signature = _job_asset_signature(
            job_type, self.research_root, self.project_root
        )
        if current_signature != self._capability_signatures.get(job_type):
            self._capabilities_cache = None
            self._capability_signatures = {}
        capabilities = self.capabilities()
        expected = (capabilities.get("job_contracts") or {}).get(job_type)
        if job_type not in capabilities.get("supported_job_types", []):
            raise RuntimeError(
                (capabilities.get("gated_job_types") or {}).get(
                    job_type, "compute job is no longer available"
                )
            )
        for name in JOB_CONTRACT_FIELDS[job_type]:
            if supplied.get(name) != expected.get(name):
                raise RuntimeError(
                    f"compute-node {name} changed after health verification"
                )

    def execute(
        self,
        state: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        manifest = state["manifest"]
        job_type = str(manifest["job_type"])
        self.validate_contract(job_type, manifest.get("contract") or {})
        if job_type == STUDENT_TRAIN_JOB:
            return self._execute_student(state, progress_callback)
        if job_type not in {EDMFORMER_JOB, SONGFORMER_JOB}:
            raise ValueError(f"unsupported fixed compute job {job_type!r}")
        return self._execute_teacher(state, progress_callback)

    def _execute_teacher(
        self,
        state: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        manifest = state["manifest"]
        job_type = str(manifest["job_type"])
        audio = next(item for item in manifest["objects"] if item["role"] == "audio")
        audio_path = self.spool.object_path(audio["sha256"])
        output = self.spool.results / (self.spool.job_path(manifest["job_id"]).stem + ".raw.json")
        threads = max(
            1,
            min(
                self.max_threads,
                int(
                    manifest.get("resources", {}).get("threads")
                    or self.max_threads
                ),
            ),
        )
        if job_type == EDMFORMER_JOB:
            # The validated CPU runner deliberately rejects larger values;
            # SongFormer and student training can still use the worker's
            # wider Threadripper ceiling.
            threads = min(threads, EDMFORMER_MAX_THREADS)
            source = self.research_root / "sources" / "edm98"
            checkpoint = source / "data" / "checkpoints"
            command = [
                str(self.research_root / "environments" / "edmformer" / "bin" / "python"),
                str(self.project_root / "scripts" / "edmformer-cpu-runner.py"),
                str(audio_path),
                "--checkpoint", str(checkpoint / "model.pt"),
                "--config", str(source / "configs" / "edmformer.yaml"),
                "--musicfm-stat", str(checkpoint / "msd_stats.json"),
                "--musicfm-model", str(checkpoint / "pretrained_msd.pt"),
                "--musicfm-source", str(self.research_root / "sources" / "musicfm"),
                "--hf-cache-dir", str(self.research_root / "cache" / "huggingface"),
                "--threads", str(threads),
                "--output", str(output),
            ]
            required_indices = (0, 1, 4, 6, 8, 10, 12)
            display_name = "EDMFormer"
        else:
            command = [
                str(self.research_root / "environments" / "songformer" / "bin" / "python"),
                str(self.project_root / "scripts" / "songformer-cpu-runner.py"),
                str(audio_path),
                "--research-root", str(self.research_root),
                "--window-seconds", str(SONGFORMER_WINDOW_SECONDS),
                "--threads", str(threads),
                "--output", str(output),
            ]
            required_indices = (0, 1)
            display_name = "SongFormer"
        missing = [Path(command[index]) for index in required_indices if not Path(command[index]).exists()]
        if missing:
            raise RuntimeError(display_name + " compute node is not provisioned: " + ", ".join(map(str, missing)))
        started = time.monotonic()
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "NUMEXPR_NUM_THREADS": str(threads),
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
                            "threads": threads,
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
                        f"{display_name} exceeded the compute-node memory limit"
                    )
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or f"{display_name} exited {process.returncode}")[-4000:])
        segments = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"{display_name} returned no timeline segments")
        contract = self.capabilities()["job_contracts"][job_type]
        return {
            "schema": RESULT_SCHEMA,
            "job_id": manifest["job_id"],
            "job_type": manifest["job_type"],
            "manifest_sha256": state["manifest_sha256"],
            "input_sha256": audio["sha256"],
            "duration_ms": manifest["identity"]["duration_ms"],
            **{
                name: contract.get(name)
                for name in JOB_CONTRACT_FIELDS[job_type]
            },
            "segments": segments,
            "resources": {
                "elapsed_s": time.monotonic() - started,
                "threads": threads,
                "peak_rss_bytes": peak_rss,
                "memory_limit_bytes": self.max_memory_bytes,
                "returncode": process.returncode,
            },
        }

    def _execute_student(
        self,
        state: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        """Run the fixed canonical trainer in a monitorable child process."""
        manifest = state["manifest"]
        examples_object = next(
            item
            for item in manifest["objects"]
            if item["role"] == "student_examples"
        )
        work = self.spool.results / (
            self.spool.job_path(manifest["job_id"]).stem + ".student"
        )
        work.mkdir(parents=True, exist_ok=True)
        training = dict(manifest.get("training") or {})
        allowed = {
            "epochs",
            "hidden_size",
            "source_scope",
            "teacher_run_ids",
            "source_files",
            "split_counts",
            "split_group_counts",
            "label_balance",
            "teacher_merge",
            "teacher_fusion_version",
            "operator_consensus",
            "operator_consensus_revision",
            "operator_timeline_corrections",
            "trainer_version",
            "applicable_axes",
        }
        unknown = set(training) - allowed
        if unknown:
            raise ValueError(
                "student manifest contains unsupported training fields: "
                + ", ".join(sorted(unknown))
            )
        output = work / "lumen-structure-student.npz"
        prepared = work / "student-training.prepared.jsonl"
        progress_path = work / "progress.json"
        runner_result_path = work / "runner-result.json"
        spec_path = work / "specification.json"
        _atomic_json(
            spec_path,
            {
                "schema": "lumen.link.student-runner.v1",
                "work_root": str(work),
                "progress_path": str(progress_path),
                "examples_path": str(
                    self.spool.object_path(examples_object["sha256"])
                ),
                "examples_sha256": str(examples_object["sha256"]),
                "prepared_examples_path": str(prepared),
                "output_path": str(output),
                "result_path": str(runner_result_path),
                "recordings": [
                    {
                        "recording_id": str(item["recording_id"]),
                        "audio_path": str(
                            self.spool.object_path(item["sha256"])
                        ),
                        "sha256": str(item["sha256"]),
                    }
                    for item in manifest["objects"]
                    if item.get("role") == "recording_audio"
                ],
                "training": training,
            },
        )
        started = time.monotonic()
        threads = max(
            1,
            min(
                self.max_threads,
                int(
                    manifest.get("resources", {}).get("threads")
                    or self.max_threads
                ),
            ),
        )
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "NUMEXPR_NUM_THREADS": str(threads),
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "elapsed_s": 0.0,
                    "rss_bytes": 0,
                    "peak_rss_bytes": 0,
                    "memory_limit_bytes": self.max_memory_bytes,
                    "threads": threads,
                    "stage": "student_feature_preparation",
                }
            )
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.project_root / "scripts" / "lumen-link-student-runner.py"),
                str(spec_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        peak_rss = 0
        stage = "student_feature_preparation"
        if progress_callback is not None:
            progress_callback(
                {
                    "elapsed_s": time.monotonic() - started,
                    "rss_bytes": 0,
                    "peak_rss_bytes": 0,
                    "memory_limit_bytes": self.max_memory_bytes,
                    "threads": threads,
                    "stage": "student_training",
                }
            )
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
                                peak_rss, int(line.split()[1]) * 1024
                            )
                            break
                except (OSError, ValueError, IndexError):
                    pass
                try:
                    stage = str(_read_json(progress_path).get("stage") or stage)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                if progress_callback is not None:
                    progress_callback(
                        {
                            "elapsed_s": time.monotonic() - started,
                            "rss_bytes": peak_rss,
                            "peak_rss_bytes": peak_rss,
                            "memory_limit_bytes": self.max_memory_bytes,
                            "threads": threads,
                            "stage": stage,
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
                        "student training exceeded the compute-node memory limit"
                    )
        if process.returncode != 0:
            raise RuntimeError(
                (stderr or stdout or "student runner failed")[-4000:]
            )
        runner_result = _read_json(runner_result_path)
        trained = dict(runner_result["result"])
        if progress_callback is not None:
            progress_callback(
                {
                    "elapsed_s": time.monotonic() - started,
                    "rss_bytes": peak_rss,
                    "peak_rss_bytes": peak_rss,
                    "memory_limit_bytes": self.max_memory_bytes,
                    "threads": threads,
                    "stage": "student_validation",
                }
            )
        candidate = Path(str(trained["candidate_model_path"])).resolve()
        evaluation = Path(str(trained["evaluation_path"])).resolve()
        artifacts = {
            "candidate_model": {
                **self.spool.publish_file(candidate),
                "format": "numpy-npz",
            },
            "evaluation": {
                **self.spool.publish_file(evaluation),
                "format": "json",
            },
            "prepared_examples": {
                **self.spool.publish_file(prepared),
                "format": "jsonl",
            },
        }
        if progress_callback is not None:
            progress_callback(
                {
                    "elapsed_s": time.monotonic() - started,
                    "rss_bytes": peak_rss,
                    "peak_rss_bytes": peak_rss,
                    "memory_limit_bytes": self.max_memory_bytes,
                    "threads": threads,
                    "stage": "student_artifacts",
                }
            )
        contract = self.capabilities()["job_contracts"][STUDENT_TRAIN_JOB]
        return {
            "schema": RESULT_SCHEMA,
            "job_id": manifest["job_id"],
            "job_type": STUDENT_TRAIN_JOB,
            "manifest_sha256": state["manifest_sha256"],
            "input_sha256": str(examples_object["sha256"]),
            **{
                name: contract.get(name)
                for name in STUDENT_CONTRACT_FIELDS
            },
            "artifacts": artifacts,
            "report": json.loads(evaluation.read_text(encoding="utf-8")),
            "resources": {
                "elapsed_s": time.monotonic() - started,
                "threads": threads,
                "peak_rss_bytes": peak_rss,
                "memory_limit_bytes": self.max_memory_bytes,
            },
        }


class LinkNodeRuntime:
    def __init__(
        self,
        spool: LinkSpool,
        executor: LinkNodeExecutor,
        *,
        maximum_parallel_jobs: int = DEFAULT_PARALLEL_JOBS,
    ) -> None:
        self.spool = spool
        self.executor = executor
        self.maximum_parallel_jobs = max(1, int(maximum_parallel_jobs))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.job_threads: dict[str, threading.Thread] = {}
        self.started_at = time.time()
        self.last_coordinator_contact_at: float | None = None

    def note_coordinator_contact(self) -> None:
        """Record a successfully authenticated request from the Lumen PC."""

        self.last_coordinator_contact_at = time.time()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.spool.recover_running()
        self.executor.capabilities()
        self.thread = threading.Thread(target=self._run, name="lumen-link-node", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        for thread in list(self.job_threads.values()):
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.wait(0.5):
            self.job_threads = {
                job_id: thread
                for job_id, thread in self.job_threads.items()
                if thread.is_alive()
            }
            running = self.spool.list_jobs(limit=1_000)
            running_types = {
                str(job.get("job_type"))
                for job in running
                if job.get("status") == "running"
            }
            if STUDENT_TRAIN_JOB in running_types:
                continue
            capacity = self.maximum_parallel_jobs - len(self.job_threads)
            if capacity <= 0:
                continue
            queued = [
                job for job in reversed(running)
                if job.get("status") == "queued"
            ]
            if not queued:
                continue
            if queued[0].get("job_type") == STUDENT_TRAIN_JOB:
                if self.job_threads:
                    continue
                queued = queued[:1]
            else:
                queued = [
                    job for job in queued
                    if job.get("job_type") != STUDENT_TRAIN_JOB
                ][:capacity]
            for state in queued:
                job_id = str(state["job_id"])
                self.spool.update_job(
                    job_id,
                    status="running",
                    stage="inference",
                    progress=None,
                    progress_kind="indeterminate",
                )
                thread = threading.Thread(
                    target=self._execute_job,
                    args=(state,),
                    name=f"lumen-link-job-{job_id[-8:]}",
                    daemon=True,
                )
                self.job_threads[job_id] = thread
                thread.start()

    def _execute_job(self, state: dict[str, Any]) -> None:
        job_id = str(state["job_id"])
        try:
            def publish_progress(resources: dict[str, Any]) -> None:
                stage = str(resources.get("stage") or "inference")
                self.spool.update_job(
                    job_id,
                    status="running",
                    stage=stage,
                    progress=None,
                    progress_kind="indeterminate",
                    resources=resources,
                )

            result = self.executor.execute(
                state,
                progress_callback=publish_progress,
            )
            self.spool.save_result(job_id, result)
            self.spool.update_job(job_id, status="complete", stage="complete", progress=1.0, resources=result.get("resources", {}))
        except Exception as error:
            self.spool.update_job(job_id, status="failed", stage="failed", error=str(error))

    def health(self) -> dict[str, Any]:
        jobs = self.spool.list_jobs(limit=100)
        counts = {state: sum(job.get("status") == state for job in jobs) for state in ("queued", "running", "complete", "failed", "canceled")}
        return {
            "schema": LINK_SCHEMA,
            "service": "Lumen Link compute node",
            "authenticated": True,
            "uptime_s": max(0.0, time.time() - self.started_at),
            "capabilities": {
                **self.executor.capabilities(),
                "maximum_parallel_jobs": self.maximum_parallel_jobs,
            },
            "node": _node_resources(),
            "queue": counts,
            "jobs": jobs[:50],
            "active_slots": len(self.job_threads),
            "maximum_parallel_jobs": self.maximum_parallel_jobs,
        }

    def dashboard_status(self) -> dict[str, Any]:
        """Read-only operational telemetry with no recording identity."""

        health = self.health()
        jobs = []
        for item in health["jobs"][:24]:
            resources = item.get("resources") or {}
            jobs.append({
                "job_type": item.get("job_type"),
                "status": item.get("status"),
                "stage": item.get("stage"),
                "progress": item.get("progress"),
                "created_unix_ms": item.get("created_unix_ms"),
                "updated_unix_ms": item.get("updated_unix_ms"),
                "elapsed_s": resources.get("elapsed_s"),
                "peak_rss_bytes": resources.get("peak_rss_bytes"),
                "threads": resources.get("threads"),
            })
        contact_age = (
            None
            if self.last_coordinator_contact_at is None
            else max(0.0, time.time() - self.last_coordinator_contact_at)
        )
        return {
            "service": "Lumen Link",
            "node": health["node"],
            "uptime_s": health["uptime_s"],
            "connection": {
                "state": (
                    "connected"
                    if contact_age is not None and contact_age <= 10.0
                    else "waiting"
                ),
                "last_contact_age_s": contact_age,
            },
            "queue": health["queue"],
            "active_slots": health["active_slots"],
            "maximum_parallel_jobs": health["maximum_parallel_jobs"],
            "jobs": jobs,
        }


_LINK_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lumen Link · Threadripper</title><style>
:root{color-scheme:dark;--bg:#071012;--panel:#101d20;--line:#30484a;--cyan:#69cfc2;--gold:#d7a85e;--red:#d87870;--text:#bdd1cd;--muted:#718681}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#173236 0,transparent 38%),var(--bg);color:var(--text);font:15px system-ui,sans-serif}header{padding:28px 4vw 18px;border-bottom:1px solid var(--line)}h1{margin:4px 0;font-size:30px;letter-spacing:.04em}header span,.muted{color:var(--muted)}#heartbeat.connected{color:var(--cyan)}#heartbeat.waiting{color:var(--gold)}main{padding:22px 4vw;display:grid;gap:16px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.card,.jobs{background:rgba(16,29,32,.9);border:1px solid var(--line);border-radius:5px;padding:16px;box-shadow:0 12px 35px #0005}.card b{display:block;color:var(--cyan);font:27px ui-monospace,monospace;margin-top:7px}.slots{height:10px;background:#071012;border:1px solid var(--line);margin-top:12px}.slots i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--gold));transition:width .4s}.job{display:grid;grid-template-columns:170px 120px 1fr 130px;gap:12px;padding:11px 0;border-top:1px solid #253a3c;align-items:center}.job:first-child{border:0}.state{color:var(--gold);font-family:ui-monospace,monospace}.bar{height:7px;background:#071012;overflow:hidden}.bar i{display:block;height:100%;background:var(--cyan)}.bar.indeterminate i{width:32%!important;animation:scan 1.2s ease-in-out infinite}@keyframes scan{0%{transform:translateX(-110%)}100%{transform:translateX(325%)}}@media(max-width:950px){.metrics{grid-template-columns:1fr 1fr}.job{grid-template-columns:1fr 1fr}.bar{grid-column:1/-1}}
</style></head><body><header><span>REMOTE COMPUTE NODE</span><h1>Lumen Link · Threadripper</h1><span id="heartbeat">Waiting for worker telemetry…</span></header><main><section class="metrics"><div class="card">Lumen contact<b id="contact">—</b></div><div class="card">Worker uptime<b id="uptime">—</b></div><div class="card">CPU utilization<b id="cpu">—</b></div><div class="card">Parallel slots<b id="slots">—</b><div class="slots"><i id="slotbar"></i></div></div><div class="card">Queued<b id="queued">—</b></div><div class="card">Completed<b id="complete">—</b></div><div class="card">Memory available<b id="memory">—</b></div></section><section class="jobs"><h2>Compute flow</h2><div id="jobs" class="muted">No work has arrived.</div></section></main><script>
const fmt=s=>s==null?'—':s<60?`${Math.round(s)}s`:s<3600?`${Math.floor(s/60)}m ${Math.round(s%60)}s`:`${Math.floor(s/3600)}h ${Math.round(s%3600/60)}m`;const gib=n=>n==null?'—':`${(n/1073741824).toFixed(1)} GiB`;
async function tick(){try{const d=await fetch('/dashboard/status',{cache:'no-store'}).then(r=>r.json());const linked=d.connection.state==='connected';heartbeat.className=d.connection.state;heartbeat.textContent=`WORKER ONLINE · LUMEN ${linked?'CONNECTED':'WAITING'} · ${new Date().toLocaleTimeString()}`;contact.textContent=d.connection.last_contact_age_s==null?'never':fmt(d.connection.last_contact_age_s);uptime.textContent=fmt(d.uptime_s);cpu.textContent=`${Math.round(d.node.cpu_usage_percent==null?100*(d.node.load_1m||0)/Math.max(1,d.node.cpu_logical||1):d.node.cpu_usage_percent)}%`;slots.textContent=`${d.active_slots} / ${d.maximum_parallel_jobs}`;slotbar.style.width=`${100*d.active_slots/Math.max(1,d.maximum_parallel_jobs)}%`;queued.textContent=d.queue.queued||0;complete.textContent=d.queue.complete||0;memory.textContent=gib(d.node.memory_available_bytes);jobs.innerHTML=d.jobs.length?d.jobs.map(j=>`<div class="job"><b>${j.job_type||'compute'}</b><span class="state">${j.stage||j.status}</span><div class="bar ${j.progress==null&&j.status==='running'?'indeterminate':''}"><i style="width:${j.progress==null?(j.status==='complete'?100:12):100*j.progress}%"></i></div><span>${fmt(j.elapsed_s)} · ${gib(j.peak_rss_bytes)}</span></div>`).join(''):'No work has arrived.'}catch(e){heartbeat.className='waiting';heartbeat.textContent=`WORKER OFFLINE · ${e.message}`}}tick();setInterval(tick,1000);
</script></body></html>'''


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
            if self.path in {"/", "/dashboard"}:
                body = _LINK_DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/dashboard/status":
                body = _json_bytes(self.server.runtime.dashboard_status())
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._authenticate(_sha256_bytes(b""))
            if self.path == "/v1/health":
                self._json(HTTPStatus.OK, self.server.runtime.health())
                return
            if self.path.startswith("/v1/objects/"):
                digest = self.server.spool.validate_digest(
                    self.path.rsplit("/", 1)[-1]
                )
                source = self.server.spool.object_path(digest)
                if not source.is_file():
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "immutable object is not available"},
                    )
                    return
                total = source.stat().st_size
                offset = 0
                range_header = str(self.headers.get("Range") or "")
                if range_header:
                    if not range_header.startswith("bytes=") or not range_header.endswith("-"):
                        raise ValueError("only resumable bytes=N- ranges are supported")
                    offset = int(range_header[6:-1])
                    if offset < 0 or offset >= total:
                        raise ValueError("artifact range offset is invalid")
                length = total - offset
                self.send_response(
                    HTTPStatus.PARTIAL_CONTENT if offset else HTTPStatus.OK
                )
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                if offset:
                    self.send_header(
                        "Content-Range", f"bytes {offset}-{total - 1}/{total}"
                    )
                self.send_header("Cache-Control", "private, immutable")
                for key, value in self.server.authenticator.headers(
                    "RESPONSE", self.path, digest
                ).items():
                    self.send_header(key, value)
                self.end_headers()
                with source.open("rb") as input_file:
                    input_file.seek(offset)
                    shutil.copyfileobj(input_file, self.wfile, 1024 * 1024)
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
                job_type = str(value.get("job_type") or "")
                fields = JOB_CONTRACT_FIELDS.get(job_type)
                expected = (capabilities.get("job_contracts") or {}).get(
                    job_type
                )
                if fields is None or not isinstance(expected, dict):
                    raise ValueError("compute node does not support this job")
                if job_type not in capabilities.get(
                    "supported_job_types", []
                ):
                    raise ValueError(
                        (capabilities.get("gated_job_types") or {}).get(
                            job_type, "compute job is gated"
                        )
                    )
                validate_contract = getattr(
                    self.server.runtime.executor,
                    "validate_contract",
                    None,
                )
                if validate_contract is not None:
                    validate_contract(job_type, contract)
                    capabilities = (
                        self.server.runtime.executor.capabilities()
                    )
                    expected = (
                        capabilities.get("job_contracts") or {}
                    ).get(job_type)
                for name in fields:
                    if contract.get(name) != expected.get(name):
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
        self.server.runtime.note_coordinator_contact()

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

    def upload(
        self,
        path: Path,
        digest: str | None = None,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        chunk_guard: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        with chunk_guard() if chunk_guard else nullcontext():
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
            if progress_callback is not None:
                progress_callback(total, total)
            return {"complete": True, "bytes": total, "resumed_from": offset}
        with path.open("rb") as source:
            source.seek(offset)
            while offset < total:
                with chunk_guard() if chunk_guard else nullcontext():
                    body = source.read(
                        min(DEFAULT_CHUNK_BYTES, total - offset)
                    )
                    end = offset + len(body) - 1
                    result = self.request("PUT", f"/v1/objects/{digest}", body, {"Content-Range": f"bytes {offset}-{end}/{total}"})
                offset = int(result["bytes"])
                if progress_callback is not None:
                    progress_callback(offset, total)
        return {"complete": True, "bytes": total, "resumed_from": remote["offset"]}

    def download(self, digest: str, byte_count: int, target: Path) -> Path:
        """Authenticated atomic download of a content-addressed result."""
        digest = LinkSpool.validate_digest(digest)
        path = f"/v1/objects/{digest}"
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > int(byte_count):
            partial.unlink()
            offset = 0
        if offset == int(byte_count):
            if _hash_file(partial) != digest:
                partial.unlink()
                offset = 0
            else:
                partial.replace(target)
                return target
        headers = self.authenticator.headers(
            "GET", path, _sha256_bytes(b"")
        )
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(self.endpoint + path, method="GET", headers=headers)
        try:
            with urlopen(request, timeout=max(self.timeout_s, 60.0)) as response:
                declared = int(response.headers.get("Content-Length", "-1"))
                if declared != int(byte_count) - offset:
                    raise LinkProtocolError(
                        "remote artifact byte length does not match result"
                    )
                if offset:
                    expected_range = (
                        f"bytes {offset}-{int(byte_count) - 1}/{int(byte_count)}"
                    )
                    if (
                        response.status != HTTPStatus.PARTIAL_CONTENT
                        or response.headers.get("Content-Range")
                        != expected_range
                    ):
                        partial.unlink(missing_ok=True)
                        raise LinkProtocolError(
                            "remote artifact did not honor resume range"
                        )
                response_headers = response.headers
                written = offset
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        written += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if written != int(byte_count) or _hash_file(partial) != digest:
                    raise LinkProtocolError(
                        "downloaded artifact checksum or length does not match"
                    )
                self.authenticator.verify(
                    "RESPONSE", path, response_headers, digest
                )
        except Exception:
            # Keep a valid-length prefix for a later authenticated Range
            # request. Final publication still requires full SHA-256 match.
            raise
        partial.replace(target)
        return target

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
        on_import: Callable[[dict[str, Any], dict[str, Any]], None]
        | None = None,
    ) -> None:
        self.store = store
        self.research_root = research_root.resolve()
        self.state_root = state_root.resolve()
        self.config_path = config_path.resolve()
        self.state_path = self.state_root / "coordinator.json"
        self.contract_cache_path = self.state_root / "asset-contract.json"
        self.lock = threading.RLock()
        self.standby_work_lock = threading.Lock()
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
        self.on_import = on_import
        self.configuration = LinkConfiguration.load(self.config_path)
        self._local_contract_cache: dict[str, dict[str, Any]] = {}
        self._local_contract_signatures: dict[str, dict[str, Any]] = {}
        self._status_cache: dict[str, Any] | None = None
        self._status_cached_at = 0.0
        self._job_snapshot: list[dict[str, Any]] = []
        self._job_snapshot_at = 0.0
        self._prepared_manifests: dict[str, dict[str, Any]] = {}
        self._prepared_sources: dict[
            str, list[tuple[dict[str, Any], Path]]
        ] = {}
        # A prefilled remote job remains locally queued until its result is
        # claimed and imported. Track that intermediate state explicitly so
        # completed remote work is not resubmitted and mistaken for new slot
        # occupancy on every coordinator cycle.
        self._submitted_local_job_ids: set[str] = set()
        self._queue_summary = {"queued": 0, "bytes_pending": 0}
        self.recent_imports_path = self.state_root / "recent-imports.json"
        self.recent_imports: deque[dict[str, Any]] = deque(maxlen=20)
        try:
            recent = _read_json(self.recent_imports_path).get("imports") or []
            self.recent_imports.extend(
                item for item in recent if isinstance(item, dict)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

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

    @contextmanager
    def live_transition_guard(self):
        """Let engine start wait until Link reaches an I/O checkpoint."""
        acquired = self.standby_work_lock.acquire(timeout=30.0)
        if not acquired:
            raise RuntimeError(
                "Lumen Link did not reach a standby I/O checkpoint"
            )
        try:
            yield
        finally:
            self.standby_work_lock.release()

    @contextmanager
    def _standby_guard(self):
        with self.standby_work_lock:
            if not self.can_import():
                raise LinkStandbyRequired(
                    "Lumen Link heavy work is parked until standby"
                )
            yield

    @contextmanager
    def standby_task_guard(self):
        """Serialize application-side offline work with the Live transition."""
        with self._standby_guard():
            yield

    def control(self, action: str) -> dict[str, Any]:
        action = str(action).casefold()
        if action == "test":
            if not self.can_import():
                raise RuntimeError(
                    "Stop Live before verifying multi-gigabyte model assets"
                )
            self._poll_health()
            for job_type in SUPPORTED_JOB_TYPES:
                self._remote_is_compatible(
                    job_type, allow_contract_scan=True
                )
        elif action in {"enable", "pause", "resume", "disable"}:
            if action == "enable":
                if not self.can_import():
                    raise RuntimeError(
                        "Stop Live before enabling the offline compute link"
                    )
                # Enable is also a fresh authenticated compatibility check. A
                # button press must not claim success while an old worker
                # revision gates every compute contract.
                self._poll_health()
                compatible = {
                    job_type: self._remote_is_compatible(
                        job_type, allow_contract_scan=True
                    )
                    for job_type in SUPPORTED_JOB_TYPES
                }
                if not any(compatible.values()):
                    remote_contracts = (
                        ((self.remote_status or {}).get("capabilities") or {})
                        .get("job_contracts") or {}
                    )
                    remote_revisions = {
                        str((remote_contracts.get(job_type) or {}).get("code_revision") or "")
                        for job_type in SUPPORTED_JOB_TYPES
                    }
                    remote_revision = next(
                        (value for value in remote_revisions if value), "unknown"
                    )
                    raise RuntimeError(
                        "Threadripper authenticated, but its compute contract "
                        f"is incompatible (worker code {remote_revision[:7]}). "
                        "Update and restart the worker, then Test Connection again."
                    )
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

    def _job_threads(self, job_type: str) -> int:
        """Choose the remote worker's current CPU ceiling for a manifest."""

        capabilities = (self.remote_status or {}).get("capabilities") or {}
        maximum = int(capabilities.get("max_threads") or 24)
        parallel = max(
            1, int(capabilities.get("maximum_parallel_jobs") or 1)
        )
        if job_type in {EDMFORMER_JOB, SONGFORMER_JOB}:
            # Leave each concurrently running teacher a fair share of the
            # machine. Student training runs alone and may use every thread.
            maximum = max(1, maximum // parallel)
        if job_type == EDMFORMER_JOB:
            # The validated EDMFormer runner has its own bounded attention
            # implementation and currently accepts at most eight threads.
            return max(1, min(maximum, EDMFORMER_MAX_THREADS))
        return max(1, maximum)

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
        while not self.stop_event.wait(COORDINATOR_POLL_SECONDS):
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
                        for job_type in SUPPORTED_JOB_TYPES:
                            self._remote_is_compatible(
                                job_type, allow_contract_scan=True
                            )
                self.route_queued_jobs(refresh=False)
                self._prefill_remote_queue()
                self._advance()
            except LinkStandbyRequired:
                continue
            except Exception as error:
                # Health polling owns connection state. A malformed or stale
                # individual job must remain a job error instead of making a
                # healthy authenticated Link flash between Ready and Error.
                self._event("error", f"Link job cycle: {error}")

    def _refresh_job_snapshot(
        self, *, force: bool = False, precompute_students: bool = False
    ) -> list[dict[str, Any]]:
        if not self.can_import():
            return list(self._job_snapshot)
        if (
            not force
            and self._job_snapshot
            and time.monotonic() - self._job_snapshot_at < JOB_SNAPSHOT_SECONDS
        ):
            return list(self._job_snapshot)
        with self._standby_guard():
            jobs = self.store.list_analysis_jobs(limit=10_000)
            self._job_snapshot = jobs
            self._job_snapshot_at = time.monotonic()
            if precompute_students:
                for job in jobs:
                    if (
                        job.get("status") != "queued"
                        or job.get("job_type") != STUDENT_TRAIN_JOB
                    ):
                        continue
                    job_id = str(job["id"])
                    if job_id in self._prepared_manifests:
                        continue
                    try:
                        manifest = self._manifest(job, jobs=jobs)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        message = f"Lumen Link preparation failed: {error}"
                        self.store.update_analysis_job(
                            job_id,
                            status="failed",
                            error=message,
                        )
                        job["status"] = "failed"
                        job["error"] = message
                        self._event("error", f"{job_id}: {message}")
                        continue
                    self._prepared_manifests[job_id] = manifest
                    self._prepared_sources[job_id] = self._object_sources(
                        job, manifest, jobs=jobs
                    )
            seen_objects: set[str] = set()
            pending_bytes = 0
            queued_count = 0
            for job in jobs:
                if (
                    job.get("status") != "queued"
                    or job.get("job_type") not in SUPPORTED_JOB_TYPES
                ):
                    continue
                queued_count += 1
                manifest = self._prepared_manifests.get(str(job["id"]))
                if manifest is not None:
                    objects = manifest.get("objects") or []
                else:
                    payload = job.get("payload") or {}
                    audio_path = Path(str(payload.get("audio_path") or ""))
                    objects = [
                        {
                            "sha256": payload.get("content_sha256")
                            or str(audio_path),
                            "bytes": (
                                audio_path.stat().st_size
                                if audio_path.is_file()
                                else 0
                            ),
                        }
                    ]
                for item in objects:
                    identity = str(item.get("sha256") or "")
                    if not identity or identity in seen_objects:
                        continue
                    seen_objects.add(identity)
                    pending_bytes += int(item.get("bytes") or 0)
            self._queue_summary = {
                "queued": queued_count,
                "bytes_pending": pending_bytes,
            }
        self._invalidate_status()
        return list(jobs)

    def route_queued_jobs(self, *, refresh: bool = True) -> int:
        """Maintain a standby buffer after authenticated contract agreement."""
        if (
            not self.configuration.enabled
            or self.configuration.paused
            or not self.can_import()
        ):
            return 0
        jobs = self._refresh_job_snapshot(
            force=refresh, precompute_students=True
        )
        eligible = sorted(
            (
                job
                for job in jobs
                if job.get("status") == "queued"
                and job.get("job_type") in SUPPORTED_JOB_TYPES
                and str(
                    (job.get("payload") or {}).get("execution_target")
                    or "automatic"
                )
                == "automatic"
                and self._remote_is_compatible(str(job["job_type"]))
            ),
            key=lambda item: (
                -int(item.get("priority") or 0),
                int(item.get("created_unix_ms") or 0),
            ),
        )
        maximum = max(
            1,
            int(
                ((self.remote_status or {}).get("capabilities") or {}).get(
                    "maximum_parallel_jobs", 1
                )
            ),
        )
        already_routed = sum(
            job.get("status") == "queued"
            and (job.get("payload") or {}).get("execution_target")
            == "threadripper"
            and str(job.get("id") or "")
            not in self._submitted_local_job_ids
            for job in jobs
        )
        routed = 0
        for job in eligible[: max(0, maximum - already_routed)]:
            target = str(
                (job.get("payload") or {}).get("execution_target")
                or "automatic"
            )
            if (
                target == "automatic"
                and self.store.set_analysis_job_execution_target(
                    str(job["id"]), execution_target="threadripper"
                )
            ):
                job.setdefault("payload", {})[
                    "execution_target"
                ] = "threadripper"
                self._invalidate_status()
                routed += 1
        return routed

    def _prefill_remote_queue(self) -> int:
        """Keep the Threadripper's parallel slots supplied ahead of import."""

        if not self.can_import() or not self.remote_status:
            return 0
        capabilities = (self.remote_status.get("capabilities") or {})
        maximum = max(1, int(capabilities.get("maximum_parallel_jobs") or 1))
        remote_jobs = self.remote_status.get("jobs") or []
        occupied_ids = {
            str(job.get("job_id") or "")
            for job in remote_jobs
            if job.get("status") in {"queued", "running"}
        }
        capacity = maximum - len(occupied_ids)
        if capacity <= 0:
            return 0
        active_id = str(((self.active or {}).get("job") or {}).get("id") or "")
        candidates = [
            job for job in self._job_snapshot
            if job.get("status") == "queued"
            and (job.get("payload") or {}).get("execution_target") == "threadripper"
            and str(job.get("id") or "") not in occupied_ids
            and str(job.get("id") or "") != active_id
            and str(job.get("id") or "")
            not in self._submitted_local_job_ids
            and self._remote_is_compatible(str(job.get("job_type") or ""))
        ]
        candidates.sort(key=lambda item: (-int(item.get("priority") or 0), int(item.get("created_unix_ms") or 0)))
        if candidates and candidates[0].get("job_type") == STUDENT_TRAIN_JOB:
            if occupied_ids:
                return 0
            candidates = candidates[:1]
        else:
            candidates = [job for job in candidates if job.get("job_type") != STUDENT_TRAIN_JOB]
        submitted = 0
        client = self._client()
        for job in candidates:
            if submitted >= capacity:
                break
            manifest = self._prepared_manifests.get(str(job["id"])) or self._manifest(job, jobs=self._job_snapshot)
            sources = self._prepared_sources.get(str(job["id"])) or self._object_sources(job, manifest, jobs=self._job_snapshot)
            # upload() acquires the standby guard once per chunk so Live can
            # preempt a long transfer. Do not nest that non-reentrant guard.
            for item, source in sources:
                client.upload(
                    source,
                    str(item["sha256"]),
                    chunk_guard=self._standby_guard,
                )
            with self._standby_guard():
                manifest, remote = self._submit_manifest(
                    client, job, manifest
                )
            self._prepared_manifests[str(job["id"])] = manifest
            self._prepared_sources[str(job["id"])] = sources
            self._submitted_local_job_ids.add(str(job["id"]))
            if remote.get("status") in {"queued", "running"}:
                submitted += 1
                self._event(
                    "prefill",
                    f"submitted {job['id']} to a parallel slot",
                )
            elif remote.get("status") == "complete":
                self._event(
                    "prefill",
                    f"found completed remote result for {job['id']}",
                )
        return submitted

    def _submit_manifest(
        self,
        client: LinkClient,
        job: dict[str, Any],
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Submit immutably, re-keying a changed manifest after an upgrade."""

        try:
            state = client.submit(manifest)
            return manifest, state
        except LinkProtocolError as error:
            if "job ID already has a different manifest" not in str(error):
                raise
        canonical_id = str(job["id"])
        fingerprint_source = {**manifest, "job_id": canonical_id}
        fingerprint = _sha256_bytes(_json_bytes(fingerprint_source))[:16]
        retry_manifest = {
            **manifest,
            "job_id": f"{canonical_id}.manifest-{fingerprint}",
        }
        state = client.submit(retry_manifest)
        self._event(
            "job",
            f"re-keyed {canonical_id} after an immutable manifest conflict",
        )
        return retry_manifest, state

    def _restore_queued_jobs(self) -> int:
        if not self.can_import():
            return 0
        restored = 0
        self._submitted_local_job_ids.clear()
        for job in self._refresh_job_snapshot(force=True):
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
        self._job_snapshot_at = 0.0
        return restored

    def _local_contract(self, job_type: str) -> dict[str, Any]:
        # Accept the original test/upgrade cache shape as the EDM contract.
        if (
            "code_revision" in self._local_contract_cache
            and EDMFORMER_JOB not in self._local_contract_cache
        ):
            legacy = dict(self._local_contract_cache)  # type: ignore[arg-type]
            self._local_contract_cache = {EDMFORMER_JOB: legacy}
        project_root = Path(__file__).resolve().parents[2]
        signature = _job_asset_signature(
            job_type, self.research_root, project_root
        )
        prior_signature = self._local_contract_signatures.get(job_type)
        if prior_signature is not None and prior_signature != signature:
            self._local_contract_cache.pop(job_type, None)
        if job_type not in self._local_contract_cache:
            cached: dict[str, Any] | None = None
            cache_path = self.contract_cache_path.with_name(
                f"asset-contract-{job_type.replace('.', '-')}.json"
            )
            try:
                receipt = _read_json(cache_path)
                if receipt.get("signature") == signature and isinstance(
                    receipt.get("contract"), dict
                ):
                    cached = dict(receipt["contract"])
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            contract = cached or _job_asset_contract(
                job_type, self.research_root, project_root
            )
            self._local_contract_cache[job_type] = contract
            self._local_contract_signatures[job_type] = signature
            if cached is None:
                _atomic_json(
                    cache_path,
                    {
                        "schema": "lumen.link.asset-contract.v1",
                        "signature": signature,
                        "job_type": job_type,
                        "contract": contract,
                        "created_unix_ms": int(time.time() * 1000),
                    },
                )
        return dict(self._local_contract_cache[job_type])

    def _remote_is_compatible(
        self,
        job_type: str = EDMFORMER_JOB,
        *,
        allow_contract_scan: bool = False,
    ) -> bool:
        if self.last_error or not self.remote_status:
            return False
        if not self.remote_status.get("authenticated"):
            return False
        capabilities = self.remote_status.get("capabilities") or {}
        if job_type not in capabilities.get("supported_job_types", []):
            return False
        local = self._local_contract_cache.get(job_type)
        if local is None:
            # Compatibility for pre-job-contract tests/nodes is EDM-only.
            if (
                job_type == EDMFORMER_JOB
                and "code_revision" in self._local_contract_cache
            ):
                local = dict(self._local_contract_cache)  # type: ignore[arg-type]
            if not allow_contract_scan:
                if local is None:
                    return False
            if local is None:
                local = self._local_contract(job_type)
        remote_contract = (capabilities.get("job_contracts") or {}).get(
            job_type
        )
        if not isinstance(remote_contract, dict):
            remote_contract = capabilities if job_type == EDMFORMER_JOB else {}
        return all(
            local.get(name) == remote_contract.get(name)
            for name in JOB_CONTRACT_FIELDS[job_type]
        )

    def ready_for_offload(self, job_type: str = EDMFORMER_JOB) -> bool:
        return bool(
            self.configuration.enabled
            and not self.configuration.paused
            and self._remote_is_compatible(job_type)
        )

    def _reconcile_active(self) -> None:
        if self.active is None:
            return
        job_id = str((self.active.get("job") or {}).get("id") or "")
        jobs = {
            str(job["id"]): job
            for job in self._refresh_job_snapshot(force=True)
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
            if not self.can_import():
                return
            with self._standby_guard():
                capabilities = (self.remote_status or {}).get("capabilities") or {}
                candidate = next(
                    (
                        item
                        for item in self._job_snapshot
                        if item.get("status") == "queued"
                        and item.get("job_type") in SUPPORTED_JOB_TYPES
                        and (item.get("payload") or {}).get(
                            "execution_target"
                        )
                        == "threadripper"
                        and self._remote_is_compatible(
                            str(item["job_type"])
                        )
                    ),
                    None,
                )
                if candidate is None:
                    return
                manifest = self._prepared_manifests.get(
                    str(candidate["id"])
                ) or self._manifest(candidate, jobs=self._job_snapshot)
                remote_contract = (
                    capabilities.get("job_contracts") or {}
                ).get(candidate["job_type"])
                if not isinstance(remote_contract, dict):
                    remote_contract = (
                        capabilities
                        if candidate["job_type"] == EDMFORMER_JOB
                        else {}
                    )
                for name in JOB_CONTRACT_FIELDS[candidate["job_type"]]:
                    if manifest["contract"].get(name) != remote_contract.get(name):
                        raise RuntimeError(
                            f"Threadripper {name} does not match Lumen"
                        )
                job = self.store.claim_analysis_job_by_id(
                    str(candidate["id"]),
                    worker_id=f"lumen-link:{platform.node()}",
                    worker_pid=os.getpid(),
                )
                if job is None:
                    self._job_snapshot_at = 0.0
                    return
                self.active = {"job": job, "manifest": manifest, "stage": "transfer", "progress": 0.0}
                self._persist()
                self._event("job", f"claimed {job['id']}")
        job = self.active["job"]
        manifest = self.active["manifest"]
        if self.active["stage"] == "transfer":
            if not self.can_import():
                return
            transferred = 0
            transfer_total = sum(
                int(item.get("bytes") or 0)
                for item in manifest["objects"]
            )
            sources = self._prepared_sources.get(str(job["id"]))
            if sources is None:
                with self._standby_guard():
                    sources = self._object_sources(
                        job, manifest, jobs=self._job_snapshot
                    )
                    self._prepared_sources[str(job["id"])] = sources
            for item, source in sources:
                base = transferred

                def transfer_progress(current: int, total: int) -> None:
                    del total
                    self.active["transferred_bytes"] = base + current
                    self.active["progress"] = (
                        (base + current) / transfer_total
                        if transfer_total
                        else 1.0
                    )
                    # This callback runs just after the guarded network chunk.
                    # Keep its high-frequency progress in memory so a Live
                    # transition cannot inherit one more mechanical-disk write.
                    # The remote content-addressed offset is the durable resume
                    # point; coordinator state is persisted at the next stage.
                    self._invalidate_status()

                transfer = client.upload(
                    source,
                    str(item["sha256"]),
                    progress_callback=transfer_progress,
                    chunk_guard=self._standby_guard,
                )
                transferred += int(transfer.get("bytes") or 0)
                self.active["transferred_bytes"] = transferred
            with self._standby_guard():
                manifest, _remote_state = self._submit_manifest(
                    client, job, manifest
                )
                self._submitted_local_job_ids.add(str(job["id"]))
                self.active["manifest"] = manifest
                self.active.update(stage="remote", progress=0.05)
                self._persist()
        # Older persisted coordinator states predate an explicit transport ID
        # in the saved manifest; their canonical job ID remains the fallback.
        remote_job_id = str(manifest.get("job_id") or job["id"])
        remote = client.request("GET", f"/v1/jobs/{remote_job_id}")
        self.active["remote"] = remote
        self.active["stage"] = str(remote.get("stage") or remote.get("status"))
        remote_progress = remote.get("progress")
        self.active["progress"] = (
            float(remote_progress)
            if remote_progress is not None
            else None
        )
        if self.can_import():
            with self._standby_guard():
                self.store.heartbeat_analysis_job(job["id"], worker_id=str(job["worker_id"]), progress={"execution_target": "threadripper", "stage": self.active["stage"], "progress": self.active["progress"]})
                self._persist()
        else:
            self._invalidate_status()
        if remote.get("status") == "failed":
            if not self.can_import():
                self.active["stage"] = "remote_failed_awaiting_standby"
                self._invalidate_status()
                return
            with self._standby_guard():
                self.store.update_analysis_job(job["id"], status="failed", error=f"Threadripper: {remote.get('error')}")
                self._event("error", f"remote job {job['id']} failed")
                self.active = None
                self._persist()
        elif remote.get("status") == "complete":
            if not self.can_import():
                self.active["stage"] = "awaiting_local_import"
                self.active["progress"] = 1.0
                self._invalidate_status()
                return
            with self._standby_guard():
                self.active["stage"] = "returning"
                self.active["progress"] = None
                self._persist()
                result = client.request(
                    "GET", f"/v1/jobs/{remote_job_id}/result"
                )
                self.active["stage"] = "importing"
                self._persist()
                if job["job_type"] == EDMFORMER_JOB:
                    imported = self._import_teacher(
                        job, manifest, result, teacher="EDMFormer"
                    )
                elif job["job_type"] == SONGFORMER_JOB:
                    imported = self._import_teacher(
                        job, manifest, result, teacher="SongFormer"
                    )
                elif job["job_type"] == STUDENT_TRAIN_JOB:
                    imported = self._import_student(
                        client, job, manifest, result
                    )
                else:
                    raise LinkProtocolError("unsupported completed remote job")
                self.store.update_analysis_job(job["id"], status="complete", result=imported)
                self._record_import(job, manifest, imported)
                self._event("complete", f"imported {job['id']}")
                self.active = None
                self._prepared_manifests.pop(str(job["id"]), None)
                self._prepared_sources.pop(str(job["id"]), None)
                self._submitted_local_job_ids.discard(str(job["id"]))
                self._job_snapshot_at = 0.0
                self._persist()
            if self.on_import is not None:
                try:
                    self.on_import(dict(job), dict(imported))
                except Exception as error:
                    self._event(
                        "error",
                        f"post-import notification failed: {error}",
                    )

    def _manifest(
        self,
        job: dict[str, Any],
        *,
        jobs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = job["payload"]
        if job["job_type"] == STUDENT_TRAIN_JOB:
            return self._student_manifest(job, jobs=jobs)
        audio_path = Path(str(payload["audio_path"])).resolve()
        digest = _audio_object_digest(
            audio_path, str(payload.get("content_sha256") or "") or None
        )
        contract = self._local_contract(str(job["job_type"]))
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
            "resources": {"threads": self._job_threads(str(job["job_type"]))},
            # This must survive transfer retry and both-machine restart so
            # submitting the same canonical job remains byte-for-byte
            # idempotent.
            "created_unix_ms": int(job["created_unix_ms"]),
        }

    def _student_manifest(
        self,
        job: dict[str, Any],
        *,
        jobs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = dict(job["payload"])
        examples_path = Path(str(payload["examples_path"])).resolve()
        examples_digest = str(
            payload.get("examples_sha256") or _hash_file(examples_path)
        )
        if _hash_file(examples_path) != examples_digest:
            raise ValueError("student examples changed after queueing")
        recording_ids: set[str] = set()
        with examples_path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    recording_id = str(row.get("recording_id") or "")
                    if recording_id:
                        recording_ids.add(recording_id)
        audio_by_recording: dict[str, tuple[Path, str]] = {}
        candidates = jobs if jobs is not None else self._job_snapshot
        if not candidates:
            if not self.can_import():
                raise LinkStandbyRequired(
                    "student manifest requires standby"
                )
            candidates = self.store.list_analysis_jobs(limit=10_000)
        for candidate in candidates:
            candidate_payload = candidate.get("payload") or {}
            recording_id = str(candidate_payload.get("recording_id") or "")
            raw_path = str(candidate_payload.get("audio_path") or "")
            if recording_id not in recording_ids or not raw_path:
                continue
            if recording_id in audio_by_recording:
                continue
            path = Path(raw_path).resolve()
            if not path.is_file():
                continue
            digest = _audio_object_digest(
                path,
                str(candidate_payload.get("content_sha256") or "") or None,
            )
            audio_by_recording[recording_id] = (path, digest)
        missing = recording_ids - set(audio_by_recording)
        if missing:
            raise ValueError(
                "student offload is missing coherent WAVs for: "
                + ", ".join(sorted(missing)[:10])
            )
        objects = [
            {
                "role": "student_examples",
                "sha256": examples_digest,
                "bytes": examples_path.stat().st_size,
                "format": "jsonl",
            }
        ]
        objects.extend(
            {
                "role": "recording_audio",
                "recording_id": recording_id,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "format": "wav-pcm",
            }
            for recording_id, (path, digest) in sorted(
                audio_by_recording.items()
            )
        )
        allowed_training = {
            "epochs",
            "hidden_size",
            "source_scope",
            "teacher_run_ids",
            "source_files",
            "split_counts",
            "split_group_counts",
            "label_balance",
            "teacher_merge",
            "teacher_fusion_version",
            "operator_consensus",
            "operator_consensus_revision",
            "operator_timeline_corrections",
            "trainer_version",
            "applicable_axes",
        }
        training = {
            key: payload[key]
            for key in allowed_training
            if key in payload
        }
        return {
            "schema": MANIFEST_SCHEMA,
            "job_id": job["id"],
            "job_type": STUDENT_TRAIN_JOB,
            "identity": {"recording_ids": sorted(recording_ids)},
            "objects": objects,
            "contract": {
                **self._local_contract(STUDENT_TRAIN_JOB),
                "result_schema": RESULT_SCHEMA,
            },
            "training": training,
            "resources": {"threads": self._job_threads(STUDENT_TRAIN_JOB)},
            "created_unix_ms": int(job["created_unix_ms"]),
        }

    def _object_sources(
        self,
        job: dict[str, Any],
        manifest: dict[str, Any],
        *,
        jobs: list[dict[str, Any]] | None = None,
    ) -> list[tuple[dict[str, Any], Path]]:
        if job["job_type"] != STUDENT_TRAIN_JOB:
            return [
                (
                    manifest["objects"][0],
                    Path(str(job["payload"]["audio_path"])).resolve(),
                )
            ]
        examples = Path(str(job["payload"]["examples_path"])).resolve()
        audio_paths: dict[str, Path] = {}
        candidates = jobs if jobs is not None else self._job_snapshot
        if not candidates:
            if not self.can_import():
                raise LinkStandbyRequired("student sources require standby")
            candidates = self.store.list_analysis_jobs(limit=10_000)
        for candidate in candidates:
            payload = candidate.get("payload") or {}
            recording_id = str(payload.get("recording_id") or "")
            if recording_id and payload.get("audio_path"):
                if recording_id in audio_paths:
                    continue
                audio_paths[recording_id] = Path(
                    str(payload["audio_path"])
                ).resolve()
        result: list[tuple[dict[str, Any], Path]] = []
        for item in manifest["objects"]:
            if item["role"] == "student_examples":
                result.append((item, examples))
            else:
                result.append((item, audio_paths[str(item["recording_id"])]))
        return result

    def _import_edmformer(self, job: dict[str, Any], manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wrapper retained for receipt recovery tests."""
        return self._import_teacher(
            job, manifest, result, teacher="EDMFormer"
        )

    def _import_teacher(
        self,
        job: dict[str, Any],
        manifest: dict[str, Any],
        result: dict[str, Any],
        *,
        teacher: str,
    ) -> dict[str, Any]:
        # Keep health checks and pairing dependency-free. Heavy audio/model
        # modules are needed only for the standby-time canonical import.
        from lumen_engine.offline import (
            _mean_confidence,
            _normalize_teacher_segments,
            _validate_teacher_coverage,
            build_student_examples,
        )

        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("job_id")
            != str(manifest.get("job_id") or job["id"])
            or result.get("job_type") != job["job_type"]
        ):
            raise LinkProtocolError("remote result identity does not match local job")
        if result.get("manifest_sha256") != _sha256_bytes(_json_bytes(manifest)):
            raise LinkProtocolError("remote result manifest checksum does not match")
        audio = manifest["objects"][0]
        if result.get("input_sha256") != audio["sha256"]:
            raise LinkProtocolError("remote result audio checksum does not match")
        job_type = str(job["job_type"])
        fields = JOB_CONTRACT_FIELDS[job_type]
        teacher_revision_field = (
            "teacher_revision"
            if teacher == "EDMFormer"
            else "songformer_revision"
        )
        preprocessing_key = (
            "edmformer_preprocessing_version"
            if teacher == "EDMFormer"
            else "songformer_preprocessing_version"
        )
        preprocessing_version = (
            EDMFORMER_PREPROCESSING_VERSION
            if teacher == "EDMFormer"
            else SONGFORMER_PREPROCESSING_VERSION
        )
        provenance = (
            "edmformer_teacher"
            if teacher == "EDMFormer"
            else "songformer_teacher"
        )
        for name in fields:
            if result.get(name) != manifest["contract"].get(name):
                raise LinkProtocolError(
                    f"remote result {name} does not match its manifest"
                )
        payload = job["payload"]
        if int(result.get("duration_ms") or 0) != int(
            payload["duration_ms"]
        ):
            raise LinkProtocolError("remote result duration does not match input")
        teacher_revision = str(
            result.get(teacher_revision_field) or "unknown"
        )
        segments = _normalize_teacher_segments(
            result.get("segments") or [],
            source=teacher,
            source_version=teacher_revision,
        )
        _validate_teacher_coverage(
            segments,
            source=teacher,
            duration_ms=int(payload["duration_ms"]),
        )
        manifest_sha256 = str(result["manifest_sha256"])

        # Import is a transactionally recoverable receipt keyed by the
        # canonical analysis job and immutable manifest. A crash after saving
        # the timeline or completing the teacher run must resume that same run
        # instead of manufacturing duplicate authority.
        for existing in self.store.list_teacher_runs():
            if (
                str(existing.get("analysis_job_id") or "") != str(job["id"])
                or str(existing.get("teacher_name") or "").casefold()
                != teacher.casefold()
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
                    preprocessing_key: preprocessing_version,
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
        run_id = self.store.begin_teacher_run(teacher_name=teacher, teacher_version=teacher_revision, device="threadripper-cpu", preprocessing_version=preprocessing_version, recording_id=payload.get("recording_id"), capture_session_id=payload.get("capture_session_id"), analysis_job_id=str(job["id"]))
        try:
            timeline_id = self.store.save_structure_timeline(recording_id=payload.get("recording_id"), song_id=payload.get("song_id"), capture_session_id=payload.get("capture_session_id"), teacher_run_id=run_id, provenance=provenance, timeline_version=TEACHER_NORMALIZATION_VERSION, confidence=_mean_confidence(segments), segments=segments, metadata={"audio_path": str(payload["audio_path"]), "content_sha256": audio["sha256"], "execution_target": "threadripper", "link_manifest_sha256": manifest_sha256, "structure_supervision": payload.get("structure_supervision"), "cpu_context_window_seconds": (SONGFORMER_WINDOW_SECONDS if teacher == "SongFormer" else None)})
            examples = build_student_examples(self.store, research_root=self.research_root, recording_id=str(payload["recording_id"]), timeline_id=timeline_id)
            metrics = {"elapsed_s": result.get("resources", {}).get("elapsed_s"), "segments": len(segments), "timeline_id": timeline_id, "student_examples": examples, "execution_target": "threadripper", "link_manifest_sha256": manifest_sha256, "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION, preprocessing_key: preprocessing_version, "resources": result.get("resources", {})}
            self.store.finish_teacher_run(run_id, status="complete", metrics=metrics)
            return metrics
        except Exception as error:
            self.store.finish_teacher_run(run_id, status="failed", error=str(error))
            raise

    def _import_student(
        self,
        client: LinkClient,
        job: dict[str, Any],
        manifest: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify, independently gate, and atomically activate a candidate."""
        from lumen_engine.student import StreamingStructureStudent

        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("job_id")
            != str(manifest.get("job_id") or job["id"])
            or result.get("job_type") != STUDENT_TRAIN_JOB
        ):
            raise LinkProtocolError("remote student result identity is invalid")
        manifest_sha256 = _sha256_bytes(_json_bytes(manifest))
        if result.get("manifest_sha256") != manifest_sha256:
            raise LinkProtocolError("remote student manifest receipt is invalid")
        examples_object = next(
            item
            for item in manifest["objects"]
            if item["role"] == "student_examples"
        )
        if result.get("input_sha256") != examples_object["sha256"]:
            raise LinkProtocolError("remote student input receipt is invalid")
        for name in STUDENT_CONTRACT_FIELDS:
            if result.get(name) != manifest["contract"].get(name):
                raise LinkProtocolError(
                    f"remote student {name} does not match its manifest"
                )
        artifacts = result.get("artifacts") or {}
        if set(artifacts) != {
            "candidate_model",
            "evaluation",
            "prepared_examples",
        }:
            raise LinkProtocolError("remote student artifact set is incomplete")
        receipt_path = self.state_root / "imports" / (
            hashlib.sha256(str(job["id"]).encode()).hexdigest()
            + ".student.json"
        )
        try:
            receipt = _read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            receipt = {}
        output_path = Path(str(job["payload"]["output_path"])).resolve()
        candidate_meta = artifacts["candidate_model"]
        if (
            receipt.get("manifest_sha256") == manifest_sha256
            and receipt.get("candidate_sha256")
            == candidate_meta.get("sha256")
            and Path(str(receipt.get("candidate_model_path") or "")).is_file()
            and _hash_file(Path(str(receipt["candidate_model_path"])))
            == receipt.get("candidate_sha256")
            and Path(str(receipt.get("evaluation_path") or "")).is_file()
            and _hash_file(Path(str(receipt["evaluation_path"])))
            == receipt.get("local_evaluation_sha256")
            and (
                not receipt.get("activated")
                or (
                    output_path.is_file()
                    and _hash_file(output_path)
                    == candidate_meta.get("sha256")
                )
            )
        ):
            return {**receipt, "import_reused": True}

        import_root = self.state_root / "imports" / hashlib.sha256(
            manifest_sha256.encode()
        ).hexdigest()
        downloaded: dict[str, Path] = {}
        suffixes = {
            "candidate_model": ".npz",
            "evaluation": ".evaluation.json",
            "prepared_examples": ".prepared.jsonl",
        }
        for name, metadata in artifacts.items():
            if not isinstance(metadata, dict):
                raise LinkProtocolError("remote artifact descriptor is invalid")
            downloaded[name] = client.download(
                str(metadata.get("sha256") or ""),
                int(metadata.get("bytes") or 0),
                import_root / (name + suffixes[name]),
            )
        remote_report = _read_json(downloaded["evaluation"])
        if remote_report != result.get("report"):
            raise LinkProtocolError(
                "remote evaluation artifact does not match result envelope"
            )

        def load_rows(path: Path) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise LinkProtocolError(
                                "student examples contain a non-object row"
                            )
                        rows.append(value)
            return rows

        original_path = Path(
            str(job["payload"]["examples_path"])
        ).resolve()
        if _hash_file(original_path) != examples_object["sha256"]:
            raise LinkProtocolError(
                "local student examples changed during remote execution"
            )
        original_rows = load_rows(original_path)
        prepared_rows = load_rows(downloaded["prepared_examples"])
        if len(original_rows) != len(prepared_rows):
            raise LinkProtocolError("prepared student row count changed")
        for original, prepared in zip(original_rows, prepared_rows):
            original_target = {
                key: value
                for key, value in original.items()
                if key not in {"features", "feature_preprocessing_version"}
            }
            prepared_target = {
                key: value
                for key, value in prepared.items()
                if key not in {"features", "feature_preprocessing_version"}
            }
            if original_target != prepared_target:
                raise LinkProtocolError(
                    "remote feature preparation changed student supervision"
                )
            if prepared.get("feature_preprocessing_version") != result.get(
                "student_audio_feature_version"
            ):
                raise LinkProtocolError(
                    "prepared student feature contract is invalid"
                )
        model = StreamingStructureStudent.load(
            downloaded["candidate_model"]
        )
        local_gate = _student_gate_assessment(
            model, prepared_rows, job["payload"]
        )
        if sorted(model.approved_axes) != local_gate["approved_axes"]:
            raise LinkProtocolError(
                "candidate approved axes fail local held-out validation"
            )
        if (
            bool(remote_report.get("activated"))
            != local_gate["activated"]
            or sorted(remote_report.get("approved_axes") or [])
            != local_gate["approved_axes"]
            or remote_report.get("evaluation") != local_gate["evaluation"]
        ):
            raise LinkProtocolError(
                "remote student gate does not reproduce on the Lumen PC"
            )
        evaluation_report = {
            **remote_report,
            **local_gate,
            "link_manifest_sha256": manifest_sha256,
            "candidate_sha256": str(candidate_meta["sha256"]),
            "execution_target": "threadripper",
            "local_revalidated": True,
        }
        candidate_path = output_path.with_name(
            output_path.stem + ".candidate" + output_path.suffix
        )
        candidate_evaluation = candidate_path.with_name(
            candidate_path.stem + ".evaluation.json"
        )

        def atomic_copy(source: Path, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".link-import")
            shutil.copyfile(source, partial)
            partial.replace(target)

        atomic_copy(downloaded["candidate_model"], candidate_path)
        _atomic_json(candidate_evaluation, evaluation_report)
        if local_gate["activated"]:
            active_evaluation = output_path.with_name(
                output_path.stem + ".evaluation.json"
            )
            if (
                output_path.is_file()
                and _hash_file(output_path) != candidate_meta["sha256"]
            ):
                shutil.copyfile(
                    output_path,
                    output_path.with_name(output_path.stem + ".previous.npz"),
                )
                if active_evaluation.is_file():
                    shutil.copyfile(
                        active_evaluation,
                        active_evaluation.with_name(
                            output_path.stem
                            + ".previous.evaluation.json"
                        ),
                    )
            # Publish evaluation before the model. Live sees either the old
            # model or the fully verified new model; the model rename is the
            # activation commit point.
            _atomic_json(active_evaluation, evaluation_report)
            atomic_copy(downloaded["candidate_model"], output_path)
        receipt = {
            "schema": "lumen.link.student-import.v1",
            "job_id": str(job["id"]),
            "manifest_sha256": manifest_sha256,
            "candidate_sha256": str(candidate_meta["sha256"]),
            "evaluation_sha256": str(artifacts["evaluation"]["sha256"]),
            "local_evaluation_sha256": _hash_file(candidate_evaluation),
            "prepared_examples_sha256": str(
                artifacts["prepared_examples"]["sha256"]
            ),
            "activated": local_gate["activated"],
            "approved_axes": local_gate["approved_axes"],
            "candidate_model_path": str(candidate_path),
            "evaluation_path": str(candidate_evaluation),
            "execution_target": "threadripper",
            "local_revalidated": True,
            "resources": result.get("resources", {}),
        }
        _atomic_json(receipt_path, receipt)
        return receipt

    def _persist(self) -> None:
        _atomic_json(self.state_path, {"schema": LINK_SCHEMA, "active": self.active, "updated_unix_ms": int(time.time() * 1000)})
        self._invalidate_status()

    def _record_import(
        self,
        job: dict[str, Any],
        manifest: dict[str, Any],
        imported: dict[str, Any],
    ) -> None:
        receipt = {
            "job_id": str(job["id"]),
            "job_type": str(job["job_type"]),
            "imported_unix_ms": int(time.time() * 1000),
            "manifest_sha256": _sha256_bytes(_json_bytes(manifest)),
            "activated": imported.get("activated"),
            "approved_axes": imported.get("approved_axes", []),
            "local_revalidated": bool(imported.get("local_revalidated")),
            "result": imported,
        }
        self.recent_imports.appendleft(receipt)
        for cached in self._job_snapshot:
            if str(cached.get("id") or "") == str(job["id"]):
                cached["status"] = "complete"
                cached["result"] = imported
                break
        _atomic_json(
            self.recent_imports_path,
            {
                "schema": "lumen.link.recent-imports.v1",
                "imports": list(self.recent_imports),
            },
        )
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
        jobs = [deepcopy(self.active)] if self.active else []
        remote_jobs = list((self.remote_status or {}).get("jobs") or [])[:50]
        local_by_id = {
            str(job["id"]): job for job in self._job_snapshot[:1_000]
        }
        imports_by_id = {
            str(item.get("job_id") or ""): item
            for item in self.recent_imports
        }
        known_ids = {
            str(
                (item.get("job") or item).get("job_id")
                or (item.get("job") or item).get("id")
                or ""
            )
            for item in jobs
        }
        jobs.extend(
            deepcopy(item)
            for item in remote_jobs
            if str(item.get("job_id") or "") not in known_ids
        )
        for item in jobs:
            raw = item.get("job") or item
            job_id = str(raw.get("job_id") or raw.get("id") or "")
            receipt = imports_by_id.get(job_id)
            canonical = local_by_id.get(job_id)
            item["local_import_state"] = (
                "imported" if receipt is not None else "pending"
            )
            item["locally_imported"] = receipt is not None
            item["canonical_status"] = (
                "complete"
                if receipt is not None
                else (canonical.get("status") if canonical else None)
            )
        known_ids = {
            str((item.get("job") or item).get("job_id") or "")
            for item in jobs
        }
        for receipt in self.recent_imports:
            if str(receipt.get("job_id") or "") in known_ids:
                continue
            jobs.append(
                {
                    "job_id": receipt.get("job_id"),
                    "job_type": receipt.get("job_type"),
                    "status": "complete",
                    "stage": "locally_imported",
                    "local_import_state": "imported",
                    "locally_imported": True,
                    "canonical_status": "complete",
                    "import_receipt": receipt,
                }
            )
        local_jobs = self._job_snapshot
        queued = [
            job
            for job in local_jobs
            if job["status"] == "queued"
            and job["job_type"] in SUPPORTED_JOB_TYPES
        ]
        queue_bytes = int(self._queue_summary.get("bytes_pending") or 0)
        compatibility = {
            job_type: self._remote_is_compatible(job_type)
            for job_type in SUPPORTED_JOB_TYPES
        }
        any_compatible = any(compatibility.values())
        remote_contracts = (
            ((self.remote_status or {}).get("capabilities") or {})
            .get("job_contracts") or {}
        )
        local_revisions = {
            str((self._local_contract_cache.get(job_type) or {}).get("code_revision") or "")
            for job_type in SUPPORTED_JOB_TYPES
            if isinstance(self._local_contract_cache.get(job_type), dict)
        }
        remote_revisions = {
            str((remote_contracts.get(job_type) or {}).get("code_revision") or "")
            for job_type in SUPPORTED_JOB_TYPES
            if isinstance(remote_contracts.get(job_type), dict)
        }
        local_revision = next((value for value in local_revisions if value), "")
        remote_revision = next((value for value in remote_revisions if value), "")
        compatibility_detail = None
        if not configuration.secret:
            setup_state = "needs_secret"
            next_action = (
                "Run the Lumen Link setup script on both computers."
            )
        elif self.last_error:
            setup_state = "connection_error"
            next_action = "Check the direct cable, WSL service, and firewall."
        elif self.remote_status and not any_compatible:
            setup_state = "incompatible"
            if local_revision and remote_revision and local_revision != remote_revision:
                compatibility_detail = (
                    f"Authenticated worker code {remote_revision[:7]} does not "
                    f"match Lumen code {local_revision[:7]}."
                )
                next_action = (
                    f"Update the Threadripper worker from {remote_revision[:7]} "
                    f"to {local_revision[:7]}, restart it, then Test Connection."
                )
            else:
                compatibility_detail = (
                    "Authenticated worker model or preprocessing assets do not "
                    "match Lumen's compute contract."
                )
                next_action = (
                    "Reconfigure and verify the Threadripper worker, then Test "
                    "Connection again."
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
            next_action = "Compatible offline jobs will run sequentially on the Threadripper."
        result = {
            "schema": LINK_SCHEMA,
            "configured": bool(configuration.secret),
            "enabled": configuration.enabled,
            "paused": configuration.paused,
            "connection": {"state": "ready" if any_compatible else ("error" if self.last_error else ("incompatible" if self.remote_status else "offline")), "authenticated": bool(self.remote_status and not self.last_error), "latency_ms": self.latency_ms, "last_seen_unix_ms": int(self.last_seen * 1000) if self.last_seen else None, "error": self.last_error, "detail": compatibility_detail, "endpoint": configuration.endpoint, "address": configuration.endpoint.removeprefix("http://").removeprefix("https://"), "compatibility_by_job_type": compatibility},
            "local_node": {**_node_resources(), "address": "192.168.50.2"},
            "remote_node": {**((self.remote_status or {}).get("node") or {}), "address": configuration.endpoint.removeprefix("http://").removeprefix("https://")},
            "capabilities": (self.remote_status or {}).get("capabilities", {"supported_job_types": [], "gated_job_types": {job_type: "awaiting authenticated compatible health" for job_type in SUPPORTED_JOB_TYPES}, "job_contracts": {}}),
            "queue": {"queued": int(self._queue_summary.get("queued") or len(queued)), "bytes_pending": queue_bytes, "bytes_transferred": int((self.active or {}).get("transferred_bytes") or 0), "locally_imported": len(self.recent_imports), "remote": (self.remote_status or {}).get("queue", {})},
            "jobs": jobs,
            "recent_imports": list(self.recent_imports),
            "events": list(self.events),
            "setup": {
                "state": setup_state,
                "compatible": any_compatible,
                "compatibility_by_job_type": compatibility,
                "next_action": next_action,
                "endpoint": configuration.endpoint,
                "config_path": str(self.config_path),
                "direct_link": (
                    "Threadripper 192.168.50.1 ↔ Lumen 192.168.50.2"
                ),
                "port": 8765,
                "commands": (
                    [
                        "cd ~/lumenengine",
                        "./scripts/lumen-link-wsl stop",
                        "git pull --ff-only",
                        "./scripts/lumen-link-wsl configure --apply && ./scripts/lumen-link-wsl verify && ./scripts/lumen-link-wsl start",
                    ]
                    if setup_state == "incompatible"
                    else [
                        "./scripts/lumen-link status",
                        "./scripts/lumen-link test",
                        "./scripts/lumen-link setup",
                    ]
                ),
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
    maximum_parallel_jobs: int = DEFAULT_PARALLEL_JOBS,
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
        maximum_parallel_jobs=maximum_parallel_jobs,
    )
    server = LinkNodeServer((host, port), spool=spool, authenticator=LinkAuthenticator(secret), runtime=runtime)
    runtime.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        runtime.stop()
        server.server_close()
