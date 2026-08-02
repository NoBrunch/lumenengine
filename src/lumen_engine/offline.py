"""Offline teacher orchestration for Lumen's captured audio.

This module is deliberately dependency-light.  It prepares coherent WAV files,
queues work in SQLite, and invokes heavyweight teachers in their own isolated
Python environments.  Nothing here runs in the audio or DMX timing thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Iterable
import uuid
import wave

from lumen_engine.datasets import normalize_structure_label
from lumen_engine.memory import SongMemoryStore
from lumen_engine.student import (
    StreamingStructureStudent,
    semantic_frame_features,
)
from lumen_engine.training import structure_supervision_completeness


EDMFORMER_JOB = "teacher.edmformer"
SONGFORMER_JOB = "teacher.songformer"
STUDENT_TRAIN_JOB = "student.train"
MIN_TEACHER_DURATION_MS = 10_000
DEFAULT_OFFLINE_MAX_RSS_GIB = 8.0


class OfflineJobCancelled(RuntimeError):
    """A requested cancellation that leaves durable work retryable."""


class OfflineMemoryLimitExceeded(RuntimeError):
    """A teacher was stopped before it could exhaust the host."""


def _offline_memory_limit_bytes() -> int:
    raw = os.environ.get("LUMEN_OFFLINE_MAX_RSS_GIB", "").strip()
    if not raw:
        gib = DEFAULT_OFFLINE_MAX_RSS_GIB
    else:
        try:
            gib = float(raw)
        except ValueError:
            gib = DEFAULT_OFFLINE_MAX_RSS_GIB
    return int(max(0.25, gib) * 1024**3)


def _format_gib(byte_count: int) -> str:
    return f"{max(0, int(byte_count)) / 1024**3:.2f}"


def _process_group_rss_bytes(process_group_id: int) -> int:
    """Return resident bytes for every process in the isolated teacher group."""

    total_kib = 0
    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # The comm field can contain spaces and parentheses. Everything
            # after its final ')' starts with state, ppid, then process group.
            stat_tail = (entry / "stat").read_text(
                encoding="utf-8"
            ).rsplit(")", 1)[1].split()
            if int(stat_tail[2]) != int(process_group_id):
                continue
            for line in (entry / "status").read_text(
                encoding="utf-8"
            ).splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        except OSError:
            continue
    return total_kib * 1024


def _terminate_process_group(
    process: subprocess.Popen[str], *, grace_s: float = 10.0
) -> None:
    """Terminate the complete isolated teacher tree, escalating if needed."""

    if process.poll() is not None:
        process.communicate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=max(0.1, float(grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


@dataclass(frozen=True, slots=True)
class PreparedRecording:
    recording_id: str
    audio_path: Path
    content_sha256: str
    duration_ms: int
    song_id: int | None
    capture_session_id: str
    split_group_id: str
    split: str
    structure_supervision: dict[str, Any]


class ResearchJobCoordinator:
    """Prepare captured recordings and enqueue versioned offline work."""

    def __init__(
        self,
        store: SongMemoryStore,
        *,
        training_root: str | Path,
        research_root: str | Path,
    ) -> None:
        self.store = store
        self.training_root = Path(training_root).resolve()
        self.research_root = Path(research_root).resolve()
        self.audio_root = self.research_root / "audio"

    def prepare_export(
        self,
        export_directory: str | Path,
        *,
        queue_edmformer: bool = True,
        queue_songformer: bool = False,
    ) -> dict[str, Any]:
        """Materialize verified recording rows and queue available teachers."""
        export_root = Path(export_directory).resolve()
        manifest_path = export_root / "dataset.json"
        recordings_path = export_root / "recordings.jsonl"
        sessions_path = export_root / "sessions.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "lumen_training_dataset":
            raise ValueError("not a Lumen training export")
        validation = manifest.get("validation") or {}
        if not validation.get("valid", False):
            raise ValueError(
                "training export failed integrity validation; teachers were not queued"
            )
        self.audio_root.mkdir(parents=True, exist_ok=True)
        session_audio_complete: dict[str, bool] = {}
        if sessions_path.is_file():
            with sessions_path.open("r", encoding="utf-8") as rows:
                for line in rows:
                    if not line.strip():
                        continue
                    session = json.loads(line)
                    integrity = session.get("integrity")
                    session_audio_complete[str(session.get("id"))] = bool(
                        isinstance(integrity, dict)
                        and integrity.get("source_audio_complete")
                    )
        prepared: list[PreparedRecording] = []
        jobs: list[str] = []
        skipped: list[dict[str, Any]] = []
        with recordings_path.open("r", encoding="utf-8") as rows:
            for line in rows:
                if not line.strip():
                    continue
                recording = json.loads(line)
                recording.setdefault(
                    "source_audio_complete",
                    session_audio_complete.get(
                        str(recording.get("session_id")), False
                    ),
                )
                item = self._prepare_recording(recording)
                prepared.append(item)
                common_payload = {
                    "recording_id": item.recording_id,
                    "audio_path": str(item.audio_path),
                    "content_sha256": item.content_sha256,
                    "duration_ms": item.duration_ms,
                    "song_id": item.song_id,
                    "capture_session_id": item.capture_session_id,
                    "split_group_id": item.split_group_id,
                    "split": item.split,
                    "structure_supervision": item.structure_supervision,
                    "export_path": str(export_root),
                }
                requested_teachers = []
                if queue_edmformer:
                    requested_teachers.append(EDMFORMER_JOB)
                if queue_songformer:
                    requested_teachers.append(SONGFORMER_JOB)
                if item.duration_ms < MIN_TEACHER_DURATION_MS:
                    skipped.extend(
                        {
                            "recording_id": item.recording_id,
                            "job_type": job_type,
                            "reason": "recording_too_short",
                            "duration_ms": item.duration_ms,
                            "minimum_duration_ms": MIN_TEACHER_DURATION_MS,
                        }
                        for job_type in requested_teachers
                    )
                    continue
                if not item.structure_supervision["eligible"]:
                    skipped.extend(
                        {
                            "recording_id": item.recording_id,
                            "job_type": job_type,
                            "reason": (
                                "recording_incomplete_for_structure_supervision"
                            ),
                            "structure_supervision": (
                                item.structure_supervision
                            ),
                        }
                        for job_type in requested_teachers
                    )
                    continue
                edm_priority = 20 + (100 if item.split != "train" else 0)
                song_priority = 10 + (100 if item.split != "train" else 0)
                if queue_edmformer and not self._already_queued(
                    EDMFORMER_JOB,
                    item.recording_id,
                    item.content_sha256,
                    priority=edm_priority,
                ):
                    jobs.append(
                        self.store.enqueue_analysis_job(
                            job_type=EDMFORMER_JOB,
                            payload=common_payload,
                            priority=edm_priority,
                        )
                    )
                if queue_songformer and not self._already_queued(
                    SONGFORMER_JOB,
                    item.recording_id,
                    item.content_sha256,
                    priority=song_priority,
                ):
                    jobs.append(
                        self.store.enqueue_analysis_job(
                            job_type=SONGFORMER_JOB,
                            payload=common_payload,
                            priority=song_priority,
                        )
                    )
        return {
            "export_path": str(export_root),
            "recordings": len(prepared),
            "jobs_queued": len(jobs),
            "job_ids": jobs,
            "teachers_skipped": skipped,
            "prepared": [
                {
                    "recording_id": item.recording_id,
                    "audio_path": str(item.audio_path),
                    "content_sha256": item.content_sha256,
                    "duration_ms": item.duration_ms,
                    "split_group_id": item.split_group_id,
                    "split": item.split,
                    "structure_supervision": item.structure_supervision,
                }
                for item in prepared
            ],
        }

    def _prepare_recording(
        self, recording: dict[str, Any]
    ) -> PreparedRecording:
        clips = recording.get("audio_clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("recording has no reconstructable audio clips")
        sample_rate = int(recording["sample_rate"])
        channels = int(recording["channels"])
        sample_width = int(recording["sample_width"])
        if sample_width != 2:
            raise ValueError("teacher preparation currently requires PCM16")
        fingerprint = str(recording.get("content_fingerprint") or "")
        fingerprint_value = fingerprint.removeprefix("sha256:")
        if len(fingerprint_value) != 64:
            raise ValueError("recording content fingerprint is invalid")
        target = self.audio_root / f"{fingerprint_value}.wav"
        structure_supervision = recording.get("structure_supervision")
        if not isinstance(structure_supervision, dict):
            media = recording.get("track_identity")
            media = media if isinstance(media, dict) else {}
            structure_supervision = structure_supervision_completeness(
                track_duration_ms=media.get("duration_ms"),
                start_position_ms=recording.get("first_position_ms"),
                end_position_ms=recording.get("last_position_ms"),
                captured_duration_ms=round(
                    int(recording["frame_count"]) * 1000 / sample_rate
                ),
                source_audio_complete=bool(
                    recording.get("source_audio_complete", False)
                ),
            )
        def source_pcm(
            output: wave.Wave_write | None = None,
        ) -> tuple[str, int]:
            source_digest = hashlib.sha256()
            source_frames = 0
            for clip in clips:
                source = self._resolve_training_path(
                    str(clip["relative_path"])
                )
                with wave.open(str(source), "rb") as input_file:
                    if (
                        input_file.getframerate() != sample_rate
                        or input_file.getnchannels() != channels
                        or input_file.getsampwidth() != sample_width
                    ):
                        raise ValueError(
                            f"incompatible WAV clip metadata: {source}"
                        )
                    start_frame = int(clip["start_frame"])
                    frame_count = int(clip["frame_count"])
                    input_file.setpos(start_frame)
                    pcm = input_file.readframes(frame_count)
                expected_bytes = frame_count * channels * sample_width
                if len(pcm) != expected_bytes:
                    raise ValueError(f"truncated WAV clip: {source}")
                if output is not None:
                    output.writeframesraw(pcm)
                source_digest.update(pcm)
                source_frames += frame_count
            return source_digest.hexdigest(), source_frames

        if not target.is_file():
            partial = target.with_suffix(".wav.partial")
            with wave.open(str(partial), "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(sample_width)
                output.setframerate(sample_rate)
                pcm_sha256, total_frames = source_pcm(output)
            partial.replace(target)
        else:
            expected_sha256, expected_frames = source_pcm()
            cached_sha256, total_frames = _hash_wav_pcm(
                target,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )
            if (
                cached_sha256 != expected_sha256
                or total_frames != expected_frames
            ):
                raise ValueError(
                    "cached teacher WAV does not match its verified source clips: "
                    + str(target)
                )
            pcm_sha256 = cached_sha256
        media = recording.get("track_identity") or {}
        provider = str(media.get("provider") or "lumen-capture")
        provider_item_id = str(
            media.get("provider_item_id")
            or recording.get("split_group_id")
            or fingerprint_value
        )
        split_group_id = str(
            recording.get("split_group_id")
            or (
                f"{provider}:{provider_item_id}"
                if provider != "lumen-capture"
                else f"unidentified-session:{recording['session_id']}"
            )
        )
        split = _recording_split(split_group_id)
        exported_split = recording.get("split")
        if exported_split is not None and str(exported_split) != split:
            raise ValueError(
                "recording split does not match its stable split group"
            )
        duration_ms = round(total_frames * 1000 / sample_rate)
        stable_recording_id = self.store.remember_recording_version(
            provider=provider,
            provider_item_id=provider_item_id,
            song_id=recording.get("song_id"),
            duration_ms=duration_ms,
            audio_fingerprint=f"pcm-sha256:{pcm_sha256}",
            metadata={
                "capture_recording_id": recording.get("recording_id"),
                "content_fingerprint": fingerprint,
                "audio_path": str(target),
                "track_identity": media,
                "split_group_id": split_group_id,
                "split": split,
                "structure_supervision": structure_supervision,
            },
        )
        self.store.add_capture_track_span(
            capture_session_id=str(recording["session_id"]),
            start_audio_frame=int(recording["start_frame"]),
            end_audio_frame=int(recording["end_frame"]),
            recording_id=stable_recording_id,
            song_id=recording.get("song_id"),
            start_position_ms=recording.get("first_position_ms"),
            end_position_ms=recording.get("last_position_ms"),
            identity_source=(
                "spotify_metadata" if media.get("provider_item_id")
                else "capture_content_fingerprint"
            ),
            identity_confidence=0.99 if media.get("provider_item_id") else 0.75,
            metadata={
                "content_sha256": pcm_sha256,
                "audio_path": str(target),
                "split_group_id": split_group_id,
                "split": split,
                "structure_supervision": structure_supervision,
            },
        )
        return PreparedRecording(
            recording_id=stable_recording_id,
            audio_path=target,
            content_sha256=pcm_sha256,
            duration_ms=duration_ms,
            song_id=recording.get("song_id"),
            capture_session_id=str(recording["session_id"]),
            split_group_id=split_group_id,
            split=split,
            structure_supervision=structure_supervision,
        )

    def _resolve_training_path(self, relative_path: str) -> Path:
        path = (self.training_root / relative_path).resolve()
        try:
            path.relative_to(self.training_root)
        except ValueError as error:
            raise ValueError("audio clip escapes the training root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _already_queued(
        self,
        job_type: str,
        recording_id: str,
        content_sha256: str,
        *,
        priority: int,
    ) -> bool:
        for job in self.store.list_analysis_jobs(limit=100_000):
            if (
                job["job_type"] == job_type
                and job["status"] in {"queued", "running", "complete"}
                and job["payload"].get("recording_id") == recording_id
                and job["payload"].get("content_sha256") == content_sha256
            ):
                if job["status"] == "queued" and int(job["priority"]) != priority:
                    self.store.update_analysis_job_priority(
                        str(job["id"]), priority=priority
                    )
                return True
        return False


class OfflineResearchWorker:
    """Execute one queued teacher/training job at a time."""

    def __init__(
        self,
        store: SongMemoryStore,
        *,
        research_root: str | Path,
        timeout_s: float = 4 * 60 * 60,
        cancel_event: threading.Event | None = None,
        max_rss_bytes: int | None = None,
        abandoned_after_s: float = 120.0,
    ) -> None:
        self.store = store
        self.root = Path(research_root).resolve()
        self.timeout_s = timeout_s
        self.cancel_event = cancel_event
        self.worker_id = f"worker:{uuid.uuid4()}"
        self.worker_pid = os.getpid()
        self.max_rss_bytes = (
            _offline_memory_limit_bytes()
            if max_rss_bytes is None
            else max(1, int(max_rss_bytes))
        )
        self.abandoned_after_ms = max(
            1_000, int(float(abandoned_after_s) * 1000)
        )
        self.last_recovery: list[dict[str, Any]] = []
        self._active_job_id: str | None = None
        self._last_subprocess_metrics: dict[str, Any] = {}

    def run_once(
        self, job_types: Iterable[str] = ()
    ) -> dict[str, Any] | None:
        self.last_recovery = self.store.recover_abandoned_analysis_jobs(
            stale_after_ms=self.abandoned_after_ms
        )
        job = self.store.claim_analysis_job(
            tuple(job_types),
            worker_id=self.worker_id,
            worker_pid=self.worker_pid,
        )
        if job is None:
            return None
        self._active_job_id = str(job["id"])
        self._last_subprocess_metrics = {}
        if (
            job["job_type"] in {EDMFORMER_JOB, SONGFORMER_JOB}
            and int(job["payload"].get("duration_ms") or 0)
            < MIN_TEACHER_DURATION_MS
        ):
            result = {
                "reason": "recording_too_short",
                "duration_ms": int(job["payload"].get("duration_ms") or 0),
                "minimum_duration_ms": MIN_TEACHER_DURATION_MS,
            }
            self.store.update_analysis_job(
                job["id"], status="complete", result=result
            )
            self._active_job_id = None
            return {
                "job_id": job["id"],
                "job_type": job["job_type"],
                "status": "skipped",
                "result": result,
            }
        if job["job_type"] in {EDMFORMER_JOB, SONGFORMER_JOB}:
            try:
                supervision = _job_structure_supervision(self.store, job)
            except Exception as error:
                self.store.update_analysis_job(
                    job["id"], status="failed", error=str(error)
                )
                self._active_job_id = None
                return {
                    "job_id": job["id"],
                    "job_type": job["job_type"],
                    "status": "failed",
                    "error": str(error),
                }
            if not supervision["eligible"]:
                result = {
                    "reason": (
                        "recording_incomplete_for_structure_supervision"
                    ),
                    "structure_supervision": supervision,
                }
                self.store.update_analysis_job(
                    job["id"], status="complete", result=result
                )
                self._active_job_id = None
                return {
                    "job_id": job["id"],
                    "job_type": job["job_type"],
                    "status": "skipped",
                    "result": result,
                }
        try:
            if job["job_type"] == EDMFORMER_JOB:
                result = self._run_edmformer(job)
            elif job["job_type"] == SONGFORMER_JOB:
                result = self._run_songformer(job)
            elif job["job_type"] == STUDENT_TRAIN_JOB:
                result = self._train_student(job)
            else:
                raise ValueError(f"unsupported analysis job {job['job_type']!r}")
        except OfflineJobCancelled as error:
            self.store.update_analysis_job(
                job["id"],
                status="queued",
                result=self._last_subprocess_metrics or None,
                error=str(error),
            )
            self._active_job_id = None
            return {
                "job_id": job["id"],
                "job_type": job["job_type"],
                "status": "canceled",
                "error": str(error),
            }
        except Exception as error:
            self.store.update_analysis_job(
                job["id"],
                status="failed",
                result=self._last_subprocess_metrics or None,
                error=str(error),
            )
            self._active_job_id = None
            return {
                "job_id": job["id"],
                "job_type": job["job_type"],
                "status": "failed",
                "error": str(error),
            }
        self.store.update_analysis_job(
            job["id"], status="complete", result=result
        )
        self._active_job_id = None
        return {
            "job_id": job["id"],
            "job_type": job["job_type"],
            "status": "complete",
            "result": result,
        }

    def run_until_empty(
        self,
        job_types: Iterable[str] = (),
        *,
        maximum_jobs: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while maximum_jobs is None or len(results) < maximum_jobs:
            result = self.run_once(job_types)
            if result is None:
                break
            results.append(result)
        return results

    def _run_edmformer(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        paths = self._edmformer_paths()
        audio_path = Path(str(payload["audio_path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        output_dir = self.root / "exports" / "teacher-results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job['id'].replace(':', '-')}.json"
        window_seconds = max(
            30, min(60, int(payload.get("edmformer_window_seconds", 60)))
        )
        run_id = self.store.begin_teacher_run(
            teacher_name="EDMFormer",
            teacher_version=paths["revision"],
            device="cpu",
            preprocessing_version=(
                "lumen_edmformer_cpu_runner_v1:"
                f"{window_seconds}s"
            ),
            recording_id=payload.get("recording_id"),
            capture_session_id=payload.get("capture_session_id"),
            analysis_job_id=str(job["id"]),
        )
        command = [
            str(paths["python"]),
            str(paths["runner"]),
            str(audio_path),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--config",
            str(paths["config"]),
            "--musicfm-stat",
            str(paths["musicfm_stat"]),
            "--musicfm-model",
            str(paths["musicfm_model"]),
            "--musicfm-source",
            str(paths["musicfm_source"]),
            "--hf-cache-dir",
            str(paths["hf_cache"]),
            "--window-seconds",
            str(window_seconds),
            "--threads",
            "4",
            "--output",
            str(output_path),
        ]
        started = time.monotonic()
        try:
            completed = self._run_subprocess(command, dict(os.environ))
            if completed.returncode != 0:
                message = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"EDMFormer exited {completed.returncode}"
                )
                raise RuntimeError(message[-4000:])
            raw_segments = json.loads(output_path.read_text(encoding="utf-8"))
            segments = _normalize_teacher_segments(
                raw_segments,
                source="EDMFormer",
                source_version=paths["revision"],
            )
            _validate_teacher_coverage(
                segments,
                source="EDMFormer",
                duration_ms=int(payload["duration_ms"]),
            )
            timeline_id = self.store.save_structure_timeline(
                recording_id=payload.get("recording_id"),
                song_id=payload.get("song_id"),
                capture_session_id=payload.get("capture_session_id"),
                teacher_run_id=run_id,
                provenance="edmformer_teacher",
                timeline_version="lumen_normalized_structure_v1",
                confidence=_mean_confidence(segments),
                segments=segments,
                metadata={
                    "audio_path": str(audio_path),
                    "content_sha256": payload.get("content_sha256"),
                    "raw_output": str(output_path),
                    "command": command,
                    "bounded_window_seconds": window_seconds,
                    "runner": "lumen_edmformer_cpu_runner_v1",
                    "structure_supervision": payload.get(
                        "structure_supervision"
                    ),
                },
            )
            student_examples = build_student_examples(
                self.store,
                research_root=self.root,
                recording_id=str(payload["recording_id"]),
                timeline_id=timeline_id,
            )
            metrics = {
                "elapsed_s": time.monotonic() - started,
                "segments": len(segments),
                "timeline_id": timeline_id,
                "student_examples": student_examples,
                "structure_supervision": payload.get(
                    "structure_supervision"
                ),
                "bounded_window_seconds": window_seconds,
                "subprocess": dict(self._last_subprocess_metrics),
            }
            self.store.finish_teacher_run(
                run_id, status="complete", metrics=metrics
            )
            return metrics
        except Exception as error:
            self.store.finish_teacher_run(
                run_id,
                status=(
                    "canceled"
                    if isinstance(error, OfflineJobCancelled)
                    else "failed"
                ),
                metrics={"elapsed_s": time.monotonic() - started},
                error=str(error),
            )
            raise

    def _run_subprocess(
        self,
        command: list[str],
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.timeout_s
        started = time.monotonic()
        peak_rss_bytes = 0
        last_heartbeat = 0.0
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.5)
                elapsed_s = time.monotonic() - started
                self._last_subprocess_metrics = {
                    "elapsed_s": elapsed_s,
                    "peak_rss_bytes": peak_rss_bytes,
                    "memory_limit_bytes": self.max_rss_bytes,
                    "returncode": process.returncode,
                }
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                rss_bytes = _process_group_rss_bytes(process.pid)
                peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
                self._last_subprocess_metrics = {
                    "elapsed_s": now - started,
                    "rss_bytes": rss_bytes,
                    "peak_rss_bytes": peak_rss_bytes,
                    "memory_limit_bytes": self.max_rss_bytes,
                }
                if (
                    self._active_job_id is not None
                    and now - last_heartbeat >= 2.0
                ):
                    self.store.heartbeat_analysis_job(
                        self._active_job_id,
                        worker_id=self.worker_id,
                        progress=self._last_subprocess_metrics,
                    )
                    last_heartbeat = now
                if (
                    self.cancel_event is not None
                    and self.cancel_event.is_set()
                ):
                    _terminate_process_group(process)
                    raise OfflineJobCancelled(
                        "offline teacher canceled at requested checkpoint"
                    )
                if rss_bytes > self.max_rss_bytes:
                    _terminate_process_group(process)
                    raise OfflineMemoryLimitExceeded(
                        "offline teacher exceeded the local memory limit "
                        f"({_format_gib(peak_rss_bytes)} GiB peak; "
                        f"{_format_gib(self.max_rss_bytes)} GiB limit). "
                        "The job was stopped before system-wide memory "
                        "exhaustion and remains failed for inspection instead "
                        "of repeating automatically."
                    )
                if now >= deadline:
                    _terminate_process_group(process)
                    raise TimeoutError(
                        f"offline teacher exceeded {self.timeout_s:.0f}s "
                        f"({_format_gib(peak_rss_bytes)} GiB peak RSS)"
                    )

    def _run_songformer(self, job: dict[str, Any]) -> dict[str, Any]:
        """Run the isolated, bounded-memory SongFormer CPU adaptation."""
        payload = job["payload"]
        paths = self._songformer_paths()
        audio_path = Path(str(payload["audio_path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        output_dir = self.root / "exports" / "teacher-results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job['id'].replace(':', '-')}.json"
        window_seconds = max(
            30, min(60, int(payload.get("songformer_window_seconds", 60)))
        )
        run_id = self.store.begin_teacher_run(
            teacher_name="SongFormer",
            teacher_version=paths["revision"],
            device="cpu",
            preprocessing_version=(
                f"songformer_official_features_cpu_windowed_v1:"
                f"{window_seconds}s"
            ),
            recording_id=payload.get("recording_id"),
            capture_session_id=payload.get("capture_session_id"),
            analysis_job_id=str(job["id"]),
        )
        command = [
            str(paths["python"]),
            str(paths["runner"]),
            str(audio_path),
            "--research-root",
            str(self.root),
            "--window-seconds",
            str(window_seconds),
            "--threads",
            "4",
            "--output",
            str(output_path),
        ]
        started = time.monotonic()
        try:
            completed = self._run_subprocess(command, dict(os.environ))
            if completed.returncode != 0:
                message = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"SongFormer exited {completed.returncode}"
                )
                raise RuntimeError(message[-4000:])
            raw_segments = json.loads(output_path.read_text(encoding="utf-8"))
            segments = _normalize_teacher_segments(
                raw_segments,
                source="SongFormer",
                source_version=paths["revision"],
            )
            _validate_teacher_coverage(
                segments,
                source="SongFormer",
                duration_ms=int(payload["duration_ms"]),
            )
            timeline_id = self.store.save_structure_timeline(
                recording_id=payload.get("recording_id"),
                song_id=payload.get("song_id"),
                capture_session_id=payload.get("capture_session_id"),
                teacher_run_id=run_id,
                provenance="songformer_teacher",
                timeline_version="lumen_normalized_structure_v1",
                confidence=_mean_confidence(segments),
                segments=segments,
                metadata={
                    "audio_path": str(audio_path),
                    "content_sha256": payload.get("content_sha256"),
                    "raw_output": str(output_path),
                    "command": command,
                    "cpu_context_window_seconds": window_seconds,
                    "upstream_context_window_seconds": 420,
                    "adaptation": (
                        "official SongFormer feature/head path with a "
                        "bounded CPU context window"
                    ),
                    "structure_supervision": payload.get(
                        "structure_supervision"
                    ),
                },
            )
            student_examples = build_student_examples(
                self.store,
                research_root=self.root,
                recording_id=str(payload["recording_id"]),
                timeline_id=timeline_id,
            )
            metrics = {
                "elapsed_s": time.monotonic() - started,
                "segments": len(segments),
                "timeline_id": timeline_id,
                "student_examples": student_examples,
                "cpu_context_window_seconds": window_seconds,
                "structure_supervision": payload.get(
                    "structure_supervision"
                ),
                "subprocess": dict(self._last_subprocess_metrics),
            }
            self.store.finish_teacher_run(
                run_id, status="complete", metrics=metrics
            )
            return metrics
        except Exception as error:
            self.store.finish_teacher_run(
                run_id,
                status=(
                    "canceled"
                    if isinstance(error, OfflineJobCancelled)
                    else "failed"
                ),
                metrics={"elapsed_s": time.monotonic() - started},
                error=str(error),
            )
            raise

    def _train_student(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        examples_path = Path(str(payload["examples_path"])).resolve()
        expected_examples_sha256 = payload.get("examples_sha256")
        if (
            expected_examples_sha256
            and _hash_file(examples_path) != expected_examples_sha256
        ):
            raise ValueError(
                "student training examples changed after the job was queued"
            )
        examples = _load_jsonl(examples_path)
        gated = bool(payload.get("require_activation_gate", False))
        statistics = _student_example_statistics(
            examples, require_group_identity=gated
        )
        train_examples = [
            row for row in examples if row.get("split", "train") == "train"
        ]
        if not train_examples:
            if gated:
                raise ValueError(
                    "automatic student training requires a train split"
                )
            train_examples = examples

        def cancel_check() -> None:
            if self._active_job_id is not None:
                self.store.heartbeat_analysis_job(
                    self._active_job_id,
                    worker_id=self.worker_id,
                    progress={"stage": "student_training"},
                )
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise OfflineJobCancelled(
                    "offline student canceled at requested checkpoint"
                )

        model = StreamingStructureStudent(
            hidden_size=int(payload.get("hidden_size", 32))
        )
        training = model.train(
            train_examples,
            epochs=int(payload.get("epochs", 30)),
            learning_rate=float(payload.get("learning_rate", 0.025)),
            cancel_check=cancel_check,
        )
        cancel_check()
        evaluation = {
            split: model.evaluate(
                [row for row in examples if row.get("split", "train") == split]
            )
            for split in ("train", "validation", "test")
        }
        output_path = Path(
            str(
                payload.get("output_path")
                or self.root / "models" / "lumen-structure-student.npz"
            )
        )
        candidate_path = (
            output_path.with_name(
                output_path.stem + ".candidate" + output_path.suffix
            )
            if gated
            else output_path
        )
        model.save(candidate_path)
        cancel_check()
        held_out_name = (
            "test"
            if evaluation["test"]["energy"]["examples"]
            else "validation"
        )
        held_out = evaluation[held_out_name]
        gate_reasons: list[str] = []
        if gated:
            energy = held_out["energy"]
            functional = held_out["functional"]
            boundary = held_out["boundary"]
            if int(energy["examples"] or 0) < 10:
                gate_reasons.append("held-out set has fewer than 10 frames")
            energy_floor = max(
                0.35,
                float(energy.get("majority_baseline") or 0.0),
            )
            if float(energy.get("accuracy") or 0.0) < energy_floor:
                gate_reasons.append(
                    "held-out energy accuracy did not meet its baseline gate"
                )
            functional_floor = max(
                0.25,
                float(functional.get("majority_baseline") or 0.0),
            )
            if (
                int(functional["examples"] or 0) >= 10
                and float(functional.get("accuracy") or 0.0)
                < functional_floor
            ):
                gate_reasons.append(
                    "held-out functional-section accuracy did not meet "
                    "its baseline gate"
                )
            boundary_positives = int(boundary["tp"]) + int(boundary["fn"])
            if (
                boundary_positives >= 5
                and float(boundary.get("f1") or 0.0) < 0.10
            ):
                gate_reasons.append(
                    "held-out boundary detection did not meet its F1 gate"
                )
        activated = not gate_reasons
        if activated and candidate_path != output_path:
            cancel_check()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            activation_partial = output_path.with_suffix(
                output_path.suffix + ".activation"
            )
            shutil.copyfile(candidate_path, activation_partial)
            activation_partial.replace(output_path)
        evaluation_path = output_path.with_name(
            output_path.stem + ".evaluation.json"
        )
        evaluation_path.write_text(
            json.dumps(
                {
                    "activated": activated,
                    "gate_reasons": gate_reasons,
                    "held_out_split": held_out_name,
                    "evaluation": evaluation,
                    "training": training,
                    "source_sha256": _hash_file(examples_path),
                    "source_scope": payload.get("source_scope"),
                    "teacher_run_ids": payload.get(
                        "teacher_run_ids", []
                    ),
                    "source_files": payload.get("source_files", []),
                    "split_counts": statistics["split_counts"],
                    "split_group_counts": statistics[
                        "split_group_counts"
                    ],
                    "label_balance": statistics["label_balance"],
                    "teacher_merge": payload.get("teacher_merge"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "model_path": str(output_path),
            "candidate_model_path": str(candidate_path),
            "evaluation_path": str(evaluation_path),
            "activated": activated,
            "activation_gate_reasons": gate_reasons,
            "held_out_split": held_out_name,
            "training": training,
            "evaluation": evaluation,
            "split_counts": statistics["split_counts"],
            "split_group_counts": statistics["split_group_counts"],
            "label_balance": statistics["label_balance"],
            "teacher_merge": payload.get("teacher_merge"),
            "examples_sha256": _hash_file(examples_path),
            "source_scope": payload.get("source_scope"),
            "teacher_run_ids": payload.get("teacher_run_ids", []),
        }

    def _edmformer_paths(self) -> dict[str, Any]:
        source = self.root / "sources" / "edm98"
        musicfm_source = self.root / "sources" / "musicfm"
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_root = source / "data" / "checkpoints"
        paths: dict[str, Any] = {
            "python": self.root / "environments" / "edmformer" / "bin" / "python",
            "runner": project_root / "scripts" / "edmformer-cpu-runner.py",
            "checkpoint": checkpoint_root / "model.pt",
            "config": source / "configs" / "edmformer.yaml",
            "musicfm_stat": checkpoint_root / "msd_stats.json",
            "musicfm_model": checkpoint_root / "pretrained_msd.pt",
            "musicfm_source": musicfm_source,
            "hf_cache": self.root / "cache" / "huggingface",
            "revision": _git_revision(source),
        }
        missing = [
            str(value)
            for key, value in paths.items()
            if key != "revision" and isinstance(value, Path)
            and not value.exists()
        ]
        if missing:
            raise RuntimeError(
                "EDMFormer is not provisioned: " + ", ".join(missing)
            )
        return paths

    def _songformer_paths(self) -> dict[str, Any]:
        source = self.root / "sources" / "songformer"
        project_root = Path(__file__).resolve().parents[2]
        paths: dict[str, Any] = {
            "python": (
                self.root
                / "environments"
                / "songformer"
                / "bin"
                / "python"
            ),
            "runner": project_root / "scripts" / "songformer-cpu-runner.py",
            "source": source,
            "head": (
                self.root
                / "models"
                / "songformer"
                / "SongFormer.safetensors"
            ),
            "muq": self.root / "models" / "muq" / "model.safetensors",
            "musicfm": (
                self.root
                / "sources"
                / "edm98"
                / "data"
                / "checkpoints"
                / "pretrained_msd.pt"
            ),
            "revision": _git_revision(source),
        }
        missing = [
            str(value)
            for key, value in paths.items()
            if key != "revision"
            and isinstance(value, Path)
            and not value.exists()
        ]
        if missing:
            raise RuntimeError(
                "SongFormer CPU gate is not provisioned: "
                + ", ".join(missing)
                + ". Run scripts/setup-research provision."
            )
        if not paths["revision"]:
            raise RuntimeError(
                "SongFormer CPU gate cannot verify the pinned source revision"
            )
        return paths


_AXIS_TEACHER_PRIORITY = {
    "functional": {"songformer": 20, "edmformer": 10},
    "energy": {"edmformer": 20, "songformer": 10},
    "content": {"songformer": 20, "edmformer": 10},
}


def _teacher_source(row: dict[str, Any]) -> str:
    details = row.get("target_provenance_details")
    if isinstance(details, dict) and details.get("source"):
        return str(details["source"])
    return str(row.get("target_provenance") or "unknown")


def _merge_teacher_example_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse complementary teachers into one target per captured audio frame."""

    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for order, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("student example row is not an object")
        if (
            row.get("recording_id") is None
            or row.get("capture_session_id") is None
            or row.get("audio_frame_index") is None
        ):
            passthrough.append((order, dict(row)))
            continue
        key = (
            str(row["recording_id"]),
            str(row["capture_session_id"]),
            int(row["audio_frame_index"]),
        )
        buckets.setdefault(key, []).append(row)

    merged: list[dict[str, Any]] = []
    axis_conflicts = {axis: 0 for axis in _AXIS_TEACHER_PRIORITY}
    for key, candidates in buckets.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                _teacher_source(row).casefold(),
                str(row.get("teacher_run_id") or ""),
                str(row.get("timeline_id") or ""),
            ),
        )
        base = dict(ordered[0])
        for other in ordered[1:]:
            for field in (
                "recording_id",
                "capture_session_id",
                "audio_frame_index",
                "recording_offset_ms",
                "position_ms",
                "split_group_id",
                "split",
                "features",
            ):
                if other.get(field) != base.get(field):
                    raise ValueError(
                        f"teacher rows disagree on frame invariant {field}"
                    )

        # Early teacher exports predate the per-row supervision snapshot.
        # A later complementary teacher may therefore carry the verified
        # recording-level value while the older row has no value at all.
        # Missing legacy data is not a disagreement: preserve the available
        # snapshot.  Two explicit snapshots must still agree exactly so a
        # genuinely inconsistent capture cannot enter student training.
        supervision_snapshots: list[dict[str, Any]] = []
        for row in ordered:
            snapshot = row.get("structure_supervision")
            if snapshot is None:
                continue
            if not isinstance(snapshot, dict):
                raise ValueError(
                    "teacher row has invalid structure_supervision"
                )
            supervision_snapshots.append(snapshot)
        if supervision_snapshots:
            supervision = supervision_snapshots[0]
            if any(
                snapshot != supervision
                for snapshot in supervision_snapshots[1:]
            ):
                raise ValueError(
                    "teacher rows disagree on frame invariant "
                    "structure_supervision"
                )
            base["structure_supervision"] = dict(supervision)

        provenance_by_axis: dict[str, dict[str, Any]] = {}
        for axis, source_priority in _AXIS_TEACHER_PRIORITY.items():
            labeled = []
            for row in ordered:
                label = str(row.get(axis) or "unknown")
                if label == "unknown":
                    continue
                source = _teacher_source(row)
                confidence = float(row.get("target_confidence") or 0.0)
                labeled.append(
                    (
                        source_priority.get(source.casefold(), 0),
                        confidence,
                        source.casefold(),
                        str(row.get("teacher_run_id") or ""),
                        label,
                        row,
                    )
                )
            if not labeled:
                base[axis] = "unknown"
                continue
            if len({item[4] for item in labeled}) > 1:
                axis_conflicts[axis] += 1
            selected = max(labeled, key=lambda item: item[:4])
            selected_row = selected[5]
            base[axis] = selected[4]
            details = selected_row.get("target_provenance_details")
            provenance_by_axis[axis] = {
                **(details if isinstance(details, dict) else {}),
                "teacher_run_id": selected_row.get("teacher_run_id"),
                "timeline_id": selected_row.get("timeline_id"),
                "label": selected[4],
            }

        base["boundary"] = max(
            int(float(row.get("boundary") or 0) >= 0.5)
            for row in ordered
        )
        boundary_distances = [
            int(row.get("milliseconds_since_boundary") or 0)
            for row in ordered
            if int(float(row.get("boundary") or 0) >= 0.5)
        ]
        base["milliseconds_since_boundary"] = (
            min(boundary_distances) if boundary_distances else 0
        )
        run_ids = sorted(
            {
                str(row["teacher_run_id"])
                for row in ordered
                if row.get("teacher_run_id")
            }
        )
        timeline_ids = sorted(
            {
                str(row["timeline_id"])
                for row in ordered
                if row.get("timeline_id")
            }
        )
        base["teacher_run_ids"] = run_ids
        base["timeline_ids"] = timeline_ids
        base["teacher_run_id"] = run_ids[0] if len(run_ids) == 1 else None
        base["timeline_id"] = timeline_ids[0] if len(timeline_ids) == 1 else None
        base["target_provenance"] = (
            str(ordered[0].get("target_provenance") or "teacher_prediction")
            if len(run_ids) <= 1
            else "teacher_ensemble"
        )
        base["target_provenance_by_axis"] = provenance_by_axis
        base["target_confidence"] = max(
            float(row.get("target_confidence") or 0.0) for row in ordered
        )
        merged.append(base)

    merged.extend(row for _, row in passthrough)
    merged.sort(
        key=lambda row: (
            str(row.get("split") or "train"),
            str(row.get("split_group_id") or ""),
            str(row.get("recording_id") or ""),
            str(row.get("capture_session_id") or ""),
            int(row.get("audio_frame_index") or 0),
        )
    )
    return merged, {
        "source_rows": len(rows),
        "merged_rows": len(merged),
        "duplicates_collapsed": len(rows) - len(merged),
        "axis_conflicts": axis_conflicts,
        "axis_precedence": _AXIS_TEACHER_PRIORITY,
        "boundary_merge": "maximum",
    }


def _student_example_statistics(
    rows: list[dict[str, Any]],
    *,
    require_group_identity: bool = True,
) -> dict[str, Any]:
    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_groups: dict[str, set[str]] = {
        split: set() for split in split_counts
    }
    label_balance: dict[str, dict[str, int]] = {
        axis: {} for axis in _AXIS_TEACHER_PRIORITY
    }
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        split = str(row.get("split") or "train")
        if split not in split_counts:
            raise ValueError(f"unknown student dataset split {split!r}")
        group = str(row.get("split_group_id") or row.get("recording_id") or "")
        if not group:
            if require_group_identity:
                raise ValueError("student example has no split group identity")
            group = f"explicit-unidentified:{split}"
        split_counts[split] += 1
        split_groups[split].add(group)
        group_splits.setdefault(group, set()).add(split)
        for axis in label_balance:
            label = str(row.get(axis) or "unknown")
            if label != "unknown":
                label_balance[axis][label] = (
                    label_balance[axis].get(label, 0) + 1
                )
    leaking = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if leaking:
        raise ValueError(
            "student split groups cross dataset partitions: "
            + ", ".join(sorted(leaking)[:5])
        )
    return {
        "split_counts": split_counts,
        "split_group_counts": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "label_balance": label_balance,
    }


def enqueue_student_training(
    store: SongMemoryStore,
    *,
    research_root: str | Path,
    example_paths: Iterable[str | Path] = (),
    epochs: int = 30,
) -> dict[str, Any]:
    """Combine teacher examples and enqueue a reproducible CPU training job."""
    root = Path(research_root).resolve()
    paths = [Path(path).resolve() for path in example_paths]
    provenance: dict[str, Any]
    if not paths:
        trusted = trusted_student_examples(store, research_root=root)
        paths = [Path(path) for path in trusted["paths"]]
        provenance = trusted
    else:
        provenance = {
            "scope": "explicit_paths",
            "paths": [str(path) for path in paths],
        }
    if not paths:
        raise RuntimeError(
            "no completed teacher runs in this Lumen database have usable "
            "student examples"
        )
    combined = root / "exports" / "student-training.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    partial = combined.with_suffix(".jsonl.partial")
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        source_rows.extend(_load_jsonl(path))
    merged_rows, merge_report = _merge_teacher_example_rows(source_rows)
    rows = len(merged_rows)
    with partial.open("w", encoding="utf-8") as output:
        for row in merged_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    if rows == 0:
        raise RuntimeError("teacher example files contain no training rows")
    statistics = _student_example_statistics(
        merged_rows,
        require_group_identity=not bool(example_paths),
    )
    split_counts = statistics["split_counts"]
    split_group_counts = statistics["split_group_counts"]
    label_balance = statistics["label_balance"]
    held_out_examples = split_counts["validation"] + split_counts["test"]
    if not example_paths and held_out_examples == 0:
        raise RuntimeError(
            "teacher examples do not yet include a held-out song; process "
            "more recordings before training"
        )
    if not example_paths and split_group_counts["train"] < 2:
        raise RuntimeError(
            "teacher examples require at least two complete training songs"
        )
    partial.replace(combined)
    job_id = store.enqueue_analysis_job(
        job_type=STUDENT_TRAIN_JOB,
        payload={
            "examples_path": str(combined),
            "examples_sha256": _hash_file(combined),
            "epochs": max(1, int(epochs)),
            "output_path": str(
                root / "models" / "lumen-structure-student.npz"
            ),
            "source_files": [str(path) for path in paths],
            "source_sha256": {
                str(path): _hash_file(path) for path in paths
            },
            "source_scope": provenance.get("scope"),
            "teacher_run_ids": provenance.get("teacher_run_ids", []),
            "split_counts": split_counts,
            "split_group_counts": {
                **split_group_counts,
            },
            "label_balance": label_balance,
            "teacher_merge": merge_report,
            "require_activation_gate": not bool(example_paths),
        },
        priority=50,
    )
    return {
        "job_id": job_id,
        "examples_path": str(combined),
        "source_files": len(paths),
        "examples": rows,
        "split_counts": split_counts,
        "split_group_counts": {
            **split_group_counts,
        },
        "label_balance": label_balance,
        "teacher_merge": merge_report,
        "teacher_run_ids": provenance.get("teacher_run_ids", []),
    }


def trusted_student_examples(
    store: SongMemoryStore,
    *,
    research_root: str | Path,
) -> dict[str, Any]:
    """Resolve only example files proven by completed runs in this database."""

    root = Path(research_root).resolve()
    examples_root = (root / "exports" / "student-examples").resolve()
    paths: list[str] = []
    run_ids: list[str] = []
    errors: list[dict[str, str]] = []
    excluded_runs: list[dict[str, Any]] = []
    excluded_examples = 0
    total_examples = 0
    label_balance: dict[str, dict[str, int]] = {
        axis: {} for axis in ("functional", "energy", "content")
    }
    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_groups: dict[str, set[str]] = {
        "train": set(), "validation": set(), "test": set()
    }
    seen: set[Path] = set()
    completed_runs = store.list_teacher_runs(status="complete")
    for run in completed_runs:
        summary = (run.get("metrics") or {}).get("student_examples") or {}
        raw_path = summary.get("path")
        if not raw_path or int(summary.get("examples") or 0) <= 0:
            continue
        supervision = _recording_structure_supervision(
            store,
            recording_id=run.get("recording_id"),
            capture_session_id=run.get("capture_session_id"),
            declared=(run.get("metrics") or {}).get(
                "structure_supervision"
            ),
        )
        if not supervision["eligible"]:
            count = int(summary.get("examples") or 0)
            excluded_examples += count
            excluded_runs.append(
                {
                    "teacher_run_id": str(run["id"]),
                    "recording_id": run.get("recording_id"),
                    "examples": count,
                    "reason": (
                        "recording_incomplete_for_structure_supervision"
                    ),
                    "structure_supervision": supervision,
                }
            )
            continue
        path = Path(str(raw_path)).resolve()
        reason: str | None = None
        if not path.is_relative_to(examples_root):
            reason = "example file is outside the research export directory"
        elif not path.is_file():
            reason = "example file is missing"
        elif summary.get("sha256") and _hash_file(path) != summary["sha256"]:
            reason = "example file checksum does not match its teacher run"
        if reason is not None:
            errors.append({"teacher_run_id": str(run["id"]), "error": reason})
            continue
        if path in seen:
            continue
        valid_rows = 0
        file_label_balance: dict[str, dict[str, int]] = {
            axis: {} for axis in label_balance
        }
        file_split_counts = {split: 0 for split in split_counts}
        file_split_groups: dict[str, set[str]] = {
            split: set() for split in split_groups
        }
        try:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("teacher_run_id") != run["id"]:
                        raise ValueError(
                            "row teacher_run_id does not match database run"
                        )
                    if (
                        run.get("recording_id")
                        and row.get("recording_id") != run["recording_id"]
                    ):
                        raise ValueError(
                            "row recording_id does not match database run"
                        )
                    if (
                        run.get("capture_session_id")
                        and row.get("capture_session_id")
                        != run["capture_session_id"]
                    ):
                        raise ValueError(
                            "row capture_session_id does not match database run"
                        )
                    valid_rows += 1
                    split = str(row.get("split", "train"))
                    if split not in split_counts:
                        raise ValueError("row split is not recognized")
                    split_group_id = str(
                        row.get("split_group_id")
                        or row.get("recording_id")
                        or ""
                    )
                    if not split_group_id:
                        raise ValueError("row split group is missing")
                    if split != _recording_split(split_group_id):
                        raise ValueError(
                            "row split does not match its stable song group"
                        )
                    file_split_counts[split] += 1
                    file_split_groups[split].add(split_group_id)
                    for axis in label_balance:
                        label = row.get(axis)
                        if label is not None and str(label) != "unknown":
                            key = str(label)
                            file_label_balance[axis][key] = (
                                file_label_balance[axis].get(key, 0) + 1
                            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(
                {"teacher_run_id": str(run["id"]), "error": str(error)}
            )
            continue
        if valid_rows != int(summary.get("examples") or 0):
            errors.append(
                {
                    "teacher_run_id": str(run["id"]),
                    "error": "example row count does not match teacher metrics",
                }
            )
            continue
        seen.add(path)
        paths.append(str(path))
        run_ids.append(str(run["id"]))
        total_examples += valid_rows
        for split in split_counts:
            split_counts[split] += file_split_counts[split]
            split_groups[split].update(file_split_groups[split])
        for axis in label_balance:
            for label, count in file_label_balance[axis].items():
                label_balance[axis][label] = (
                    label_balance[axis].get(label, 0) + count
                )
    merged_rows: list[dict[str, Any]] = []
    merge_report: dict[str, Any] = {
        "source_rows": 0,
        "merged_rows": 0,
        "duplicates_collapsed": 0,
        "axis_conflicts": {},
    }
    if paths:
        try:
            source_rows = [
                row for path in paths for row in _load_jsonl(Path(path))
            ]
            merged_rows, merge_report = _merge_teacher_example_rows(
                source_rows
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(
                {"teacher_run_id": "merged_teacher_examples", "error": str(error)}
            )
            paths = []
            run_ids = []
            merged_rows = []
    merged_statistics = _student_example_statistics(merged_rows)
    return {
        "scope": "active_database_completed_teacher_runs",
        "paths": paths,
        "teacher_run_ids": run_ids,
        "completed_teacher_runs": len(completed_runs),
        "usable_teacher_runs": len(run_ids),
        "examples": len(merged_rows),
        "raw_examples": total_examples,
        "label_balance": merged_statistics["label_balance"],
        "split_counts": merged_statistics["split_counts"],
        "split_group_counts": merged_statistics["split_group_counts"],
        "teacher_merge": merge_report,
        "errors": errors,
        "excluded_teacher_runs": excluded_runs,
        "excluded_examples": excluded_examples,
    }


def training_readiness(
    store: SongMemoryStore,
    *,
    research_root: str | Path,
) -> dict[str, Any]:
    """Describe exactly what remains before a student can be trained safely."""

    trusted = trusted_student_examples(store, research_root=research_root)
    jobs = store.list_analysis_jobs(limit=100_000)
    teacher_types = {EDMFORMER_JOB, SONGFORMER_JOB}
    teacher_jobs = [job for job in jobs if job["job_type"] in teacher_types]
    job_counts: dict[str, dict[str, int]] = {
        job_type: {status: 0 for status in ("queued", "running", "complete", "failed")}
        for job_type in teacher_types
    }
    captured_recording_ids: set[str] = set()
    eligible_recording_ids: set[str] = set()
    partial_recording_ids: set[str] = set()
    unknown_recording_ids: set[str] = set()
    recording_job_statuses: dict[str, dict[str, set[str]]] = {}
    capture_inventory: dict[str, list[dict[str, Any]]] = {}
    for span in store.capture_track_spans():
        recording_id = span.get("recording_id")
        inventory_id = (
            str(recording_id)
            if recording_id
            else (
                f"capture:{span['capture_session_id']}:"
                f"{span['start_audio_frame']}"
            )
        )
        capture_inventory.setdefault(inventory_id, []).append(
            _span_structure_supervision(span)
        )
    for recording_id, supervision_rows in capture_inventory.items():
        captured_recording_ids.add(recording_id)
        if any(item["eligible"] for item in supervision_rows):
            eligible_recording_ids.add(recording_id)
        elif any(
            item["classification"] == "partial"
            for item in supervision_rows
        ):
            partial_recording_ids.add(recording_id)
        else:
            unknown_recording_ids.add(recording_id)
    inventory_from_captures = bool(capture_inventory)
    eligible_teacher_jobs = 0
    eligible_jobs_complete = 0
    eligible_jobs_remaining = 0
    excluded_teacher_jobs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    elapsed: list[float] = []
    for job in teacher_jobs:
        job_type = str(job["job_type"])
        status = str(job["status"])
        job_counts[job_type].setdefault(status, 0)
        job_counts[job_type][status] += 1
        recording_id = job.get("payload", {}).get("recording_id")
        if recording_id:
            recording_key = str(recording_id)
            if not inventory_from_captures:
                captured_recording_ids.add(recording_key)
            recording_job_statuses.setdefault(recording_key, {}).setdefault(
                job_type, set()
            ).add(status)
        supervision = _job_structure_supervision(store, job)
        if not supervision["eligible"]:
            if (
                not inventory_from_captures
                and recording_id
                and str(recording_id) not in eligible_recording_ids
            ):
                if supervision["classification"] == "unknown":
                    unknown_recording_ids.add(str(recording_id))
                else:
                    partial_recording_ids.add(str(recording_id))
            excluded_teacher_jobs.append(
                {
                    "job_id": str(job["id"]),
                    "job_type": job_type,
                    "status": status,
                    "recording_id": job.get("payload", {}).get(
                        "recording_id"
                    ),
                    "structure_supervision": supervision,
                }
            )
            continue
        eligible_teacher_jobs += 1
        if recording_id and not inventory_from_captures:
            recording_key = str(recording_id)
            eligible_recording_ids.add(recording_key)
            partial_recording_ids.discard(recording_key)
            unknown_recording_ids.discard(recording_key)
        if status == "complete":
            eligible_jobs_complete += 1
        elif status in {"queued", "running"}:
            eligible_jobs_remaining += 1
        if status == "failed":
            failures.append(
                {
                    "job_type": job_type,
                    "recording_id": str(recording_id or "unknown"),
                    "error": str(job.get("error") or "teacher failed"),
                }
            )
        result = job.get("result") or {}
        if status == "complete" and result.get("elapsed_s") is not None:
            elapsed.append(float(result["elapsed_s"]))
    queued = sum(
        counts.get("queued", 0) + counts.get("running", 0)
        for counts in job_counts.values()
    )
    completed = sum(counts.get("complete", 0) for counts in job_counts.values())
    total = len(teacher_jobs)
    completed_recordings = {
        recording_id
        for recording_id in eligible_recording_ids
        if all(
            "complete" in recording_job_statuses.get(recording_id, {}).get(
                job_type, set()
            )
            for job_type in teacher_types
        )
    }
    held_out = (
        trusted["split_counts"]["validation"]
        + trusted["split_counts"]["test"]
    )
    blockers: list[str] = []
    if trusted["examples"] == 0:
        blockers.append("no completed teacher run has produced usable examples")
    if held_out == 0:
        blockers.append("no held-out song is available for validation")
    if trusted["split_group_counts"]["train"] < 2:
        blockers.append(
            "at least two complete training songs are required"
        )
    average_elapsed = sum(elapsed) / len(elapsed) if elapsed else None
    model_root = Path(research_root).resolve() / "models"
    active_model = model_root / "lumen-structure-student.npz"
    candidate_model = model_root / "lumen-structure-student.candidate.npz"
    evaluation_path = model_root / "lumen-structure-student.evaluation.json"
    evaluation: dict[str, Any] | None = None
    if evaluation_path.is_file():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evaluation = {"error": "saved evaluation report is unreadable"}
    student_jobs = [
        job
        for job in jobs
        if job["job_type"] == STUDENT_TRAIN_JOB
        and job["status"] == "complete"
    ]
    latest_student_job = student_jobs[0] if student_jobs else None
    trusted_run_ids = set(trusted["teacher_run_ids"])
    trained_run_ids = set(
        (latest_student_job or {}).get("payload", {}).get(
            "teacher_run_ids", []
        )
    )
    candidate_provenance_current = bool(latest_student_job) and bool(
        trained_run_ids
    ) and trained_run_ids.issubset(trusted_run_ids)
    return {
        # These explicit counts keep the operator display honest.  Partial and
        # unidentified captures remain useful local data, but they are not
        # silently included in the denominator for whole-song teachers.
        "recordings_captured": len(captured_recording_ids),
        "recordings_eligible": len(eligible_recording_ids),
        "recordings_partial": len(partial_recording_ids),
        "recordings_unknown": len(unknown_recording_ids),
        "recordings_planned": len(eligible_recording_ids),
        "recordings_processed": len(completed_recordings),
        "teacher_jobs": job_counts,
        "teacher_jobs_total": total,
        "teacher_jobs_complete": completed,
        "teacher_jobs_remaining": queued,
        "eligible_teacher_jobs": eligible_teacher_jobs,
        "eligible_teacher_jobs_complete": eligible_jobs_complete,
        "eligible_teacher_jobs_remaining": eligible_jobs_remaining,
        "excluded_teacher_jobs": excluded_teacher_jobs,
        "collection_complete": eligible_jobs_remaining == 0,
        "progress": (
            eligible_jobs_complete / eligible_teacher_jobs
            if eligible_teacher_jobs
            else 0.0
        ),
        "estimated_remaining_seconds": (
            average_elapsed * eligible_jobs_remaining
            if average_elapsed is not None
            else None
        ),
        "usable_examples": trusted["examples"],
        "usable_teacher_runs": trusted["usable_teacher_runs"],
        "split_counts": trusted["split_counts"],
        "split_group_counts": trusted["split_group_counts"],
        "label_balance": trusted["label_balance"],
        "provenance_errors": trusted["errors"],
        "excluded_teacher_runs": trusted["excluded_teacher_runs"],
        "excluded_examples": trusted["excluded_examples"],
        "teacher_errors": failures,
        "train_ready": not blockers,
        "blockers": blockers,
        "model": {
            "active": active_model.is_file(),
            "active_path": str(active_model),
            "candidate": candidate_model.is_file(),
            "candidate_path": str(candidate_model),
            "evaluation": evaluation,
            "candidate_provenance_current": candidate_provenance_current,
            "candidate_teacher_run_ids": sorted(trained_run_ids),
            "trusted_teacher_run_ids": sorted(trusted_run_ids),
        },
    }


def _normalize_teacher_segments(
    raw_segments: Any,
    *,
    source: str,
    source_version: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{source} returned no structure segments")
    normalized: list[dict[str, Any]] = []
    previous_end_ms: int | None = None
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ValueError(f"{source} segment {index} is not an object")
        label = normalize_structure_label(str(item.get("label") or "unknown"))
        start_s = float(item["start"])
        end_s = float(item["end"])
        confidence_value = float(item.get("confidence", 0.72))
        if not all(
            math.isfinite(value)
            for value in (start_s, end_s, confidence_value)
        ):
            raise ValueError(f"{source} returned non-finite segment data")
        start_ms = round(start_s * 1000)
        end_ms = round(end_s * 1000)
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"{source} returned an invalid segment range")
        if previous_end_ms is not None:
            discontinuity_ms = start_ms - previous_end_ms
            if abs(discontinuity_ms) > 2:
                problem = "gap" if discontinuity_ms > 0 else "overlap"
                raise ValueError(
                    f"{source} returned a {problem} between segments "
                    f"{index - 1} and {index}"
                )
            start_ms = previous_end_ms
            if end_ms <= start_ms:
                raise ValueError(f"{source} returned an invalid segment range")
        confidence = max(0.0, min(1.0, confidence_value))
        normalized.append(
            {
                "segment_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "functional_label": label.functional.value,
                "energy_label": label.energy.value,
                "content_label": label.content.value,
                "boundary_confidence": confidence,
                "label_confidence": confidence,
                "raw_label": label.raw,
                "provenance": {
                    "source": source,
                    "source_version": source_version,
                    "annotation_type": "teacher_prediction",
                    "confidence": confidence,
                },
            }
        )
        previous_end_ms = end_ms
    return normalized


def _validate_teacher_coverage(
    segments: list[dict[str, Any]],
    *,
    source: str,
    duration_ms: int,
) -> None:
    """Reject timelines that silently describe only part of the queued WAV."""
    if duration_ms <= 0:
        raise ValueError("teacher input duration must be positive")
    tolerance_ms = max(250, round(duration_ms * 0.005))
    first_ms = int(segments[0]["start_ms"])
    last_ms = int(segments[-1]["end_ms"])
    if first_ms > tolerance_ms:
        raise ValueError(
            f"{source} timeline starts {first_ms} ms into its input"
        )
    if abs(last_ms - duration_ms) > tolerance_ms:
        raise ValueError(
            f"{source} timeline ends at {last_ms} ms for a "
            f"{duration_ms} ms input"
        )


def build_student_examples(
    store: SongMemoryStore,
    *,
    research_root: str | Path,
    recording_id: str,
    timeline_id: str,
) -> dict[str, Any]:
    """Align causal 10 Hz Lumen frames with a teacher's normalized timeline."""
    timeline = store.structure_timeline(timeline_id)
    if timeline is None:
        raise ValueError(f"unknown structure timeline {timeline_id}")
    if timeline.get("recording_id") != recording_id:
        raise ValueError("timeline and recording identity do not match")
    segments = timeline["segments"]
    spans = store.capture_spans_for_recording(recording_id)
    timeline_capture_session = timeline.get("capture_session_id")
    if timeline_capture_session is not None:
        spans = [
            span
            for span in spans
            if str(span.get("capture_session_id"))
            == str(timeline_capture_session)
        ]
    if not spans:
        raise ValueError(
            "teacher timeline has no matching capture span in this database"
        )
    examples: list[dict[str, Any]] = []
    # Empty/example-less teacher results retain the legacy deterministic value
    # in their summary. Any real capture span replaces this with the stable
    # song/provider group used by every emitted row.
    split = _recording_split(recording_id)
    for span in spans:
        structure_supervision = _span_structure_supervision(span)
        if not structure_supervision["eligible"]:
            raise ValueError(
                "partial or unknown recording cannot produce whole-song "
                "structure supervision"
            )
        start_frame = int(span["start_audio_frame"])
        end_frame = span.get("end_audio_frame")
        end_frame = int(end_frame) if end_frame is not None else None
        sample_rate = int(span["sample_rate"])
        start_position_ms = span.get("start_position_ms")
        split_group_id = _capture_split_group(span)
        split = _recording_split(split_group_id)
        stored_split = span.get("metadata", {}).get("split")
        if stored_split is not None and str(stored_split) != split:
            raise ValueError("capture span split metadata is inconsistent")
        for frame in store.training_frames(
            str(span["capture_session_id"])
        ):
            frame_index = int(frame["audio_frame_index"])
            if frame_index < start_frame or (
                end_frame is not None and frame_index >= end_frame
            ):
                continue
            # Teacher timelines start at zero for the reconstructed capture,
            # even when Spotify reports that capture as beginning partway
            # through the original track.  Preserve the provider position for
            # provenance, but align targets on the capture-relative clock.
            recording_offset_ms = round(
                (frame_index - start_frame) * 1000 / sample_rate
            )
            position_ms = frame.get("position_ms")
            if position_ms is None:
                position_ms = (
                    int(start_position_ms) + recording_offset_ms
                    if start_position_ms is not None
                    else recording_offset_ms
                )
            target = _segment_at_position(segments, recording_offset_ms)
            if target is None:
                continue
            boundary_distance_ms = recording_offset_ms - int(target["start_ms"])
            is_boundary = int(
                int(target["start_ms"]) > 0
                and 0 <= boundary_distance_ms < 1_500
            )
            features = semantic_frame_features(frame["payload"])
            examples.append(
                {
                    "recording_id": recording_id,
                    "timeline_id": timeline_id,
                    "capture_session_id": span["capture_session_id"],
                    "audio_frame_index": frame_index,
                    "position_ms": int(position_ms),
                    "recording_offset_ms": recording_offset_ms,
                    "features": [float(value) for value in features],
                    "functional": target["functional_label"],
                    "energy": target["energy_label"],
                    "content": target["content_label"],
                    "boundary": is_boundary,
                    "milliseconds_since_boundary": max(
                        0, boundary_distance_ms
                    ),
                    "target_confidence": target["label_confidence"],
                    "target_provenance": timeline["provenance"],
                    "target_provenance_details": target["provenance"],
                    "timeline_version": timeline["timeline_version"],
                    "teacher_run_id": timeline.get("teacher_run_id"),
                    "split_group_id": split_group_id,
                    "split": split,
                    "structure_supervision": structure_supervision,
                }
            )
    destination_root = (
        Path(research_root) / "exports" / "student-examples"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    material = f"{recording_id}\x1f{timeline_id}"
    filename = hashlib.sha256(material.encode("utf-8")).hexdigest() + ".jsonl"
    destination = destination_root / filename
    partial = destination.with_suffix(".jsonl.partial")
    partial.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in examples),
        encoding="utf-8",
    )
    partial.replace(destination)
    return {
        "path": str(destination),
        "examples": len(examples),
        "split": split,
        "sha256": _hash_file(destination),
    }


def _mean_confidence(segments: list[dict[str, Any]]) -> float:
    return sum(float(item["label_confidence"]) for item in segments) / len(
        segments
    )


def _segment_at_position(
    segments: list[dict[str, Any]], position_ms: int
) -> dict[str, Any] | None:
    for segment in segments:
        end_ms = segment.get("end_ms")
        if int(segment["start_ms"]) <= position_ms and (
            end_ms is None or position_ms < int(end_ms)
        ):
            return segment
    return None


def _recording_split(recording_id: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(recording_id.encode("utf-8")).digest()[:2], "big"
    ) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _capture_split_group(span: dict[str, Any]) -> str:
    """Resolve a song-stable split group, never an analog PCM identity."""
    metadata = span.get("metadata")
    if isinstance(metadata, dict) and metadata.get("split_group_id"):
        return str(metadata["split_group_id"])
    provider = span.get("recording_provider")
    provider_item_id = span.get("recording_provider_item_id")
    if provider and provider_item_id and provider != "lumen-capture":
        return f"{provider}:{provider_item_id}"
    return f"unidentified-session:{span['capture_session_id']}"


def _span_structure_supervision(span: dict[str, Any]) -> dict[str, Any]:
    """Classify current and pre-classification capture-span records."""

    metadata = span.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    stored = metadata.get("structure_supervision")
    if isinstance(stored, dict) and isinstance(stored.get("eligible"), bool):
        return stored
    recording_metadata = span.get("recording_metadata")
    recording_metadata = (
        recording_metadata if isinstance(recording_metadata, dict) else {}
    )
    track_identity = recording_metadata.get("track_identity")
    track_identity = track_identity if isinstance(track_identity, dict) else {}
    start_frame = int(span.get("start_audio_frame") or 0)
    end_frame = span.get("end_audio_frame")
    sample_rate = int(span.get("sample_rate") or 0)
    captured_duration_ms = (
        round((int(end_frame) - start_frame) * 1000 / sample_rate)
        if end_frame is not None
        and sample_rate > 0
        and int(end_frame) >= start_frame
        else 0
    )
    source_audio_complete = bool(
        span.get("capture_session_status") == "complete"
        and int(span.get("frames_received") or 0)
        == int(span.get("frames_written") or 0)
        and int(span.get("dropped_frames") or 0) == 0
    )
    return structure_supervision_completeness(
        track_duration_ms=track_identity.get("duration_ms"),
        start_position_ms=span.get("start_position_ms"),
        end_position_ms=span.get("end_position_ms"),
        captured_duration_ms=captured_duration_ms,
        source_audio_complete=source_audio_complete,
    )


def _recording_structure_supervision(
    store: SongMemoryStore,
    *,
    recording_id: str | None,
    capture_session_id: str | None,
    declared: object = None,
) -> dict[str, Any]:
    if isinstance(declared, dict) and isinstance(
        declared.get("eligible"), bool
    ):
        return declared
    if not recording_id:
        return {
            "eligible": False,
            "classification": "unknown",
            "reason_codes": ["recording_identity_unknown"],
            "evidence": {},
        }
    spans = store.capture_spans_for_recording(str(recording_id))
    if capture_session_id is not None:
        spans = [
            span
            for span in spans
            if str(span.get("capture_session_id"))
            == str(capture_session_id)
        ]
    if not spans:
        return {
            "eligible": False,
            "classification": "unknown",
            "reason_codes": ["capture_span_not_found"],
            "evidence": {
                "recording_id": recording_id,
                "capture_session_id": capture_session_id,
            },
        }
    statuses = [_span_structure_supervision(span) for span in spans]
    eligible = [status for status in statuses if status["eligible"]]
    if eligible:
        return eligible[0]
    # A recording/content identity should normally resolve to one span. If it
    # does not, fail closed and expose every classification for diagnosis.
    if len(statuses) == 1:
        return statuses[0]
    return {
        "eligible": False,
        "classification": "partial",
        "reason_codes": ["no_complete_capture_span"],
        "evidence": {"span_classifications": statuses},
    }


def _job_structure_supervision(
    store: SongMemoryStore, job: dict[str, Any]
) -> dict[str, Any]:
    payload = job.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    return _recording_structure_supervision(
        store,
        recording_id=payload.get("recording_id"),
        capture_session_id=payload.get("capture_session_id"),
        declared=payload.get("structure_supervision"),
    )


def _hash_wav_pcm(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as input_file:
        if (
            input_file.getframerate() != sample_rate
            or input_file.getnchannels() != channels
            or input_file.getsampwidth() != sample_width
        ):
            raise ValueError(f"cached teacher WAV metadata changed: {path}")
        total_frames = input_file.getnframes()
        while True:
            pcm = input_file.readframes(65_536)
            if not pcm:
                break
            digest.update(pcm)
    return digest.hexdigest(), total_frames


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
