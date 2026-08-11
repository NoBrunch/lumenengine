"""Offline teacher orchestration for Lumen's captured audio.

This module is deliberately dependency-light.  It prepares coherent WAV files,
queues work in SQLite, and invokes heavyweight teachers in their own isolated
Python environments.  Nothing here runs in the audio or DMX timing thread.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
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

from lumen_engine.datasets import (
    normalize_structure_label,
    normalize_techno_structure_label,
)
from lumen_engine.audio import RealtimeAudioAnalyzer
from lumen_engine.memory import (
    EDMFORMER_PREPROCESSING_VERSION,
    SongMemoryStore,
    TEACHER_NORMALIZATION_VERSION,
    current_songformer_preprocessing,
)
from lumen_engine.student import (
    LABELS,
    StreamingStructureStudent,
    semantic_frame_features,
)
from lumen_engine.training import structure_supervision_completeness
from lumen_engine.structure import (
    CANONICAL_TECHNO_SECTIONS,
    TransitionEvent,
    transition_event_for,
)


EDMFORMER_JOB = "teacher.edmformer"
SONGFORMER_JOB = "teacher.songformer"
STUDENT_TRAIN_JOB = "student.train"
MIN_TEACHER_DURATION_MS = 10_000
DEFAULT_OFFLINE_MAX_RSS_GIB = 5.5
APPLIANCE_OFFLINE_MEMORY_FILE = Path(
    "/etc/lumen-appliance/offline-memory-gib"
)
STUDENT_AUDIO_FEATURE_VERSION = "realtime_audio_analyzer_causal_10hz_v2"
STUDENT_EXAMPLE_VERSION = "teacher_timeline_examples_v4"
STUDENT_ACTIVATION_GATE_VERSION = "heldout_song_event_gate_v3"
TEACHER_FUSION_VERSION = "edmformer_energy_songformer_form_v1"
MIN_CLASSIFIER_BASELINE_MARGIN = 0.005
MIN_BALANCED_ACCURACY_MARGIN = 0.05
MIN_ACTIVATION_TEST_GROUPS = 5
BOUNDARY_TARGET_WINDOW_MS = 1_500
COMBINED_STUDENT_AXES = frozenset(
    {"functional", "energy", "content", "boundary"}
)


class OfflineJobCancelled(RuntimeError):
    """A requested cancellation that leaves durable work retryable."""


class OfflineMemoryLimitExceeded(RuntimeError):
    """A teacher was stopped before it could exhaust the host."""


def _offline_memory_limit_bytes() -> int:
    raw = os.environ.get("LUMEN_OFFLINE_MAX_RSS_GIB", "").strip()
    if not raw:
        try:
            raw = APPLIANCE_OFFLINE_MEMORY_FILE.read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            raw = ""
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
        retained_without_audio: list[dict[str, Any]] = []
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
                requested_teachers = []
                if queue_edmformer:
                    requested_teachers.append(EDMFORMER_JOB)
                if queue_songformer:
                    requested_teachers.append(SONGFORMER_JOB)
                raw_supervision = recording.get("structure_supervision")
                raw_duration_ms = round(
                    int(recording.get("frame_count") or 0)
                    * 1000
                    / max(1, int(recording.get("sample_rate") or 1))
                )
                # A verified but incomplete/short capture remains indexed for
                # local choreography. Do not duplicate hundreds of megabytes
                # into the teacher cache when no teacher may consume it.
                if (
                    requested_teachers
                    and isinstance(raw_supervision, dict)
                    and (
                        not raw_supervision.get("eligible")
                        or raw_duration_ms < MIN_TEACHER_DURATION_MS
                    )
                ):
                    reason = (
                        "recording_too_short"
                        if raw_duration_ms < MIN_TEACHER_DURATION_MS
                        else "recording_incomplete_for_structure_supervision"
                    )
                    skipped.extend(
                        {
                            "recording_id": recording.get("recording_id"),
                            "job_type": job_type,
                            "reason": reason,
                            "duration_ms": raw_duration_ms,
                            "minimum_duration_ms": MIN_TEACHER_DURATION_MS,
                            "structure_supervision": raw_supervision,
                        }
                        for job_type in requested_teachers
                    )
                    retained_without_audio.append(
                        {
                            "recording_id": recording.get("recording_id"),
                            "duration_ms": raw_duration_ms,
                            "split_group_id": recording.get("split_group_id"),
                            "split": recording.get("split"),
                            "structure_supervision": raw_supervision,
                            "materialized": False,
                        }
                    )
                    self._inventory_retained_recording(
                        recording,
                        structure_supervision=raw_supervision,
                    )
                    continue
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
                    "teacher_normalization_version": (
                        TEACHER_NORMALIZATION_VERSION
                    ),
                }
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
            "recordings": len(prepared) + len(retained_without_audio),
            "jobs_queued": len(jobs),
            "job_ids": jobs,
            "teachers_skipped": skipped,
            "recordings_ineligible": len(retained_without_audio),
            "recordings_partial": sum(
                item.get("structure_supervision", {}).get("classification")
                == "partial"
                for item in retained_without_audio
            ),
            "recordings_unknown": sum(
                item.get("structure_supervision", {}).get("classification")
                == "unknown"
                for item in retained_without_audio
            ),
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
            ] + retained_without_audio,
        }

    def _inventory_retained_recording(
        self,
        recording: dict[str, Any],
        *,
        structure_supervision: dict[str, Any],
    ) -> None:
        """Persist an ineligible capture without duplicating its WAV audio."""

        session_id = recording.get("session_id")
        start_frame = recording.get("start_frame")
        if session_id is None or start_frame is None:
            return
        media = recording.get("track_identity")
        media = media if isinstance(media, dict) else {}
        self.store.add_capture_track_span(
            capture_session_id=str(session_id),
            start_audio_frame=int(start_frame),
            end_audio_frame=(
                int(recording["end_frame"])
                if recording.get("end_frame") is not None
                else None
            ),
            recording_id=None,
            song_id=recording.get("song_id"),
            start_position_ms=recording.get("first_position_ms"),
            end_position_ms=recording.get("last_position_ms"),
            identity_source=(
                "spotify_metadata"
                if media.get("provider_item_id")
                else "capture_identity"
            ),
            identity_confidence=0.99 if media.get("provider_item_id") else 0.5,
            metadata={
                "export_recording_id": recording.get("recording_id"),
                "split_group_id": recording.get("split_group_id"),
                "split": recording.get("split"),
                "track_identity": media or None,
                "source_audio_complete": bool(
                    recording.get("source_audio_complete", False)
                ),
                "source_gap_frames": int(
                    recording.get("source_gap_frames") or 0
                ),
                "structure_supervision": structure_supervision,
            },
        )

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
            completed_version = (job.get("result") or {}).get(
                "teacher_normalization_version"
            )
            if (
                job["job_type"] == job_type
                and job["status"] in {"queued", "running", "complete"}
                and job["payload"].get("recording_id") == recording_id
                and job["payload"].get("content_sha256") == content_sha256
            ):
                # Queued and running work executes the installed runner, not
                # the version that happened to stamp its old payload.  Treat
                # it as current-capable and validate the produced artifacts
                # only after completion.  Otherwise a contract upgrade can
                # duplicate every durable queued recording.
                if job["status"] in {"queued", "running"}:
                    if (
                        job["status"] == "queued"
                        and int(job["priority"]) != priority
                    ):
                        self.store.update_analysis_job_priority(
                            str(job["id"]), priority=priority
                        )
                    return True
                if (
                    job["payload"].get("teacher_normalization_version")
                    != TEACHER_NORMALIZATION_VERSION
                    and completed_version != TEACHER_NORMALIZATION_VERSION
                ):
                    continue
                if (
                    job_type == EDMFORMER_JOB
                    and not self._completed_edmformer_job_is_trusted(job)
                ):
                    continue
                if (
                    job_type == SONGFORMER_JOB
                    and not self._completed_songformer_job_is_trusted(job)
                ):
                    continue
                return True
        return False

    def requeue_obsolete_edmformer_jobs(self) -> dict[str, Any]:
        """Queue full-song replacements for preserved short-context results.

        Capture sessions are marked prepared after their coherent WAV has
        been materialized.  An inference-contract upgrade must therefore be
        able to reuse that verified local WAV without rebuilding the original
        segmented capture or asking the operator to record the song again.
        """

        queued: list[str] = []
        unavailable: list[dict[str, str]] = []
        for job in self.store.list_analysis_jobs(limit=100_000):
            if job.get("job_type") != EDMFORMER_JOB:
                continue
            payload = dict(job.get("payload") or {})
            recording_id = str(payload.get("recording_id") or "")
            content_sha256 = str(payload.get("content_sha256") or "")
            audio_path = Path(str(payload.get("audio_path") or ""))
            if not recording_id or not content_sha256:
                continue
            if self._already_queued(
                EDMFORMER_JOB,
                recording_id,
                content_sha256,
                priority=int(job.get("priority") or 20),
            ):
                continue
            supervision = payload.get("structure_supervision") or {}
            if not bool(supervision.get("eligible", True)):
                continue
            if not audio_path.is_file():
                unavailable.append({
                    "recording_id": recording_id,
                    "reason": "materialized_teacher_audio_missing",
                    "audio_path": str(audio_path),
                })
                continue
            replacement = dict(payload)
            replacement.pop("edmformer_window_seconds", None)
            replacement["teacher_normalization_version"] = (
                TEACHER_NORMALIZATION_VERSION
            )
            replacement["edmformer_preprocessing_version"] = (
                EDMFORMER_PREPROCESSING_VERSION
            )
            queued.append(self.store.enqueue_analysis_job(
                job_type=EDMFORMER_JOB,
                payload=replacement,
                priority=int(job.get("priority") or 20),
            ))
        return {
            "jobs_queued": len(queued),
            "job_ids": queued,
            "unavailable": unavailable,
        }

    def requeue_obsolete_songformer_jobs(self) -> dict[str, Any]:
        """Queue current functional-teacher replacements from cached WAVs."""

        queued: list[str] = []
        unavailable: list[dict[str, str]] = []
        for job in self.store.list_analysis_jobs(limit=100_000):
            if job.get("job_type") != SONGFORMER_JOB:
                continue
            payload = dict(job.get("payload") or {})
            recording_id = str(payload.get("recording_id") or "")
            content_sha256 = str(payload.get("content_sha256") or "")
            audio_path = Path(str(payload.get("audio_path") or ""))
            if not recording_id or not content_sha256:
                continue
            if self._already_queued(
                SONGFORMER_JOB,
                recording_id,
                content_sha256,
                priority=int(job.get("priority") or 10),
            ):
                continue
            supervision = payload.get("structure_supervision") or {}
            if not bool(supervision.get("eligible", True)):
                continue
            if not audio_path.is_file():
                unavailable.append({
                    "recording_id": recording_id,
                    "reason": "materialized_teacher_audio_missing",
                    "audio_path": str(audio_path),
                })
                continue
            replacement = dict(payload)
            replacement["teacher_normalization_version"] = (
                TEACHER_NORMALIZATION_VERSION
            )
            replacement["songformer_window_seconds"] = 60
            queued.append(self.store.enqueue_analysis_job(
                job_type=SONGFORMER_JOB,
                payload=replacement,
                priority=int(job.get("priority") or 10),
            ))
        return {
            "jobs_queued": len(queued),
            "job_ids": queued,
            "unavailable": unavailable,
        }

    def _completed_edmformer_job_is_trusted(
        self, job: dict[str, Any]
    ) -> bool:
        """Return whether a completed job still owns usable local artifacts."""
        return self._completed_structure_job_is_trusted(
            job,
            teacher_name=ACTIVE_TECHNO_TEACHER,
            preprocessing_matches=lambda value: (
                value == EDMFORMER_PREPROCESSING_VERSION
            ),
        )

    def _completed_songformer_job_is_trusted(
        self, job: dict[str, Any]
    ) -> bool:
        """Return whether a completed functional-teacher job is reusable."""
        return self._completed_structure_job_is_trusted(
            job,
            teacher_name=ACTIVE_FUNCTION_TEACHER,
            preprocessing_matches=current_songformer_preprocessing,
        )

    def _completed_structure_job_is_trusted(
        self,
        job: dict[str, Any],
        *,
        teacher_name: str,
        preprocessing_matches: Any,
    ) -> bool:
        """Verify a completed teacher's DB ownership and derived JSONL."""
        job_id = str(job.get("id") or "")
        payload = job.get("payload") or {}
        recording_id = str(payload.get("recording_id") or "")
        content_sha256 = str(payload.get("content_sha256") or "")
        examples_root = (
            self.research_root / "exports" / "student-examples"
        ).resolve()
        for run in self.store.list_teacher_runs(status="complete"):
            if (
                str(run.get("analysis_job_id") or "") != job_id
                or str(run.get("teacher_name") or "").casefold()
                != teacher_name
                or str(run.get("recording_id") or "") != recording_id
                or not preprocessing_matches(
                    str(run.get("preprocessing_version") or "")
                )
            ):
                continue
            metrics = run.get("metrics") or {}
            summary = metrics.get("student_examples") or {}
            timeline_id = str(metrics.get("timeline_id") or "")
            timeline = (
                self.store.structure_timeline(timeline_id)
                if timeline_id else None
            )
            if (
                timeline is None
                or str(timeline.get("recording_id") or "") != recording_id
                or str(timeline.get("teacher_run_id") or "")
                != str(run["id"])
                or str(timeline.get("timeline_version") or "")
                != TEACHER_NORMALIZATION_VERSION
                or str(
                    (timeline.get("metadata") or {}).get("content_sha256")
                    or ""
                ) != content_sha256
            ):
                continue
            try:
                supervision = _recording_structure_supervision(
                    self.store,
                    recording_id=run.get("recording_id"),
                    capture_session_id=run.get("capture_session_id"),
                    declared=metrics.get("structure_supervision"),
                )
                path = Path(str(summary["path"])).resolve()
                if (
                    not supervision["eligible"]
                    or summary.get("schema_version")
                    != STUDENT_EXAMPLE_VERSION
                    or not path.is_relative_to(examples_root)
                    or not path.is_file()
                    or _hash_file(path) != str(summary.get("sha256") or "")
                ):
                    continue
                rows = _load_jsonl(path)
                if not rows or len(rows) != int(
                    summary.get("examples") or 0
                ):
                    continue
                if all(
                    row.get("teacher_run_id") == run["id"]
                    and row.get("recording_id") == recording_id
                    and (
                        not run.get("capture_session_id")
                        or row.get("capture_session_id")
                        == run.get("capture_session_id")
                    )
                    and str(row.get("timeline_version") or "")
                    == TEACHER_NORMALIZATION_VERSION
                    and str(row.get("split") or "train")
                    == _recording_split(
                        str(
                            row.get("split_group_id")
                            or row.get("recording_id")
                            or ""
                        )
                    )
                    for row in rows
                ):
                    return True
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
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
            execution_targets=("automatic", "local"),
        )
        if job is None:
            return None
        self._active_job_id = str(job["id"])
        self._last_subprocess_metrics = {}
        if job["job_type"] == EDMFORMER_JOB:
            reused = self._reusable_edmformer_result(job)
            if reused is not None:
                self.store.update_analysis_job(
                    job["id"], status="complete", result=reused
                )
                self._active_job_id = None
                return {
                    "job_id": job["id"],
                    "job_type": job["job_type"],
                    "status": "skipped",
                    "result": reused,
                }
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

    def _reusable_edmformer_result(
        self, claimed_job: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Reuse a complete current-v3 EDMFormer result after job claim.

        Preparation-time deduplication cannot prevent a stale queued row from
        surviving an ontology upgrade or process restart. This preflight is
        deliberately strict: the candidate job, completed teacher run,
        timeline, audio identity, and checksum-verified example file must all
        agree before expensive inference is skipped.
        """

        payload = claimed_job.get("payload") or {}
        recording_id = str(payload.get("recording_id") or "")
        content_sha256 = str(payload.get("content_sha256") or "")
        if not recording_id or not content_sha256:
            return None
        completed_jobs = [
            job
            for job in self.store.list_analysis_jobs(limit=100_000)
            if job["id"] != claimed_job["id"]
            and job["job_type"] == EDMFORMER_JOB
            and job["status"] == "complete"
            and str((job.get("payload") or {}).get("recording_id") or "")
            == recording_id
            and str((job.get("payload") or {}).get("content_sha256") or "")
            == content_sha256
            and str((job.get("result") or {}).get(
                "teacher_normalization_version"
            ) or "") == TEACHER_NORMALIZATION_VERSION
            and (job.get("result") or {}).get("reason")
            != "reused_completed_edmformer"
        ]
        if not completed_jobs:
            return None
        trusted = trusted_student_examples(
            self.store, research_root=self.root
        )
        trusted_run_ids = set(trusted.get("teacher_run_ids") or ())
        runs = self.store.list_teacher_runs(status="complete")
        for completed_job in sorted(
            completed_jobs,
            key=lambda item: (
                -int(item.get("updated_unix_ms") or 0),
                str(item["id"]),
            ),
        ):
            candidate_runs = [
                run
                for run in runs
                if str(run.get("analysis_job_id") or "")
                == str(completed_job["id"])
                and str(run.get("teacher_name") or "").casefold()
                == ACTIVE_TECHNO_TEACHER
                and str(run.get("recording_id") or "") == recording_id
                and str(run.get("preprocessing_version") or "")
                == EDMFORMER_PREPROCESSING_VERSION
                and str(run["id"]) in trusted_run_ids
            ]
            for run in candidate_runs:
                metrics = dict(run.get("metrics") or {})
                timeline_id = str(metrics.get("timeline_id") or "")
                timeline = (
                    self.store.structure_timeline(timeline_id)
                    if timeline_id
                    else None
                )
                if (
                    timeline is None
                    or str(timeline.get("recording_id") or "")
                    != recording_id
                    or str(timeline.get("teacher_run_id") or "")
                    != str(run["id"])
                    or str(timeline.get("timeline_version") or "")
                    != TEACHER_NORMALIZATION_VERSION
                    or str(
                        (timeline.get("metadata") or {}).get(
                            "content_sha256"
                        )
                        or ""
                    )
                    != content_sha256
                ):
                    continue
                student_examples = metrics.get("student_examples")
                if not isinstance(student_examples, dict):
                    continue
                return {
                    "reason": "reused_completed_edmformer",
                    "teacher_normalization_version": (
                        TEACHER_NORMALIZATION_VERSION
                    ),
                    "recording_id": recording_id,
                    "content_sha256": content_sha256,
                    "timeline_id": timeline_id,
                    "teacher_run_id": str(run["id"]),
                    "student_examples": dict(student_examples),
                    "reused_from_job_id": str(completed_job["id"]),
                    "reuse_provenance": {
                        "source_job_id": str(completed_job["id"]),
                        "source_teacher_run_id": str(run["id"]),
                        "source_timeline_id": timeline_id,
                        "audio_identity": "recording_id+content_sha256",
                        "artifacts_reused_in_place": True,
                    },
                }
        return None

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
        run_id = self.store.begin_teacher_run(
            teacher_name="EDMFormer",
            teacher_version=paths["revision"],
            device="cpu",
            preprocessing_version=EDMFORMER_PREPROCESSING_VERSION,
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
                timeline_version=TEACHER_NORMALIZATION_VERSION,
                confidence=_mean_confidence(segments),
                segments=segments,
                metadata={
                    "audio_path": str(audio_path),
                    "content_sha256": payload.get("content_sha256"),
                    "raw_output": str(output_path),
                    "command": command,
                    "local_feature_chunk_seconds": 30,
                    "global_context_seconds": 420,
                    "inference_scope": "one_full_song_sequence",
                    "runner": (
                        "lumen_edmformer_full_song_multiresolution_v3_cpu_sdpa"
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
                "structure_supervision": payload.get(
                    "structure_supervision"
                ),
                "local_feature_chunk_seconds": 30,
                "global_context_seconds": 420,
                "inference_scope": "one_full_song_sequence",
                "edmformer_preprocessing_version": (
                    EDMFORMER_PREPROCESSING_VERSION
                ),
                "teacher_normalization_version": (
                    TEACHER_NORMALIZATION_VERSION
                ),
                "subprocess": dict(self._last_subprocess_metrics),
            }
            self.store.finish_teacher_run(
                run_id, status="complete", metrics=metrics
            )
            # The operator view is sparse and derived. Rebase it now that the
            # authoritative full-song boundary map is committed.
            if payload.get("song_id") is not None:
                metrics["operator_consensus"] = (
                    self.store.refresh_operator_structure_consensus(
                        song_ids={int(payload["song_id"])}
                    )
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
                f"{window_seconds}s:{TEACHER_NORMALIZATION_VERSION}"
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
                timeline_version=TEACHER_NORMALIZATION_VERSION,
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
                "teacher_normalization_version": (
                    TEACHER_NORMALIZATION_VERSION
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

    def _train_student(
        self,
        job: dict[str, Any],
        progress_callback: Any = None,
    ) -> dict[str, Any]:
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
        validation_examples = [
            row for row in examples if row.get("split") == "validation"
        ]
        if not train_examples:
            if gated:
                raise ValueError(
                    "automatic student training requires a train split"
                )
            train_examples = examples

        last_heartbeat_at = 0.0
        heartbeat_stage = "student_feature_preparation"

        def cancel_check() -> None:
            nonlocal last_heartbeat_at
            now = time.monotonic()
            # The trainer checks for cancellation every 64 frames.  Writing a
            # SQLite lease heartbeat at that same cadence turns a CPU-bound
            # training run into thousands of synchronous journal commits on
            # the target HDD.  Keep cancellation responsive in memory while
            # renewing the durable worker lease at a human-scale cadence.
            if (
                self._active_job_id is not None
                and now - last_heartbeat_at >= 5.0
            ):
                self.store.heartbeat_analysis_job(
                    self._active_job_id,
                    worker_id=self.worker_id,
                    progress={"stage": heartbeat_stage},
                )
                last_heartbeat_at = now
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise OfflineJobCancelled(
                    "offline student canceled at requested checkpoint"
                )

        feature_preprocessing: dict[str, Any] = {
            "version": "stored_semantic_frames"
        }
        if progress_callback is not None:
            progress_callback("student_feature_preparation")
        if bool(payload.get("refresh_audio_features", False)):
            feature_preprocessing = _refresh_student_audio_features(
                examples,
                jobs=self.store.list_analysis_jobs(limit=100_000),
                research_root=self.root,
                cancel_check=cancel_check,
            )
        heartbeat_stage = "student_training"
        if progress_callback is not None:
            progress_callback("student_training")
        model = StreamingStructureStudent(
            hidden_size=int(payload.get("hidden_size", 32))
        )
        training = model.train(
            train_examples,
            epochs=int(payload.get("epochs", 30)),
            learning_rate=float(payload.get("learning_rate", 0.001)),
            validation_examples=validation_examples,
            cancel_check=cancel_check,
        )
        cancel_check()
        heartbeat_stage = "student_validation"
        if progress_callback is not None:
            progress_callback("student_validation")
        evaluation = {
            split: model.evaluate(
                [row for row in examples if row.get("split", "train") == split]
            )
            for split in ("train", "validation", "test")
        }
        song_evaluation = _student_song_evaluation(
            model, examples, store=self.store
        )
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
        held_out_name = (
            "test"
            if evaluation["test"]["energy"]["examples"]
            else "validation"
        )
        held_out = evaluation[held_out_name]
        all_axes = {*LABELS.keys(), "boundary"}
        declared_applicable_axes = payload.get("applicable_axes")
        if (
            declared_applicable_axes is None
            and payload.get("source_scope")
            == "active_database_completed_teacher_runs"
        ):
            declared_applicable_axes = COMBINED_STUDENT_AXES
        applicable_axes = {
            str(axis)
            for axis in (declared_applicable_axes or all_axes)
            if str(axis) in all_axes
        }
        if not applicable_axes:
            applicable_axes = set(all_axes)
        not_applicable_axes = all_axes - applicable_axes
        gate_reasons: list[str] = []
        axis_gate_reasons: dict[str, list[str]] = {
            axis: [] for axis in all_axes
        }
        test_group_count = int(
            statistics["split_group_counts"].get("test") or 0
        )
        test_population_reliable = (
            test_group_count >= MIN_ACTIVATION_TEST_GROUPS
        )
        if gated:
            energy = held_out["energy"]
            functional = held_out["functional"]
            content = held_out["content"]
            boundary = held_out["boundary"]
            if not test_population_reliable:
                gate_reasons.append(
                    "held-out test contains "
                    f"{test_group_count} independent songs; at least "
                    f"{MIN_ACTIVATION_TEST_GROUPS} are required"
                )
            if "energy" in applicable_axes and int(
                energy["examples"] or 0
            ) < 10:
                axis_gate_reasons["energy"].append(
                    "held-out set has fewer than 10 energy frames"
                )
            elif "energy" in applicable_axes and float(
                energy.get("majority_baseline") or 0.0
            ) >= 0.999:
                axis_gate_reasons["energy"].append(
                    "held-out energy set does not contain multiple classes"
                )
            energy_floor = max(
                0.35,
                float(energy.get("majority_baseline") or 0.0)
                + MIN_CLASSIFIER_BASELINE_MARGIN,
            )
            if (
                "energy" in applicable_axes
                and not axis_gate_reasons["energy"]
                and float(energy.get("accuracy") or 0.0) < energy_floor
            ):
                axis_gate_reasons["energy"].append(
                    "held-out energy accuracy did not meet its baseline gate"
                )
            balanced_energy_floor = max(
                0.25,
                float(energy.get("balanced_baseline") or 0.0)
                + MIN_BALANCED_ACCURACY_MARGIN,
            )
            if (
                "energy" in applicable_axes
                and not axis_gate_reasons["energy"]
                and float(energy.get("balanced_accuracy") or 0.0)
                < balanced_energy_floor
            ):
                axis_gate_reasons["energy"].append(
                    "held-out energy balanced accuracy did not meet its "
                    "per-class gate"
                )
            functional_floor = max(
                0.25,
                float(functional.get("majority_baseline") or 0.0)
                + MIN_CLASSIFIER_BASELINE_MARGIN,
            )
            if "functional" in applicable_axes and int(
                functional["examples"] or 0
            ) < 10:
                axis_gate_reasons["functional"].append(
                    "held-out set has fewer than 10 functional frames"
                )
            elif "functional" in applicable_axes and (
                float(functional.get("majority_baseline") or 0.0) >= 0.999
            ):
                axis_gate_reasons["functional"].append(
                    "held-out functional set does not contain multiple "
                    "classes"
                )
            elif "functional" in applicable_axes and (
                float(functional.get("accuracy") or 0.0)
                < functional_floor
            ):
                axis_gate_reasons["functional"].append(
                    "held-out functional-section accuracy did not meet "
                    "its baseline gate"
                )
            content_floor = max(
                0.35,
                float(content.get("majority_baseline") or 0.0)
                + MIN_CLASSIFIER_BASELINE_MARGIN,
            )
            if "content" in applicable_axes and int(
                content["examples"] or 0
            ) < 10:
                axis_gate_reasons["content"].append(
                    "held-out set has fewer than 10 content frames"
                )
            elif "content" in applicable_axes and float(
                content.get("majority_baseline") or 0.0
            ) >= 0.999:
                axis_gate_reasons["content"].append(
                    "held-out content set does not contain multiple classes"
                )
            elif "content" in applicable_axes and float(
                content.get("accuracy") or 0.0
            ) < content_floor:
                axis_gate_reasons["content"].append(
                    "held-out content-role accuracy did not meet its "
                    "baseline gate"
                )
            boundary_positives = int(
                boundary.get("event_tp") or 0
            ) + int(boundary.get("event_fn") or 0)
            if "boundary" in applicable_axes and int(
                boundary.get("examples") or 0
            ) < 10:
                axis_gate_reasons["boundary"].append(
                    "held-out set has fewer than 10 boundary frames"
                )
            elif "boundary" in applicable_axes and boundary_positives < 5:
                axis_gate_reasons["boundary"].append(
                    "held-out set has fewer than 5 positive boundaries"
                )
            elif "boundary" in applicable_axes and (
                float(boundary.get("event_f1") or 0.0) < 0.20
                or float(boundary.get("event_precision") or 0.0) < 0.12
            ):
                axis_gate_reasons["boundary"].append(
                    "held-out boundary events did not meet their tolerant "
                    "precision/F1 gate"
                )
        approved_axes = {
            axis
            for axis in applicable_axes
            for reasons in (axis_gate_reasons[axis],)
            if not reasons
        }
        if gated and not test_population_reliable:
            approved_axes.clear()
        # Each semantic axis has an independent held-out gate. Energy is the
        # only head allowed to replace Live's energy-section decision, but a
        # failed energy head must not discard a proven functional/content head
        # that can still provide non-authoritative planning context. Persist
        # only the approved heads in the artifact; Live already checks this
        # set before accepting any prediction.
        if gated and not approved_axes and not gate_reasons:
            gate_reasons.append(
                "no student axis passed its held-out activation gate"
            )
        activated = not gated or bool(approved_axes)
        model.approved_axes = approved_axes if gated else set(all_axes)
        heartbeat_stage = "student_artifacts"
        if progress_callback is not None:
            progress_callback("student_artifacts")
        model.save(candidate_path)
        cancel_check()
        report = {
            "activated": activated,
            "gate_reasons": gate_reasons,
            "axis_gate_reasons": axis_gate_reasons,
            "approved_axes": sorted(model.approved_axes),
            "inactive_axes": sorted(
                applicable_axes - model.approved_axes
            ),
            "not_applicable_axes": sorted(not_applicable_axes),
            "held_out_split": held_out_name,
            "evaluation": evaluation,
            "song_evaluation": song_evaluation,
            "minimum_test_song_groups": MIN_ACTIVATION_TEST_GROUPS,
            "test_population_reliable": test_population_reliable,
            "training": training,
            "source_sha256": _hash_file(examples_path),
            "source_scope": payload.get("source_scope"),
            "teacher_normalization_version": TEACHER_NORMALIZATION_VERSION,
            "edmformer_preprocessing_version": (
                EDMFORMER_PREPROCESSING_VERSION
            ),
            "activation_gate_version": STUDENT_ACTIVATION_GATE_VERSION,
            "teacher_fusion_version": payload.get(
                "teacher_fusion_version", "legacy_single_teacher"
            ),
            "teacher_run_ids": payload.get("teacher_run_ids", []),
            "source_files": payload.get("source_files", []),
            "split_counts": statistics["split_counts"],
            "split_group_counts": statistics["split_group_counts"],
            "label_balance": statistics["label_balance"],
            "teacher_merge": payload.get("teacher_merge"),
            "operator_consensus": payload.get("operator_consensus"),
            "operator_consensus_revision": payload.get(
                "operator_consensus_revision"
            ),
            "operator_timeline_corrections": payload.get(
                "operator_timeline_corrections"
            ),
            "feature_preprocessing": feature_preprocessing,
        }
        candidate_evaluation_path = candidate_path.with_name(
            candidate_path.stem + ".evaluation.json"
        )
        candidate_evaluation_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        quarantined_active_artifact = False
        if (
            gated
            and not activated
            and candidate_path != output_path
            and output_path.is_file()
        ):
            active_evaluation_path = output_path.with_name(
                output_path.stem + ".evaluation.json"
            )
            active_gate_version = None
            active_fusion_version = None
            if active_evaluation_path.is_file():
                try:
                    active_report = json.loads(
                        active_evaluation_path.read_text(encoding="utf-8")
                    )
                    active_gate_version = active_report.get(
                        "activation_gate_version"
                    )
                    active_fusion_version = active_report.get(
                        "teacher_fusion_version"
                    )
                except (OSError, ValueError, TypeError):
                    active_gate_version = None
                    active_fusion_version = None
            if (
                active_gate_version != STUDENT_ACTIVATION_GATE_VERSION
                or active_fusion_version != TEACHER_FUSION_VERSION
            ):
                # Preserve the old artifact for diagnosis, but do not let an
                # approval made by an obsolete gate continue controlling
                # Live. The rejected candidate already carries an empty
                # approved-axis set, so it is a safe fallback artifact.
                gate_slug = STUDENT_ACTIVATION_GATE_VERSION.replace("/", "-")
                shutil.copyfile(
                    output_path,
                    output_path.with_name(
                        f"{output_path.stem}.pre-{gate_slug}.npz"
                    ),
                )
                if active_evaluation_path.is_file():
                    shutil.copyfile(
                        active_evaluation_path,
                        active_evaluation_path.with_name(
                            f"{output_path.stem}.pre-{gate_slug}.evaluation.json"
                        ),
                    )
                activation_partial = output_path.with_suffix(
                    output_path.suffix + ".activation"
                )
                shutil.copyfile(candidate_path, activation_partial)
                activation_partial.replace(output_path)
                evaluation_partial = active_evaluation_path.with_suffix(
                    active_evaluation_path.suffix + ".activation"
                )
                shutil.copyfile(candidate_evaluation_path, evaluation_partial)
                evaluation_partial.replace(active_evaluation_path)
                quarantined_active_artifact = True
        if activated and candidate_path != output_path:
            cancel_check()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            active_evaluation_path = output_path.with_name(
                output_path.stem + ".evaluation.json"
            )
            if output_path.is_file():
                shutil.copyfile(
                    output_path,
                    output_path.with_name(output_path.stem + ".previous.npz"),
                )
            if active_evaluation_path.is_file():
                shutil.copyfile(
                    active_evaluation_path,
                    active_evaluation_path.with_name(
                        output_path.stem + ".previous.evaluation.json"
                    ),
                )
            activation_partial = output_path.with_suffix(
                output_path.suffix + ".activation"
            )
            shutil.copyfile(candidate_path, activation_partial)
            activation_partial.replace(output_path)
            evaluation_partial = active_evaluation_path.with_suffix(
                active_evaluation_path.suffix + ".activation"
            )
            evaluation_partial.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            evaluation_partial.replace(active_evaluation_path)
        return {
            "model_path": str(output_path),
            "candidate_model_path": str(candidate_path),
            "evaluation_path": str(candidate_evaluation_path),
            "activated": activated,
            "quarantined_active_artifact": quarantined_active_artifact,
            "activation_gate_reasons": gate_reasons,
            "axis_gate_reasons": axis_gate_reasons,
            "approved_axes": sorted(model.approved_axes),
            "inactive_axes": sorted(
                applicable_axes - model.approved_axes
            ),
            "not_applicable_axes": sorted(not_applicable_axes),
            "held_out_split": held_out_name,
            "training": training,
            "evaluation": evaluation,
            "song_evaluation": song_evaluation,
            "minimum_test_song_groups": MIN_ACTIVATION_TEST_GROUPS,
            "test_population_reliable": test_population_reliable,
            "split_counts": statistics["split_counts"],
            "split_group_counts": statistics["split_group_counts"],
            "label_balance": statistics["label_balance"],
            "teacher_merge": payload.get("teacher_merge"),
            "operator_consensus": payload.get("operator_consensus"),
            "operator_consensus_revision": payload.get(
                "operator_consensus_revision"
            ),
            "feature_preprocessing": feature_preprocessing,
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


ACTIVE_TECHNO_TEACHER = "edmformer"
ACTIVE_FUNCTION_TEACHER = "songformer"
ACTIVE_STRUCTURE_TEACHERS = frozenset(
    {ACTIVE_TECHNO_TEACHER, ACTIVE_FUNCTION_TEACHER}
)
_AXIS_TEACHER_PRIORITY = {
    "functional": {
        ACTIVE_FUNCTION_TEACHER: 30,
        ACTIVE_TECHNO_TEACHER: 10,
    },
    "energy": {
        ACTIVE_TECHNO_TEACHER: 30,
    },
    "content": {
        ACTIVE_FUNCTION_TEACHER: 30,
        ACTIVE_TECHNO_TEACHER: 10,
    },
}


def _teacher_source(row: dict[str, Any]) -> str:
    details = row.get("target_provenance_details")
    if isinstance(details, dict):
        if details.get("source"):
            return str(details["source"])
        if details.get("teacher_name"):
            return str(details["teacher_name"])
    return str(row.get("target_provenance") or "unknown")


_TEACHER_MERGE_FIELDS = frozenset({
    "recording_id", "capture_session_id", "audio_frame_index",
    "recording_offset_ms", "position_ms", "split_group_id", "split",
    "features", "functional", "energy", "content", "boundary",
    "boundary_supervised", "milliseconds_since_boundary",
    "target_confidence", "target_provenance", "target_provenance_details",
    "teacher_run_id", "timeline_id", "structure_supervision",
})

_STUDENT_TRAINING_FIELDS = frozenset({
    "recording_id", "capture_session_id", "audio_frame_index",
    "recording_offset_ms", "position_ms", "split_group_id", "split",
    "features", "functional", "energy", "content", "boundary",
    "boundary_supervised", "milliseconds_since_boundary",
})


def _compact_example_row(
    row: dict[str, Any], fields: frozenset[str]
) -> dict[str, Any]:
    """Keep only fields needed by fusion or numerical student training."""

    return {name: row[name] for name in fields if name in row}


def _authoritative_structure_row(row: dict[str, Any]) -> bool:
    source = _teacher_source(row).casefold()
    return any(teacher in source for teacher in ACTIVE_STRUCTURE_TEACHERS)


def _merge_teacher_example_rows(
    rows: list[dict[str, Any]],
    *,
    authoritative_only: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse axis-specific teacher targets per captured audio frame.

    EDMFormer has precedence for techno energy; SongFormer has precedence for
    functional/content form. Boundaries are the union of either current
    teacher, while explicit developer paths may opt out of authority filtering.
    """

    source_count = len(rows)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("student example row is not an object")
    authoritative_rows = (
        [row for row in rows if _authoritative_structure_row(row)]
        if authoritative_only
        else list(rows)
    )
    excluded_non_authoritative = source_count - len(authoritative_rows)
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for order, row in enumerate(authoritative_rows):
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

        # Early exports predate the per-row supervision snapshot. Duplicate
        # EDMFormer exports may therefore differ only by the presence of the
        # verified recording-level value.
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
                if source.casefold() not in source_priority:
                    continue
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

        boundary_rows = [
            row
            for row in ordered
            if bool(row.get("boundary_supervised", "boundary" in row))
            and "boundary" in row
        ]
        if boundary_rows:
            base["boundary"] = max(
                int(float(row.get("boundary") or 0) >= 0.5)
                for row in boundary_rows
            )
            base["boundary_supervised"] = True
        else:
            base.pop("boundary", None)
            base["boundary_supervised"] = False
        boundary_distances = [
            int(row.get("milliseconds_since_boundary") or 0)
            for row in boundary_rows
            if int(float(row.get("boundary") or 0) >= 0.5)
        ]
        if boundary_rows:
            base["milliseconds_since_boundary"] = (
                min(boundary_distances) if boundary_distances else 0
            )
        else:
            base.pop("milliseconds_since_boundary", None)
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
        "source_rows": source_count,
        "authoritative_source_rows": len(authoritative_rows),
        "excluded_non_authoritative_rows": excluded_non_authoritative,
        "merged_rows": len(merged),
        "duplicates_collapsed": len(authoritative_rows) - len(merged),
        "axis_conflicts": axis_conflicts,
        "axis_precedence": _AXIS_TEACHER_PRIORITY,
        "boundary_merge": "maximum",
        "active_teacher_authority": (
            "axis_specific_edmformer_songformer"
            if authoritative_only else "explicit_paths"
        ),
    }


def _apply_operator_consensus_rows(
    store: SongMemoryStore,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Overlay sparse, participant-consensus corrections on teacher rows."""

    recording_ids = {
        str(row["recording_id"])
        for row in rows
        if row.get("recording_id")
    }
    timelines = store.operator_consensus_for_recordings(recording_ids)
    corrected_rows = 0
    corrected_axes = {"functional": 0, "energy": 0, "content": 0}
    boundary_rows = 0
    songs: set[str] = set()
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        recording_id = str(row.get("recording_id") or "")
        timeline = timelines.get(recording_id)
        position = row.get("position_ms")
        changed = False
        if timeline is not None and position is not None:
            position_ms = max(0, int(position))
            segment = next(
                (
                    item
                    for item in timeline.get("segments", ())
                    if int(item["start_ms"]) <= position_ms
                    and (
                        item.get("end_ms") is None
                        or position_ms < int(item["end_ms"])
                    )
                ),
                None,
            )
            if segment is not None:
                provenance_by_axis = dict(
                    row.get("target_provenance_by_axis") or {}
                )
                for axis in ("functional", "energy", "content"):
                    label = segment.get(f"{axis}_label")
                    if label is None or str(label) == "unknown":
                        continue
                    row[axis] = str(label)
                    provenance_by_axis[axis] = {
                        "source": "operator_annotation_consensus",
                        "timeline_id": timeline["id"],
                        "label": str(label),
                        "confidence": float(segment["label_confidence"]),
                        "evidence": segment.get("provenance") or {},
                    }
                    corrected_axes[axis] += 1
                    changed = True
                row["target_provenance_by_axis"] = provenance_by_axis
                events = (segment.get("provenance") or {}).get(
                    "transition_events"
                ) or []
                if (
                    events
                    and 0
                    <= position_ms - int(segment["start_ms"])
                    <= BOUNDARY_TARGET_WINDOW_MS
                ):
                    row["boundary"] = 1
                    row["boundary_supervised"] = True
                    row["milliseconds_since_boundary"] = (
                        position_ms - int(segment["start_ms"])
                    )
                    row["boundary_provenance"] = {
                        "source": "operator_annotation_consensus",
                        "timeline_id": timeline["id"],
                        "events": events,
                    }
                    boundary_rows += 1
                    changed = True
                if changed:
                    row["target_provenance"] = (
                        "edmformer_with_operator_consensus"
                    )
                    row["target_confidence"] = max(
                        float(row.get("target_confidence") or 0.0),
                        float(segment["label_confidence"]),
                    )
                    row["operator_consensus_revision"] = (
                        timeline.get("metadata", {}).get(
                            "consensus_revision"
                        )
                    )
                    songs.add(recording_id)
                    corrected_rows += 1
        result.append(row)
    return result, {
        "revision": store.operator_consensus_revision(),
        "timelines": len(timelines),
        "recordings_corrected": len(songs),
        "rows_corrected": corrected_rows,
        "axis_rows_corrected": corrected_axes,
        "boundary_rows_corrected": boundary_rows,
    }


def _apply_operator_timeline_corrections(
    store: SongMemoryStore,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Overlay the latest immutable desktop correction per recording."""

    correction_by_recording: dict[str, dict[str, Any]] = {}
    for recording_id in {
        str(row["recording_id"])
        for row in rows
        if row.get("recording_id")
    }:
        correction = next(
            (
                timeline
                for timeline in store.structure_timelines_for_recording(
                    recording_id
                )
                if str(timeline.get("provenance") or "").casefold()
                == "operator_correction"
            ),
            None,
        )
        if correction is not None:
            correction_by_recording[recording_id] = correction

    corrected_rows = 0
    corrected_axes = {
        axis: 0 for axis in ("functional", "energy", "content")
    }
    corrected_boundaries = 0
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        recording_id = str(row.get("recording_id") or "")
        timeline = correction_by_recording.get(recording_id)
        if timeline is None:
            result.append(row)
            continue
        position_ms = int(row.get("recording_offset_ms") or 0)
        segment = _segment_at_position(
            list(timeline.get("segments") or ()), position_ms
        )
        if segment is None:
            result.append(row)
            continue
        changed = False
        provenance_by_axis = dict(
            row.get("target_provenance_by_axis") or {}
        )
        for axis in ("functional", "energy", "content"):
            value = str(segment.get(f"{axis}_label") or "unknown")
            if value == "unknown":
                continue
            if row.get(axis) != value:
                corrected_axes[axis] += 1
                changed = True
            row[axis] = value
            provenance_by_axis[axis] = {
                "source": "operator_correction",
                "timeline_id": timeline.get("id"),
                "label": value,
                "note": (timeline.get("metadata") or {}).get("note"),
            }
        boundary_distance = abs(
            position_ms - int(segment.get("start_ms") or 0)
        )
        if boundary_distance <= BOUNDARY_TARGET_WINDOW_MS:
            if int(row.get("boundary") or 0) != 1:
                corrected_boundaries += 1
                changed = True
            row["boundary"] = 1
            row["boundary_supervised"] = True
            row["milliseconds_since_boundary"] = boundary_distance
            row["boundary_provenance"] = {
                "source": "operator_correction",
                "timeline_id": timeline.get("id"),
                "event_tolerance_ms": BOUNDARY_TARGET_WINDOW_MS,
            }
        row["target_provenance_by_axis"] = provenance_by_axis
        if changed:
            corrected_rows += 1
        result.append(row)
    return result, {
        "recordings": len(correction_by_recording),
        "rows_corrected": corrected_rows,
        "axis_rows_corrected": corrected_axes,
        "boundary_rows_corrected": corrected_boundaries,
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


def _process_teacher_example_groups(
    store: SongMemoryStore,
    path_groups: dict[str, list[Path]],
    *,
    authoritative_only: bool,
    apply_operator_overlays: bool,
    row_consumer: Any = None,
) -> dict[str, Any]:
    """Fuse one recording at a time so corpus size cannot dictate RAM use."""

    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_groups: dict[str, set[str]] = {
        split: set() for split in split_counts
    }
    group_splits: dict[str, set[str]] = {}
    label_balance: dict[str, dict[str, int]] = {
        axis: {} for axis in _AXIS_TEACHER_PRIORITY
    }
    merge_report: dict[str, Any] = {
        "source_rows": 0,
        "authoritative_source_rows": 0,
        "excluded_non_authoritative_rows": 0,
        "merged_rows": 0,
        "duplicates_collapsed": 0,
        "axis_conflicts": {
            axis: 0 for axis in _AXIS_TEACHER_PRIORITY
        },
        "axis_precedence": _AXIS_TEACHER_PRIORITY,
        "boundary_merge": "maximum",
        "active_teacher_authority": (
            "axis_specific_edmformer_songformer"
            if authoritative_only
            else "explicit_paths"
        ),
    }
    consensus_report: dict[str, Any] = {
        "revision": store.operator_consensus_revision(),
        "timelines": 0,
        "recordings_corrected": 0,
        "rows_corrected": 0,
        "axis_rows_corrected": {
            axis: 0 for axis in ("functional", "energy", "content")
        },
        "boundary_rows_corrected": 0,
    }
    correction_report: dict[str, Any] = {
        "recordings": 0,
        "rows_corrected": 0,
        "axis_rows_corrected": {
            axis: 0 for axis in ("functional", "energy", "content")
        },
        "boundary_rows_corrected": 0,
    }

    for recording_key in sorted(path_groups):
        source_rows = [
            _compact_example_row(row, _TEACHER_MERGE_FIELDS)
            for path in path_groups[recording_key]
            for row in _iter_jsonl(path)
        ]
        merged_rows, group_merge = _merge_teacher_example_rows(
            source_rows,
            authoritative_only=authoritative_only,
        )
        del source_rows
        for name in (
            "source_rows",
            "authoritative_source_rows",
            "excluded_non_authoritative_rows",
            "merged_rows",
            "duplicates_collapsed",
        ):
            merge_report[name] += int(group_merge.get(name) or 0)
        for axis, count in (group_merge.get("axis_conflicts") or {}).items():
            merge_report["axis_conflicts"][axis] = (
                merge_report["axis_conflicts"].get(axis, 0) + int(count)
            )

        if apply_operator_overlays:
            merged_rows, group_consensus = _apply_operator_consensus_rows(
                store, merged_rows
            )
            merged_rows, group_corrections = (
                _apply_operator_timeline_corrections(store, merged_rows)
            )
            for name in (
                "timelines",
                "recordings_corrected",
                "rows_corrected",
                "boundary_rows_corrected",
            ):
                consensus_report[name] += int(
                    group_consensus.get(name) or 0
                )
            for axis, count in (
                group_consensus.get("axis_rows_corrected") or {}
            ).items():
                consensus_report["axis_rows_corrected"][axis] += int(count)
            for name in (
                "recordings",
                "rows_corrected",
                "boundary_rows_corrected",
            ):
                correction_report[name] += int(
                    group_corrections.get(name) or 0
                )
            for axis, count in (
                group_corrections.get("axis_rows_corrected") or {}
            ).items():
                correction_report["axis_rows_corrected"][axis] += int(count)

        for row in merged_rows:
            split = str(row.get("split") or "train")
            if split not in split_counts:
                raise ValueError(f"unknown student dataset split {split!r}")
            group = str(
                row.get("split_group_id") or row.get("recording_id") or ""
            )
            if not group:
                raise ValueError("student example has no split group identity")
            split_counts[split] += 1
            split_groups[split].add(group)
            group_splits.setdefault(group, set()).add(split)
            for axis in label_balance:
                label = str(row.get(axis) or "unknown")
                if label != "unknown":
                    label_balance[axis][label] = (
                        label_balance[axis].get(label, 0) + 1
                    )
        if row_consumer is not None:
            row_consumer(merged_rows)
        del merged_rows

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
        "examples": int(merge_report["merged_rows"]),
        "split_counts": split_counts,
        "split_group_counts": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "label_balance": label_balance,
        "teacher_merge": merge_report,
        "operator_consensus": consensus_report,
        "operator_timeline_corrections": correction_report,
    }


def _student_song_evaluation(
    model: StreamingStructureStudent,
    rows: list[dict[str, Any]],
    *,
    store: SongMemoryStore,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate each validation/test song independently and identify it."""

    catalog = {
        str(item["recording_id"]): item
        for item in store.structure_timeline_catalog()
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "train")
        if split not in {"validation", "test"}:
            continue
        group = str(
            row.get("split_group_id") or row.get("recording_id") or ""
        )
        if not group:
            continue
        grouped.setdefault((split, group), []).append(row)
    result: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    for (split, group), group_rows in sorted(grouped.items()):
        recording_ids = sorted({
            str(row["recording_id"])
            for row in group_rows
            if row.get("recording_id")
        })
        identity = next(
            (
                catalog[recording_id]
                for recording_id in recording_ids
                if recording_id in catalog
            ),
            {},
        )
        result[split].append(
            {
                "split_group_id": group,
                "recording_ids": recording_ids,
                "title": identity.get("title") or group,
                "artists": list(identity.get("artists") or []),
                "review_status": identity.get("review_status"),
                "examples": len(group_rows),
                "metrics": model.evaluate(group_rows),
            }
        )
    return result


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
    explicit_paths = bool(paths)
    provenance: dict[str, Any]
    if not paths:
        # This is a derived materialized view. Refresh it only in the explicit
        # offline training action, never from Live or a status poll.
        store.refresh_operator_structure_consensus()
        refresh_current_student_examples(store, research_root=root)
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
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    partial = exports / f".student-training-{uuid.uuid4().hex}.partial"
    if not explicit_paths:
        path_groups = {
            str(key): [Path(path) for path in values]
            for key, values in (provenance.get("path_groups") or {}).items()
        }
        with partial.open("w", encoding="utf-8") as output:
            def write_group(group_rows: list[dict[str, Any]]) -> None:
                for row in group_rows:
                    output.write(
                        json.dumps(
                            _compact_example_row(
                                row, _STUDENT_TRAINING_FIELDS
                            ),
                            sort_keys=True,
                        )
                        + "\n"
                    )

            materialized = _process_teacher_example_groups(
                store,
                path_groups,
                authoritative_only=True,
                apply_operator_overlays=True,
                row_consumer=write_group,
            )
        rows = int(materialized["examples"])
        statistics = {
            "split_counts": materialized["split_counts"],
            "split_group_counts": materialized["split_group_counts"],
            "label_balance": materialized["label_balance"],
        }
        merge_report = materialized["teacher_merge"]
        consensus_report = materialized["operator_consensus"]
        correction_report = materialized[
            "operator_timeline_corrections"
        ]
    else:
        source_rows = [
            _compact_example_row(row, _TEACHER_MERGE_FIELDS)
            for path in paths
            for row in _iter_jsonl(path)
        ]
        merged_rows, merge_report = _merge_teacher_example_rows(
            source_rows,
            authoritative_only=False,
        )
        del source_rows
        merged_rows = [
            _compact_example_row(row, _STUDENT_TRAINING_FIELDS)
            for row in merged_rows
        ]
        rows = len(merged_rows)
        with partial.open("w", encoding="utf-8") as output:
            for row in merged_rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
        statistics = _student_example_statistics(
            merged_rows,
            require_group_identity=False,
        )
        consensus_report = {
            "revision": store.operator_consensus_revision(),
            "rows_corrected": 0,
        }
        correction_report = {"rows_corrected": 0}
        del merged_rows
    if rows == 0:
        raise RuntimeError("teacher example files contain no training rows")
    split_counts = statistics["split_counts"]
    split_group_counts = statistics["split_group_counts"]
    label_balance = statistics["label_balance"]
    held_out_examples = split_counts["validation"] + split_counts["test"]
    if not explicit_paths and held_out_examples == 0:
        raise RuntimeError(
            "teacher examples do not yet include a held-out song; process "
            "more recordings before training"
        )
    if not explicit_paths and split_group_counts["train"] < 2:
        raise RuntimeError(
            "teacher examples require at least two complete training songs"
        )
    combined_sha256 = _hash_file(partial)
    combined = exports / f"student-training-{combined_sha256}.jsonl"
    if combined.is_file():
        if _hash_file(combined) != combined_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("content-addressed student dataset is corrupt")
        partial.unlink(missing_ok=True)
    else:
        partial.replace(combined)
    # Only the newest explicit request remains runnable. Previously every job
    # referenced one mutable student-training.jsonl, so later preparation
    # silently invalidated earlier queued manifests.
    for existing in store.list_analysis_jobs(limit=10_000):
        if (
            existing.get("job_type") == STUDENT_TRAIN_JOB
            and existing.get("status") == "queued"
        ):
            store.update_analysis_job(
                str(existing["id"]),
                status="canceled",
                error="superseded by a newer trusted training snapshot",
            )
    job_id = store.enqueue_analysis_job(
        job_type=STUDENT_TRAIN_JOB,
        payload={
            "examples_path": str(combined),
            "examples_sha256": combined_sha256,
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
            "teacher_fusion_version": TEACHER_FUSION_VERSION,
            "operator_consensus": consensus_report,
            "operator_consensus_revision": consensus_report.get("revision"),
            "operator_timeline_corrections": correction_report,
            "trainer_version": StreamingStructureStudent.format_version,
            "refresh_audio_features": not explicit_paths,
            "feature_preprocessing_version": (
                STUDENT_AUDIO_FEATURE_VERSION
                if not explicit_paths
                else "stored_semantic_frames"
            ),
            "require_activation_gate": not explicit_paths,
            "applicable_axes": (
                sorted(COMBINED_STUDENT_AXES)
                if not explicit_paths
                else sorted({*LABELS.keys(), "boundary"})
            ),
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
        "operator_consensus": consensus_report,
        "operator_consensus_revision": consensus_report.get("revision"),
        "operator_timeline_corrections": correction_report,
        "teacher_run_ids": provenance.get("teacher_run_ids", []),
    }


def refresh_current_student_examples(
    store: SongMemoryStore,
    *,
    research_root: str | Path,
) -> dict[str, int]:
    """Upgrade current teacher examples without rerunning either model.

    Teacher timelines are the durable model result. Student JSONL files are a
    derived training view and can be rebuilt when that view's schema changes.
    This migration runs only when training is explicitly requested; status
    inspection remains read-only.
    """
    root = Path(research_root).resolve()
    rebuilt = 0
    current = 0
    for run in store.list_teacher_runs(status="complete"):
        teacher_name = str(run.get("teacher_name") or "").casefold()
        preprocessing_version = str(run.get("preprocessing_version") or "")
        current_preprocessing = (
            teacher_name == ACTIVE_TECHNO_TEACHER
            and preprocessing_version == EDMFORMER_PREPROCESSING_VERSION
        ) or (
            teacher_name == ACTIVE_FUNCTION_TEACHER
            and current_songformer_preprocessing(preprocessing_version)
        )
        if not current_preprocessing:
            continue
        metrics = dict(run.get("metrics") or {})
        supervision = _recording_structure_supervision(
            store,
            recording_id=run.get("recording_id"),
            capture_session_id=run.get("capture_session_id"),
            declared=metrics.get("structure_supervision"),
        )
        if not supervision["eligible"]:
            continue
        summary = dict(metrics.get("student_examples") or {})
        path = Path(str(summary.get("path") or ""))
        if (
            summary.get("schema_version") == STUDENT_EXAMPLE_VERSION
            and path.is_file()
            and summary.get("sha256") == _hash_file(path)
        ):
            current += 1
            continue
        timeline_id = str(metrics.get("timeline_id") or "")
        recording_id = str(run.get("recording_id") or "")
        if not timeline_id or not recording_id:
            raise RuntimeError(
                f"teacher run {run['id']} cannot rebuild student examples: "
                "recording or timeline identity is missing"
            )
        rebuilt_summary = build_student_examples(
            store,
            research_root=root,
            recording_id=recording_id,
            timeline_id=timeline_id,
        )
        metrics["student_examples"] = rebuilt_summary
        store.finish_teacher_run(
            str(run["id"]), status="complete", metrics=metrics
        )
        rebuilt += 1
    return {"rebuilt": rebuilt, "current": current}


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
    path_groups: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    completed_runs = store.list_teacher_runs(status="complete")
    for run in completed_runs:
        teacher_name = str(run.get("teacher_name") or "").casefold()
        summary = (run.get("metrics") or {}).get("student_examples") or {}
        if teacher_name not in ACTIVE_STRUCTURE_TEACHERS:
            count = int(summary.get("examples") or 0)
            excluded_examples += count
            excluded_runs.append(
                {
                    "teacher_run_id": str(run["id"]),
                    "recording_id": run.get("recording_id"),
                    "examples": count,
                    "reason": "non_authoritative_structure_teacher",
                    "teacher_name": run.get("teacher_name"),
                    "active_teacher_authority": (
                        "axis_specific_edmformer_songformer"
                    ),
                    "artifacts_preserved": True,
                }
            )
            continue
        timeline_id = str(
            (run.get("metrics") or {}).get("timeline_id") or ""
        )
        review = (
            store.structure_timeline_review(timeline_id)
            if timeline_id else None
        )
        if str((review or {}).get("status") or "").casefold() == "rejected":
            count = int(summary.get("examples") or 0)
            excluded_examples += count
            excluded_runs.append({
                "teacher_run_id": str(run["id"]),
                "recording_id": run.get("recording_id"),
                "timeline_id": timeline_id,
                "examples": count,
                "reason": "operator_rejected_teacher_target",
                "artifacts_preserved": True,
            })
            continue
        raw_path = summary.get("path")
        if not raw_path or int(summary.get("examples") or 0) <= 0:
            continue
        preprocessing_version = str(
            run.get("preprocessing_version") or ""
        )
        current_preprocessing = (
            teacher_name == ACTIVE_TECHNO_TEACHER
            and preprocessing_version == EDMFORMER_PREPROCESSING_VERSION
        ) or (
            teacher_name == ACTIVE_FUNCTION_TEACHER
            and current_songformer_preprocessing(preprocessing_version)
        )
        if not current_preprocessing:
            count = int(summary.get("examples") or 0)
            excluded_examples += count
            excluded_runs.append(
                {
                    "teacher_run_id": str(run["id"]),
                    "recording_id": run.get("recording_id"),
                    "examples": count,
                    "reason": "obsolete_teacher_normalization_version",
                    "preprocessing_version": preprocessing_version,
                    "required_normalization_version": TEACHER_NORMALIZATION_VERSION,
                    "required_preprocessing_version": (
                        EDMFORMER_PREPROCESSING_VERSION
                        if teacher_name == ACTIVE_TECHNO_TEACHER
                        else (
                            "songformer_official_features_cpu_windowed_v1:"
                            f"30-60s:{TEACHER_NORMALIZATION_VERSION}"
                        )
                    ),
                }
            )
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
                    if (
                        str(row.get("timeline_version") or "")
                        != TEACHER_NORMALIZATION_VERSION
                    ):
                        raise ValueError(
                            "example row uses an obsolete teacher "
                            "normalization version"
                        )
                    if teacher_name == ACTIVE_TECHNO_TEACHER:
                        energy_label = str(
                            row.get("energy") or "unknown"
                        )
                        if energy_label not in {
                            "unknown", *CANONICAL_TECHNO_SECTIONS
                        }:
                            raise ValueError(
                                "example row uses a noncanonical techno "
                                f"state: {energy_label}"
                            )
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
            if "noncanonical techno state" in str(error):
                count = int(summary.get("examples") or 0)
                excluded_examples += count
                excluded_runs.append(
                    {
                        "teacher_run_id": str(run["id"]),
                        "recording_id": run.get("recording_id"),
                        "examples": count,
                        "reason": "noncanonical_techno_ontology",
                        "required_states": list(
                            CANONICAL_TECHNO_SECTIONS
                        ),
                    }
                )
                continue
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
        path_groups.setdefault(
            str(run.get("recording_id") or path), []
        ).append(path)
        total_examples += valid_rows
        for split in split_counts:
            split_counts[split] += file_split_counts[split]
            split_groups[split].update(file_split_groups[split])
        for axis in label_balance:
            for label, count in file_label_balance[axis].items():
                label_balance[axis][label] = (
                    label_balance[axis].get(label, 0) + count
                )
    merged_summary: dict[str, Any] = {
        "examples": 0,
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "split_group_counts": {"train": 0, "validation": 0, "test": 0},
        "label_balance": {
            axis: {} for axis in _AXIS_TEACHER_PRIORITY
        },
        "teacher_merge": {
            "source_rows": 0,
            "merged_rows": 0,
            "duplicates_collapsed": 0,
            "axis_conflicts": {},
        },
        "operator_consensus": {
            "revision": store.operator_consensus_revision(),
            "rows_corrected": 0,
        },
        "operator_timeline_corrections": {"rows_corrected": 0},
    }
    if paths:
        try:
            merged_summary = _process_teacher_example_groups(
                store,
                path_groups,
                authoritative_only=True,
                apply_operator_overlays=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(
                {"teacher_run_id": "merged_teacher_examples", "error": str(error)}
            )
            paths = []
            run_ids = []
            path_groups = {}
    merge_report = merged_summary["teacher_merge"]
    consensus_report = merged_summary["operator_consensus"]
    correction_report = merged_summary["operator_timeline_corrections"]
    return {
        "scope": "active_database_completed_teacher_runs",
        "active_teacher_authority": "axis_specific_edmformer_songformer",
        "paths": paths,
        "path_groups": {
            key: [str(path) for path in values]
            for key, values in path_groups.items()
        },
        "teacher_run_ids": run_ids,
        "completed_teacher_runs": len(completed_runs),
        "preserved_non_authoritative_runs": sum(
            1
            for item in excluded_runs
            if item.get("reason") == "non_authoritative_structure_teacher"
        ),
        "usable_teacher_runs": len(run_ids),
        "examples": int(merged_summary["examples"]),
        "raw_examples": total_examples,
        "label_balance": merged_summary["label_balance"],
        "split_counts": merged_summary["split_counts"],
        "split_group_counts": merged_summary["split_group_counts"],
        "teacher_merge": merge_report,
        "operator_consensus": consensus_report,
        "operator_consensus_revision": consensus_report.get("revision"),
        "operator_timeline_corrections": correction_report,
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
    active_teacher_types = {EDMFORMER_JOB, SONGFORMER_JOB}
    teacher_jobs = [job for job in jobs if job["job_type"] in teacher_types]
    trusted_run_ids = set(trusted.get("teacher_run_ids") or ())
    trusted_teacher_job_ids = {
        str(run.get("analysis_job_id") or "")
        for run in store.list_teacher_runs(status="complete")
        if str(run.get("id") or "") in trusted_run_ids
        and run.get("analysis_job_id")
    }
    job_counts: dict[str, dict[str, int]] = {
        job_type: {status: 0 for status in ("queued", "running", "complete", "failed")}
        for job_type in teacher_types
    }
    active_job_counts = {
        job_type: {
            status: 0
            for status in ("queued", "running", "complete", "failed")
        }
        for job_type in active_teacher_types
    }
    excluded_job_reasons: dict[str, str] = {}
    active_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    current_teacher_inventory_jobs: list[dict[str, Any]] = []
    for job in teacher_jobs:
        job_id = str(job["id"])
        job_type = str(job["job_type"])
        payload = job.get("payload") or {}
        result = job.get("result") or {}
        version = str(
            result.get("teacher_normalization_version")
            or payload.get("teacher_normalization_version")
            or ""
        )
        if (
            version != TEACHER_NORMALIZATION_VERSION
            and str(job.get("status") or "") not in {"queued", "running"}
        ):
            excluded_job_reasons[job_id] = (
                "obsolete_teacher_normalization_version"
            )
            continue
        current_teacher_inventory_jobs.append(job)
        if result.get("reason") == "reused_completed_edmformer":
            excluded_job_reasons[job_id] = "reused_duplicate_edmformer"
            continue
        if (
            str(job.get("status") or "") == "complete"
            and job_id not in trusted_teacher_job_ids
        ):
            excluded_job_reasons[job_id] = (
                "completed_without_usable_current_examples"
            )
            continue
        recording_key = str(payload.get("recording_id") or "")
        content_key = str(payload.get("content_sha256") or "")
        if not recording_key or not content_key:
            recording_key = f"job:{job_id}"
            content_key = "unidentified"
        active_groups.setdefault(
            (job_type, recording_key, content_key), []
        ).append(job)
    active_job_ids: set[str] = set()
    status_rank = {"complete": 4, "running": 3, "queued": 2, "failed": 1}
    for grouped_jobs in active_groups.values():
        representative = max(
            grouped_jobs,
            key=lambda job: (
                status_rank.get(str(job.get("status") or ""), 0),
                -int(job.get("created_unix_ms") or 0),
                str(job["id"]),
            ),
        )
        active_job_ids.add(str(representative["id"]))
        for duplicate in grouped_jobs:
            duplicate_id = str(duplicate["id"])
            if duplicate_id != str(representative["id"]):
                excluded_job_reasons[duplicate_id] = (
                    "duplicate_active_teacher_work_item"
                )
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
    if not inventory_from_captures:
        for job in current_teacher_inventory_jobs:
            recording_id = (job.get("payload") or {}).get("recording_id")
            if not recording_id:
                continue
            recording_key = str(recording_id)
            captured_recording_ids.add(recording_key)
            try:
                supervision = _job_structure_supervision(store, job)
            except Exception:
                unknown_recording_ids.add(recording_key)
                continue
            if supervision["eligible"]:
                eligible_recording_ids.add(recording_key)
                partial_recording_ids.discard(recording_key)
                unknown_recording_ids.discard(recording_key)
            elif supervision["classification"] == "partial":
                partial_recording_ids.add(recording_key)
            else:
                unknown_recording_ids.add(recording_key)
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
        excluded_reason = excluded_job_reasons.get(str(job["id"]))
        if excluded_reason is not None:
            excluded_teacher_jobs.append(
                {
                    "job_id": str(job["id"]),
                    "job_type": job_type,
                    "status": status,
                    "recording_id": recording_id,
                    "reason": excluded_reason,
                    "teacher_normalization_version": str(
                        (job.get("result") or {}).get(
                            "teacher_normalization_version"
                        )
                        or (job.get("payload") or {}).get(
                            "teacher_normalization_version"
                        )
                        or "unknown"
                    ),
                    "artifacts_preserved": True,
                }
            )
            continue
        if str(job["id"]) not in active_job_ids:
            continue
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
        active_job_counts[job_type].setdefault(status, 0)
        active_job_counts[job_type][status] += 1
        eligible_teacher_jobs += 1
        if recording_id:
            recording_key = str(recording_id)
            recording_job_statuses.setdefault(recording_key, {}).setdefault(
                job_type, set()
            ).add(status)
            if not inventory_from_captures:
                captured_recording_ids.add(recording_key)
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
        for job_type, counts in active_job_counts.items()
        if job_type in active_teacher_types
    )
    completed = sum(
        counts.get("complete", 0)
        for job_type, counts in active_job_counts.items()
        if job_type in active_teacher_types
    )
    total = sum(
        sum(counts.values())
        for job_type, counts in active_job_counts.items()
        if job_type in active_teacher_types
    )
    completed_recordings = {
        recording_id
        for recording_id in eligible_recording_ids
        if all(
            recording_job_statuses.get(recording_id, {}).get(
                job_type, set()
            ) == {"complete"}
            for job_type in active_teacher_types
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
    test_song_groups = int(trusted["split_group_counts"]["test"] or 0)
    activation_blockers: list[str] = []
    if test_song_groups < MIN_ACTIVATION_TEST_GROUPS:
        activation_blockers.append(
            f"{MIN_ACTIVATION_TEST_GROUPS - test_song_groups} more "
            "independent test songs are required for activation"
        )
    average_elapsed = sum(elapsed) / len(elapsed) if elapsed else None
    model_root = Path(research_root).resolve() / "models"
    active_model = model_root / "lumen-structure-student.npz"
    candidate_model = model_root / "lumen-structure-student.candidate.npz"
    active_evaluation_path = (
        model_root / "lumen-structure-student.evaluation.json"
    )
    candidate_evaluation_path = (
        model_root / "lumen-structure-student.candidate.evaluation.json"
    )
    evaluation: dict[str, Any] | None = None
    latest_evaluation_path = (
        candidate_evaluation_path
        if candidate_evaluation_path.is_file()
        else active_evaluation_path
    )
    if latest_evaluation_path.is_file():
        try:
            evaluation = json.loads(
                latest_evaluation_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            evaluation = {"error": "saved evaluation report is unreadable"}
    active_evaluation: dict[str, Any] | None = None
    if active_evaluation_path.is_file():
        try:
            active_evaluation = json.loads(
                active_evaluation_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            active_evaluation = {
                "error": "active evaluation report is unreadable"
            }
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
    trusted_consensus_revision = str(
        trusted.get("operator_consensus_revision") or ""
    )
    trained_consensus_revision = str(
        (latest_student_job or {}).get("payload", {}).get(
            "operator_consensus_revision"
        ) or ""
    )
    # A candidate is current only when it was trained from exactly the trusted
    # run set.  A subset means new teacher results arrived afterward and the
    # UI must ask for retraining instead of presenting the older candidate as
    # up to date.
    candidate_provenance_current = bool(latest_student_job) and bool(
        trained_run_ids
    ) and trained_run_ids == trusted_run_ids and (
        trained_consensus_revision == trusted_consensus_revision
    ) and isinstance(evaluation, dict) and (
        evaluation.get("activation_gate_version")
        == STUDENT_ACTIVATION_GATE_VERSION
    ) and (
        evaluation.get("teacher_fusion_version")
        == TEACHER_FUSION_VERSION
    )
    newly_trusted_run_ids = trusted_run_ids - trained_run_ids
    retired_trained_run_ids = trained_run_ids - trusted_run_ids
    candidate_stale_reasons: list[str] = []
    if newly_trusted_run_ids:
        candidate_stale_reasons.append(
            f"{len(newly_trusted_run_ids)} newly trusted teacher run(s) arrived"
        )
    if retired_trained_run_ids:
        candidate_stale_reasons.append(
            f"{len(retired_trained_run_ids)} prior teacher run(s) are no longer trusted"
        )
    if trained_consensus_revision != trusted_consensus_revision:
        candidate_stale_reasons.append("operator timeline corrections changed")
    if isinstance(evaluation, dict) and evaluation.get(
        "activation_gate_version"
    ) != STUDENT_ACTIVATION_GATE_VERSION:
        candidate_stale_reasons.append("qualification gate version changed")
    if isinstance(evaluation, dict) and evaluation.get(
        "teacher_fusion_version"
    ) != TEACHER_FUSION_VERSION:
        candidate_stale_reasons.append("teacher fusion version changed")
    active_provenance_current = bool(
        active_model.is_file()
        and isinstance(active_evaluation, dict)
        and active_evaluation.get("activated") is True
        and active_evaluation.get("teacher_normalization_version")
        == TEACHER_NORMALIZATION_VERSION
        and active_evaluation.get("edmformer_preprocessing_version")
        == EDMFORMER_PREPROCESSING_VERSION
        and active_evaluation.get("activation_gate_version")
        == STUDENT_ACTIVATION_GATE_VERSION
        and active_evaluation.get("teacher_fusion_version")
        == TEACHER_FUSION_VERSION
        and str(active_evaluation.get("operator_consensus_revision") or "")
        == trusted_consensus_revision
    )
    return {
        # These explicit counts keep the operator display honest.  Partial and
        # unidentified captures remain useful local data, but they are not
        # silently included in the denominator for whole-song teachers.
        "recordings_captured": len(captured_recording_ids),
        "ontology": {
            "version": TEACHER_NORMALIZATION_VERSION,
            "sustained_states": list(CANONICAL_TECHNO_SECTIONS),
            "transition_events": [
                event.value for event in TransitionEvent
            ],
            "active_teacher_authority": "axis_specific_edmformer_songformer",
            "axis_teachers": {
                "functional": "SongFormer",
                "energy": "EDMFormer",
                "content": "SongFormer",
                "boundary": "EDMFormer + SongFormer",
            },
            "songformer_role": "functional_content_and_boundary_teacher",
            "edmformer_preprocessing_version": (
                EDMFORMER_PREPROCESSING_VERSION
            ),
        },
        "recordings_eligible": len(eligible_recording_ids),
        "recordings_partial": len(partial_recording_ids),
        "recordings_unknown": len(unknown_recording_ids),
        "recordings_planned": len(eligible_recording_ids),
        "recordings_processed": len(completed_recordings),
        "teacher_jobs": job_counts,
        "active_teacher_jobs": active_job_counts,
        "active_teacher_authority": "axis_specific_edmformer_songformer",
        "active_teacher_job_types": sorted(active_teacher_types),
        "preserved_non_authoritative_job_types": [],
        "teacher_jobs_all_total": len(teacher_jobs),
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
        "operator_consensus": trusted.get("operator_consensus"),
        "provenance_errors": trusted["errors"],
        "excluded_teacher_runs": trusted["excluded_teacher_runs"],
        "excluded_examples": trusted["excluded_examples"],
        "teacher_errors": failures,
        "train_ready": not blockers,
        "blockers": blockers,
        "activation_ready": not activation_blockers,
        "activation_blockers": activation_blockers,
        "minimum_test_song_groups": MIN_ACTIVATION_TEST_GROUPS,
        "model": {
            "active": active_provenance_current,
            "active_artifact_exists": active_model.is_file(),
            "active_provenance_current": active_provenance_current,
            "active_path": str(active_model),
            "candidate": candidate_model.is_file(),
            "candidate_path": str(candidate_model),
            "evaluation": evaluation,
            "active_evaluation": active_evaluation,
            "candidate_provenance_current": candidate_provenance_current,
            "candidate_stale_reasons": candidate_stale_reasons,
            "newly_trusted_teacher_runs": len(newly_trusted_run_ids),
            "candidate_teacher_run_ids": sorted(trained_run_ids),
            "trusted_teacher_run_ids": sorted(trusted_run_ids),
            "candidate_operator_consensus_revision": trained_consensus_revision,
            "trusted_operator_consensus_revision": trusted_consensus_revision,
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
    previous_energy: str | None = None
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ValueError(f"{source} segment {index} is not an object")
        raw_label = str(item.get("label") or "unknown")
        label = (
            normalize_techno_structure_label(raw_label)
            if "edmformer" in source.casefold()
            else normalize_structure_label(raw_label)
        )
        start_s = float(item["start"])
        end_s = float(item["end"])
        raw_confidence = item.get("confidence")
        confidence_provided = raw_confidence is not None
        # Several upstream runners return categorical sections without a
        # calibrated probability. Preserve those labels, but represent the
        # missing score honestly instead of manufacturing live authority.
        confidence_value = (
            float(raw_confidence) if confidence_provided else 0.0
        )
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
        transition_event = transition_event_for(
            previous_energy, label.energy
        ).value
        raw_prediction = {
            "label": raw_label,
            "start": start_s,
            "end": end_s,
            "confidence": (
                float(raw_confidence) if confidence_provided else None
            ),
        }
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
                "transition_event": transition_event,
                "provenance": {
                    "source": source,
                    "source_version": source_version,
                    "annotation_type": "teacher_prediction",
                    "confidence": confidence,
                    "confidence_provided": confidence_provided,
                    "confidence_kind": (
                        "model_score" if confidence_provided else "unscored"
                    ),
                    "raw_predictions": [raw_prediction],
                    "raw_labels": [raw_label],
                    "transition_event": transition_event,
                    "ontology": TEACHER_NORMALIZATION_VERSION,
                },
            }
        )
        previous_end_ms = end_ms
        previous_energy = label.energy.value
    merged: list[dict[str, Any]] = []
    label_fields = (
        "functional_label", "energy_label", "content_label"
    )
    for segment in normalized:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and all(
                previous[field] == segment[field]
                for field in label_fields
            )
            and int(previous["end_ms"]) == int(segment["start_ms"])
        ):
            previous_duration = (
                int(previous["end_ms"]) - int(previous["start_ms"])
            )
            segment_duration = (
                int(segment["end_ms"]) - int(segment["start_ms"])
            )
            total_duration = previous_duration + segment_duration
            previous["end_ms"] = segment["end_ms"]
            previous["label_confidence"] = (
                float(previous["label_confidence"]) * previous_duration
                + float(segment["label_confidence"]) * segment_duration
            ) / max(1, total_duration)
            previous["provenance"] = {
                **dict(previous["provenance"]),
                "merged_identical_segments": int(
                    previous["provenance"].get(
                        "merged_identical_segments", 1
                    )
                ) + 1,
                "raw_predictions": [
                    *list(
                        previous["provenance"].get(
                            "raw_predictions", []
                        )
                    ),
                    *list(
                        segment["provenance"].get(
                            "raw_predictions", []
                        )
                    ),
                ],
                "raw_labels": [
                    *list(previous["provenance"].get("raw_labels", [])),
                    *list(segment["provenance"].get("raw_labels", [])),
                ],
            }
            continue
        copied = dict(segment)
        copied["segment_index"] = len(merged)
        merged.append(copied)
    return merged


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


def _student_audio_feature_cache(
    audio_path: Path,
    *,
    research_root: Path,
    content_sha256: str | None = None,
    cancel_check: Any = None,
    include_features: bool = True,
) -> dict[str, Any]:
    """Build reusable causal 10 Hz features from a coherent captured WAV."""
    cache_root = research_root / "features" / STUDENT_AUDIO_FEATURE_VERSION
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = str(content_sha256 or audio_path.stem)
    cache_path = cache_root / f"{identity}.json"
    metadata_path = cache_path.with_suffix(".metadata.json")
    if cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("version") == STUDENT_AUDIO_FEATURE_VERSION
                and metadata.get("audio_identity") == identity
                and int(metadata.get("step_ms") or 0) == 100
                and int(metadata.get("feature_rows") or 0) > 0
            ):
                if not include_features:
                    return {
                        **metadata,
                        "path": str(cache_path),
                        "cached": True,
                    }
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return {**cached, "path": str(cache_path), "cached": True}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("version") == STUDENT_AUDIO_FEATURE_VERSION
                and cached.get("audio_identity") == identity
                and int(cached.get("step_ms") or 0) == 100
            ):
                metadata = {
                    name: value
                    for name, value in cached.items()
                    if name != "features"
                }
                metadata["feature_rows"] = len(cached.get("features") or ())
                metadata_partial = metadata_path.with_suffix(
                    metadata_path.suffix + ".partial"
                )
                metadata_partial.write_text(
                    json.dumps(metadata, separators=(",", ":")),
                    encoding="utf-8",
                )
                metadata_partial.replace(metadata_path)
                if include_features:
                    return {
                        **cached,
                        "path": str(cache_path),
                        "cached": True,
                    }
                return {
                    **metadata,
                    "path": str(cache_path),
                    "cached": True,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        total_frames = source.getnframes()
        if sample_width != 2:
            raise ValueError("student audio features require PCM16 WAV input")
        analyzer = RealtimeAudioAnalyzer(
            sample_rate=sample_rate, channels=channels
        )
        chunk_frames = 2_048
        feature_rows: list[list[float]] = []
        next_offset_ms = 0
        analyzed_frames = 0
        latest = [0.0] * len(semantic_frame_features({}))
        while analyzed_frames < total_frames:
            if cancel_check is not None:
                cancel_check()
            pcm = source.readframes(
                min(chunk_frames, total_frames - analyzed_frames)
            )
            if not pcm:
                break
            chunk_frame_count = len(pcm) // (sample_width * channels)
            chunk_end_frames = analyzed_frames + chunk_frame_count
            chunk_end_ms = chunk_end_frames * 1000 / sample_rate
            # Emit only observations whose audio is fully in the past. This
            # preserves the same causal contract used by Live.
            while next_offset_ms < chunk_end_ms:
                feature_rows.append(list(latest))
                next_offset_ms += 100
            observation = analyzer.analyze_pcm16(
                pcm, timestamp_s=chunk_end_frames / sample_rate
            )
            metrics = analyzer.last_metrics
            clipping = metrics.clipped_samples / max(
                1, metrics.frame_count * channels
            )
            latest = [
                float(value)
                for value in semantic_frame_features(
                    {
                        "observation": asdict(observation),
                        "audio": {**asdict(metrics), "clipping": clipping},
                    }
                )
            ]
            analyzed_frames = chunk_end_frames
        duration_ms = round(total_frames * 1000 / sample_rate)
        while next_offset_ms <= duration_ms:
            feature_rows.append(list(latest))
            next_offset_ms += 100
    result = {
        "version": STUDENT_AUDIO_FEATURE_VERSION,
        "audio_identity": identity,
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "step_ms": 100,
        "duration_ms": duration_ms,
        "features": feature_rows,
    }
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    partial.write_text(
        json.dumps(result, separators=(",", ":")), encoding="utf-8"
    )
    partial.replace(cache_path)
    metadata = {
        name: value for name, value in result.items() if name != "features"
    }
    metadata["feature_rows"] = len(feature_rows)
    metadata_partial = metadata_path.with_suffix(
        metadata_path.suffix + ".partial"
    )
    metadata_partial.write_text(
        json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
    )
    metadata_partial.replace(metadata_path)
    if include_features:
        return {**result, "path": str(cache_path), "cached": False}
    return {**metadata, "path": str(cache_path), "cached": False}


def _student_audio_feature_cache_task(
    task: tuple[str, str, str | None],
) -> dict[str, Any]:
    """Process-pool boundary for one immutable recording feature cache."""

    audio_path, research_root, content_sha256 = task
    return _student_audio_feature_cache(
        Path(audio_path),
        research_root=Path(research_root),
        content_sha256=content_sha256,
        include_features=False,
    )


def _refresh_student_audio_features(
    examples: list[dict[str, Any]],
    *,
    jobs: list[dict[str, Any]],
    research_root: Path,
    cancel_check: Any = None,
    maximum_workers: int = 1,
    progress_callback: Any = None,
) -> dict[str, Any]:
    audio_by_recording: dict[str, tuple[Path, str | None]] = {}
    for job in jobs:
        payload = job.get("payload") or {}
        recording_id = payload.get("recording_id")
        audio_path = payload.get("audio_path")
        if recording_id and audio_path and Path(str(audio_path)).is_file():
            audio_by_recording[str(recording_id)] = (
                Path(str(audio_path)).resolve(),
                (
                    str(payload["content_sha256"])
                    if payload.get("content_sha256")
                    else None
                ),
            )
    recording_ids = sorted(
        {str(row.get("recording_id") or "") for row in examples}
    )
    missing = [
        recording_id
        for recording_id in recording_ids
        if recording_id and recording_id not in audio_by_recording
    ]
    if missing:
        raise ValueError(
            "current-audio student preprocessing is missing coherent WAVs "
            f"for {len(missing)} recording(s)"
        )
    active_recording_ids = [item for item in recording_ids if item]
    rows_by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        recording_id = str(row.get("recording_id") or "")
        if recording_id:
            rows_by_recording.setdefault(recording_id, []).append(row)

    tasks: dict[str, tuple[str, str, str | None]] = {}
    identity_by_recording: dict[str, str] = {}
    for recording_id in active_recording_ids:
        audio_path, content_sha256 = audio_by_recording[recording_id]
        identity = str(content_sha256 or audio_path.stem)
        identity_by_recording[recording_id] = identity
        tasks.setdefault(
            identity,
            (str(audio_path), str(research_root), content_sha256),
        )

    cache_metadata: dict[str, dict[str, Any]] = {}
    completed = 0
    worker_count = max(1, min(int(maximum_workers), len(tasks) or 1))

    def publish_progress() -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "student_feature_preparation",
                    "recordings_complete": completed,
                    "recordings_total": len(tasks),
                    "progress": (
                        completed / len(tasks) if tasks else 1.0
                    ),
                }
            )

    publish_progress()
    if worker_count == 1:
        for identity, task in tasks.items():
            if cancel_check is not None:
                cancel_check()
            cache_metadata[identity] = _student_audio_feature_cache_task(task)
            completed += 1
            publish_progress()
    else:
        thread_variables = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        previous_thread_values = {
            name: os.environ.get(name) for name in thread_variables
        }
        try:
            # Spawn clean workers so the parent's multi-gigabyte examples list
            # is not inherited by every child. Give each recording analyzer
            # one numerical thread; parallelism comes from recordings, while
            # the later model trainer retains the parent's full thread pool.
            for name in thread_variables:
                os.environ[name] = "1"
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(
                        _student_audio_feature_cache_task, task
                    ): identity
                    for identity, task in tasks.items()
                }
                for future in as_completed(futures):
                    if cancel_check is not None:
                        cancel_check()
                    identity = futures[future]
                    cache_metadata[identity] = future.result()
                    completed += 1
                    publish_progress()
        finally:
            for name, value in previous_thread_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    for index, recording_id in enumerate(active_recording_ids, start=1):
        metadata = cache_metadata[identity_by_recording[recording_id]]
        cache = json.loads(Path(str(metadata["path"])).read_text(encoding="utf-8"))
        features = cache["features"]
        step_ms = int(cache["step_ms"])
        for row in rows_by_recording.get(recording_id, ()):
            offset_ms = max(0, int(row.get("recording_offset_ms") or 0))
            feature_index = min(
                len(features) - 1,
                round(offset_ms / step_ms),
            )
            row["features"] = list(features[feature_index])
            row["feature_preprocessing_version"] = (
                STUDENT_AUDIO_FEATURE_VERSION
            )
        if cancel_check is not None and index % 4 == 0:
            cancel_check()
    return {
        "version": STUDENT_AUDIO_FEATURE_VERSION,
        "recordings": len(active_recording_ids),
        "examples": len(examples),
        "cache_paths": [
            cache_metadata[key]["path"] for key in sorted(cache_metadata)
        ],
        "cache_hits": sum(
            bool(cache["cached"]) for cache in cache_metadata.values()
        ),
        "feature_workers": worker_count,
    }


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
                and 0 <= boundary_distance_ms < BOUNDARY_TARGET_WINDOW_MS
            )
            features = semantic_frame_features(frame["payload"])
            example = {
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
                "target_confidence": target["label_confidence"],
                "target_provenance": timeline["provenance"],
                "target_provenance_details": target["provenance"],
                "timeline_version": timeline["timeline_version"],
                "teacher_run_id": timeline.get("teacher_run_id"),
                "transition_event": (
                    (target.get("provenance") or {}).get(
                        "transition_event"
                    )
                ),
                "split_group_id": split_group_id,
                "split": split,
                "structure_supervision": structure_supervision,
            }
            # The normalized teacher timeline itself supplies a categorical
            # boundary target: every non-initial segment begins at a teacher
            # transition, whether or not the upstream model exposes a
            # calibrated probability. Missing model confidence must prevent
            # that score from becoming live authority, but it must not erase
            # the deterministic transition encoded by the timeline.
            example["boundary_supervised"] = True
            example["boundary"] = is_boundary
            example["milliseconds_since_boundary"] = max(
                0, boundary_distance_ms
            )
            example["boundary_provenance"] = {
                "source": "teacher_timeline_transition",
                "target_window_ms": BOUNDARY_TARGET_WINDOW_MS,
                "teacher_run_id": timeline.get("teacher_run_id"),
                "timeline_id": timeline_id,
                "confidence_calibrated": bool(
                    (target.get("provenance") or {}).get(
                        "confidence_provided", False
                    )
                ),
            }
            examples.append(example)
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
        "schema_version": STUDENT_EXAMPLE_VERSION,
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
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row in {path} is not an object")
            yield value


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
