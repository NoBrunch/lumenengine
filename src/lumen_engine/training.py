"""Local, synchronized training-data capture for future neural models.

Audio is kept as ordinary PCM WAV files while compact semantic frames and
operator feedback live in the existing private SQLite memory. Disk writes and
database batches happen on a dedicated worker so the audio/DMX loop only copies
one packet into a bounded queue.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Callable
import wave

from lumen_engine.memory import SongMemoryStore


STRUCTURE_START_TOLERANCE_MS = 10_000
STRUCTURE_END_TOLERANCE_MS = 10_000
STRUCTURE_CAPTURE_TOLERANCE_MS = 20_000


def structure_supervision_completeness(
    *,
    track_duration_ms: int | None,
    start_position_ms: int | None,
    end_position_ms: int | None,
    captured_duration_ms: int,
    source_audio_complete: bool,
) -> dict[str, Any]:
    """Classify whether a capture can teach whole-song structure.

    Partial recordings remain valuable for choreography and preference
    learning, but a teacher timeline beginning at capture time zero cannot be
    treated as a complete song map unless the capture reaches both ends of the
    provider track and contains an almost complete duration of source audio.
    """

    evidence = {
        "track_duration_ms": (
            int(track_duration_ms) if track_duration_ms is not None else None
        ),
        "start_position_ms": (
            int(start_position_ms) if start_position_ms is not None else None
        ),
        "end_position_ms": (
            int(end_position_ms) if end_position_ms is not None else None
        ),
        "captured_duration_ms": max(0, int(captured_duration_ms)),
        "source_audio_complete": bool(source_audio_complete),
        "start_tolerance_ms": STRUCTURE_START_TOLERANCE_MS,
        "end_tolerance_ms": STRUCTURE_END_TOLERANCE_MS,
        "capture_tolerance_ms": STRUCTURE_CAPTURE_TOLERANCE_MS,
    }
    reasons: list[str] = []
    if not source_audio_complete:
        reasons.append("source_audio_incomplete")
    if track_duration_ms is None or int(track_duration_ms) <= 0:
        reasons.append("track_duration_unknown")
    if start_position_ms is None:
        reasons.append("track_start_position_unknown")
    if end_position_ms is None:
        reasons.append("track_end_position_unknown")
    if reasons and all(reason.endswith("unknown") for reason in reasons):
        return {
            "eligible": False,
            "classification": "unknown",
            "reason_codes": reasons,
            "evidence": evidence,
        }
    if track_duration_ms is not None and int(track_duration_ms) > 0:
        duration = int(track_duration_ms)
        if (
            start_position_ms is not None
            and int(start_position_ms) > STRUCTURE_START_TOLERANCE_MS
        ):
            reasons.append("capture_started_after_track_beginning")
        if (
            end_position_ms is not None
            and int(end_position_ms)
            < duration - STRUCTURE_END_TOLERANCE_MS
        ):
            reasons.append("capture_ended_before_track_end")
        if (
            int(captured_duration_ms)
            < duration - STRUCTURE_CAPTURE_TOLERANCE_MS
        ):
            reasons.append("captured_audio_too_short_for_track")
        if (
            int(captured_duration_ms)
            > duration + STRUCTURE_CAPTURE_TOLERANCE_MS
        ):
            reasons.append("captured_audio_too_long_for_track")
    eligible = not reasons
    return {
        "eligible": eligible,
        "classification": "complete" if eligible else "partial",
        "reason_codes": reasons,
        "evidence": evidence,
    }


@dataclass(frozen=True, slots=True)
class TrainingCaptureConfig:
    root: Path
    sample_rate: int = 48_000
    channels: int = 2
    sample_width: int = 2
    segment_seconds: int = 60
    feature_rate_hz: float = 10.0
    max_bytes: int = 100 * 1024**3
    minimum_free_bytes: int = 5 * 1024**3
    # Roughly three minutes of ordinary 2,048-frame ALSA packets.  The target
    # machine has ample RAM; this queue is deliberately large enough to absorb
    # a prolonged mechanical-disk stall without making the sample consumer
    # wait for storage.
    queue_packets: int = 4_096
    ram_spool_segments: int = 16
    ram_spool_root: Path | None = Path("/dev/shm")

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("training sample rate must be positive")
        if self.channels <= 0:
            raise ValueError("training channel count must be positive")
        if self.sample_width != 2:
            raise ValueError("training capture currently requires signed PCM16")
        if self.segment_seconds < 10:
            raise ValueError("training segments must be at least 10 seconds")
        if not 1.0 <= self.feature_rate_hz <= 50.0:
            raise ValueError("training feature rate must be between 1 and 50 Hz")
        if self.max_bytes <= 0:
            raise ValueError("training storage limit must be positive")
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum free storage must be non-negative")
        if self.queue_packets <= 0:
            raise ValueError("training packet queue must be bounded and positive")
        if self.ram_spool_segments <= 0:
            raise ValueError("RAM segment queue must be bounded and positive")

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * self.sample_width

    @property
    def segment_frames(self) -> int:
        return self.sample_rate * self.segment_seconds

    @property
    def feature_interval_frames(self) -> int:
        return max(1, round(self.sample_rate / self.feature_rate_hz))


@dataclass(frozen=True, slots=True)
class _AudioPacket:
    start_frame: int
    pcm: bytes
    created_unix_ms: int
    song_id: int | None
    position_ms: int | None
    feature_frames: tuple[int, ...]
    semantic_payload: dict[str, Any] | Callable[[], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _PendingSegment:
    spool_path: Path
    final_path: Path
    segment_index: int
    start_frame: int
    frame_count: int
    started_unix_ms: int
    byte_count: int


class TrainingDataRecorder:
    """Write one lossless PCM session plus synchronized semantic annotations."""

    def __init__(
        self,
        store: SongMemoryStore,
        *,
        session_id: str,
        mode: str,
        config: TrainingCaptureConfig,
        metadata: dict[str, Any],
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.mode = mode
        self.config = config
        self.metadata = metadata
        self._lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._queue: queue.Queue[_AudioPacket | None] = queue.Queue(
            maxsize=config.queue_packets
        )
        self._segment_queue: queue.Queue[_PendingSegment | None] = queue.Queue(
            maxsize=config.ram_spool_segments
        )
        self._thread: threading.Thread | None = None
        self._persist_thread: threading.Thread | None = None
        self._spool_directory: Path | None = None
        self._accepting = False
        self._state = "standby"
        self._error: str | None = None
        self._started_unix_ms: int | None = None
        self._frames_received = 0
        self._frames_written = 0
        self._dropped_packets = 0
        self._dropped_frames = 0
        self._segment_count = 0
        self._bytes_written = 0
        self._pending_segment_bytes = 0
        self._pending_segment_count = 0
        self._persisted_segment_count = 0
        self._last_persist_duration_ms: float | None = None
        self._maximum_persist_duration_ms = 0.0
        # Ten-Hz examples live on an exact sample-clock grid. The half-window
        # offset represents each interval by its center instead of quantizing
        # cadence to whatever ALSA packet size happens to be in use.
        self._next_feature_frame = config.feature_interval_frames // 2
        self._semantic_frames = 0
        self._latest_audio_frame: int | None = None
        self._base_bytes = 0
        self._dropped_ranges: list[dict[str, Any]] = []
        self._observed_gaps: list[dict[str, Any]] = []
        self._annotation_errors = 0
        self._last_annotation_error: str | None = None

    @property
    def latest_audio_frame(self) -> int | None:
        with self._lock:
            return self._latest_audio_frame

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("training recorder is already started")
            self.config.root.mkdir(parents=True, exist_ok=True)
            existing = int(self.store.training_summary().get("bytes", 0))
            self._base_bytes = existing
            available = shutil.disk_usage(self.config.root).free
            if existing >= self.config.max_bytes:
                self._state = "quota"
                self._error = (
                    "Training storage limit reached; capture was not started."
                )
                return
            if available < self.config.minimum_free_bytes:
                self._state = "quota"
                self._error = (
                    "Less than 5 GiB is free; training capture was not started."
                )
                return
            started = int(time.time() * 1000)
            self._started_unix_ms = started
            relative_directory = self._session_relative_directory(started)
            (self.config.root / relative_directory).mkdir(
                parents=True, exist_ok=True
            )
            spool_root = self.config.ram_spool_root
            if spool_root is not None:
                spool_root = Path(spool_root)
                spool_root.mkdir(parents=True, exist_ok=True)
                self._spool_directory = Path(
                    tempfile.mkdtemp(
                        prefix="lumen-audio-",
                        dir=str(spool_root),
                    )
                )
            else:
                self._spool_directory = Path(
                    tempfile.mkdtemp(prefix="lumen-audio-")
                )
            self.store.begin_training_session(
                session_id=self.session_id,
                mode=self.mode,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                sample_width=self.config.sample_width,
                relative_path=relative_directory.as_posix(),
                metadata=self.metadata,
            )
            self._accepting = True
            self._state = "recording"
            self._persist_thread = threading.Thread(
                target=self._persist_segments,
                name="lumen-training-persist",
                daemon=True,
            )
            self._persist_thread.start()
            self._thread = threading.Thread(
                target=self._writer,
                name="lumen-training-audio",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        pcm: bytes,
        *,
        song_id: int | None,
        position_ms: int | None,
        payload: dict[str, Any] | Callable[[], dict[str, Any]],
        source_start_frame: int | None = None,
        captured_unix_ms: int | None = None,
    ) -> int | None:
        """Queue one packet and return its center frame on the PCM timeline.

        Continuously drained capture packets carry their ALSA source frame as
        metadata on the bytes object. An explicit argument takes precedence;
        ordinary callers retain the original contiguous allocation behavior.
        """
        # Payload snapshots are generated outside the main state lock, but
        # submissions must remain serialized. Otherwise two capture callers
        # could allocate ordered frame indexes and enqueue their PCM backward.
        with self._submit_lock:
            bytes_per_frame = self.config.bytes_per_frame
            if len(pcm) % bytes_per_frame:
                raise ValueError("training PCM packet is not frame-aligned")
            frame_count = len(pcm) // bytes_per_frame
            if frame_count == 0:
                return None
            if source_start_frame is None:
                tagged_start = getattr(pcm, "source_start_frame", None)
                if tagged_start is not None:
                    source_start_frame = int(tagged_start)
            if source_start_frame is not None:
                source_start_frame = int(source_start_frame)
            if source_start_frame is not None and source_start_frame < 0:
                raise ValueError("source_start_frame must not be negative")
            with self._lock:
                if not self._accepting:
                    return None
                start_frame = (
                    self._frames_received
                    if source_start_frame is None
                    else int(source_start_frame)
                )
                if start_frame < self._frames_received:
                    raise ValueError(
                        "source audio packets must not overlap or move backward"
                    )
                center_frame = start_frame + frame_count // 2
                packet_end = start_frame + frame_count
                self._frames_received = packet_end
                self._latest_audio_frame = center_frame
                next_feature_frame = self._next_feature_frame
                feature_frames: list[int] = []
                # Advance the ten-Hz grid across missing source audio without
                # inventing semantic labels for the zero-filled reconstruction.
                while next_feature_frame < start_frame:
                    next_feature_frame += self.config.feature_interval_frames
                while next_feature_frame < packet_end:
                    feature_frames.append(next_feature_frame)
                    next_feature_frame += self.config.feature_interval_frames
            packet = _AudioPacket(
                start_frame=start_frame,
                pcm=pcm,
                created_unix_ms=(
                    int(time.time() * 1000)
                    if captured_unix_ms is None
                    else int(captured_unix_ms)
                ),
                song_id=song_id,
                position_ms=position_ms,
                feature_frames=tuple(feature_frames),
                semantic_payload=payload,
            )
            with self._lock:
                self._next_feature_frame = next_feature_frame
                if not self._accepting:
                    self._dropped_packets += 1
                    self._dropped_frames += frame_count
                    self._append_range(
                        self._dropped_ranges,
                        start_frame=start_frame,
                        frame_count=frame_count,
                        reason="capture_stopped_during_submission",
                    )
                    return None
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                with self._lock:
                    self._dropped_packets += 1
                    self._dropped_frames += frame_count
                    self._append_range(
                        self._dropped_ranges,
                        start_frame=start_frame,
                        frame_count=frame_count,
                        reason="capture_queue_overflow",
                    )
                return None
            return center_frame

    def stop(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        submit_lock_acquired = self._submit_lock.acquire(
            timeout=max(0.0, timeout)
        )
        try:
            with self._lock:
                self._accepting = False
                thread = self._thread
                if thread is None:
                    return
                if not submit_lock_acquired:
                    self._state = "error"
                    self._error = (
                        "Training capture timed out waiting for an in-flight "
                        "semantic submission."
                    )
                elif self._state == "recording":
                    self._state = "finalizing"
        finally:
            if submit_lock_acquired:
                self._submit_lock.release()
        while thread.is_alive() and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                self._queue.put(None, timeout=min(0.10, remaining))
                break
            except queue.Full:
                continue
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            with self._lock:
                self._state = "error"
                self._error = "Training audio writer did not finish in time."
            raise RuntimeError(self._error)

    def status(self) -> dict[str, Any]:
        with self._lock:
            duration_s = self._frames_received / self.config.sample_rate
            return {
                "session_id": self.session_id,
                "state": self._state,
                "error": self._error,
                "recording": self._accepting,
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "codec": "PCM16 WAV",
                "duration_s": duration_s,
                "frames_received": self._frames_received,
                "frames_written": self._frames_written,
                "dropped_packets": self._dropped_packets,
                "dropped_frames": self._dropped_frames,
                "gap_ranges": len(self._dropped_ranges),
                "segments": self._segment_count,
                "semantic_frames": self._semantic_frames,
                "annotation_errors": self._annotation_errors,
                "last_annotation_error": self._last_annotation_error,
                "bytes_written": self._bytes_written,
                "queue_packets": self._queue.qsize(),
                "queue_capacity_packets": self.config.queue_packets,
                "ram_spool": {
                    "active": self._spool_directory is not None,
                    "path": (
                        None
                        if self._spool_directory is None
                        else str(self._spool_directory)
                    ),
                    "pending_segments": self._pending_segment_count,
                    "pending_bytes": self._pending_segment_bytes,
                    "queue_capacity_segments": self.config.ram_spool_segments,
                    "persisted_segments": self._persisted_segment_count,
                    "last_persist_duration_ms": self._last_persist_duration_ms,
                    "maximum_persist_duration_ms": (
                        self._maximum_persist_duration_ms
                    ),
                },
                "path": str(self.config.root),
                "feature_rate_hz": self.config.feature_rate_hz,
            }

    @staticmethod
    def _append_range(
        ranges: list[dict[str, Any]],
        *,
        start_frame: int,
        frame_count: int,
        reason: str,
    ) -> None:
        if frame_count <= 0:
            return
        if (
            ranges
            and ranges[-1]["reason"] == reason
            and ranges[-1]["start_frame"] + ranges[-1]["frame_count"]
            == start_frame
        ):
            ranges[-1]["frame_count"] += frame_count
            return
        ranges.append(
            {
                "start_frame": start_frame,
                "frame_count": frame_count,
                "reason": reason,
            }
        )

    def _session_relative_directory(self, started_unix_ms: int) -> Path:
        date = time.strftime(
            "%Y-%m-%d", time.localtime(started_unix_ms / 1000.0)
        )
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in self.session_id
        )
        return Path("audio") / date / safe_id

    def _writer(self) -> None:
        writer: wave.Wave_write | None = None
        raw_file: Any = None
        spool_path: Path | None = None
        final_path: Path | None = None
        segment_index = 0
        segment_start_frame = 0
        segment_frame_count = 0
        expected_frame = 0
        segment_started_unix_ms = int(time.time() * 1000)
        feature_rows: list[dict[str, Any]] = []
        status = "complete"
        quota_reached = False
        try:
            assert self._started_unix_ms is not None
            relative_directory = self._session_relative_directory(
                self._started_unix_ms
            )

            def open_segment() -> None:
                nonlocal writer, raw_file, spool_path, final_path
                nonlocal segment_start_frame, segment_frame_count
                nonlocal segment_started_unix_ms
                segment_start_frame = expected_frame
                segment_frame_count = 0
                segment_started_unix_ms = int(
                    self._started_unix_ms
                    + segment_start_frame * 1000 / self.config.sample_rate
                )
                stem = f"segment-{segment_index:05d}"
                final_path = self.config.root / relative_directory / f"{stem}.wav"
                assert self._spool_directory is not None
                spool_path = self._spool_directory / f"{stem}.wav"
                raw_file = spool_path.open("wb")
                writer = wave.open(raw_file, "wb")
                writer.setnchannels(self.config.channels)
                writer.setsampwidth(self.config.sample_width)
                writer.setframerate(self.config.sample_rate)

            def close_segment() -> None:
                nonlocal writer, raw_file, spool_path, final_path
                if writer is None or raw_file is None:
                    return
                writer.close()
                raw_file.flush()
                raw_file.close()
                assert spool_path is not None and final_path is not None
                byte_count = spool_path.stat().st_size
                pending = _PendingSegment(
                    spool_path=spool_path,
                    final_path=final_path,
                    segment_index=segment_index,
                    start_frame=segment_start_frame,
                    frame_count=segment_frame_count,
                    started_unix_ms=segment_started_unix_ms,
                    byte_count=byte_count,
                )
                with self._lock:
                    self._pending_segment_count += 1
                    self._pending_segment_bytes += byte_count
                self._segment_queue.put(pending)
                writer = None
                raw_file = None
                spool_path = None
                final_path = None

            def write_aligned(data: bytes) -> None:
                nonlocal writer, segment_frame_count, segment_index
                nonlocal expected_frame
                offset = 0
                bytes_per_frame = self.config.bytes_per_frame
                total_frames = len(data) // bytes_per_frame
                while total_frames:
                    if writer is None:
                        open_segment()
                    available = (
                        self.config.segment_frames - segment_frame_count
                    )
                    take = min(total_frames, available)
                    end = offset + take * bytes_per_frame
                    assert writer is not None
                    writer.writeframesraw(data[offset:end])
                    segment_frame_count += take
                    expected_frame += take
                    offset = end
                    total_frames -= take
                    if segment_frame_count >= self.config.segment_frames:
                        close_segment()
                        segment_index += 1

            def projected_storage_bytes(additional_frames: int = 0) -> int:
                """Conservatively include open PCM and future WAV headers."""
                with self._lock:
                    durable_or_queued = (
                        self._bytes_written + self._pending_segment_bytes
                    )
                projected = self._base_bytes + durable_or_queued
                frames_left = max(0, int(additional_frames))
                if writer is not None:
                    projected += (
                        44
                        + segment_frame_count * self.config.bytes_per_frame
                    )
                    available = (
                        self.config.segment_frames - segment_frame_count
                    )
                    take = min(frames_left, available)
                    projected += take * self.config.bytes_per_frame
                    frames_left -= take
                while frames_left:
                    take = min(frames_left, self.config.segment_frames)
                    projected += 44 + take * self.config.bytes_per_frame
                    frames_left -= take
                return projected

            def enter_quota(reason: str) -> None:
                nonlocal quota_reached
                quota_reached = True
                with self._lock:
                    self._accepting = False
                    self._state = "quota"
                    self._error = reason

            last_free_space_check = 0.0
            cached_free_space = shutil.disk_usage(self.config.root).free

            def available_storage_bytes(*, force: bool = False) -> int:
                nonlocal last_free_space_check, cached_free_space
                now = time.monotonic()
                if force or now - last_free_space_check >= 5.0:
                    cached_free_space = shutil.disk_usage(
                        self.config.root
                    ).free
                    last_free_space_check = now
                return cached_free_space

            while True:
                try:
                    packet = self._queue.get(timeout=0.25)
                except queue.Empty:
                    with self._lock:
                        accepting = self._accepting
                    if not accepting:
                        # A submission may already own a frame range before it
                        # reaches the queue. Do not finalize ahead of it merely
                        # because shutdown made the queue temporarily empty.
                        if self._submit_lock.acquire(timeout=0.25):
                            self._submit_lock.release()
                            break
                        continue
                    continue
                try:
                    if packet is None:
                        break
                    packet_frames = (
                        len(packet.pcm) // self.config.bytes_per_frame
                    )
                    if quota_reached:
                        with self._lock:
                            self._dropped_packets += 1
                            self._dropped_frames += packet_frames
                            self._append_range(
                                self._dropped_ranges,
                                start_frame=packet.start_frame,
                                frame_count=packet_frames,
                                reason="storage_quota",
                            )
                        continue
                    projected_bytes = projected_storage_bytes(packet_frames)
                    current_bytes = projected_storage_bytes()
                    required_bytes = max(0, projected_bytes - current_bytes)
                    if projected_bytes > self.config.max_bytes:
                        enter_quota(
                            "Training capture reached its storage limit."
                        )
                        with self._lock:
                            self._dropped_packets += 1
                            self._dropped_frames += packet_frames
                            self._append_range(
                                self._dropped_ranges,
                                start_frame=packet.start_frame,
                                frame_count=packet_frames,
                                reason="storage_quota",
                            )
                        continue
                    if (
                        available_storage_bytes()
                        < self.config.minimum_free_bytes + required_bytes
                    ):
                        enter_quota(
                            "Training capture reached its free-space reserve."
                        )
                        with self._lock:
                            self._dropped_packets += 1
                            self._dropped_frames += packet_frames
                            self._append_range(
                                self._dropped_ranges,
                                start_frame=packet.start_frame,
                                frame_count=packet_frames,
                                reason="free_space_reserve",
                            )
                        continue
                    if packet.start_frame > expected_frame:
                        missing = packet.start_frame - expected_frame
                        with self._lock:
                            self._append_range(
                                self._observed_gaps,
                                start_frame=expected_frame,
                                frame_count=missing,
                                reason="missing_capture_packet",
                            )
                        silence_frames = min(
                            missing, self.config.sample_rate
                        )
                        silence = bytes(
                            silence_frames * self.config.bytes_per_frame
                        )
                        while missing:
                            take = min(missing, silence_frames)
                            write_aligned(
                                silence[
                                    : take * self.config.bytes_per_frame
                                ]
                            )
                            missing -= take
                    packet_pcm = packet.pcm
                    if packet.start_frame < expected_frame:
                        overlap_frames = min(
                            expected_frame - packet.start_frame,
                            len(packet_pcm) // self.config.bytes_per_frame,
                        )
                        packet_pcm = packet_pcm[
                            overlap_frames * self.config.bytes_per_frame :
                        ]
                    write_aligned(packet_pcm)
                    center_frame = packet.start_frame + packet_frames // 2
                    with self._lock:
                        self._frames_written = expected_frame
                    annotation: dict[str, Any] | None = None
                    if packet.feature_frames:
                        try:
                            candidate = packet.semantic_payload
                            annotation = (
                                candidate()
                                if callable(candidate)
                                else candidate
                            )
                            if not isinstance(annotation, dict):
                                raise TypeError(
                                    "semantic payload must be a dictionary"
                                )
                        except Exception as error:
                            # PCM is authoritative and has already been written.
                            # Semantic enrichment is deliberately best-effort
                            # and cannot delay or fail the live submitter.
                            annotation = None
                            with self._lock:
                                self._annotation_errors += 1
                                self._last_annotation_error = str(error)
                    semantic_samples = (
                        tuple(
                            (
                                feature_frame,
                                {
                                    **annotation,
                                    "semantic_resampling": {
                                        "method": (
                                            "nearest_packet_sample_hold"
                                        ),
                                        "target_audio_frame": feature_frame,
                                        "source_audio_frame": center_frame,
                                        "source_offset_frames": (
                                            center_frame - feature_frame
                                        ),
                                        "interval_frames": (
                                            self.config.feature_interval_frames
                                        ),
                                    },
                                },
                            )
                            for feature_frame in packet.feature_frames
                        )
                        if annotation is not None
                        else ()
                    )
                    if semantic_samples:
                        with self._lock:
                            self._semantic_frames += len(semantic_samples)
                    for feature_frame, semantic_payload in semantic_samples:
                        offset_ms = round(
                            (feature_frame - center_frame)
                            * 1000
                            / self.config.sample_rate
                        )
                        feature_rows.append(
                            {
                                "session_id": self.session_id,
                                "audio_frame_index": feature_frame,
                                "segment_index": (
                                    feature_frame
                                    // self.config.segment_frames
                                ),
                                "segment_frame_index": (
                                    feature_frame
                                    % self.config.segment_frames
                                ),
                                "created_unix_ms": (
                                    packet.created_unix_ms + offset_ms
                                ),
                                "song_id": packet.song_id,
                                "position_ms": (
                                    None
                                    if packet.position_ms is None
                                    else max(
                                        0,
                                        packet.position_ms + offset_ms,
                                    )
                                ),
                                "payload": semantic_payload,
                            }
                        )
                    if len(feature_rows) >= 50:
                        self.store.add_training_frames(feature_rows)
                        feature_rows.clear()
                    if projected_storage_bytes() >= self.config.max_bytes:
                        enter_quota(
                            "Training capture reached its storage limit."
                        )
                    elif (
                        available_storage_bytes()
                        < self.config.minimum_free_bytes
                    ):
                        enter_quota(
                            "Training capture reached its free-space reserve."
                        )
                finally:
                    self._queue.task_done()
            with self._lock:
                received_frames = self._frames_received
            if not quota_reached and expected_frame < received_frames:
                missing = received_frames - expected_frame
                with self._lock:
                    self._append_range(
                        self._observed_gaps,
                        start_frame=expected_frame,
                        frame_count=missing,
                        reason="missing_capture_packet",
                    )
                zero_block_frames = min(missing, self.config.sample_rate)
                zero_block = bytes(
                    zero_block_frames * self.config.bytes_per_frame
                )
                while missing:
                    take = min(missing, zero_block_frames)
                    write_aligned(
                        zero_block[: take * self.config.bytes_per_frame]
                    )
                    missing -= take
                with self._lock:
                    self._frames_written = expected_frame
            if feature_rows:
                self.store.add_training_frames(feature_rows)
            close_segment()
        except Exception as error:
            status = "error"
            with self._lock:
                self._error = str(error)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if raw_file is not None:
                try:
                    raw_file.close()
                except Exception:
                    pass
        finally:
            # The audio writer only stages complete WAVs in RAM.  Let the
            # persistence worker drain every staged segment before publishing
            # the final session manifest and database summary.
            self._segment_queue.put(None)
            persist_thread = self._persist_thread
            if persist_thread is not None:
                persist_thread.join()
            with self._lock:
                if self._state == "quota":
                    status = "quota"
                elif self._error is not None:
                    status = "error"
                self._state = status
                self._accepting = False
                summary = {
                    "frames_received": self._frames_received,
                    "frames_written": self._frames_written,
                    "dropped_packets": self._dropped_packets,
                    "dropped_frames": self._dropped_frames,
                    "segment_count": self._segment_count,
                    "bytes_written": self._bytes_written,
                }
                dropped_ranges = [dict(item) for item in self._dropped_ranges]
                observed_gaps = [dict(item) for item in self._observed_gaps]
                annotation_errors = self._annotation_errors
                last_annotation_error = self._last_annotation_error
            try:
                assert self._started_unix_ms is not None
                self._write_capture_manifest(
                    status=status,
                    summary=summary,
                    dropped_ranges=dropped_ranges,
                    observed_gaps=observed_gaps,
                    annotation_errors=annotation_errors,
                    last_annotation_error=last_annotation_error,
                )
            except Exception as error:
                with self._lock:
                    self._state = "error"
                    self._error = (
                        self._error
                        or f"Could not write capture manifest: {error}"
                    )
                status = "error"
            try:
                self.store.finish_training_session(
                    self.session_id, status=status, **summary
                )
            except Exception as error:
                with self._lock:
                    self._state = "error"
                    self._error = (
                        self._error
                        or f"Could not finalize training index: {error}"
                    )

            spool_directory = self._spool_directory
            if spool_directory is not None:
                try:
                    spool_directory.rmdir()
                except OSError:
                    # A retained spool file is evidence of an incomplete disk
                    # transfer and must not be silently deleted.
                    pass

    def _persist_segments(self) -> None:
        """Move completed RAM-spooled WAVs to durable storage sequentially."""

        while True:
            pending = self._segment_queue.get()
            try:
                if pending is None:
                    return
                started = time.monotonic()
                partial_path = pending.final_path.with_suffix(".wav.partial")
                digest = hashlib.sha256()
                with pending.spool_path.open("rb") as source, partial_path.open(
                    "wb"
                ) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                partial_path.replace(pending.final_path)
                relative_path = pending.final_path.relative_to(
                    self.config.root
                ).as_posix()
                self.store.add_training_segment(
                    session_id=self.session_id,
                    segment_index=pending.segment_index,
                    relative_path=relative_path,
                    start_frame=pending.start_frame,
                    frame_count=pending.frame_count,
                    started_unix_ms=pending.started_unix_ms,
                    ended_unix_ms=int(
                        pending.started_unix_ms
                        + pending.frame_count
                        * 1000
                        / self.config.sample_rate
                    ),
                    byte_count=pending.byte_count,
                    sha256=digest.hexdigest(),
                )
                duration_ms = (time.monotonic() - started) * 1000.0
                with self._lock:
                    self._segment_count += 1
                    self._persisted_segment_count += 1
                    self._bytes_written += pending.byte_count
                    self._pending_segment_count -= 1
                    self._pending_segment_bytes -= pending.byte_count
                    self._last_persist_duration_ms = round(duration_ms, 2)
                    self._maximum_persist_duration_ms = max(
                        self._maximum_persist_duration_ms,
                        duration_ms,
                    )
                pending.spool_path.unlink(missing_ok=True)
            except Exception as error:
                with self._lock:
                    self._accepting = False
                    self._state = "error"
                    self._error = self._error or (
                        f"Could not persist RAM-spooled audio: {error}"
                    )
            finally:
                self._segment_queue.task_done()

    def _write_capture_manifest(
        self,
        *,
        status: str,
        summary: dict[str, int],
        dropped_ranges: list[dict[str, Any]],
        observed_gaps: list[dict[str, Any]],
        annotation_errors: int,
        last_annotation_error: str | None,
    ) -> None:
        assert self._started_unix_ms is not None
        relative_directory = self._session_relative_directory(
            self._started_unix_ms
        )
        directory = self.config.root / relative_directory
        gaps_by_location: dict[tuple[int, int], dict[str, Any]] = {}
        for item in observed_gaps:
            key = (int(item["start_frame"]), int(item["frame_count"]))
            gaps_by_location[key] = {
                **item,
                "representation": "pcm_silence",
            }
        for item in dropped_ranges:
            key = (int(item["start_frame"]), int(item["frame_count"]))
            existing = gaps_by_location.get(key)
            if existing is None:
                gaps_by_location[key] = {
                    **item,
                    "representation": (
                        "pcm_silence"
                        if int(item["start_frame"]) + int(item["frame_count"])
                        <= summary["frames_written"]
                        else "not_written"
                    ),
                }
            else:
                existing["reason"] = item["reason"]
        if summary["frames_written"] < summary["frames_received"]:
            key = (summary["frames_written"], (
                summary["frames_received"] - summary["frames_written"]
            ))
            gaps_by_location.setdefault(
                key,
                {
                    "start_frame": key[0],
                    "frame_count": key[1],
                    "reason": "capture_ended_before_write",
                    "representation": "not_written",
                },
            )
        gaps = sorted(
            gaps_by_location.values(),
            key=lambda item: int(item["start_frame"]),
        )
        capture_identity = hashlib.sha256(
            (
                f"{self.session_id}\x1f{self._started_unix_ms}\x1f"
                f"{self.config.sample_rate}\x1f{self.config.channels}"
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "format": "lumen_pcm_capture",
            "version": 2,
            "capture_id": f"capture:{capture_identity[:32]}",
            "session_id": self.session_id,
            "started_unix_ms": self._started_unix_ms,
            "status": status,
            "audio_format": {
                "container": "wav",
                "encoding": "signed_pcm_16_little_endian",
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "sample_width": self.config.sample_width,
            },
            "timeline": {
                **summary,
                "sample_accurate": (
                    summary["frames_received"] == summary["frames_written"]
                ),
                "gap_policy": "queue losses are replaced with PCM silence",
                "gaps": gaps,
            },
            "semantic_capture": {
                "frames": self._semantic_frames,
                "annotation_errors": annotation_errors,
                "last_annotation_error": last_annotation_error,
            },
        }
        temporary = directory / "capture.json.partial"
        final = directory / "capture.json"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(final)


def export_training_dataset(
    store: SongMemoryStore,
    root: Path,
) -> dict[str, Any]:
    """Create verified, track-separated indexes for model training."""
    sessions = store.training_sessions()
    if not sessions:
        raise RuntimeError("No recorded training sessions are available yet.")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    export_name = (
        f"dataset-{timestamp}-{time.time_ns() % 1_000_000:06d}"
    )
    export_directory = root / "exports" / export_name
    working_directory = export_directory.with_name(
        f".{export_directory.name}.partial"
    )
    collision_index = 0
    while export_directory.exists() or working_directory.exists():
        collision_index += 1
        export_directory = (
            root / "exports" / f"{export_name}-{collision_index:02d}"
        )
        working_directory = export_directory.with_name(
            f".{export_directory.name}.partial"
        )
    working_directory.mkdir(parents=True, exist_ok=False)
    integrity_errors: list[dict[str, Any]] = []
    capture_warnings: list[dict[str, Any]] = []
    total_segments = 0
    total_frames = 0
    total_feedback = 0
    total_annotations = 0
    total_recordings = 0
    total_student_sequences = 0
    total_choreography_sequences = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    structure_supervision_counts = {
        "complete": 0,
        "partial": 0,
        "unknown": 0,
    }

    with (
        (working_directory / "sessions.jsonl").open("w", encoding="utf-8")
        as session_file,
        (working_directory / "segments.jsonl").open("w", encoding="utf-8")
        as segment_file,
        (working_directory / "frames.jsonl").open("w", encoding="utf-8")
        as frame_file,
        (working_directory / "feedback_examples.jsonl").open(
            "w", encoding="utf-8"
        ) as feedback_file,
        (working_directory / "annotation_examples.jsonl").open(
            "w", encoding="utf-8"
        ) as annotation_file,
        (working_directory / "recordings.jsonl").open(
            "w", encoding="utf-8"
        ) as recording_file,
        (working_directory / "student_sequences.jsonl").open(
            "w", encoding="utf-8"
        ) as student_file,
        (working_directory / "choreography_sequences.jsonl").open(
            "w", encoding="utf-8"
        ) as choreography_file,
    ):
        for session in sessions:
            session_id = str(session["id"])
            segments = store.training_segments(session_id)
            frames = store.training_frames(session_id)
            feedback = store.training_feedback(session_id)
            annotations = store.training_annotations(session_id)
            capture = _load_capture_manifest(root, session)
            verified_segments, session_errors = _verify_segments(
                root, session, segments
            )
            integrity_errors.extend(session_errors)
            continuity = _session_continuity(
                session, verified_segments, capture
            )
            integrity_errors.extend(continuity["errors"])
            capture_warnings.extend(continuity["warnings"])
            if capture is None:
                capture_warnings.append(
                    {
                        "session_id": session_id,
                        "code": "capture_manifest_unavailable",
                        "message": (
                            "This capture predates the explicit gap manifest "
                            "or its capture.json could not be read."
                        ),
                    }
                )
            exported_session = {
                **session,
                "capture_id": (
                    capture.get("capture_id") if capture is not None else None
                ),
                "capture_manifest": (
                    (
                        Path(str(session["relative_path"])) / "capture.json"
                    ).as_posix()
                    if capture is not None
                    else None
                ),
                "integrity": {
                    "timeline_complete": continuity["timeline_complete"],
                    "sample_accurate": continuity["sample_accurate"],
                    "source_audio_complete": continuity[
                        "source_audio_complete"
                    ],
                    "substituted_silence_frames": continuity[
                        "substituted_silence_frames"
                    ],
                },
            }
            session_file.write(
                json.dumps(exported_session, sort_keys=True) + "\n"
            )
            for segment in verified_segments:
                segment_file.write(
                    json.dumps(segment, sort_keys=True) + "\n"
                )
            for frame in frames:
                # Runtime output is retained as baseline context, never declared
                # to be a correct target. Human feedback supplies supervision.
                exported_frame = {
                    **frame,
                    "target_provenance": (
                        "heuristic_runtime_baseline_not_ground_truth"
                    ),
                }
                frame_file.write(
                    json.dumps(exported_frame, sort_keys=True) + "\n"
                )
            feedback_examples = _feedback_examples(
                session, verified_segments, frames, feedback
            )
            annotation_examples = _labeled_examples(
                session,
                verified_segments,
                frames,
                annotations,
                supervision_kind="operator_annotation",
            )
            for example in feedback_examples:
                feedback_file.write(
                    json.dumps(example, sort_keys=True) + "\n"
                )
            for example in annotation_examples:
                annotation_file.write(
                    json.dumps(example, sort_keys=True) + "\n"
                )
            recordings = _recording_sequences(
                exported_session,
                verified_segments,
                frames,
                capture=capture,
            )
            for recording in recordings:
                recording_file.write(
                    json.dumps(recording, sort_keys=True) + "\n"
                )
                split_counts[str(recording["split"])] += 1
                classification = str(
                    recording["structure_supervision"]["classification"]
                )
                structure_supervision_counts[classification] += 1
                student = _student_sequence(
                    recording, frames, annotations
                )
                student_file.write(
                    json.dumps(student, sort_keys=True) + "\n"
                )
                choreography = _choreography_sequence(
                    recording, frames, feedback, annotations
                )
                choreography_file.write(
                    json.dumps(choreography, sort_keys=True) + "\n"
                )
                total_student_sequences += 1
                total_choreography_sequences += 1
            total_recordings += len(recordings)
            total_segments += len(segments)
            total_frames += len(frames)
            total_feedback += len(feedback_examples)
            total_annotations += len(annotation_examples)

    manifest = {
        "format": "lumen_training_dataset",
        "version": 2,
        "created_unix_ms": int(time.time() * 1000),
        "dataset_root": str(root),
        "audio_format": {
            "container": "wav",
            "encoding": "signed_pcm_16_little_endian",
            "timing_authority": "audio_frame_index",
        },
        "files": {
            "sessions": "sessions.jsonl",
            "segments": "segments.jsonl",
            "frames": "frames.jsonl",
            "feedback_examples": "feedback_examples.jsonl",
            "annotation_examples": "annotation_examples.jsonl",
            "recordings": "recordings.jsonl",
            "student_sequences": "student_sequences.jsonl",
            "choreography_sequences": "choreography_sequences.jsonl",
        },
        "counts": {
            "sessions": len(sessions),
            "segments": total_segments,
            "semantic_frames": total_frames,
            "feedback_examples": total_feedback,
            "annotation_examples": total_annotations,
            "recordings": total_recordings,
            "student_sequences": total_student_sequences,
            "choreography_sequences": total_choreography_sequences,
            "splits": split_counts,
            "structure_supervision": structure_supervision_counts,
        },
        "validation": {
            "errors": integrity_errors,
            "warnings": capture_warnings,
            "valid": not integrity_errors,
        },
        "guidance": {
            "input": (
                "Use recordings.jsonl for track-safe audio reconstruction and "
                "the task-specific sequence index for each model."
            ),
            "supervision": (
                "Feedback and preferred actions are human labels. The recorded "
                "runtime decision is context only and must not be treated as truth."
            ),
            "split_rule": (
                "The provided deterministic split is grouped by provider track "
                "identity, falling back to song_id and then capture session. Never "
                "randomly split neighboring frames."
            ),
            "gap_policy": (
                "Queue losses remain on the sample clock and are explicitly "
                "represented as PCM silence plus capture-manifest gap ranges."
            ),
            "structure_supervision": (
                "Only complete provider-aligned tracks are eligible for "
                "whole-song structure teachers. Partial and unknown captures "
                "remain available for choreography and local preference data."
            ),
        },
    }
    (working_directory / "dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # A completed export appears atomically. Interrupted work remains hidden
    # under a .partial name and is never mistaken for a usable dataset.
    working_directory.replace(export_directory)
    return {
        "path": str(export_directory),
        "manifest": manifest,
    }


def export_research_session_index(
    store: SongMemoryStore,
    root: Path,
    session_id: str,
) -> dict[str, Any]:
    """Build the bounded index needed to queue teachers for one new capture.

    The full operator export intentionally contains every semantic frame and
    derived choreography sequence. Rebuilding that historical dataset after
    each Live stop scales quadratically and can take longer than application
    shutdown. Teacher preparation only needs verified segment clips and track
    boundaries, so this path projects compact identities directly in SQLite.
    """

    session = next(
        (
            item
            for item in store.training_sessions()
            if str(item["id"]) == str(session_id)
        ),
        None,
    )
    if session is None:
        raise ValueError(f"unknown training session: {session_id}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = (
        f"research-{timestamp}-"
        f"{hashlib.sha256(str(session_id).encode()).hexdigest()[:10]}"
    )
    export_directory = root / "exports" / stem
    working_directory = export_directory.with_name(
        f".{export_directory.name}.partial"
    )
    collision = 0
    while export_directory.exists() or working_directory.exists():
        collision += 1
        export_directory = root / "exports" / f"{stem}-{collision:02d}"
        working_directory = export_directory.with_name(
            f".{export_directory.name}.partial"
        )
    working_directory.mkdir(parents=True, exist_ok=False)

    segments = store.training_segments(str(session_id))
    frames = store.training_frame_identities(str(session_id))
    capture = _load_capture_manifest(root, session)
    verified_segments, integrity_errors = _verify_segments(
        root, session, segments
    )
    continuity = _session_continuity(
        session, verified_segments, capture
    )
    integrity_errors.extend(continuity["errors"])
    warnings = list(continuity["warnings"])
    if capture is None:
        warnings.append(
            {
                "session_id": str(session_id),
                "code": "capture_manifest_unavailable",
                "message": "The session capture manifest could not be read.",
            }
        )
    exported_session = {
        **session,
        "capture_id": (
            capture.get("capture_id") if capture is not None else None
        ),
        "capture_manifest": (
            (
                Path(str(session["relative_path"])) / "capture.json"
            ).as_posix()
            if capture is not None
            else None
        ),
        "integrity": {
            "timeline_complete": continuity["timeline_complete"],
            "sample_accurate": continuity["sample_accurate"],
            "source_audio_complete": continuity["source_audio_complete"],
            "substituted_silence_frames": continuity[
                "substituted_silence_frames"
            ],
        },
    }
    recordings = _recording_sequences(
        exported_session,
        verified_segments,
        frames,
        capture=capture,
    )
    files = {
        "sessions": "sessions.jsonl",
        "segments": "segments.jsonl",
        "frames": "frames.jsonl",
        "feedback_examples": "feedback_examples.jsonl",
        "annotation_examples": "annotation_examples.jsonl",
        "recordings": "recordings.jsonl",
        "student_sequences": "student_sequences.jsonl",
        "choreography_sequences": "choreography_sequences.jsonl",
    }
    for filename in files.values():
        (working_directory / filename).touch()
    with (working_directory / files["sessions"]).open(
        "w", encoding="utf-8"
    ) as target:
        target.write(json.dumps(exported_session, sort_keys=True) + "\n")
    with (working_directory / files["segments"]).open(
        "w", encoding="utf-8"
    ) as target:
        for segment in verified_segments:
            target.write(json.dumps(segment, sort_keys=True) + "\n")
    with (working_directory / files["recordings"]).open(
        "w", encoding="utf-8"
    ) as target:
        for recording in recordings:
            target.write(json.dumps(recording, sort_keys=True) + "\n")
    split_counts = {"train": 0, "validation": 0, "test": 0}
    supervision = {"complete": 0, "partial": 0, "unknown": 0}
    for recording in recordings:
        split_counts[str(recording["split"])] += 1
        classification = str(
            recording["structure_supervision"]["classification"]
        )
        supervision[classification] = supervision.get(classification, 0) + 1
    manifest = {
        "format": "lumen_training_dataset",
        "version": 2,
        "purpose": "incremental_teacher_preparation",
        "created_unix_ms": int(time.time() * 1000),
        "dataset_root": str(root),
        "audio_format": {
            "container": "wav",
            "encoding": "signed_pcm_16_little_endian",
            "timing_authority": "audio_frame_index",
        },
        "files": files,
        "counts": {
            "sessions": 1,
            "segments": len(segments),
            "semantic_frames": len(frames),
            "feedback_examples": 0,
            "annotation_examples": 0,
            "recordings": len(recordings),
            "student_sequences": 0,
            "choreography_sequences": 0,
            "splits": split_counts,
            "structure_supervision": supervision,
        },
        "validation": {
            "errors": integrity_errors,
            "warnings": warnings,
            "valid": not integrity_errors,
        },
    }
    (working_directory / "dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    working_directory.replace(export_directory)
    return {"path": str(export_directory), "manifest": manifest}


def _load_capture_manifest(
    root: Path,
    session: dict[str, Any],
) -> dict[str, Any] | None:
    path = root / str(session["relative_path"]) / "capture.json"
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verify_segments(
    root: Path,
    session: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verified: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    for segment in segments:
        item = dict(segment)
        relative_path = str(item["relative_path"])
        integrity: dict[str, Any] = {
            "exists": False,
            "path_within_dataset_root": False,
            "byte_count_matches": False,
            "sha256_matches": False,
            "wav_metadata_matches": False,
        }
        try:
            path = (root / relative_path).resolve()
            path.relative_to(root_resolved)
            integrity["path_within_dataset_root"] = True
        except (OSError, ValueError):
            errors.append(
                {
                    "session_id": session["id"],
                    "segment_index": item["segment_index"],
                    "code": "audio_path_outside_dataset_root",
                    "relative_path": relative_path,
                }
            )
            item["integrity"] = integrity
            verified.append(item)
            continue
        if not path.is_file():
            errors.append(
                {
                    "session_id": session["id"],
                    "segment_index": item["segment_index"],
                    "code": "audio_file_missing",
                    "relative_path": relative_path,
                }
            )
            item["integrity"] = integrity
            verified.append(item)
            continue
        integrity["exists"] = True
        actual_bytes = path.stat().st_size
        integrity["actual_byte_count"] = actual_bytes
        integrity["byte_count_matches"] = (
            actual_bytes == int(item["byte_count"])
        )
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        integrity["actual_sha256"] = actual_sha256
        integrity["sha256_matches"] = actual_sha256 == str(item["sha256"])
        try:
            with wave.open(str(path), "rb") as recording:
                wav_metadata = {
                    "frame_count": recording.getnframes(),
                    "sample_rate": recording.getframerate(),
                    "channels": recording.getnchannels(),
                    "sample_width": recording.getsampwidth(),
                    "compression": recording.getcomptype(),
                }
            integrity["wav"] = wav_metadata
            integrity["wav_metadata_matches"] = (
                wav_metadata["frame_count"] == int(item["frame_count"])
                and wav_metadata["sample_rate"] == int(session["sample_rate"])
                and wav_metadata["channels"] == int(session["channels"])
                and wav_metadata["sample_width"]
                == int(session["sample_width"])
                and wav_metadata["compression"] == "NONE"
            )
        except (OSError, EOFError, wave.Error) as error:
            integrity["wav_error"] = str(error)
        for field, code in (
            ("byte_count_matches", "audio_byte_count_mismatch"),
            ("sha256_matches", "audio_sha256_mismatch"),
            ("wav_metadata_matches", "audio_wav_metadata_mismatch"),
        ):
            if not integrity[field]:
                errors.append(
                    {
                        "session_id": session["id"],
                        "segment_index": item["segment_index"],
                        "code": code,
                        "relative_path": relative_path,
                    }
                )
        item["integrity"] = integrity
        verified.append(item)
    return verified, errors


def _session_continuity(
    session: dict[str, Any],
    segments: list[dict[str, Any]],
    capture: dict[str, Any] | None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    expected_start = 0
    for expected_index, segment in enumerate(segments):
        if int(segment["segment_index"]) != expected_index:
            errors.append(
                {
                    "session_id": session["id"],
                    "code": "segment_index_discontinuity",
                    "expected": expected_index,
                    "actual": int(segment["segment_index"]),
                }
            )
        if int(segment["start_frame"]) != expected_start:
            errors.append(
                {
                    "session_id": session["id"],
                    "code": "segment_timeline_discontinuity",
                    "expected": expected_start,
                    "actual": int(segment["start_frame"]),
                }
            )
        expected_start = int(segment["start_frame"]) + int(
            segment["frame_count"]
        )
    frames_written = int(session["frames_written"])
    frames_received = int(session["frames_received"])
    if expected_start != frames_written:
        errors.append(
            {
                "session_id": session["id"],
                "code": "indexed_frame_count_mismatch",
                "indexed": expected_start,
                "frames_written": frames_written,
            }
        )
    if frames_written > frames_received:
        errors.append(
            {
                "session_id": session["id"],
                "code": "written_frames_exceed_capture_clock",
                "frames_received": frames_received,
                "frames_written": frames_written,
            }
        )
    gap_rows: list[dict[str, Any]] = []
    if capture is not None:
        timeline = capture.get("timeline")
        if isinstance(timeline, dict) and isinstance(
            timeline.get("gaps"), list
        ):
            gap_rows = [
                item for item in timeline["gaps"] if isinstance(item, dict)
            ]
    substituted_silence = sum(
        int(item.get("frame_count", 0))
        for item in gap_rows
        if item.get("representation") == "pcm_silence"
    )
    if gap_rows:
        warnings.append(
            {
                "session_id": session["id"],
                "code": "capture_contains_explicit_gap_ranges",
                "gap_ranges": len(gap_rows),
                "substituted_silence_frames": substituted_silence,
            }
        )
    elif int(session.get("dropped_frames", 0)) > 0:
        warnings.append(
            {
                "session_id": session["id"],
                "code": "legacy_capture_has_unlocated_dropped_frames",
                "dropped_frames": int(session["dropped_frames"]),
            }
        )
    timeline_complete = expected_start == frames_received
    return {
        "errors": errors,
        "warnings": warnings,
        "timeline_complete": timeline_complete,
        "sample_accurate": timeline_complete,
        "source_audio_complete": (
            timeline_complete
            and substituted_silence == 0
            and int(session.get("dropped_frames", 0)) == 0
        ),
        "substituted_silence_frames": substituted_silence,
    }


def _recording_sequences(
    session: dict[str, Any],
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    capture: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    total_frames = int(session["frames_written"])
    if total_frames <= 0:
        return []
    runs: list[dict[str, Any]] = []
    for frame in frames:
        media = frame.get("payload", {}).get("media")
        media = media if isinstance(media, dict) else None
        song_id = frame.get("song_id")
        if (
            media is not None
            and media.get("provider")
            and media.get("provider_item_id")
        ):
            track_key = (
                f"{media['provider']}:{media['provider_item_id']}"
            )
        elif song_id is not None:
            track_key = f"lumen-song:{song_id}"
        else:
            track_key = f"unidentified-session:{session['id']}"
        position_ms = frame.get("position_ms")
        starts_new_play = False
        if runs:
            previous = runs[-1]
            previous_position = previous.get("last_position_ms")
            starts_new_play = track_key != previous["track_key"] or (
                position_ms is not None
                and previous_position is not None
                and int(position_ms) + 2_000 < int(previous_position)
            )
        if not runs or starts_new_play:
            runs.append(
                {
                    "track_key": track_key,
                    "song_id": song_id,
                    "media": media,
                    "first_frame": int(frame["audio_frame_index"]),
                    "last_frame": int(frame["audio_frame_index"]),
                    "first_position_ms": position_ms,
                    "last_position_ms": position_ms,
                    "semantic_frame_count": 1,
                }
            )
        else:
            run = runs[-1]
            run["last_frame"] = int(frame["audio_frame_index"])
            run["last_position_ms"] = position_ms
            run["semantic_frame_count"] += 1
            if run["media"] is None and media is not None:
                run["media"] = media
            if run["song_id"] is None and song_id is not None:
                run["song_id"] = song_id
    if not runs:
        runs.append(
            {
                "track_key": f"unidentified-session:{session['id']}",
                "song_id": None,
                "media": None,
                "first_frame": 0,
                "last_frame": total_frames,
                "first_position_ms": None,
                "last_position_ms": None,
                "semantic_frame_count": 0,
            }
        )
    recordings: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        start_frame = 0 if index == 0 else int(run["first_frame"])
        end_frame = (
            int(runs[index + 1]["first_frame"])
            if index + 1 < len(runs)
            else total_frames
        )
        start_frame = min(max(0, start_frame), total_frames)
        end_frame = min(max(start_frame, end_frame), total_frames)
        identity_material = (
            f"{session['id']}\x1f{run['track_key']}\x1f{index}\x1f"
            f"{start_frame}\x1f{end_frame}"
        )
        recording_digest = hashlib.sha256(
            identity_material.encode("utf-8")
        ).hexdigest()
        split_group = str(run["track_key"])
        split_digest = hashlib.sha256(
            split_group.encode("utf-8")
        ).digest()
        split_bucket = int.from_bytes(split_digest[:2], "big") % 100
        split = (
            "train"
            if split_bucket < 80
            else "validation" if split_bucket < 90 else "test"
        )
        clips = _clips_for_range(
            segments,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_rate=int(session["sample_rate"]),
            channels=int(session["channels"]),
        )
        content_material = "\x1e".join(
            "\x1f".join(
                (
                    str(clip.get("sha256", "")),
                    str(clip["start_frame"]),
                    str(clip["frame_count"]),
                )
            )
            for clip in clips
        )
        content_digest = hashlib.sha256(
            content_material.encode("utf-8")
        ).hexdigest()
        media = run["media"] if isinstance(run["media"], dict) else {}
        captured_duration_ms = round(
            (end_frame - start_frame) * 1000 / int(session["sample_rate"])
        )
        session_integrity = session.get("integrity")
        source_audio_complete = bool(
            isinstance(session_integrity, dict)
            and session_integrity.get("source_audio_complete")
        )
        source_gap_frames = 0
        timeline = capture.get("timeline") if isinstance(capture, dict) else None
        raw_gaps = timeline.get("gaps") if isinstance(timeline, dict) else None
        if isinstance(raw_gaps, list):
            for gap in raw_gaps:
                if not isinstance(gap, dict):
                    continue
                gap_start = max(0, int(gap.get("start_frame") or 0))
                gap_end = gap_start + max(0, int(gap.get("frame_count") or 0))
                source_gap_frames += max(
                    0,
                    min(end_frame, gap_end) - max(start_frame, gap_start),
                )
            # A capture can have a gap in one song while later songs remain
            # sample-complete. Session-wide completeness must not discard the
            # unaffected recordings. Missing legacy gap locations remain
            # conservative because raw_gaps is unavailable in that case.
            source_audio_complete = bool(
                isinstance(session_integrity, dict)
                and session_integrity.get("timeline_complete")
                and int(session.get("dropped_frames") or 0) == 0
                and source_gap_frames == 0
            )
        structure_supervision = structure_supervision_completeness(
            track_duration_ms=media.get("duration_ms"),
            start_position_ms=run["first_position_ms"],
            end_position_ms=run["last_position_ms"],
            captured_duration_ms=captured_duration_ms,
            source_audio_complete=source_audio_complete,
        )
        recordings.append(
            {
                "recording_id": f"recording:{recording_digest[:32]}",
                "capture_id": session.get("capture_id"),
                "session_id": session["id"],
                "play_index": index,
                "song_id": run["song_id"],
                "track_identity": run["media"],
                "split_group_id": split_group,
                "split": split,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": end_frame - start_frame,
                "sample_rate": int(session["sample_rate"]),
                "channels": int(session["channels"]),
                "sample_width": int(session["sample_width"]),
                "first_position_ms": run["first_position_ms"],
                "last_position_ms": run["last_position_ms"],
                "semantic_frame_count": run["semantic_frame_count"],
                "source_audio_complete": source_audio_complete,
                "source_gap_frames": source_gap_frames,
                "audio_clips": clips,
                "content_fingerprint": (
                    f"sha256:{content_digest}"
                ),
                "structure_supervision": structure_supervision,
            }
        )
    return recordings


def _clips_for_range(
    segments: list[dict[str, Any]],
    *,
    start_frame: int,
    end_frame: int,
    sample_rate: int,
    channels: int,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for segment in segments:
        segment_start = int(segment["start_frame"])
        segment_end = segment_start + int(segment["frame_count"])
        overlap_start = max(start_frame, segment_start)
        overlap_end = min(end_frame, segment_end)
        if overlap_start >= overlap_end:
            continue
        clips.append(
            {
                "relative_path": segment["relative_path"],
                "start_frame": overlap_start - segment_start,
                "frame_count": overlap_end - overlap_start,
                "timeline_start_frame": overlap_start,
                "sample_rate": sample_rate,
                "channels": channels,
                "sha256": segment.get("sha256"),
            }
        )
    return clips


def _student_sequence(
    recording: dict[str, Any],
    frames: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    start = int(recording["start_frame"])
    end = int(recording["end_frame"])
    selected_frames = [
        {
            "audio_frame_index": int(frame["audio_frame_index"]),
            "position_ms": frame.get("position_ms"),
            "observation": frame.get("payload", {}).get("observation"),
            "provenance": "lumen_live_analyzer_heuristic",
        }
        for frame in frames
        if start <= int(frame["audio_frame_index"]) < end
    ]
    human_targets = [
        {
            "audio_frame_index": int(item["audio_frame_index"]),
            "position_ms": item.get("position_ms"),
            "label": item.get("label"),
            "intensity": item.get("intensity"),
            "provenance": "operator_musical_context",
        }
        for item in annotations
        if item.get("kind") == "musical_context"
        and item.get("audio_frame_index") is not None
        and start <= int(item["audio_frame_index"]) < end
    ]
    return {
        "recording_id": recording["recording_id"],
        "session_id": recording["session_id"],
        "song_id": recording["song_id"],
        "track_identity": recording["track_identity"],
        "split_group_id": recording["split_group_id"],
        "split": recording["split"],
        "audio_clips": recording["audio_clips"],
        "timeline_start_frame": start,
        "timeline_end_frame": end,
        "live_analyzer_context": selected_frames,
        "human_musical_targets": human_targets,
        "teacher_targets": [],
        "structure_supervision": recording["structure_supervision"],
        "target_notice": (
            "Live analyzer context is not ground truth. Train only from human "
            "musical targets or future versioned teacher predictions."
        ),
    }


def _choreography_sequence(
    recording: dict[str, Any],
    frames: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    start = int(recording["start_frame"])
    end = int(recording["end_frame"])
    performed_frames = [
        {
            "audio_frame_index": int(frame["audio_frame_index"]),
            "position_ms": frame.get("position_ms"),
            "observation": frame.get("payload", {}).get("observation"),
            "decision": frame.get("payload", {}).get("decision"),
            "controls": frame.get("payload", {}).get("controls"),
            "fixture_dmx": frame.get("payload", {}).get("fixture_dmx"),
            "target_provenance": (
                "heuristic_runtime_baseline_not_ground_truth"
            ),
        }
        for frame in frames
        if start <= int(frame["audio_frame_index"]) < end
    ]
    performed_indexes = [
        int(frame["audio_frame_index"]) for frame in performed_frames
    ]

    def context_link(frame_index: int) -> dict[str, int | None]:
        if not performed_indexes:
            return {
                "context_frame_audio_index": None,
                "context_frame_distance": None,
            }
        insertion = bisect_left(performed_indexes, frame_index)
        candidate_indexes = {
            max(0, min(len(performed_indexes) - 1, insertion)),
            max(0, min(len(performed_indexes) - 1, insertion - 1)),
        }
        nearest = min(
            (performed_indexes[index] for index in candidate_indexes),
            key=lambda candidate: abs(candidate - frame_index),
        )
        return {
            "context_frame_audio_index": nearest,
            "context_frame_distance": abs(nearest - frame_index),
        }

    events: list[dict[str, Any]] = []
    for item in feedback:
        frame_index = item.get("audio_frame_index")
        if frame_index is None or not start <= int(frame_index) < end:
            continue
        events.append(
            {
                "audio_frame_index": int(frame_index),
                **context_link(int(frame_index)),
                "event_id": item.get("id"),
                "created_unix_ms": item.get("created_unix_ms"),
                "position_ms": item.get("position_ms"),
                "kind": "operator_feedback",
                "label": item.get("label"),
                "scope": item.get("scope"),
                "fixture_id": item.get("fixture_id"),
                "intensity": item.get("value"),
                "note": item.get("note"),
                "context": {
                    key: item.get(key)
                    for key in (
                        "gesture", "section", "energy", "motion",
                        "tension", "confidence", "bpm", "routine",
                    )
                },
            }
        )
    for item in annotations:
        frame_index = item.get("audio_frame_index")
        if (
            item.get("kind") != "preferred_action"
            or frame_index is None
            or not start <= int(frame_index) < end
        ):
            continue
        events.append(
            {
                "audio_frame_index": int(frame_index),
                **context_link(int(frame_index)),
                "event_id": item.get("id"),
                "created_unix_ms": item.get("created_unix_ms"),
                "position_ms": item.get("position_ms"),
                "kind": "operator_preferred_action",
                "label": item.get("label"),
                "scope": item.get("scope"),
                "fixture_id": item.get("fixture_id"),
                "intensity": item.get("intensity"),
                "note": item.get("note"),
                "context": item.get("context"),
            }
        )
    events.sort(
        key=lambda item: (
            int(item["audio_frame_index"]),
            int(item.get("created_unix_ms") or 0),
        )
    )
    performed_routine_runs = _performed_routine_runs(
        performed_frames,
        recording_end_frame=end,
        sample_rate=int(recording["sample_rate"]),
    )
    preferred_action_sequence = [
        {
            "audio_frame_index": event["audio_frame_index"],
            "position_ms": event["position_ms"],
            "label": event["label"],
            "scope": event["scope"],
            "fixture_id": event["fixture_id"],
            "intensity": event["intensity"],
            "note": event.get("note"),
        }
        for event in events
        if event["kind"] == "operator_preferred_action"
    ]
    return {
        "recording_id": recording["recording_id"],
        "session_id": recording["session_id"],
        "song_id": recording["song_id"],
        "track_identity": recording["track_identity"],
        "split_group_id": recording["split_group_id"],
        "split": recording["split"],
        "audio_clips": recording["audio_clips"],
        "timeline_start_frame": start,
        "timeline_end_frame": end,
        "performed_frames": performed_frames,
        "performed_routine_runs": performed_routine_runs,
        "preferred_action_sequence": preferred_action_sequence,
        "supervision_events": events,
        "preferred_sequence_completeness": (
            "ordered_sparse_actions_without_authored_durations"
            if preferred_action_sequence
            else "none"
        ),
        "structure_supervision": recording["structure_supervision"],
        "sequence_notice": (
            "Events are sparse human preferences aligned to the uninterrupted "
            "audio clock; repeated feedback remains additive evidence. Authored "
            "durations must come from an explicit choreography sequence rather "
            "than being invented from button timing."
        ),
    }


def _performed_routine_runs(
    frames: list[dict[str, Any]],
    *,
    recording_end_frame: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for frame in frames:
        decision = frame.get("decision")
        if not isinstance(decision, dict):
            continue
        routine = str(decision.get("routine") or "auto")
        palette = decision.get("palette_hint")
        frame_index = int(frame["audio_frame_index"])
        if (
            runs
            and runs[-1]["routine"] == routine
            and runs[-1]["palette"] == palette
        ):
            runs[-1]["last_observed_frame"] = frame_index
            continue
        runs.append(
            {
                "start_frame": frame_index,
                "last_observed_frame": frame_index,
                "routine": routine,
                "palette": palette,
                "fixture_scope": "overall_semantic_decision",
                "target_provenance": (
                    "heuristic_runtime_baseline_not_ground_truth"
                ),
            }
        )
    for index, run in enumerate(runs):
        end_frame = (
            int(runs[index + 1]["start_frame"])
            if index + 1 < len(runs)
            else recording_end_frame
        )
        run["end_frame"] = max(int(run["start_frame"]), end_frame)
        run["duration_frames"] = run["end_frame"] - int(run["start_frame"])
        run["duration_s"] = run["duration_frames"] / sample_rate
    return runs


def _feedback_examples(
    session: dict[str, Any],
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _labeled_examples(
        session,
        segments,
        frames,
        feedback,
        supervision_kind="operator_preference",
        label_key="feedback",
    )


def _labeled_examples(
    session: dict[str, Any],
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    supervision_kind: str,
    label_key: str = "annotation",
) -> list[dict[str, Any]]:
    sample_rate = int(session["sample_rate"])
    frame_indexes = [int(frame["audio_frame_index"]) for frame in frames]
    examples: list[dict[str, Any]] = []
    for item in labels:
        center = item.get("audio_frame_index")
        if center is None:
            continue
        center = int(center)
        start = max(0, center - 8 * sample_rate)
        end = center + 8 * sample_rate
        nearest: dict[str, Any] | None = None
        if frame_indexes:
            insertion = bisect_left(frame_indexes, center)
            candidate_indexes = {
                max(0, min(len(frames) - 1, insertion)),
                max(0, min(len(frames) - 1, insertion - 1)),
            }
            nearest = min(
                (frames[index] for index in candidate_indexes),
                key=lambda frame: abs(
                    int(frame["audio_frame_index"]) - center
                ),
            )
        clips = _clips_for_range(
            segments,
            start_frame=start,
            end_frame=end,
            sample_rate=sample_rate,
            channels=int(session["channels"]),
        )
        examples.append(
            {
                "session_id": session["id"],
                "audio_frame_index": center,
                "window_start_frame": start,
                "window_end_frame": end,
                "audio_clips": clips,
                "audio_available": bool(clips),
                "context_frame_distance": (
                    abs(int(nearest["audio_frame_index"]) - center)
                    if nearest is not None
                    else None
                ),
                label_key: item,
                "context_frame": nearest,
                "supervision_kind": supervision_kind,
            }
        )
    return examples
