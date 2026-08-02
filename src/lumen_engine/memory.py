"""Private SQLite memory for recordings, analyses, feedback, and routines."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from lumen_engine.models import Feedback, MediaIdentity, MusicalObservation, PerformanceDecision

SCHEMA_VERSION = 5


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SongMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL,
                    provider_item_id TEXT NOT NULL,
                    title TEXT,
                    artists_json TEXT NOT NULL,
                    album TEXT,
                    duration_ms INTEGER,
                    first_seen_unix_ms INTEGER NOT NULL,
                    last_seen_unix_ms INTEGER NOT NULL,
                    play_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(provider, provider_item_id)
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    analysis_version INTEGER NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(song_id, analysis_version)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    position_ms INTEGER,
                    label TEXT NOT NULL,
                    value REAL NOT NULL,
                    note TEXT,
                    scope TEXT NOT NULL DEFAULT 'overall',
                    fixture_id TEXT,
                    gesture TEXT,
                    section TEXT,
                    energy REAL,
                    motion REAL,
                    tension REAL,
                    confidence REAL,
                    bpm REAL,
                    routine TEXT,
                    capture_session_id TEXT,
                    audio_frame_index INTEGER,
                    created_unix_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS routines (
                    song_id INTEGER PRIMARY KEY REFERENCES songs(id) ON DELETE CASCADE,
                    routine_version INTEGER NOT NULL,
                    updated_unix_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    position_ms INTEGER,
                    created_unix_ms INTEGER NOT NULL,
                    gesture TEXT NOT NULL,
                    brightness REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS feedback_song_position
                    ON feedback(song_id, position_ms);
                CREATE INDEX IF NOT EXISTS decisions_song_position
                    ON decisions(song_id, position_ms);

                CREATE TABLE IF NOT EXISTS performance_samples (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    position_ms INTEGER,
                    created_unix_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS performance_samples_session_time
                    ON performance_samples(session_id, created_unix_ms);

                CREATE TABLE IF NOT EXISTS training_sessions (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    started_unix_ms INTEGER NOT NULL,
                    ended_unix_ms INTEGER,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    sample_width INTEGER NOT NULL,
                    codec TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    frames_received INTEGER NOT NULL DEFAULT 0,
                    frames_written INTEGER NOT NULL DEFAULT 0,
                    dropped_packets INTEGER NOT NULL DEFAULT 0,
                    dropped_frames INTEGER NOT NULL DEFAULT 0,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    bytes_written INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_audio_segments (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL
                        REFERENCES training_sessions(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    relative_path TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    started_unix_ms INTEGER NOT NULL,
                    ended_unix_ms INTEGER NOT NULL,
                    byte_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    UNIQUE(session_id, segment_index)
                );

                CREATE TABLE IF NOT EXISTS training_frames (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL
                        REFERENCES training_sessions(id) ON DELETE CASCADE,
                    audio_frame_index INTEGER NOT NULL,
                    segment_index INTEGER NOT NULL,
                    segment_frame_index INTEGER NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    position_ms INTEGER,
                    payload_json TEXT NOT NULL,
                    UNIQUE(session_id, audio_frame_index)
                );

                CREATE TABLE IF NOT EXISTS training_annotations (
                    id INTEGER PRIMARY KEY,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    position_ms INTEGER,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'overall',
                    fixture_id TEXT,
                    intensity REAL NOT NULL DEFAULT 1.0,
                    note TEXT,
                    capture_session_id TEXT,
                    audio_frame_index INTEGER,
                    context_json TEXT NOT NULL,
                    created_unix_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS training_frames_session_frame
                    ON training_frames(session_id, audio_frame_index);
                CREATE INDEX IF NOT EXISTS training_frames_song_position
                    ON training_frames(song_id, position_ms);
                CREATE INDEX IF NOT EXISTS training_annotations_capture_frame
                    ON training_annotations(capture_session_id, audio_frame_index);

                CREATE TABLE IF NOT EXISTS dataset_sources (
                    source_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    version TEXT,
                    revision TEXT,
                    license TEXT,
                    root_path TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_unix_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_tracks (
                    id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL
                        REFERENCES dataset_sources(source_id) ON DELETE CASCADE,
                    source_track_id TEXT NOT NULL,
                    title TEXT,
                    artist TEXT,
                    duration_ms INTEGER,
                    split TEXT,
                    audio_path TEXT,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(source_id, source_track_id)
                );

                CREATE INDEX IF NOT EXISTS dataset_tracks_source_split
                    ON dataset_tracks(source_id, split);

                CREATE TABLE IF NOT EXISTS recording_versions (
                    id TEXT PRIMARY KEY,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    provider TEXT NOT NULL,
                    provider_item_id TEXT NOT NULL,
                    duration_ms INTEGER,
                    audio_fingerprint TEXT,
                    metadata_json TEXT NOT NULL,
                    first_seen_unix_ms INTEGER NOT NULL,
                    last_seen_unix_ms INTEGER NOT NULL,
                    UNIQUE(provider, provider_item_id, id)
                );

                CREATE TABLE IF NOT EXISTS capture_track_spans (
                    id INTEGER PRIMARY KEY,
                    capture_session_id TEXT NOT NULL
                        REFERENCES training_sessions(id) ON DELETE CASCADE,
                    recording_id TEXT
                        REFERENCES recording_versions(id) ON DELETE SET NULL,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    start_audio_frame INTEGER NOT NULL,
                    end_audio_frame INTEGER,
                    start_position_ms INTEGER,
                    end_position_ms INTEGER,
                    identity_source TEXT NOT NULL,
                    identity_confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(capture_session_id, start_audio_frame)
                );

                CREATE INDEX IF NOT EXISTS capture_track_spans_recording
                    ON capture_track_spans(recording_id, start_audio_frame);

                CREATE TABLE IF NOT EXISTS teacher_runs (
                    id TEXT PRIMARY KEY,
                    teacher_name TEXT NOT NULL,
                    teacher_version TEXT,
                    recording_id TEXT
                        REFERENCES recording_versions(id) ON DELETE SET NULL,
                    capture_session_id TEXT
                        REFERENCES training_sessions(id) ON DELETE SET NULL,
                    status TEXT NOT NULL,
                    device TEXT,
                    preprocessing_version TEXT,
                    started_unix_ms INTEGER NOT NULL,
                    ended_unix_ms INTEGER,
                    error TEXT,
                    metrics_json TEXT NOT NULL,
                    analysis_job_id TEXT
                );

                CREATE TABLE IF NOT EXISTS structure_timelines (
                    id TEXT PRIMARY KEY,
                    recording_id TEXT
                        REFERENCES recording_versions(id) ON DELETE SET NULL,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    capture_session_id TEXT
                        REFERENCES training_sessions(id) ON DELETE SET NULL,
                    dataset_source_id TEXT
                        REFERENCES dataset_sources(source_id) ON DELETE SET NULL,
                    teacher_run_id TEXT
                        REFERENCES teacher_runs(id) ON DELETE SET NULL,
                    provenance TEXT NOT NULL,
                    timeline_version TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS structure_segments (
                    timeline_id TEXT NOT NULL
                        REFERENCES structure_timelines(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER,
                    functional_label TEXT,
                    energy_label TEXT,
                    content_label TEXT,
                    beat_index INTEGER,
                    downbeat INTEGER,
                    boundary_confidence REAL NOT NULL,
                    label_confidence REAL NOT NULL,
                    raw_label TEXT,
                    provenance_json TEXT NOT NULL,
                    PRIMARY KEY(timeline_id, segment_index)
                );

                CREATE INDEX IF NOT EXISTS structure_timelines_recording
                    ON structure_timelines(recording_id, created_unix_ms);
                CREATE INDEX IF NOT EXISTS structure_segments_time
                    ON structure_segments(timeline_id, start_ms);

                CREATE TABLE IF NOT EXISTS metrical_events (
                    timeline_id TEXT NOT NULL
                        REFERENCES structure_timelines(id) ON DELETE CASCADE,
                    event_index INTEGER NOT NULL,
                    time_ms INTEGER NOT NULL,
                    position_in_bar INTEGER,
                    bar_number INTEGER,
                    confidence REAL NOT NULL,
                    provenance_json TEXT NOT NULL,
                    PRIMARY KEY(timeline_id, event_index)
                );

                CREATE INDEX IF NOT EXISTS metrical_events_time
                    ON metrical_events(timeline_id, time_ms);

                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_unix_ms INTEGER NOT NULL,
                    updated_unix_ms INTEGER NOT NULL,
                    worker_id TEXT,
                    worker_pid INTEGER,
                    heartbeat_unix_ms INTEGER
                );

                CREATE INDEX IF NOT EXISTS analysis_jobs_status_priority
                    ON analysis_jobs(status, priority DESC, created_unix_ms);

                CREATE TABLE IF NOT EXISTS choreography_sequences (
                    id TEXT PRIMARY KEY,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    timeline_id TEXT
                        REFERENCES structure_timelines(id) ON DELETE SET NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context_json TEXT NOT NULL,
                    created_unix_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS choreography_steps (
                    sequence_id TEXT NOT NULL
                        REFERENCES choreography_sequences(id) ON DELETE CASCADE,
                    step_index INTEGER NOT NULL,
                    start_beat REAL NOT NULL,
                    duration_beats REAL NOT NULL,
                    fixture_scope TEXT NOT NULL,
                    routine TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    palette TEXT,
                    parameters_json TEXT NOT NULL,
                    PRIMARY KEY(sequence_id, step_index)
                );
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(feedback)")
            }
            if "scope" not in columns:
                connection.execute(
                    "ALTER TABLE feedback ADD COLUMN scope TEXT NOT NULL DEFAULT 'overall'"
                )
            if "fixture_id" not in columns:
                connection.execute(
                    "ALTER TABLE feedback ADD COLUMN fixture_id TEXT"
                )
            for name, definition in (
                ("gesture", "TEXT"), ("section", "TEXT"),
                ("energy", "REAL"), ("motion", "REAL"),
                ("tension", "REAL"), ("confidence", "REAL"),
                ("bpm", "REAL"), ("routine", "TEXT"),
                ("capture_session_id", "TEXT"),
                ("audio_frame_index", "INTEGER"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE feedback ADD COLUMN {name} {definition}")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS feedback_capture_frame
                ON feedback(capture_session_id, audio_frame_index)
                """
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            teacher_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(teacher_runs)"
                )
            }
            if "analysis_job_id" not in teacher_columns:
                connection.execute(
                    "ALTER TABLE teacher_runs ADD COLUMN analysis_job_id TEXT"
                )
            job_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(analysis_jobs)"
                )
            }
            for name, definition in (
                ("worker_id", "TEXT"),
                ("worker_pid", "INTEGER"),
                ("heartbeat_unix_ms", "INTEGER"),
            ):
                if name not in job_columns:
                    connection.execute(
                        f"ALTER TABLE analysis_jobs ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS teacher_runs_analysis_job
                ON teacher_runs(analysis_job_id)
                """
            )

    def remember_media(self, media: MediaIdentity, count_play: bool = False) -> int:
        now_ms = int(time.time() * 1000)
        provider_item_id = media.provider_item_id or self._fallback_identity(media)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO songs(
                    provider, provider_item_id, title, artists_json, album,
                    duration_ms, first_seen_unix_ms, last_seen_unix_ms, play_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, provider_item_id) DO UPDATE SET
                    title=excluded.title,
                    artists_json=excluded.artists_json,
                    album=excluded.album,
                    duration_ms=COALESCE(excluded.duration_ms, songs.duration_ms),
                    last_seen_unix_ms=excluded.last_seen_unix_ms,
                    play_count=songs.play_count + excluded.play_count
                """,
                (
                    media.provider,
                    provider_item_id,
                    media.title,
                    json.dumps(media.artists),
                    media.album,
                    media.duration_ms,
                    now_ms,
                    now_ms,
                    1 if count_play else 0,
                ),
            )
            row = connection.execute(
                "SELECT id FROM songs WHERE provider=? AND provider_item_id=?",
                (media.provider, provider_item_id),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def get_song(self, song_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM songs WHERE id=?", (song_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["artists"] = tuple(json.loads(result.pop("artists_json")))
        return result

    def save_analysis(
        self, song_id: int, analysis_version: int, payload: dict[str, Any]
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO analyses(
                    song_id, analysis_version, created_unix_ms, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(song_id, analysis_version) DO UPDATE SET
                    created_unix_ms=excluded.created_unix_ms,
                    payload_json=excluded.payload_json
                """,
                (
                    song_id,
                    analysis_version,
                    int(time.time() * 1000),
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def latest_analysis(self, song_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT analysis_version, created_unix_ms, payload_json
                FROM analyses WHERE song_id=?
                ORDER BY analysis_version DESC LIMIT 1
                """,
                (song_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "analysis_version": row["analysis_version"],
            "created_unix_ms": row["created_unix_ms"],
            "payload": json.loads(row["payload_json"]),
        }

    def add_feedback(self, feedback: Feedback) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback(
                    song_id, position_ms, label, value, note, scope,
                    fixture_id, gesture, section, energy, motion, tension,
                    confidence, bpm, routine, capture_session_id,
                    audio_frame_index, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.song_id,
                    feedback.position_ms,
                    feedback.label,
                    feedback.value,
                    feedback.note,
                    feedback.scope,
                    feedback.fixture_id,
                    feedback.gesture,
                    feedback.section,
                    feedback.energy,
                    feedback.motion,
                    feedback.tension,
                    feedback.confidence,
                    feedback.bpm,
                    feedback.routine,
                    feedback.capture_session_id,
                    feedback.audio_frame_index,
                    int(time.time() * 1000),
                ),
            )
            return int(cursor.lastrowid)

    def list_feedback(self, song_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, position_ms, label, value, note, scope, fixture_id,
                       gesture, section, energy, motion, tension, confidence, bpm, routine,
                       capture_session_id, audio_frame_index, created_unix_ms
                FROM feedback WHERE song_id=?
                ORDER BY COALESCE(position_ms, -1), created_unix_ms
                """,
                (song_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_feedback(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, song_id, position_ms, label, value, note, scope, fixture_id, "
                "gesture, section, energy, motion, tension, confidence, bpm, routine, "
                "capture_session_id, audio_frame_index, created_unix_ms "
                "FROM feedback"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_feedback(self, feedback_id: int) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
            return cursor.rowcount > 0

    def summary(self, limit: int = 30) -> dict[str, Any]:
        """Return compact operator-facing song and feedback history."""

        if limit < 1:
            raise ValueError("limit must be at least 1")
        with closing(self._connect()) as connection:
            songs = connection.execute(
                """
                SELECT
                    songs.id,
                    songs.provider,
                    songs.provider_item_id,
                    songs.title,
                    songs.artists_json,
                    songs.album,
                    songs.duration_ms,
                    songs.first_seen_unix_ms,
                    songs.last_seen_unix_ms,
                    songs.play_count,
                    COUNT(DISTINCT feedback.id) AS feedback_count,
                    COUNT(DISTINCT decisions.id) AS decision_count
                FROM songs
                LEFT JOIN feedback ON feedback.song_id = songs.id
                LEFT JOIN decisions ON decisions.song_id = songs.id
                GROUP BY songs.id
                ORDER BY songs.last_seen_unix_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            feedback = connection.execute(
                """
                SELECT
                    feedback.id,
                    feedback.song_id,
                    songs.title AS song_title,
                    feedback.position_ms,
                    feedback.label,
                    feedback.value,
                    feedback.note,
                    feedback.scope,
                    feedback.fixture_id,
                    feedback.gesture,
                    feedback.section,
                    feedback.energy,
                    feedback.motion,
                    feedback.tension,
                    feedback.capture_session_id,
                    feedback.audio_frame_index,
                    feedback.created_unix_ms
                FROM feedback
                JOIN songs ON songs.id = feedback.song_id
                ORDER BY feedback.created_unix_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            totals = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM songs) AS songs,
                    (SELECT COUNT(*) FROM feedback) AS feedback,
                    (
                        SELECT COUNT(*) FROM decisions
                        WHERE song_id IS NOT NULL
                    ) AS decisions,
                    (SELECT COUNT(*) FROM routines) AS routines
                """
            ).fetchone()
        song_items: list[dict[str, Any]] = []
        for row in songs:
            item = dict(row)
            item["artists"] = list(json.loads(item.pop("artists_json")))
            song_items.append(item)
        return {
            "totals": dict(totals) if totals is not None else {},
            "songs": song_items,
            "recent_feedback": [dict(row) for row in feedback],
        }

    def save_routine(
        self, song_id: int, routine_version: int, payload: dict[str, Any]
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO routines(
                    song_id, routine_version, updated_unix_ms, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    routine_version=excluded.routine_version,
                    updated_unix_ms=excluded.updated_unix_ms,
                    payload_json=excluded.payload_json
                """,
                (
                    song_id,
                    routine_version,
                    int(time.time() * 1000),
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def get_routine(self, song_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT routine_version, updated_unix_ms, payload_json
                FROM routines WHERE song_id=?
                """,
                (song_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "routine_version": row["routine_version"],
            "updated_unix_ms": row["updated_unix_ms"],
            "payload": json.loads(row["payload_json"]),
        }

    def log_decision(
        self,
        decision: PerformanceDecision,
        song_id: int | None = None,
        position_ms: int | None = None,
        observation: MusicalObservation | None = None,
    ) -> int:
        payload = asdict(decision)
        payload["gesture"] = decision.gesture.value
        if observation is not None:
            payload["observation"] = asdict(observation)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO decisions(
                    song_id, position_ms, created_unix_ms, gesture, brightness,
                    confidence, reason, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    song_id,
                    position_ms,
                    int(time.time() * 1000),
                    decision.gesture.value,
                    decision.brightness,
                    decision.confidence,
                    decision.reason,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def log_performance_sample(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        song_id: int | None = None,
        position_ms: int | None = None,
    ) -> int:
        """Persist a compact time-series sample for last-run diagnosis."""
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO performance_samples(
                    session_id, song_id, position_ms, created_unix_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    song_id,
                    position_ms,
                    int(time.time() * 1000),
                    json.dumps(payload, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def latest_performance_session(self, limit: int = 2400) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT session_id FROM performance_samples
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return []
            rows = connection.execute(
                """
                SELECT id, session_id, song_id, position_ms, created_unix_ms,
                       payload_json
                FROM performance_samples WHERE session_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (row["session_id"], max(1, int(limit))),
            ).fetchall()
        result = []
        for sample in reversed(rows):
            item = dict(sample)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def begin_training_session(
        self,
        *,
        session_id: str,
        mode: str,
        sample_rate: int,
        channels: int,
        sample_width: int,
        relative_path: str,
        metadata: dict[str, Any],
    ) -> None:
        """Register a local PCM capture before its writer thread starts."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO training_sessions(
                    id, mode, started_unix_ms, sample_rate, channels,
                    sample_width, codec, relative_path, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'pcm_s16le_wav', ?, 'recording', ?)
                """,
                (
                    session_id,
                    mode,
                    int(time.time() * 1000),
                    sample_rate,
                    channels,
                    sample_width,
                    relative_path,
                    json.dumps(metadata, sort_keys=True),
                ),
            )

    def finish_training_session(
        self,
        session_id: str,
        *,
        status: str,
        frames_received: int,
        frames_written: int,
        dropped_packets: int,
        dropped_frames: int,
        segment_count: int,
        bytes_written: int,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE training_sessions SET
                    ended_unix_ms=?,
                    status=?,
                    frames_received=?,
                    frames_written=?,
                    dropped_packets=?,
                    dropped_frames=?,
                    segment_count=?,
                    bytes_written=?
                WHERE id=?
                """,
                (
                    int(time.time() * 1000),
                    status,
                    frames_received,
                    frames_written,
                    dropped_packets,
                    dropped_frames,
                    segment_count,
                    bytes_written,
                    session_id,
                ),
            )

    def add_training_segment(
        self,
        *,
        session_id: str,
        segment_index: int,
        relative_path: str,
        start_frame: int,
        frame_count: int,
        started_unix_ms: int,
        ended_unix_ms: int,
        byte_count: int,
        sha256: str,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO training_audio_segments(
                    session_id, segment_index, relative_path, start_frame,
                    frame_count, started_unix_ms, ended_unix_ms, byte_count, sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, segment_index) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    start_frame=excluded.start_frame,
                    frame_count=excluded.frame_count,
                    started_unix_ms=excluded.started_unix_ms,
                    ended_unix_ms=excluded.ended_unix_ms,
                    byte_count=excluded.byte_count,
                    sha256=excluded.sha256
                """,
                (
                    session_id,
                    segment_index,
                    relative_path,
                    start_frame,
                    frame_count,
                    started_unix_ms,
                    ended_unix_ms,
                    byte_count,
                    sha256,
                ),
            )

    def add_training_frames(self, rows: list[dict[str, Any]]) -> None:
        """Write a batch of synchronized semantic frames off the audio thread."""
        if not rows:
            return
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                """
                INSERT INTO training_frames(
                    session_id, audio_frame_index, segment_index,
                    segment_frame_index, created_unix_ms, song_id,
                    position_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, audio_frame_index) DO UPDATE SET
                    created_unix_ms=excluded.created_unix_ms,
                    song_id=excluded.song_id,
                    position_ms=excluded.position_ms,
                    payload_json=excluded.payload_json
                """,
                [
                    (
                        row["session_id"],
                        row["audio_frame_index"],
                        row["segment_index"],
                        row["segment_frame_index"],
                        row["created_unix_ms"],
                        row.get("song_id"),
                        row.get("position_ms"),
                        json.dumps(row["payload"], sort_keys=True),
                    )
                    for row in rows
                ],
            )

    def training_summary(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS sessions,
                    COALESCE(SUM(segment_count), 0) AS segments,
                    COALESCE(SUM(frames_written), 0) AS frames,
                    COALESCE(SUM(bytes_written), 0) AS bytes,
                    COALESCE(SUM(dropped_frames), 0) AS dropped_frames
                FROM training_sessions
                """
            ).fetchone()
            feature_count = connection.execute(
                "SELECT COUNT(*) AS value FROM training_frames"
            ).fetchone()
            linked_feedback = connection.execute(
                """
                SELECT COUNT(*) AS value FROM feedback
                WHERE capture_session_id IS NOT NULL
                  AND audio_frame_index IS NOT NULL
                """
            ).fetchone()
            annotation_count = connection.execute(
                "SELECT COUNT(*) AS value FROM training_annotations"
            ).fetchone()
            latest = connection.execute(
                """
                SELECT * FROM training_sessions
                ORDER BY started_unix_ms DESC LIMIT 1
                """
            ).fetchone()
        result = dict(totals) if totals is not None else {}
        result["feature_frames"] = int(feature_count["value"]) if feature_count else 0
        result["linked_feedback"] = (
            int(linked_feedback["value"]) if linked_feedback else 0
        )
        result["annotations"] = (
            int(annotation_count["value"]) if annotation_count else 0
        )
        if latest is not None:
            latest_item = dict(latest)
            latest_item["metadata"] = json.loads(
                latest_item.pop("metadata_json")
            )
            result["latest_session"] = latest_item
        else:
            result["latest_session"] = None
        return result

    def training_sessions(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM training_sessions ORDER BY started_unix_ms"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def training_segments(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM training_audio_segments
                WHERE session_id=? ORDER BY segment_index
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def training_frames(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, audio_frame_index, segment_index,
                       segment_frame_index, created_unix_ms, song_id,
                       position_ms, payload_json
                FROM training_frames
                WHERE session_id=? ORDER BY audio_frame_index
                """,
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def training_feedback(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, song_id, position_ms, label, value, note, scope,
                       fixture_id, gesture, section, energy, motion, tension,
                       confidence, bpm, routine, capture_session_id,
                       audio_frame_index, created_unix_ms
                FROM feedback
                WHERE capture_session_id=?
                ORDER BY audio_frame_index, created_unix_ms
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_training_annotation(
        self,
        *,
        song_id: int | None,
        position_ms: int | None,
        kind: str,
        label: str,
        scope: str,
        fixture_id: str | None,
        intensity: float,
        note: str | None,
        capture_session_id: str | None,
        audio_frame_index: int | None,
        context: dict[str, Any],
    ) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO training_annotations(
                    song_id, position_ms, kind, label, scope, fixture_id,
                    intensity, note, capture_session_id, audio_frame_index,
                    context_json, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    song_id,
                    position_ms,
                    kind,
                    label,
                    scope,
                    fixture_id,
                    intensity,
                    note,
                    capture_session_id,
                    audio_frame_index,
                    json.dumps(context, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )
            return int(cursor.lastrowid)

    def training_annotations(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM training_annotations
                WHERE capture_session_id=?
                ORDER BY audio_frame_index, created_unix_ms
                """,
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item.pop("context_json"))
            result.append(item)
        return result

    def register_dataset_source(
        self,
        *,
        source_id: str,
        display_name: str,
        role: str,
        status: str,
        version: str | None = None,
        revision: str | None = None,
        license_name: str | None = None,
        root_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a public annotation source without importing its audio."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO dataset_sources(
                    source_id, display_name, role, version, revision, license,
                    root_path, status, metadata_json, updated_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    role=excluded.role,
                    version=excluded.version,
                    revision=excluded.revision,
                    license=excluded.license,
                    root_path=excluded.root_path,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    updated_unix_ms=excluded.updated_unix_ms
                """,
                (
                    source_id,
                    display_name,
                    role,
                    version,
                    revision,
                    license_name,
                    root_path,
                    status,
                    json.dumps(metadata or {}, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )

    def list_dataset_sources(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_sources ORDER BY source_id"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def upsert_dataset_track(
        self,
        *,
        source_id: str,
        source_track_id: str,
        title: str | None = None,
        artist: str | None = None,
        duration_ms: int | None = None,
        split: str | None = None,
        audio_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO dataset_tracks(
                    source_id, source_track_id, title, artist, duration_ms,
                    split, audio_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, source_track_id) DO UPDATE SET
                    title=excluded.title,
                    artist=excluded.artist,
                    duration_ms=excluded.duration_ms,
                    split=excluded.split,
                    audio_path=excluded.audio_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    source_id,
                    source_track_id,
                    title,
                    artist,
                    duration_ms,
                    split,
                    audio_path,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            row = connection.execute(
                """
                SELECT id FROM dataset_tracks
                WHERE source_id=? AND source_track_id=?
                """,
                (source_id, source_track_id),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def remember_recording_version(
        self,
        *,
        provider: str,
        provider_item_id: str,
        song_id: int | None = None,
        duration_ms: int | None = None,
        audio_fingerprint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return a stable recording/master identity distinct from a capture."""
        identity_material = "\x1f".join(
            (
                provider.strip().lower(),
                provider_item_id.strip(),
                str(duration_ms or ""),
                audio_fingerprint or "",
            )
        )
        recording_id = (
            "recording:"
            + hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:32]
        )
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO recording_versions(
                    id, song_id, provider, provider_item_id, duration_ms,
                    audio_fingerprint, metadata_json, first_seen_unix_ms,
                    last_seen_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    song_id=COALESCE(excluded.song_id, recording_versions.song_id),
                    duration_ms=COALESCE(
                        excluded.duration_ms, recording_versions.duration_ms
                    ),
                    audio_fingerprint=COALESCE(
                        excluded.audio_fingerprint,
                        recording_versions.audio_fingerprint
                    ),
                    metadata_json=excluded.metadata_json,
                    last_seen_unix_ms=excluded.last_seen_unix_ms
                """,
                (
                    recording_id,
                    song_id,
                    provider,
                    provider_item_id,
                    duration_ms,
                    audio_fingerprint,
                    json.dumps(metadata or {}, sort_keys=True),
                    now_ms,
                    now_ms,
                ),
            )
        return recording_id

    def add_capture_track_span(
        self,
        *,
        capture_session_id: str,
        start_audio_frame: int,
        identity_source: str,
        identity_confidence: float,
        recording_id: str | None = None,
        song_id: int | None = None,
        end_audio_frame: int | None = None,
        start_position_ms: int | None = None,
        end_position_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO capture_track_spans(
                    capture_session_id, recording_id, song_id,
                    start_audio_frame, end_audio_frame, start_position_ms,
                    end_position_ms, identity_source, identity_confidence,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(capture_session_id, start_audio_frame) DO UPDATE SET
                    recording_id=excluded.recording_id,
                    song_id=excluded.song_id,
                    end_audio_frame=excluded.end_audio_frame,
                    start_position_ms=excluded.start_position_ms,
                    end_position_ms=excluded.end_position_ms,
                    identity_source=excluded.identity_source,
                    identity_confidence=excluded.identity_confidence,
                    metadata_json=excluded.metadata_json
                """,
                (
                    capture_session_id,
                    recording_id,
                    song_id,
                    start_audio_frame,
                    end_audio_frame,
                    start_position_ms,
                    end_position_ms,
                    identity_source,
                    max(0.0, min(1.0, identity_confidence)),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = connection.execute(
                """
                SELECT id FROM capture_track_spans
                WHERE capture_session_id=? AND start_audio_frame=?
                """,
                (capture_session_id, start_audio_frame),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def begin_teacher_run(
        self,
        *,
        teacher_name: str,
        teacher_version: str | None,
        device: str | None,
        preprocessing_version: str | None,
        recording_id: str | None = None,
        capture_session_id: str | None = None,
        analysis_job_id: str | None = None,
    ) -> str:
        run_id = f"teacher:{uuid.uuid4()}"
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO teacher_runs(
                    id, teacher_name, teacher_version, recording_id,
                    capture_session_id, status, device,
                    preprocessing_version, started_unix_ms, metrics_json,
                    analysis_job_id
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, '{}', ?)
                """,
                (
                    run_id,
                    teacher_name,
                    teacher_version,
                    recording_id,
                    capture_session_id,
                    device,
                    preprocessing_version,
                    int(time.time() * 1000),
                    analysis_job_id,
                ),
            )
        return run_id

    def finish_teacher_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE teacher_runs SET status=?, ended_unix_ms=?, error=?,
                    metrics_json=?
                WHERE id=?
                """,
                (
                    status,
                    int(time.time() * 1000),
                    error,
                    json.dumps(metrics or {}, sort_keys=True),
                    run_id,
                ),
            )

    def list_teacher_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Return teacher runs with decoded, database-owned provenance.

        Student training uses this list as its authority. Files merely present
        in the research export directory are not trusted because smoke tests,
        interrupted runs, or a different memory database may have created
        them.
        """

        with closing(self._connect()) as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT * FROM teacher_runs
                    ORDER BY started_unix_ms DESC LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM teacher_runs WHERE status=?
                    ORDER BY started_unix_ms DESC LIMIT ?
                    """,
                    (str(status), max(1, int(limit))),
                ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result

    def save_structure_timeline(
        self,
        *,
        provenance: str,
        timeline_version: str,
        confidence: float,
        segments: list[dict[str, Any]],
        beats: list[dict[str, Any]] | None = None,
        timeline_id: str | None = None,
        recording_id: str | None = None,
        song_id: int | None = None,
        capture_session_id: str | None = None,
        dataset_source_id: str | None = None,
        teacher_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a normalized, multi-axis structure timeline atomically."""
        resolved_id = timeline_id or f"timeline:{uuid.uuid4()}"
        previous_start = -1
        for index, segment in enumerate(segments):
            start_ms = int(segment["start_ms"])
            end_ms = segment.get("end_ms")
            if start_ms < 0 or start_ms < previous_start:
                raise ValueError("structure segment starts must be non-negative and ordered")
            if end_ms is not None and int(end_ms) <= start_ms:
                raise ValueError("structure segment end must be after its start")
            if int(segment.get("segment_index", index)) != index:
                raise ValueError("structure segment indexes must be contiguous")
            previous_start = start_ms
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO structure_timelines(
                    id, recording_id, song_id, capture_session_id,
                    dataset_source_id, teacher_run_id, provenance,
                    timeline_version, confidence, created_unix_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    recording_id=excluded.recording_id,
                    song_id=excluded.song_id,
                    capture_session_id=excluded.capture_session_id,
                    dataset_source_id=excluded.dataset_source_id,
                    teacher_run_id=excluded.teacher_run_id,
                    provenance=excluded.provenance,
                    timeline_version=excluded.timeline_version,
                    confidence=excluded.confidence,
                    created_unix_ms=excluded.created_unix_ms,
                    metadata_json=excluded.metadata_json
                """,
                (
                    resolved_id,
                    recording_id,
                    song_id,
                    capture_session_id,
                    dataset_source_id,
                    teacher_run_id,
                    provenance,
                    timeline_version,
                    max(0.0, min(1.0, confidence)),
                    int(time.time() * 1000),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            connection.execute(
                "DELETE FROM structure_segments WHERE timeline_id=?",
                (resolved_id,),
            )
            connection.executemany(
                """
                INSERT INTO structure_segments(
                    timeline_id, segment_index, start_ms, end_ms,
                    functional_label, energy_label, content_label, beat_index,
                    downbeat, boundary_confidence, label_confidence, raw_label,
                    provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        resolved_id,
                        index,
                        int(segment["start_ms"]),
                        (
                            int(segment["end_ms"])
                            if segment.get("end_ms") is not None
                            else None
                        ),
                        segment.get("functional_label"),
                        segment.get("energy_label"),
                        segment.get("content_label"),
                        segment.get("beat_index"),
                        (
                            int(bool(segment["downbeat"]))
                            if segment.get("downbeat") is not None
                            else None
                        ),
                        max(
                            0.0,
                            min(
                                1.0,
                                float(segment.get("boundary_confidence", confidence)),
                            ),
                        ),
                        max(
                            0.0,
                            min(
                                1.0,
                                float(segment.get("label_confidence", confidence)),
                            ),
                        ),
                        segment.get("raw_label"),
                        json.dumps(
                            segment.get("provenance", {"source": provenance}),
                            sort_keys=True,
                        ),
                    )
                    for index, segment in enumerate(segments)
                ],
            )
            connection.execute(
                "DELETE FROM metrical_events WHERE timeline_id=?",
                (resolved_id,),
            )
            connection.executemany(
                """
                INSERT INTO metrical_events(
                    timeline_id, event_index, time_ms, position_in_bar,
                    bar_number, confidence, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        resolved_id,
                        index,
                        int(beat["time_ms"]),
                        beat.get("position_in_bar"),
                        beat.get("bar_number"),
                        max(
                            0.0,
                            min(1.0, float(beat.get("confidence", 1.0))),
                        ),
                        json.dumps(
                            beat.get("provenance", {"source": provenance}),
                            sort_keys=True,
                        ),
                    )
                    for index, beat in enumerate(beats or [])
                ],
            )
        return resolved_id

    def structure_timeline(self, timeline_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            timeline = connection.execute(
                "SELECT * FROM structure_timelines WHERE id=?",
                (timeline_id,),
            ).fetchone()
            if timeline is None:
                return None
            rows = connection.execute(
                """
                SELECT * FROM structure_segments
                WHERE timeline_id=? ORDER BY segment_index
                """,
                (timeline_id,),
            ).fetchall()
            beat_rows = connection.execute(
                """
                SELECT * FROM metrical_events
                WHERE timeline_id=? ORDER BY event_index
                """,
                (timeline_id,),
            ).fetchall()
        result = dict(timeline)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["segments"] = []
        for row in rows:
            segment = dict(row)
            segment["downbeat"] = (
                bool(segment["downbeat"])
                if segment["downbeat"] is not None
                else None
            )
            segment["provenance"] = json.loads(
                segment.pop("provenance_json")
            )
            result["segments"].append(segment)
        result["beats"] = []
        for row in beat_rows:
            beat = dict(row)
            beat["provenance"] = json.loads(beat.pop("provenance_json"))
            result["beats"].append(beat)
        return result

    def structure_timelines_for_recording(
        self, recording_id: str
    ) -> list[dict[str, Any]]:
        """Return complete timelines for one stable recording identity."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id FROM structure_timelines
                WHERE recording_id=?
                ORDER BY created_unix_ms DESC, id
                """,
                (recording_id,),
            ).fetchall()
        return [
            timeline
            for row in rows
            if (timeline := self.structure_timeline(str(row["id"])))
            is not None
        ]

    def capture_spans_for_recording(
        self, recording_id: str
    ) -> list[dict[str, Any]]:
        """Return listening-run ranges associated with one recording."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT capture_track_spans.*, training_sessions.sample_rate,
                       training_sessions.channels,
                       training_sessions.sample_width,
                       training_sessions.status AS capture_session_status,
                       training_sessions.frames_received,
                       training_sessions.frames_written,
                       training_sessions.dropped_frames,
                       recording_versions.provider AS recording_provider,
                       recording_versions.provider_item_id
                           AS recording_provider_item_id,
                       recording_versions.duration_ms
                           AS recording_duration_ms,
                       recording_versions.metadata_json
                           AS recording_metadata_json
                FROM capture_track_spans
                JOIN training_sessions
                  ON training_sessions.id =
                     capture_track_spans.capture_session_id
                LEFT JOIN recording_versions
                  ON recording_versions.id = capture_track_spans.recording_id
                WHERE capture_track_spans.recording_id=?
                ORDER BY capture_track_spans.capture_session_id,
                         capture_track_spans.start_audio_frame
                """,
                (recording_id,),
            ).fetchall()
        return self._decode_capture_span_rows(rows)

    def capture_track_spans(self, *, limit: int = 100_000) -> list[dict[str, Any]]:
        """Return every captured track span with recording/session evidence.

        This is the inventory view used by training readiness.  It includes
        spans that were deliberately too short or incomplete to receive a
        teacher job, so the operator counts never silently omit them.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT capture_track_spans.*, training_sessions.sample_rate,
                       training_sessions.channels,
                       training_sessions.sample_width,
                       training_sessions.status AS capture_session_status,
                       training_sessions.frames_received,
                       training_sessions.frames_written,
                       training_sessions.dropped_frames,
                       recording_versions.provider AS recording_provider,
                       recording_versions.provider_item_id
                           AS recording_provider_item_id,
                       recording_versions.duration_ms
                           AS recording_duration_ms,
                       recording_versions.metadata_json
                           AS recording_metadata_json
                FROM capture_track_spans
                JOIN training_sessions
                  ON training_sessions.id =
                     capture_track_spans.capture_session_id
                LEFT JOIN recording_versions
                  ON recording_versions.id = capture_track_spans.recording_id
                ORDER BY capture_track_spans.capture_session_id,
                         capture_track_spans.start_audio_frame
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return self._decode_capture_span_rows(rows)

    @staticmethod
    def _decode_capture_span_rows(rows: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            recording_metadata = item.pop("recording_metadata_json")
            item["recording_metadata"] = (
                json.loads(recording_metadata) if recording_metadata else {}
            )
            result.append(item)
        return result

    def enqueue_analysis_job(
        self,
        *,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> str:
        job_id = f"job:{uuid.uuid4()}"
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs(
                    id, job_type, status, priority, payload_json,
                    created_unix_ms, updated_unix_ms
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    int(priority),
                    json.dumps(payload, sort_keys=True),
                    now_ms,
                    now_ms,
                ),
            )
        return job_id

    def update_analysis_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE analysis_jobs SET status=?, result_json=?, error=?,
                    attempts=attempts + ?, updated_unix_ms=?,
                    worker_id=NULL, worker_pid=NULL, heartbeat_unix_ms=NULL
                WHERE id=?
                """,
                (
                    status,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error,
                    1 if increment_attempts else 0,
                    int(time.time() * 1000),
                    job_id,
                ),
            )

    def update_analysis_job_priority(
        self, job_id: str, *, priority: int
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE analysis_jobs SET priority=?, updated_unix_ms=?
                WHERE id=? AND status='queued'
                """,
                (int(priority), int(time.time() * 1000), job_id),
            )

    def list_analysis_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_jobs
                ORDER BY created_unix_ms DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result_json = item.pop("result_json")
            item["result"] = json.loads(result_json) if result_json else None
            result.append(item)
        return result

    def claim_analysis_job(
        self,
        job_types: tuple[str, ...] = (),
        *,
        worker_id: str | None = None,
        worker_pid: int | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim the highest-priority queued offline job.

        Teacher workers run in separate processes from the live DMX loop.  An
        immediate SQLite transaction prevents two workers from processing the
        same recording when the operator starts more than one worker.
        """
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            parameters: list[Any] = []
            where = "status='queued'"
            if job_types:
                placeholders = ", ".join("?" for _ in job_types)
                where += f" AND job_type IN ({placeholders})"
                parameters.extend(job_types)
            row = connection.execute(
                f"""
                SELECT * FROM analysis_jobs
                WHERE {where}
                ORDER BY priority DESC, created_unix_ms, id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now_ms = int(time.time() * 1000)
            cursor = connection.execute(
                """
                UPDATE analysis_jobs
                SET status='running', attempts=attempts + 1,
                    updated_unix_ms=?, error=NULL, worker_id=?,
                    worker_pid=?, heartbeat_unix_ms=?
                WHERE id=? AND status='queued'
                """,
                (
                    now_ms,
                    worker_id,
                    int(worker_pid) if worker_pid is not None else None,
                    now_ms,
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        item = dict(row)
        item["status"] = "running"
        item["attempts"] = int(item["attempts"]) + 1
        item["payload"] = json.loads(item.pop("payload_json"))
        result_json = item.pop("result_json")
        item["result"] = json.loads(result_json) if result_json else None
        item["worker_id"] = worker_id
        item["worker_pid"] = worker_pid
        item["heartbeat_unix_ms"] = now_ms
        return item

    def heartbeat_analysis_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        """Renew an offline job lease and optionally persist safe progress."""

        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection, connection:
            if progress is None:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET heartbeat_unix_ms=?, updated_unix_ms=?
                    WHERE id=? AND status='running' AND worker_id=?
                    """,
                    (now_ms, now_ms, job_id, worker_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET heartbeat_unix_ms=?, updated_unix_ms=?, result_json=?
                    WHERE id=? AND status='running' AND worker_id=?
                    """,
                    (
                        now_ms,
                        now_ms,
                        json.dumps(progress, sort_keys=True),
                        job_id,
                        worker_id,
                    ),
                )
        return cursor.rowcount == 1

    def recover_abandoned_analysis_jobs(
        self,
        *,
        stale_after_ms: int = 120_000,
        now_unix_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Requeue jobs whose owning worker died or stopped heartbeating.

        Completed jobs and teacher results are never altered. A running
        teacher attempt tied to an abandoned job is marked failed; the
        requeued job will create a fresh run while preserving its provenance.
        Legacy jobs without lease metadata are considered abandoned.
        """

        now_ms = int(time.time() * 1000) if now_unix_ms is None else int(
            now_unix_ms
        )
        cutoff_ms = now_ms - max(1_000, int(stale_after_ms))
        recovered: list[dict[str, Any]] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM analysis_jobs WHERE status='running'"
            ).fetchall()
            for row in rows:
                heartbeat_ms = row["heartbeat_unix_ms"]
                worker_pid = row["worker_pid"]
                stale = heartbeat_ms is None or int(heartbeat_ms) < cutoff_ms
                owner_dead = worker_pid is None or not _process_is_alive(
                    int(worker_pid)
                )
                if not stale and not owner_dead:
                    continue
                reason = (
                    "Recovered after the previous offline worker stopped "
                    "without completing this job"
                )
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status='queued', error=?, updated_unix_ms=?,
                        worker_id=NULL, worker_pid=NULL,
                        heartbeat_unix_ms=NULL
                    WHERE id=? AND status='running'
                    """,
                    (reason, now_ms, row["id"]),
                )
                if cursor.rowcount != 1:
                    continue
                teacher_names = {
                    "teacher.edmformer": "EDMFormer",
                    "teacher.songformer": "SongFormer",
                }
                teacher_name = teacher_names.get(str(row["job_type"]))
                connection.execute(
                    """
                    UPDATE teacher_runs
                    SET status='failed', ended_unix_ms=?, error=?
                    WHERE status='running' AND analysis_job_id=?
                    """,
                    (now_ms, reason, row["id"]),
                )
                if teacher_name is not None:
                    payload = json.loads(row["payload_json"])
                    # Backfill the pre-lease schema's stranded runs. New runs
                    # use the exact analysis_job_id relationship above.
                    connection.execute(
                        """
                        UPDATE teacher_runs
                        SET status='failed', ended_unix_ms=?, error=?
                        WHERE status='running' AND analysis_job_id IS NULL
                          AND teacher_name=?
                          AND recording_id IS ?
                          AND capture_session_id IS ?
                        """,
                        (
                            now_ms,
                            reason,
                            teacher_name,
                            payload.get("recording_id"),
                            payload.get("capture_session_id"),
                        ),
                    )
                recovered.append(
                    {
                        "job_id": str(row["id"]),
                        "job_type": str(row["job_type"]),
                        "previous_worker_id": row["worker_id"],
                        "previous_worker_pid": worker_pid,
                        "reason": reason,
                    }
                )
            connection.commit()
        return recovered

    def save_choreography_sequence(
        self,
        *,
        source: str,
        confidence: float,
        context: dict[str, Any],
        steps: list[dict[str, Any]],
        sequence_id: str | None = None,
        song_id: int | None = None,
        timeline_id: str | None = None,
    ) -> str:
        """Store a semantic multi-step routine, never fixture DMX bytes."""
        resolved_id = sequence_id or f"choreography:{uuid.uuid4()}"
        if not steps:
            raise ValueError("a choreography sequence requires at least one step")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO choreography_sequences(
                    id, song_id, timeline_id, source, confidence, context_json,
                    created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    song_id=excluded.song_id,
                    timeline_id=excluded.timeline_id,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    context_json=excluded.context_json,
                    created_unix_ms=excluded.created_unix_ms
                """,
                (
                    resolved_id,
                    song_id,
                    timeline_id,
                    source,
                    max(0.0, min(1.0, confidence)),
                    json.dumps(context, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )
            connection.execute(
                "DELETE FROM choreography_steps WHERE sequence_id=?",
                (resolved_id,),
            )
            connection.executemany(
                """
                INSERT INTO choreography_steps(
                    sequence_id, step_index, start_beat, duration_beats,
                    fixture_scope, routine, intensity, palette, parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        resolved_id,
                        index,
                        float(step["start_beat"]),
                        float(step["duration_beats"]),
                        str(step["fixture_scope"]),
                        str(step["routine"]),
                        max(0.0, min(1.0, float(step.get("intensity", 1.0)))),
                        step.get("palette"),
                        json.dumps(step.get("parameters", {}), sort_keys=True),
                    )
                    for index, step in enumerate(steps)
                ],
            )
        return resolved_id

    def choreography_sequence(
        self, sequence_id: str
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            sequence = connection.execute(
                "SELECT * FROM choreography_sequences WHERE id=?",
                (sequence_id,),
            ).fetchone()
            if sequence is None:
                return None
            rows = connection.execute(
                """
                SELECT * FROM choreography_steps
                WHERE sequence_id=? ORDER BY step_index
                """,
                (sequence_id,),
            ).fetchall()
        result = dict(sequence)
        result["context"] = json.loads(result.pop("context_json"))
        result["steps"] = []
        for row in rows:
            step = dict(row)
            step["parameters"] = json.loads(step.pop("parameters_json"))
            result["steps"].append(step)
        return result

    def research_summary(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            totals = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM dataset_sources) AS dataset_sources,
                    (SELECT COUNT(*) FROM dataset_tracks) AS dataset_tracks,
                    (SELECT COUNT(*) FROM recording_versions) AS recordings,
                    (SELECT COUNT(*) FROM teacher_runs) AS teacher_runs,
                    (SELECT COUNT(*) FROM structure_timelines) AS timelines,
                    (SELECT COUNT(*) FROM analysis_jobs) AS jobs,
                    (
                        SELECT COUNT(*) FROM analysis_jobs
                        WHERE status='queued'
                    ) AS queued_jobs,
                    (
                        SELECT COUNT(*) FROM choreography_sequences
                    ) AS choreography_sequences
                """
            ).fetchone()
        return dict(totals) if totals is not None else {}

    @staticmethod
    def _fallback_identity(media: MediaIdentity) -> str:
        material = "\x1f".join(
            (
                media.title or "",
                "\x1e".join(media.artists),
                media.album or "",
                str(media.duration_ms or ""),
            )
        )
        return "derived:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
