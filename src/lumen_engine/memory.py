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
from lumen_engine.structure import (
    ContentRole,
    EnergySection,
    FunctionalSection,
    TransitionEvent,
)
from lumen_engine.structure_feedback import (
    OPERATOR_CONSENSUS_VERSION,
    consensus_anchors,
    consensus_segments,
)

SCHEMA_VERSION = 8
# Obsolete normalized teacher timelines remain durable evidence, but only this
# version may participate in Live recall.
TEACHER_NORMALIZATION_VERSION = "lumen_techno_ontology_v3"
# This identifies the inference semantics, not merely the output vocabulary.
# Earlier ``cpu_runner_v1:60s`` runs decoded independent short contexts and
# must never be mistaken for full-song EDMFormer evidence just because their
# labels were normalized to the current ontology.
EDMFORMER_PREPROCESSING_VERSION = (
    "lumen_edmformer_full_song_multiresolution_v3:"
    "local30s:global420s:cpu_sdpa_v1:"
    f"{TEACHER_NORMALIZATION_VERSION}"
)
SONGFORMER_PREPROCESSING_PREFIX = (
    "songformer_official_features_cpu_windowed_v1:"
)


def current_songformer_preprocessing(value: object) -> bool:
    """Return whether a SongFormer run uses the current bounded CPU contract."""

    text = str(value or "")
    prefix = SONGFORMER_PREPROCESSING_PREFIX
    suffix = f":{TEACHER_NORMALIZATION_VERSION}"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return False
    window = text[len(prefix):-len(suffix)]
    if not window.endswith("s"):
        return False
    try:
        seconds = int(window[:-1])
    except ValueError:
        return False
    return 30 <= seconds <= 60


_CANONICAL_STRUCTURE_LABELS = {
    "functional_label": {value.value for value in FunctionalSection},
    "energy_label": {value.value for value in EnergySection},
    "content_label": {value.value for value in ContentRole},
}


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
        # Journal mode is a database-level transition.  Reissuing it on every
        # short-lived read connection needlessly takes SQLite locks and was
        # especially costly once the local dataset grew beyond one gigabyte.
        with closing(sqlite3.connect(self.path, timeout=10.0)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")
        # The target machine has 16 GiB and Lumen is its only workload.  Give
        # each active SQLite worker a useful RAM page cache and allow the OS to
        # map the large read-mostly dataset instead of performing tiny HDD
        # reads through SQLite's default ~2 MiB cache.
        connection.execute("PRAGMA cache_size = -65536")
        connection.execute("PRAGMA mmap_size = 536870912")
        connection.execute("PRAGMA temp_store = MEMORY")
        # Live produces several megabytes of semantic/diagnostic rows per
        # minute. SQLite's default 1,000-page WAL auto-checkpoint therefore
        # lands in the middle of the audio consumer at a regular cadence.
        # Checkpoints are issued explicitly from standby/background paths.
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        return connection

    def checkpoint(self, mode: str = "PASSIVE") -> dict[str, int]:
        """Checkpoint accumulated WAL pages outside the live timing path."""

        normalized = str(mode).strip().upper()
        if normalized not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("unsupported SQLite checkpoint mode")
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"PRAGMA wal_checkpoint({normalized})"
            ).fetchone()
        assert row is not None
        return {
            "busy": int(row[0]),
            "log_pages": int(row[1]),
            "checkpointed_pages": int(row[2]),
        }

    def mark_research_session_prepared(
        self, session_id: str, export_path: str
    ) -> None:
        key = "research_prepared_session:" + hashlib.sha256(
            str(session_id).encode("utf-8")
        ).hexdigest()
        value = json.dumps(
            {
                "session_id": str(session_id),
                "export_path": str(export_path),
                "prepared_unix_ms": int(time.time() * 1000),
            },
            sort_keys=True,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    def research_prepared_session_ids(self) -> set[str]:
        """Return sessions durably inventoried or represented by teacher work.

        A preparation marker alone is not sufficient. Older preparation code
        could reject every recording in a damaged capture before inserting a
        capture-span inventory row, then mark the session prepared anyway.
        Such a session must be revisited so the operator can see the retained
        partial recordings and their concrete exclusion reasons.
        """

        prepared: set[str] = set()
        with closing(self._connect()) as connection:
            for row in connection.execute(
                """
                SELECT DISTINCT capture_session_id
                FROM teacher_runs
                WHERE capture_session_id IS NOT NULL
                UNION
                SELECT DISTINCT capture_session_id
                FROM capture_track_spans
                WHERE capture_session_id IS NOT NULL
                """
            ):
                if row[0]:
                    prepared.add(str(row[0]))
        return prepared

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
                    participant_id TEXT,
                    participant_name TEXT,
                    client_event_id TEXT,
                    listening_session_id TEXT,
                    lane_context_json TEXT,
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
                    participant_id TEXT,
                    participant_name TEXT,
                    client_event_id TEXT,
                    listening_session_id TEXT,
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

                CREATE TABLE IF NOT EXISTS structure_timeline_reviews (
                    id INTEGER PRIMARY KEY,
                    timeline_id TEXT NOT NULL
                        REFERENCES structure_timelines(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    participant_id TEXT,
                    participant_name TEXT,
                    note TEXT,
                    created_unix_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS structure_timeline_reviews_latest
                    ON structure_timeline_reviews(timeline_id, id DESC);

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
                    recording_id TEXT
                        REFERENCES recording_versions(id) ON DELETE SET NULL,
                    timeline_id TEXT
                        REFERENCES structure_timelines(id) ON DELETE SET NULL,
                    name TEXT,
                    fixture_scope TEXT NOT NULL DEFAULT 'overall',
                    participant_id TEXT,
                    participant_name TEXT,
                    client_event_id TEXT,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_unix_ms INTEGER NOT NULL,
                    updated_unix_ms INTEGER NOT NULL,
                    deleted_unix_ms INTEGER
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
                    strobe_json TEXT NOT NULL DEFAULT '{}',
                    entry_behavior TEXT,
                    exit_behavior TEXT,
                    parameters_json TEXT NOT NULL,
                    PRIMARY KEY(sequence_id, step_index)
                );

                CREATE TABLE IF NOT EXISTS choreography_sequence_history (
                    sequence_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    PRIMARY KEY(sequence_id, revision)
                );

                CREATE TABLE IF NOT EXISTS choreography_placements (
                    id TEXT PRIMARY KEY,
                    sequence_id TEXT NOT NULL
                        REFERENCES choreography_sequences(id) ON DELETE CASCADE,
                    song_id INTEGER REFERENCES songs(id) ON DELETE SET NULL,
                    recording_id TEXT
                        REFERENCES recording_versions(id) ON DELETE SET NULL,
                    timeline_id TEXT
                        REFERENCES structure_timelines(id) ON DELETE SET NULL,
                    fixture_scope TEXT NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    start_beat REAL,
                    duration_beats REAL,
                    section_label TEXT,
                    source TEXT NOT NULL,
                    participant_id TEXT,
                    participant_name TEXT,
                    client_event_id TEXT,
                    context_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_unix_ms INTEGER NOT NULL,
                    updated_unix_ms INTEGER NOT NULL,
                    deleted_unix_ms INTEGER
                );

                CREATE TABLE IF NOT EXISTS choreography_placement_history (
                    placement_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    PRIMARY KEY(placement_id, revision)
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
                ("participant_id", "TEXT"),
                ("participant_name", "TEXT"),
                ("client_event_id", "TEXT"),
                ("listening_session_id", "TEXT"),
                ("lane_context_json", "TEXT"),
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
                CREATE UNIQUE INDEX IF NOT EXISTS feedback_client_event
                ON feedback(
                    COALESCE(listening_session_id, ''),
                    COALESCE(participant_id, ''),
                    client_event_id
                )
                WHERE client_event_id IS NOT NULL
                """
            )
            annotation_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(training_annotations)"
                )
            }
            for name in (
                "participant_id", "participant_name", "client_event_id",
                "listening_session_id",
            ):
                if name not in annotation_columns:
                    connection.execute(
                        f"ALTER TABLE training_annotations ADD COLUMN {name} TEXT"
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS annotation_client_event
                ON training_annotations(
                    COALESCE(listening_session_id, ''),
                    COALESCE(participant_id, ''),
                    client_event_id
                )
                WHERE client_event_id IS NOT NULL
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
            sequence_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(choreography_sequences)"
                )
            }
            for name, definition in (
                ("recording_id", "TEXT"),
                ("name", "TEXT"),
                ("fixture_scope", "TEXT NOT NULL DEFAULT 'overall'"),
                ("participant_id", "TEXT"),
                ("participant_name", "TEXT"),
                ("client_event_id", "TEXT"),
                ("revision", "INTEGER NOT NULL DEFAULT 1"),
                ("updated_unix_ms", "INTEGER"),
                ("deleted_unix_ms", "INTEGER"),
            ):
                if name not in sequence_columns:
                    connection.execute(
                        f"ALTER TABLE choreography_sequences ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                UPDATE choreography_sequences
                SET updated_unix_ms=COALESCE(updated_unix_ms, created_unix_ms)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS choreography_sequence_client_event
                ON choreography_sequences(
                    COALESCE(participant_id, ''), client_event_id
                )
                WHERE client_event_id IS NOT NULL
                """
            )
            placement_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(choreography_placements)"
                )
            }
            for name, definition in (
                ("participant_id", "TEXT"),
                ("participant_name", "TEXT"),
                ("client_event_id", "TEXT"),
            ):
                if name not in placement_columns:
                    connection.execute(
                        f"ALTER TABLE choreography_placements ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS choreography_placement_client_event
                ON choreography_placements(
                    COALESCE(participant_id, ''), client_event_id
                )
                WHERE client_event_id IS NOT NULL
                """
            )
            step_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(choreography_steps)"
                )
            }
            for name, definition in (
                ("strobe_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("entry_behavior", "TEXT"),
                ("exit_behavior", "TEXT"),
            ):
                if name not in step_columns:
                    connection.execute(
                        f"ALTER TABLE choreography_steps ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS choreography_placements_recording_time
                ON choreography_placements(recording_id, start_ms)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS choreography_placements_song_time
                ON choreography_placements(song_id, start_ms)
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

    def add_feedback(
        self,
        feedback: Feedback,
        *,
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
        listening_session_id: str | None = None,
    ) -> int:
        return int(
            self.add_feedback_event(
                feedback,
                participant_id=participant_id,
                participant_name=participant_name,
                client_event_id=client_event_id,
                listening_session_id=listening_session_id,
            )["id"]
        )

    def add_feedback_event(
        self,
        feedback: Feedback,
        *,
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
        listening_session_id: str | None = None,
        lane_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store one listener event, idempotently when a client key is supplied.

        ``client_event_id`` is scoped to a participant and listening session so
        independent phones may use their own local counters.  Replaying the
        same request returns the original database id instead of amplifying its
        learning weight.  Legacy callers that omit these fields retain the old
        append-only behavior.
        """
        participant = participant_id.strip() if participant_id else None
        participant_label = participant_name.strip() if participant_name else None
        if participant_label is not None and len(participant_label) > 32:
            raise ValueError("participant_name must be 32 characters or fewer")
        client_event = client_event_id.strip() if client_event_id else None
        listening_session = (
            listening_session_id.strip() if listening_session_id else None
        )
        lane_context_json = (
            json.dumps(lane_context, sort_keys=True, separators=(",", ":"))
            if lane_context
            else None
        )
        if client_event_id is not None and not client_event:
            raise ValueError("client_event_id must not be blank")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO feedback(
                    song_id, position_ms, label, value, note, scope,
                    fixture_id, gesture, section, energy, motion, tension,
                    confidence, bpm, routine, capture_session_id,
                    audio_frame_index, participant_id, participant_name,
                    client_event_id,
                    listening_session_id, lane_context_json, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    participant,
                    participant_label,
                    client_event,
                    listening_session,
                    lane_context_json,
                    int(time.time() * 1000),
                ),
            )
            if cursor.rowcount > 0:
                feedback_id = int(cursor.lastrowid)
                created = True
            else:
                row = connection.execute(
                    """
                    SELECT id FROM feedback
                    WHERE COALESCE(listening_session_id, '') = COALESCE(?, '')
                      AND COALESCE(participant_id, '') = COALESCE(?, '')
                      AND client_event_id=?
                    """,
                    (listening_session, participant, client_event),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise RuntimeError("feedback idempotency lookup failed")
                feedback_id = int(row["id"])
                created = False
            connection.commit()
            return {"id": feedback_id, "created": created}

    def list_feedback(self, song_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, position_ms, label, value, note, scope, fixture_id,
                       gesture, section, energy, motion, tension, confidence, bpm, routine,
                       capture_session_id, audio_frame_index, participant_id,
                       participant_name, client_event_id,
                       listening_session_id, lane_context_json,
                       created_unix_ms
                FROM feedback WHERE song_id=?
                ORDER BY COALESCE(position_ms, -1), created_unix_ms
                """,
                (song_id,),
            ).fetchall()
        return [self._feedback_row(row) for row in rows]

    def all_feedback(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    feedback.id, feedback.song_id, feedback.position_ms,
                    feedback.label, feedback.value, feedback.note,
                    feedback.scope, feedback.fixture_id, feedback.gesture,
                    feedback.section, feedback.energy, feedback.motion,
                    feedback.tension, feedback.confidence, feedback.bpm,
                    feedback.routine, feedback.capture_session_id,
                    feedback.audio_frame_index, feedback.participant_id,
                    feedback.participant_name, feedback.client_event_id,
                    feedback.listening_session_id,
                    feedback.lane_context_json, feedback.created_unix_ms,
                    songs.artists_json AS song_artists_json
                FROM feedback
                LEFT JOIN songs ON songs.id = feedback.song_id
                """
            ).fetchall()
        return [self._feedback_row(row) for row in rows]

    @staticmethod
    def _feedback_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        encoded_artists = payload.pop("song_artists_json", None)
        try:
            artists = json.loads(encoded_artists) if encoded_artists else []
        except (TypeError, json.JSONDecodeError):
            artists = []
        if isinstance(artists, list):
            payload["song_artists"] = [str(artist) for artist in artists]
        encoded = payload.pop("lane_context_json", None)
        try:
            lane_context = json.loads(encoded) if encoded else None
        except (TypeError, json.JSONDecodeError):
            lane_context = None
        payload["lane_context"] = (
            lane_context if isinstance(lane_context, dict) else None
        )
        return payload

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
                    (
                        SELECT COUNT(*) FROM feedback
                        WHERE feedback.song_id = songs.id
                    ) AS feedback_count,
                    (
                        SELECT COUNT(*) FROM decisions
                        WHERE decisions.song_id = songs.id
                    ) AS decision_count
                FROM songs
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
                    feedback.participant_id,
                    feedback.participant_name,
                    feedback.client_event_id,
                    feedback.listening_session_id,
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
                        SELECT COUNT(DISTINCT participant_id) FROM feedback
                        WHERE participant_id IS NOT NULL
                    ) AS feedback_participants,
                    (
                        SELECT COUNT(DISTINCT listening_session_id) FROM feedback
                        WHERE listening_session_id IS NOT NULL
                    ) AS listening_sessions,
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

    def log_trace_batch(self, items: list[dict[str, Any]]) -> None:
        """Persist trace items in one transaction from the trace worker."""

        if not items:
            return
        performance_rows: list[tuple[Any, ...]] = []
        decision_rows: list[tuple[Any, ...]] = []
        now_ms = int(time.time() * 1000)
        for item in items:
            kind = str(item.get("_kind", "performance"))
            if kind == "decision":
                decision = item["decision"]
                observation = item.get("observation")
                payload = asdict(decision)
                payload["gesture"] = decision.gesture.value
                if observation is not None:
                    payload["observation"] = asdict(observation)
                decision_rows.append(
                    (
                        item.get("song_id"),
                        item.get("position_ms"),
                        now_ms,
                        decision.gesture.value,
                        decision.brightness,
                        decision.confidence,
                        decision.reason,
                        json.dumps(payload, sort_keys=True),
                    )
                )
            else:
                performance_rows.append(
                    (
                        item["session_id"],
                        item.get("song_id"),
                        item.get("position_ms"),
                        now_ms,
                        json.dumps(item["payload"], sort_keys=True),
                    )
                )
        with closing(self._connect()) as connection, connection:
            if performance_rows:
                connection.executemany(
                    """
                    INSERT INTO performance_samples(
                        session_id, song_id, position_ms, created_unix_ms,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    performance_rows,
                )
            if decision_rows:
                connection.executemany(
                    """
                    INSERT INTO decisions(
                        song_id, position_ms, created_unix_ms, gesture,
                        brightness, confidence, reason, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    decision_rows,
                )

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
            musical_count = connection.execute(
                """
                SELECT COUNT(*) AS value FROM training_annotations
                WHERE kind='musical_context'
                """
            ).fetchone()
            structure_participants = connection.execute(
                """
                SELECT COUNT(DISTINCT participant_id) AS value
                FROM training_annotations
                WHERE kind='musical_context' AND participant_id IS NOT NULL
                """
            ).fetchone()
            consensus_rows = connection.execute(
                """
                SELECT metadata_json FROM structure_timelines
                WHERE provenance='operator_annotation_consensus'
                """
            ).fetchall()
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
        result["musical_annotations"] = (
            int(musical_count["value"]) if musical_count else 0
        )
        result["structure_participants"] = (
            int(structure_participants["value"])
            if structure_participants else 0
        )
        consensus_metadata = [
            json.loads(row["metadata_json"]) for row in consensus_rows
        ]
        result["consensus_songs"] = len(consensus_metadata)
        result["consensus_anchors"] = sum(
            int(item.get("accepted_anchors") or 0)
            for item in consensus_metadata
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

    def training_frame_identities(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Return only fields needed to divide one capture into recordings.

        Automatic teacher preparation must not deserialize every full DMX,
        waveform, and choreography payload merely to find Spotify track
        boundaries. SQLite's JSON projection keeps this bounded even for a
        multi-hour listening session.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT audio_frame_index, song_id, position_ms,
                       json_extract(payload_json, '$.media') AS media_json
                FROM training_frames
                WHERE session_id=? ORDER BY audio_frame_index
                """,
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            media_json = row["media_json"]
            result.append(
                {
                    "audio_frame_index": int(row["audio_frame_index"]),
                    "song_id": row["song_id"],
                    "position_ms": row["position_ms"],
                    "payload": {
                        "media": (
                            json.loads(media_json)
                            if isinstance(media_json, str)
                            else None
                        )
                    },
                }
            )
        return result

    def training_feedback(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, song_id, position_ms, label, value, note, scope,
                       fixture_id, gesture, section, energy, motion, tension,
                       confidence, bpm, routine, capture_session_id,
                       audio_frame_index, participant_id, client_event_id,
                       participant_name, listening_session_id, created_unix_ms
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
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
        listening_session_id: str | None = None,
    ) -> int:
        return int(self.add_training_annotation_event(
            song_id=song_id,
            position_ms=position_ms,
            kind=kind,
            label=label,
            scope=scope,
            fixture_id=fixture_id,
            intensity=intensity,
            note=note,
            capture_session_id=capture_session_id,
            audio_frame_index=audio_frame_index,
            context=context,
            participant_id=participant_id,
            participant_name=participant_name,
            client_event_id=client_event_id,
            listening_session_id=listening_session_id,
        )["id"])

    def add_training_annotation_event(
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
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
        listening_session_id: str | None = None,
    ) -> dict[str, Any]:
        participant = participant_id.strip() if participant_id else None
        participant_label = participant_name.strip() if participant_name else None
        if participant_label is not None and len(participant_label) > 32:
            raise ValueError("participant_name must be 32 characters or fewer")
        client_event = client_event_id.strip() if client_event_id else None
        listening_session = (
            listening_session_id.strip() if listening_session_id else None
        )
        if client_event_id is not None and not client_event:
            raise ValueError("client_event_id must not be blank")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO training_annotations(
                    song_id, position_ms, kind, label, scope, fixture_id,
                    intensity, note, capture_session_id, audio_frame_index,
                    participant_id, participant_name, client_event_id,
                    listening_session_id,
                    context_json, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    participant,
                    participant_label,
                    client_event,
                    listening_session,
                    json.dumps(context, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )
            if cursor.rowcount > 0:
                annotation_id = int(cursor.lastrowid)
                created = True
            else:
                row = connection.execute(
                    """
                    SELECT id FROM training_annotations
                    WHERE COALESCE(listening_session_id, '')=COALESCE(?, '')
                      AND COALESCE(participant_id, '')=COALESCE(?, '')
                      AND client_event_id=?
                    """,
                    (listening_session, participant, client_event),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise RuntimeError("annotation idempotency lookup failed")
                annotation_id = int(row["id"])
                created = False
            connection.commit()
            return {"id": annotation_id, "created": created}

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

    def musical_structure_annotations(
        self, *, song_ids: set[int] | None = None
    ) -> list[dict[str, Any]]:
        """Return raw musical calls independently of their capture session."""

        values: list[Any] = []
        where = "WHERE kind='musical_context' AND position_ms IS NOT NULL"
        if song_ids is not None:
            if not song_ids:
                return []
            placeholders = ",".join("?" for _ in song_ids)
            where += f" AND song_id IN ({placeholders})"
            values.extend(sorted(song_ids))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM training_annotations {where}
                ORDER BY song_id, position_ms, created_unix_ms, id
                """,
                values,
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item.pop("context_json"))
            # Preserve the historically reported selector for audit while
            # presenting the corrected semantic interpretation downstream.
            item["reported_scope"] = item.get("scope")
            item["reported_fixture_id"] = item.get("fixture_id")
            item["scope"] = "overall"
            item["fixture_id"] = None
            result.append(item)
        return result

    def refresh_operator_structure_consensus(
        self, *, song_ids: set[int] | None = None
    ) -> dict[str, Any]:
        """Rebuild song-wide consensus from immutable participant evidence."""

        annotations = self.musical_structure_annotations(song_ids=song_ids)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            if annotation.get("song_id") is not None:
                grouped.setdefault(int(annotation["song_id"]), []).append(
                    annotation
                )
        refreshed: list[dict[str, Any]] = []
        for song_id, song_annotations in grouped.items():
            anchors = consensus_anchors(song_annotations)
            with closing(self._connect()) as connection:
                base_row = connection.execute(
                    """
                    SELECT structure_timelines.id
                    FROM structure_timelines
                    JOIN teacher_runs
                      ON teacher_runs.id=structure_timelines.teacher_run_id
                    LEFT JOIN structure_timeline_reviews
                      ON structure_timeline_reviews.id=(
                        SELECT review.id
                        FROM structure_timeline_reviews AS review
                        WHERE review.timeline_id=structure_timelines.id
                        ORDER BY review.id DESC LIMIT 1
                      )
                    WHERE structure_timelines.song_id=?
                      AND LOWER(teacher_runs.teacher_name)='edmformer'
                      AND teacher_runs.status='complete'
                      AND structure_timelines.timeline_version=?
                      AND teacher_runs.preprocessing_version=?
                      AND COALESCE(structure_timeline_reviews.status,
                                   'unreviewed')!='rejected'
                    ORDER BY
                      CASE COALESCE(structure_timeline_reviews.status,
                                    'unreviewed')
                        WHEN 'approved' THEN 1 ELSE 0 END DESC,
                      structure_timelines.confidence DESC,
                      structure_timelines.created_unix_ms DESC
                    LIMIT 1
                    """,
                    (
                        song_id,
                        TEACHER_NORMALIZATION_VERSION,
                        EDMFORMER_PREPROCESSING_VERSION,
                    ),
                ).fetchone()
                song_row = connection.execute(
                    "SELECT duration_ms FROM songs WHERE id=?", (song_id,)
                ).fetchone()
                recording_row = connection.execute(
                    """
                    SELECT * FROM recording_versions WHERE song_id=?
                    ORDER BY last_seen_unix_ms DESC, id LIMIT 1
                    """,
                    (song_id,),
                ).fetchone()
            base = (
                self.structure_timeline(str(base_row["id"]))
                if base_row is not None else None
            )
            duration_candidates = [
                int(value)
                for value in (
                    (base or {}).get("metadata", {}).get("duration_ms"),
                    recording_row["duration_ms"] if recording_row is not None else None,
                    song_row["duration_ms"] if song_row is not None else None,
                    max(
                        (
                            int(segment.get("end_ms") or 0)
                            for segment in (base or {}).get("segments", ())
                        ),
                        default=0,
                    ),
                )
                if value is not None and int(value) > 0
            ]
            if not duration_candidates:
                continue
            duration_ms = max(duration_candidates)
            segments = consensus_segments(
                anchors,
                duration_ms=duration_ms,
                base_segments=(base or {}).get("segments", ()),
            )
            evidence = json.dumps(
                {
                    "version": OPERATOR_CONSENSUS_VERSION,
                    "base": (base or {}).get("id"),
                    "anchors": anchors,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            revision = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
            timeline_id = f"timeline:operator-consensus:song:{song_id}"
            self.save_structure_timeline(
                timeline_id=timeline_id,
                provenance="operator_annotation_consensus",
                timeline_version=OPERATOR_CONSENSUS_VERSION,
                confidence=max(
                    (float(anchor["confidence"]) for anchor in anchors if anchor["accepted"]),
                    default=0.0,
                ),
                recording_id=(
                    (base or {}).get("recording_id")
                    or (str(recording_row["id"]) if recording_row is not None else None)
                ),
                song_id=song_id,
                capture_session_id=(base or {}).get("capture_session_id"),
                segments=segments,
                metadata={
                    "consensus_revision": revision,
                    "base_timeline_id": (base or {}).get("id"),
                    "duration_ms": duration_ms,
                    "raw_annotations": len(song_annotations),
                    "anchors": anchors,
                    "accepted_anchors": sum(bool(item["accepted"]) for item in anchors),
                    "participants": sorted({
                        participant
                        for anchor in anchors
                        for participant in anchor["participants"]
                    }),
                },
            )
            refreshed.append({
                "song_id": song_id,
                "timeline_id": timeline_id,
                "revision": revision,
                "anchors": len(anchors),
                "accepted_anchors": sum(bool(item["accepted"]) for item in anchors),
                "segments": len(segments),
            })
        return {
            "songs": len(refreshed),
            "raw_annotations": len(annotations),
            "timelines": refreshed,
            "revision": self.operator_consensus_revision(),
        }

    def operator_consensus_revision(self) -> str:
        """Fingerprint the complete correction view used for model training."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, metadata_json FROM structure_timelines
                WHERE provenance='operator_annotation_consensus'
                ORDER BY id
                """
            ).fetchall()
        payload = [
            (
                str(row["id"]),
                str(json.loads(row["metadata_json"]).get("consensus_revision") or ""),
            )
            for row in rows
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def operator_consensus_for_recordings(
        self, recording_ids: set[str]
    ) -> dict[str, dict[str, Any]]:
        """Map recording identities to the correction timeline for its song."""

        if not recording_ids:
            return {}
        placeholders = ",".join("?" for _ in recording_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT recording_versions.id AS recording_id,
                       structure_timelines.id AS timeline_id
                FROM recording_versions
                JOIN structure_timelines
                  ON structure_timelines.song_id=recording_versions.song_id
                 AND structure_timelines.provenance=
                     'operator_annotation_consensus'
                WHERE recording_versions.id IN ({placeholders})
                """,
                sorted(recording_ids),
            ).fetchall()
        return {
            str(row["recording_id"]): timeline
            for row in rows
            if (
                timeline := self.structure_timeline(str(row["timeline_id"]))
            ) is not None
        }

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
        """Return complete, auditable timelines for one recording identity.

        Teacher details are joined here instead of inferred by the interface.
        This deliberately returns obsolete and corrected timelines too: recall
        may reject an obsolete normalization, but an operator must still be
        able to see the original evidence and every later correction.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT structure_timelines.id,
                       teacher_runs.teacher_name,
                       teacher_runs.teacher_version,
                       teacher_runs.preprocessing_version,
                       teacher_runs.status AS teacher_status,
                       teacher_runs.error AS teacher_error,
                       structure_timeline_reviews.id AS review_id,
                       structure_timeline_reviews.status AS review_status,
                       structure_timeline_reviews.participant_id
                           AS review_participant_id,
                       structure_timeline_reviews.participant_name
                           AS review_participant_name,
                       structure_timeline_reviews.note AS review_note,
                       structure_timeline_reviews.created_unix_ms
                           AS review_created_unix_ms
                FROM structure_timelines
                LEFT JOIN teacher_runs
                  ON teacher_runs.id=structure_timelines.teacher_run_id
                LEFT JOIN structure_timeline_reviews
                  ON structure_timeline_reviews.id=(
                    SELECT review.id FROM structure_timeline_reviews AS review
                    WHERE review.timeline_id=structure_timelines.id
                    ORDER BY review.id DESC LIMIT 1
                  )
                WHERE structure_timelines.recording_id=?
                ORDER BY structure_timelines.created_unix_ms DESC,
                         structure_timelines.id
                """,
                (recording_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            timeline = self.structure_timeline(str(row["id"]))
            if timeline is None:
                continue
            timeline["teacher"] = {
                "name": row["teacher_name"],
                "version": row["teacher_version"],
                "preprocessing_version": row["preprocessing_version"],
                "status": row["teacher_status"],
                "error": row["teacher_error"],
            }
            timeline["review"] = (
                {
                    "id": row["review_id"],
                    "timeline_id": timeline["id"],
                    "status": row["review_status"],
                    "participant_id": row["review_participant_id"],
                    "participant_name": row["review_participant_name"],
                    "note": row["review_note"],
                    "created_unix_ms": row["review_created_unix_ms"],
                }
                if row["review_id"] is not None else None
            )
            teacher_name = str(row["teacher_name"] or "").casefold()
            provenance_name = str(
                timeline.get("provenance") or ""
            ).casefold()
            operator_authority = (
                "operator" in provenance_name
                or "correction" in provenance_name
            )
            technically_eligible = (
                (
                    timeline.get("teacher_run_id") is None
                    and operator_authority
                )
                or (
                    teacher_name in {"edmformer", "songformer"}
                    and row["teacher_status"] == "complete"
                    and str(timeline.get("timeline_version") or "")
                    == TEACHER_NORMALIZATION_VERSION
                    and (
                        (
                            teacher_name == "edmformer"
                            and str(row["preprocessing_version"] or "")
                            == EDMFORMER_PREPROCESSING_VERSION
                        )
                        or (
                            teacher_name == "songformer"
                            and current_songformer_preprocessing(
                                row["preprocessing_version"]
                            )
                        )
                    )
                )
            )
            review_status = str(
                (timeline["review"] or {}).get("status") or "unreviewed"
            )
            scored_or_approved = (
                review_status == "approved"
                or float(timeline.get("confidence") or 0.0) > 0.0
            )
            timeline["review_eligible"] = technically_eligible
            timeline["recall_eligible"] = (
                technically_eligible
                and review_status != "rejected"
                and scored_or_approved
            )
            if not technically_eligible:
                timeline["recall_authority"] = (
                    "non_authoritative_teacher"
                    if teacher_name
                    and teacher_name not in {"edmformer", "songformer"}
                    else "ineligible_version_or_run"
                )
            elif review_status == "rejected":
                timeline["recall_authority"] = "operator_rejected"
            elif review_status == "approved":
                timeline["recall_authority"] = "operator_approved"
            elif float(timeline.get("confidence") or 0.0) > 0.0:
                timeline["recall_authority"] = "scored_teacher"
            else:
                timeline["recall_authority"] = "unscored_requires_approval"
            result.append(timeline)
        return result

    def structure_timeline_catalog(self) -> list[dict[str, Any]]:
        """List every recording with a generated timeline for operator review.

        The catalog is deliberately independent of the currently playing
        Spotify item.  Playback identity is needed for Live recall, but it is
        not a reasonable prerequisite for inspecting or correcting offline
        teacher work.
        """

        with closing(self._connect()) as connection:
            recording_rows = connection.execute(
                """
                SELECT recording_versions.*,
                       COUNT(structure_timelines.id) AS timeline_count,
                       MAX(structure_timelines.created_unix_ms)
                           AS latest_timeline_unix_ms
                FROM recording_versions
                JOIN structure_timelines
                  ON structure_timelines.recording_id=recording_versions.id
                GROUP BY recording_versions.id
                ORDER BY latest_timeline_unix_ms DESC,
                         recording_versions.id
                """
            ).fetchall()
            timeline_rows = connection.execute(
                """
                SELECT structure_timelines.recording_id,
                       structure_timelines.provenance,
                       structure_timelines.timeline_version,
                       structure_timelines.confidence,
                       structure_timelines.teacher_run_id,
                       teacher_runs.teacher_name,
                       teacher_runs.preprocessing_version,
                       teacher_runs.status AS teacher_status,
                       structure_timeline_reviews.status AS review_status
                FROM structure_timelines
                LEFT JOIN teacher_runs
                  ON teacher_runs.id=structure_timelines.teacher_run_id
                LEFT JOIN structure_timeline_reviews
                  ON structure_timeline_reviews.id=(
                    SELECT review.id FROM structure_timeline_reviews AS review
                    WHERE review.timeline_id=structure_timelines.id
                    ORDER BY review.id DESC LIMIT 1
                  )
                ORDER BY structure_timelines.created_unix_ms DESC,
                         structure_timelines.id
                """
            ).fetchall()

        states: dict[str, list[dict[str, Any]]] = {}
        for row in timeline_rows:
            states.setdefault(str(row["recording_id"]), []).append(dict(row))

        catalog: list[dict[str, Any]] = []
        for row in recording_rows:
            item = dict(row)
            metadata = json.loads(item.pop("metadata_json"))
            track = metadata.get("track_identity") or {}
            candidates = states.get(str(item["id"]), [])

            def eligible_teacher(candidate: dict[str, Any]) -> bool:
                teacher = str(
                    candidate.get("teacher_name") or ""
                ).casefold()
                preprocessing = candidate.get("preprocessing_version")
                return bool(
                    teacher in {"edmformer", "songformer"}
                    and candidate.get("teacher_status") == "complete"
                    and candidate.get("timeline_version")
                    == TEACHER_NORMALIZATION_VERSION
                    and (
                        (
                            teacher == "edmformer"
                            and preprocessing
                            == EDMFORMER_PREPROCESSING_VERSION
                        )
                        or (
                            teacher == "songformer"
                            and current_songformer_preprocessing(preprocessing)
                        )
                    )
                )

            corrected = any(
                "operator" in str(candidate.get("provenance") or "").casefold()
                or "correction" in str(candidate.get("provenance") or "").casefold()
                for candidate in candidates
            )
            approved = any(
                str(candidate.get("review_status") or "").casefold()
                == "approved"
                for candidate in candidates
            )
            eligible_candidates = [
                candidate for candidate in candidates
                if eligible_teacher(candidate)
            ]
            rejected = sum(
                str(candidate.get("review_status") or "").casefold()
                == "rejected"
                for candidate in eligible_candidates
            )
            eligible_unreviewed = any(
                eligible_teacher(candidate)
                and str(
                    candidate.get("review_status") or "unreviewed"
                ).casefold() == "unreviewed"
                for candidate in candidates
            )
            if eligible_unreviewed:
                review_status = "needs_review"
            elif corrected:
                review_status = "corrected"
            elif approved:
                review_status = "approved"
            elif (
                rejected == len(eligible_candidates)
                and eligible_candidates
            ):
                review_status = "rejected"
            else:
                review_status = "diagnostic_only"
            artists = track.get("artists") or metadata.get("artists") or []
            if isinstance(artists, str):
                artists = [artists]
            supervision = metadata.get("structure_supervision") or {}
            canonical_runs = [
                candidate for candidate in candidates
                if eligible_teacher(candidate)
                and str(candidate.get("review_status") or "").casefold()
                != "rejected"
            ]
            teacher_sources = sorted({
                str(candidate.get("teacher_name") or "Local").strip()
                for candidate in candidates
            })
            catalog.append(
                {
                    "recording_id": item["id"],
                    "song_id": item.get("song_id"),
                    "provider": item["provider"],
                    "provider_item_id": item["provider_item_id"],
                    "duration_ms": item.get("duration_ms"),
                    "title": track.get("title") or metadata.get("title")
                    or item["provider_item_id"],
                    "artists": [str(artist) for artist in artists],
                    "album": track.get("album") or metadata.get("album"),
                    "timeline_count": int(item["timeline_count"]),
                    "latest_timeline_unix_ms": item["latest_timeline_unix_ms"],
                    "review_status": review_status,
                    "reviewed": review_status in {"approved", "corrected", "rejected"},
                    "teacher_sources": teacher_sources,
                    "teacher_run_ids": [
                        str(candidate["teacher_run_id"])
                        for candidate in canonical_runs
                        if candidate.get("teacher_run_id")
                    ],
                    "split": metadata.get("split"),
                    "capture_status": supervision.get(
                        "classification", "unknown"
                    ),
                    "training_eligible": bool(
                        canonical_runs and supervision.get("eligible", True)
                    ),
                }
            )
        catalog.sort(
            key=lambda item: (
                item["reviewed"],
                str(item["title"]).casefold(),
                str(item["recording_id"]),
            )
        )
        return catalog

    def review_structure_timeline(
        self,
        *,
        timeline_id: str,
        status: str,
        participant_id: str | None = None,
        participant_name: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Append an explicit human trust decision for a teacher timeline."""

        normalized = str(status).strip().casefold()
        if normalized not in {"approved", "rejected", "unreviewed"}:
            raise ValueError("timeline review must be approved, rejected, or unreviewed")
        if self.structure_timeline(timeline_id) is None:
            raise ValueError("structure timeline was not found")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO structure_timeline_reviews(
                    timeline_id, status, participant_id, participant_name,
                    note, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timeline_id,
                    normalized,
                    participant_id,
                    participant_name,
                    note,
                    int(time.time() * 1000),
                ),
            )
            review_id = int(cursor.lastrowid)
        review = self.structure_timeline_review(timeline_id)
        assert review is not None and int(review["id"]) == review_id
        return review

    def structure_timeline_review(
        self, timeline_id: str
    ) -> dict[str, Any] | None:
        """Return the latest explicit trust decision for one timeline."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM structure_timeline_reviews
                WHERE timeline_id=? ORDER BY id DESC LIMIT 1
                """,
                (timeline_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def save_structure_correction(
        self,
        *,
        base_timeline_id: str,
        segments: list[dict[str, Any]],
        participant_id: str | None = None,
        participant_name: str | None = None,
        note: str | None = None,
    ) -> str:
        """Save an immutable operator revision without altering its source.

        ``segments`` is a complete corrected copy of the base segmentation.
        Time boundaries may be adjusted, and blank labels remain blank.  The
        original raw label is always copied from the teacher timeline so later
        training and audits retain what the teacher actually emitted.
        """
        base = self.structure_timeline(base_timeline_id)
        if base is None:
            raise ValueError("base structure timeline was not found")
        if not base.get("recording_id"):
            raise ValueError("base structure timeline has no recording identity")
        if not segments:
            raise ValueError("at least one corrected segment is required")
        if len(segments) != len(base["segments"]):
            raise ValueError("correction must contain every base segment")

        corrected: list[dict[str, Any]] = []
        allowed_labels = (
            "functional_label", "energy_label", "content_label"
        )
        for index, (source, supplied) in enumerate(
            zip(base["segments"], segments, strict=True)
        ):
            if int(supplied.get("segment_index", index)) != index:
                raise ValueError("correction segment indexes must be contiguous")
            start_ms = int(supplied.get("start_ms", source["start_ms"]))
            end_value = supplied.get("end_ms", source.get("end_ms"))
            end_ms = int(end_value) if end_value is not None else None
            original_provenance = source.get("provenance") or {}
            event = supplied.get(
                "event",
                original_provenance.get(
                    "transition_event", original_provenance.get("event")
                ),
            )
            event = str(event).strip() if event is not None else None
            if event == "":
                event = None
            if event is not None and event not in {
                value.value for value in TransitionEvent
            }:
                raise ValueError("unknown canonical transition event")
            item: dict[str, Any] = {
                "segment_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "beat_index": source.get("beat_index"),
                "downbeat": source.get("downbeat"),
                "boundary_confidence": 1.0,
                "label_confidence": 1.0,
                "raw_label": source.get("raw_label"),
                "provenance": {
                    "source": "operator_correction",
                    "base_timeline_id": base_timeline_id,
                    "base_segment_index": index,
                    "base_provenance": original_provenance,
                    # ``transition_event`` is the normalized teacher/student
                    # contract. Older operator corrections used ``event``;
                    # readers retain that alias, but new evidence must use
                    # the canonical field consumed by student export.
                    "transition_event": event,
                    "participant_id": participant_id,
                    "participant_name": participant_name,
                },
            }
            for field in allowed_labels:
                value = supplied.get(field, source.get(field))
                item[field] = str(value).strip() if value is not None else None
                if item[field] == "":
                    item[field] = None
                if (
                    item[field] is not None
                    and item[field] not in _CANONICAL_STRUCTURE_LABELS[field]
                ):
                    raise ValueError(
                        f"unknown canonical {field.removesuffix('_label')} label"
                    )
            corrected.append(item)

        for index in range(1, len(corrected)):
            previous_end = corrected[index - 1].get("end_ms")
            current_start = int(corrected[index]["start_ms"])
            if previous_end is not None and int(previous_end) > current_start:
                raise ValueError("corrected structure segments must not overlap")

        return self.save_structure_timeline(
            provenance="operator_correction",
            timeline_version="lumen_operator_correction_v1",
            confidence=1.0,
            recording_id=str(base["recording_id"]),
            song_id=base.get("song_id"),
            capture_session_id=base.get("capture_session_id"),
            segments=corrected,
            beats=[
                {
                    "time_ms": beat["time_ms"],
                    "position_in_bar": beat.get("position_in_bar"),
                    "bar_number": beat.get("bar_number"),
                    "confidence": beat.get("confidence", 1.0),
                    "provenance": beat.get("provenance", {}),
                }
                for beat in base.get("beats", [])
            ],
            metadata={
                "corrects_timeline_id": base_timeline_id,
                "original_provenance": base.get("provenance"),
                "original_timeline_version": base.get("timeline_version"),
                "participant_id": participant_id,
                "participant_name": participant_name,
                "note": note,
            },
        )

    def resolve_recording_version(
        self,
        *,
        provider: str,
        provider_item_id: str,
        duration_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Resolve provider metadata to one stable captured recording.

        A provider track may have more than one observed master/duration.  The
        closest duration wins, then the most recently seen version and stable
        recording id break ties.  Spotify URI and bare track-id forms are
        treated as aliases without changing the stored canonical identity.
        """
        provider_key = provider.strip().lower()
        item_key = provider_item_id.strip()
        if not provider_key or not item_key:
            return None
        candidates = {item_key}
        if provider_key == "spotify":
            if item_key.startswith("spotify:track:"):
                candidates.add(item_key.rsplit(":", 1)[-1])
            elif ":" not in item_key:
                candidates.add(f"spotify:track:{item_key}")
        placeholders = ",".join("?" for _ in candidates)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM recording_versions
                WHERE LOWER(provider)=?
                  AND provider_item_id IN ({placeholders})
                """,
                (provider_key, *sorted(candidates)),
            ).fetchall()
        if not rows:
            return None

        def rank(row: sqlite3.Row) -> tuple[int, int, int, str]:
            stored_duration = row["duration_ms"]
            if duration_ms is None or stored_duration is None:
                duration_missing = 1
                duration_delta = 0
            else:
                duration_missing = 0
                duration_delta = abs(int(stored_duration) - int(duration_ms))
            return (
                duration_missing,
                duration_delta,
                -int(row["last_seen_unix_ms"]),
                str(row["id"]),
            )

        result = dict(min(rows, key=rank))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def cached_structure_at(
        self,
        *,
        playback_position_ms: int,
        recording_id: str | None = None,
        provider: str | None = None,
        provider_item_id: str | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Fuse completed cached teacher timelines at one playback position.

        EDMFormer supplies techno energy form while SongFormer supplies
        functional/content form. Operator corrections outrank both. Obsolete
        timelines remain durable research artifacts but cannot enter this
        lookup. Audio sample timing stays authoritative for beat output.
        """
        recording: dict[str, Any] | None
        if recording_id is not None:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM recording_versions WHERE id=?",
                    (recording_id,),
                ).fetchone()
            if row is None:
                return None
            recording = dict(row)
            recording["metadata"] = json.loads(recording.pop("metadata_json"))
        elif provider is not None and provider_item_id is not None:
            recording = self.resolve_recording_version(
                provider=provider,
                provider_item_id=provider_item_id,
                duration_ms=duration_ms,
            )
        else:
            raise ValueError(
                "recording_id or both provider and provider_item_id are required"
            )
        if recording is None:
            return None

        position_ms = max(0, int(playback_position_ms))
        recording_duration = recording.get("duration_ms")
        if recording_duration is not None:
            position_ms = min(position_ms, int(recording_duration))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT structure_timelines.id,
                       structure_timelines.timeline_version,
                       structure_timelines.recording_id AS timeline_recording_id,
                       timeline_recording.duration_ms AS timeline_duration_ms,
                       teacher_runs.teacher_name,
                       teacher_runs.teacher_version,
                       teacher_runs.preprocessing_version,
                       structure_timeline_reviews.id AS review_id,
                       structure_timeline_reviews.status AS review_status,
                       structure_timeline_reviews.participant_id
                           AS review_participant_id,
                       structure_timeline_reviews.participant_name
                           AS review_participant_name,
                       structure_timeline_reviews.note AS review_note,
                       structure_timeline_reviews.created_unix_ms
                           AS review_created_unix_ms
                FROM structure_timelines
                LEFT JOIN teacher_runs
                  ON teacher_runs.id=structure_timelines.teacher_run_id
                LEFT JOIN recording_versions AS timeline_recording
                  ON timeline_recording.id=structure_timelines.recording_id
                LEFT JOIN structure_timeline_reviews
                  ON structure_timeline_reviews.id=(
                    SELECT review.id FROM structure_timeline_reviews AS review
                    WHERE review.timeline_id=structure_timelines.id
                    ORDER BY review.id DESC LIMIT 1
                  )
                WHERE (
                    structure_timelines.recording_id=?
                    OR structure_timelines.song_id=?
                )
                  AND (
                    structure_timelines.teacher_run_id IS NULL
                    OR teacher_runs.status='complete'
                  )
                ORDER BY structure_timelines.created_unix_ms DESC,
                         structure_timelines.id
                """,
                (recording["id"], recording.get("song_id")),
            ).fetchall()
        timelines: list[dict[str, Any]] = []
        teacher_by_timeline: dict[str, dict[str, Any]] = {}
        for row in rows:
            # Reuse a song-level correction/teacher result across later PCM
            # captures of the same provider track, but never across a
            # materially different edit or master duration.
            if (
                str(row["timeline_recording_id"] or "")
                != str(recording["id"])
                and recording_duration is not None
                and row["timeline_duration_ms"] is not None
                and abs(
                    int(row["timeline_duration_ms"])
                    - int(recording_duration)
                ) > max(3_000, round(int(recording_duration) * 0.03))
            ):
                continue
            timeline = self.structure_timeline(str(row["id"]))
            if timeline is None:
                continue
            review = (
                {
                    "id": row["review_id"],
                    "timeline_id": row["id"],
                    "status": row["review_status"],
                    "participant_id": row["review_participant_id"],
                    "participant_name": row["review_participant_name"],
                    "note": row["review_note"],
                    "created_unix_ms": row["review_created_unix_ms"],
                }
                if row["review_id"] is not None else None
            )
            review_status = str(
                (review or {}).get("status") or "unreviewed"
            ).casefold()
            if review_status == "rejected":
                continue
            teacher_name = str(row["teacher_name"] or "").casefold()
            provenance_name = str(
                timeline.get("provenance") or ""
            ).casefold()
            operator_authority = (
                "operator" in provenance_name
                or "correction" in provenance_name
            )
            if (
                teacher_name not in {"edmformer", "songformer"}
                and not operator_authority
            ):
                continue
            if (
                teacher_name == "edmformer"
                and (
                    str(row["timeline_version"] or "")
                    != TEACHER_NORMALIZATION_VERSION
                    or str(row["preprocessing_version"] or "")
                    != EDMFORMER_PREPROCESSING_VERSION
                )
            ):
                # Preserve obsolete timelines for audit/reprocessing without
                # admitting their unmerged or synthetic-confidence output.
                continue
            if (
                teacher_name == "songformer"
                and (
                    str(row["timeline_version"] or "")
                    != TEACHER_NORMALIZATION_VERSION
                    or not current_songformer_preprocessing(
                        row["preprocessing_version"]
                    )
                )
            ):
                continue
            if any(
                segment.get(field) is not None
                and str(segment.get(field))
                not in _CANONICAL_STRUCTURE_LABELS[field]
                for segment in timeline.get("segments", ())
                for field in _CANONICAL_STRUCTURE_LABELS
            ):
                # A current-version marker or operator provenance is not enough
                # if persisted content carries a retired or invented label.
                continue
            timelines.append(timeline)
            teacher_by_timeline[str(row["id"])] = {
                "name": row["teacher_name"],
                "version": row["teacher_version"],
                "preprocessing_version": row["preprocessing_version"],
                "review": review,
                "operator_approved": review_status == "approved",
                "operator_authority": operator_authority,
            }
        if not timelines:
            return None

        axis_preferences = {
            "functional": ("songformer", "edmformer"),
            "energy": ("edmformer", "edm-98", "edm98"),
            "content": ("songformer", "edmformer"),
        }

        def source_priority(
            axis: str, source: str, *, operator_approved: bool = False
        ) -> int:
            if operator_approved:
                return 110
            lowered = source.lower()
            if "operator" in lowered or "correction" in lowered:
                return 100
            preferences = axis_preferences[axis]
            for offset, preferred in enumerate(preferences):
                if preferred in lowered:
                    return 80 - offset
            if "student" in lowered:
                return 30
            return 10

        axis_candidates: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
            "functional": [],
            "energy": [],
            "content": [],
        }
        future_boundaries: list[dict[str, Any]] = []
        used_timelines: dict[str, dict[str, Any]] = {}
        for timeline in timelines:
            timeline_id = str(timeline["id"])
            teacher = teacher_by_timeline[timeline_id]
            teacher_name = str(teacher.get("name") or "").casefold()
            source = " / ".join(
                str(value)
                for value in (timeline.get("provenance"), teacher.get("name"))
                if value
            )
            operator_approved = bool(teacher.get("operator_approved"))
            operator_authority = bool(teacher.get("operator_authority"))
            future_segment = next(
                (
                    segment
                    for segment in timeline["segments"]
                    if int(segment["start_ms"]) > position_ms
                ),
                None,
            )
            if (
                future_segment is not None
                and (
                    float(future_segment["boundary_confidence"]) > 0.0
                    or operator_approved
                )
            ):
                future_boundaries.append(
                    {
                        "time_ms": int(future_segment["start_ms"]),
                        "confidence": float(
                            future_segment["boundary_confidence"]
                        ),
                        "operator_trust": 1.0 if operator_approved else 0.0,
                        "recall_authority": (
                            "operator_approved"
                            if operator_approved
                            else "operator_consensus"
                            if operator_authority
                            else "scored_teacher"
                            if float(future_segment["boundary_confidence"]) > 0.0
                            else "unscored_requires_approval"
                        ),
                        "timeline_id": timeline_id,
                        "provenance": future_segment["provenance"],
                    }
                )
            for segment in timeline["segments"]:
                start_ms = int(segment["start_ms"])
                end_ms = segment.get("end_ms")
                if start_ms > position_ms:
                    break
                if end_ms is not None:
                    end_value = int(end_ms)
                    final_endpoint = (
                        recording_duration is not None
                        and position_ms == int(recording_duration)
                        and end_value == int(recording_duration)
                    )
                    if position_ms >= end_value and not final_endpoint:
                        continue
                for axis in ("functional", "energy", "content"):
                    if teacher_name == "songformer" and axis == "energy":
                        continue
                    label = segment.get(f"{axis}_label")
                    if label is None or not str(label).strip():
                        continue
                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            float(timeline["confidence"])
                            * float(segment["label_confidence"]),
                        ),
                    )
                    item = {
                        "label": str(label),
                        "confidence": confidence,
                        "model_confidence": confidence,
                        "operator_trust": 1.0 if operator_approved else 0.0,
                        "recall_authority": (
                            "operator_approved"
                            if operator_approved
                            else "operator_consensus"
                            if operator_authority
                            else "scored_teacher"
                            if confidence > 0.0
                            else "unscored_requires_approval"
                        ),
                        "boundary_confidence": float(
                            segment["boundary_confidence"]
                        ),
                        "start_ms": start_ms,
                        "end_ms": int(end_ms) if end_ms is not None else None,
                        "timeline_id": timeline_id,
                        "teacher": teacher,
                        "provenance": segment["provenance"],
                    }
                    rank = (
                        source_priority(
                            axis,
                            source,
                            operator_approved=operator_approved,
                        ),
                        confidence,
                        int(timeline["created_unix_ms"]),
                        timeline_id,
                    )
                    axis_candidates[axis].append((rank, item))
                break

        axes: dict[str, dict[str, Any] | None] = {}
        for axis, candidates in axis_candidates.items():
            if not candidates:
                axes[axis] = None
                continue
            _, chosen = max(candidates, key=lambda candidate: candidate[0])
            axes[axis] = chosen
            used_timelines[chosen["timeline_id"]] = {
                "timeline_id": chosen["timeline_id"],
                "teacher": chosen["teacher"],
                "provenance": chosen["provenance"],
                "recall_authority": chosen["recall_authority"],
                "operator_trust": chosen["operator_trust"],
            }
        populated = [axis for axis in axes.values() if axis is not None]
        if not populated:
            return None
        confidence = sum(float(axis["confidence"]) for axis in populated) / len(
            populated
        )
        next_boundary = None
        if future_boundaries:
            next_time = min(int(item["time_ms"]) for item in future_boundaries)
            same_time = [
                item
                for item in future_boundaries
                if int(item["time_ms"]) == next_time
            ]
            next_boundary = max(
                same_time,
                key=lambda item: (
                    float(item.get("operator_trust") or 0.0),
                    float(item["confidence"]), str(item["timeline_id"])
                ),
            )
            next_boundary = dict(next_boundary)
            next_boundary["in_ms"] = next_time - position_ms
        current_boundaries: list[dict[str, Any]] = []
        for axis in populated:
            decay = max(
                0.0,
                1.0
                - max(0, position_ms - int(axis["start_ms"])) / 1_500.0,
            )
            current_confidence = float(axis["boundary_confidence"]) * decay
            current_authority = float(axis.get("operator_trust") or 0.0) * decay
            if current_confidence <= 0.0 and current_authority <= 0.0:
                continue
            current_boundaries.append(
                {
                    "current_confidence": current_confidence,
                    "current_authority": current_authority,
                    "recall_authority": axis.get("recall_authority"),
                    "timeline_id": axis["timeline_id"],
                    "provenance": axis["provenance"],
                    "time_ms": int(axis["start_ms"]),
                    "age_ms": max(0, position_ms - int(axis["start_ms"])),
                }
            )
        current_boundary = (
            max(
                current_boundaries,
                key=lambda item: (
                    float(item.get("current_authority") or 0.0),
                    float(item["current_confidence"]),
                    str(item["timeline_id"]),
                ),
            )
            if current_boundaries
            else {
                "current_confidence": 0.0,
                "current_authority": 0.0,
                "recall_authority": None,
                "timeline_id": None,
                "provenance": None,
                "time_ms": None,
                "age_ms": None,
            }
        )
        return {
            "recording": recording,
            "playback_position_ms": position_ms,
            "timing_role": "structural_context_only",
            "beat_sync_authority": "audio_sample_clock",
            "axes": axes,
            "confidence": confidence,
            "boundary": {
                # A segment's boundary confidence describes its leading edge,
                # not every frame until the next segment. Emit a short
                # structural pulse so Live cannot expand motion continuously
                # through a two-minute cached section.
                **current_boundary,
                "next": next_boundary,
            },
            "provenance": [
                used_timelines[key] for key in sorted(used_timelines)
            ],
        }

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
        recording_id: str | None = None,
        timeline_id: str | None = None,
        name: str | None = None,
        fixture_scope: str = "overall",
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
    ) -> str:
        """Store or revise a semantic multi-step routine atomically.

        Revisions snapshot the prior semantic state, allowing an operator to
        undo an edit or deletion without restoring raw DMX bytes.  A listener
        client event is idempotent within that participant's identity.
        """
        resolved_id = sequence_id or f"choreography:{uuid.uuid4()}"
        if not steps:
            raise ValueError("a choreography sequence requires at least one step")
        previous_start = float("-inf")
        for step in steps:
            start_beat = float(step["start_beat"])
            duration_beats = float(step["duration_beats"])
            if start_beat < previous_start:
                raise ValueError("choreography steps must be ordered by start_beat")
            if duration_beats <= 0:
                raise ValueError("duration_beats must be positive")
            if not str(step.get("fixture_scope") or fixture_scope).strip():
                raise ValueError("fixture_scope must not be blank")
            if not str(step["routine"]).strip():
                raise ValueError("routine must not be blank")
            previous_start = start_beat
        participant = participant_id.strip() if participant_id else None
        participant_label = participant_name.strip() if participant_name else None
        if participant_label is not None and len(participant_label) > 32:
            raise ValueError("participant_name must be 32 characters or fewer")
        client_event = client_event_id.strip() if client_event_id else None
        if client_event_id is not None and not client_event:
            raise ValueError("client_event_id must not be blank")
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if client_event is not None:
                duplicate = connection.execute(
                    """
                    SELECT id FROM choreography_sequences
                    WHERE COALESCE(participant_id, '')=COALESCE(?, '')
                      AND client_event_id=?
                    """,
                    (participant, client_event),
                ).fetchone()
                if duplicate is not None:
                    connection.commit()
                    return str(duplicate["id"])
            existing = self._choreography_sequence_from_connection(
                connection, resolved_id, include_deleted=True
            )
            revision = 1
            created_ms = now_ms
            if existing is not None:
                revision = int(existing["revision"]) + 1
                created_ms = int(existing["created_unix_ms"])
                connection.execute(
                    """
                    INSERT OR REPLACE INTO choreography_sequence_history(
                        sequence_id, revision, snapshot_json, operation,
                        created_unix_ms
                    ) VALUES (?, ?, ?, 'revise', ?)
                    """,
                    (
                        resolved_id,
                        int(existing["revision"]),
                        json.dumps(existing, sort_keys=True),
                        now_ms,
                    ),
                )
            connection.execute(
                """
                INSERT INTO choreography_sequences(
                    id, song_id, recording_id, timeline_id, name,
                    fixture_scope, participant_id, participant_name,
                    client_event_id, source, confidence, context_json,
                    revision, created_unix_ms, updated_unix_ms,
                    deleted_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    song_id=excluded.song_id,
                    recording_id=excluded.recording_id,
                    timeline_id=excluded.timeline_id,
                    name=excluded.name,
                    fixture_scope=excluded.fixture_scope,
                    participant_id=excluded.participant_id,
                    participant_name=excluded.participant_name,
                    client_event_id=excluded.client_event_id,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    context_json=excluded.context_json,
                    revision=excluded.revision,
                    updated_unix_ms=excluded.updated_unix_ms,
                    deleted_unix_ms=NULL
                """,
                (
                    resolved_id,
                    song_id,
                    recording_id,
                    timeline_id,
                    name,
                    fixture_scope,
                    participant,
                    participant_label,
                    client_event,
                    source,
                    max(0.0, min(1.0, confidence)),
                    json.dumps(context, sort_keys=True),
                    revision,
                    created_ms,
                    now_ms,
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
                    fixture_scope, routine, intensity, palette, strobe_json,
                    entry_behavior, exit_behavior, parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(step.get("strobe", {}), sort_keys=True),
                        step.get("entry_behavior"),
                        step.get("exit_behavior"),
                        json.dumps(step.get("parameters", {}), sort_keys=True),
                    )
                    for index, step in enumerate(steps)
                ],
            )
            connection.commit()
        return resolved_id

    def choreography_sequence(
        self, sequence_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            return self._choreography_sequence_from_connection(
                connection, sequence_id, include_deleted=include_deleted
            )

    @staticmethod
    def _choreography_sequence_from_connection(
        connection: sqlite3.Connection,
        sequence_id: str,
        *,
        include_deleted: bool,
    ) -> dict[str, Any] | None:
        clause = "" if include_deleted else " AND deleted_unix_ms IS NULL"
        sequence = connection.execute(
            f"SELECT * FROM choreography_sequences WHERE id=?{clause}",
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
            step["strobe"] = json.loads(step.pop("strobe_json"))
            result["steps"].append(step)
        return result

    def list_choreography_sequences(
        self,
        *,
        song_id: int | None = None,
        recording_id: str | None = None,
        fixture_scope: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("song_id", song_id),
            ("recording_id", recording_id),
            ("fixture_scope", fixture_scope),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        if not include_deleted:
            clauses.append("deleted_unix_ms IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT id FROM choreography_sequences{where}
                ORDER BY updated_unix_ms DESC, id
                """,
                values,
            ).fetchall()
            return [
                sequence
                for row in rows
                if (
                    sequence := self._choreography_sequence_from_connection(
                        connection,
                        str(row["id"]),
                        include_deleted=include_deleted,
                    )
                )
                is not None
            ]

    def delete_choreography_sequence(self, sequence_id: str) -> bool:
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = self._choreography_sequence_from_connection(
                connection, sequence_id, include_deleted=False
            )
            if sequence is None:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT OR REPLACE INTO choreography_sequence_history(
                    sequence_id, revision, snapshot_json, operation,
                    created_unix_ms
                ) VALUES (?, ?, ?, 'delete', ?)
                """,
                (
                    sequence_id,
                    int(sequence["revision"]),
                    json.dumps(sequence, sort_keys=True),
                    now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE choreography_sequences
                SET revision=revision+1, updated_unix_ms=?, deleted_unix_ms=?
                WHERE id=?
                """,
                (now_ms, now_ms, sequence_id),
            )
            connection.commit()
            return True

    def undo_choreography_sequence(self, sequence_id: str) -> bool:
        """Undo the most recent sequence revision or soft deletion."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._choreography_sequence_from_connection(
                connection, sequence_id, include_deleted=True
            )
            if current is None:
                connection.rollback()
                return False
            target_revision = int(current["revision"]) - 1
            history = connection.execute(
                """
                SELECT snapshot_json FROM choreography_sequence_history
                WHERE sequence_id=? AND revision=?
                """,
                (sequence_id, target_revision),
            ).fetchone()
            if history is None:
                connection.rollback()
                return False
            snapshot = json.loads(history["snapshot_json"])
            self._restore_choreography_sequence(connection, snapshot)
            connection.execute(
                """
                DELETE FROM choreography_sequence_history
                WHERE sequence_id=? AND revision=?
                """,
                (sequence_id, target_revision),
            )
            connection.commit()
            return True

    @staticmethod
    def _restore_choreography_sequence(
        connection: sqlite3.Connection, snapshot: dict[str, Any]
    ) -> None:
        connection.execute(
            """
            UPDATE choreography_sequences SET
                song_id=?, recording_id=?, timeline_id=?, name=?,
                fixture_scope=?, participant_id=?, participant_name=?,
                client_event_id=?, source=?, confidence=?, context_json=?,
                revision=?, created_unix_ms=?, updated_unix_ms=?,
                deleted_unix_ms=?
            WHERE id=?
            """,
            (
                snapshot.get("song_id"), snapshot.get("recording_id"),
                snapshot.get("timeline_id"), snapshot.get("name"),
                snapshot.get("fixture_scope", "overall"),
                snapshot.get("participant_id"), snapshot.get("participant_name"),
                snapshot.get("client_event_id"), snapshot["source"],
                snapshot["confidence"],
                json.dumps(snapshot.get("context", {}), sort_keys=True),
                snapshot["revision"], snapshot["created_unix_ms"],
                snapshot["updated_unix_ms"], snapshot.get("deleted_unix_ms"),
                snapshot["id"],
            ),
        )
        connection.execute(
            "DELETE FROM choreography_steps WHERE sequence_id=?",
            (snapshot["id"],),
        )
        connection.executemany(
            """
            INSERT INTO choreography_steps(
                sequence_id, step_index, start_beat, duration_beats,
                fixture_scope, routine, intensity, palette, strobe_json,
                entry_behavior, exit_behavior, parameters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot["id"], index, step["start_beat"],
                    step["duration_beats"], step["fixture_scope"],
                    step["routine"], step["intensity"], step.get("palette"),
                    json.dumps(step.get("strobe", {}), sort_keys=True),
                    step.get("entry_behavior"), step.get("exit_behavior"),
                    json.dumps(step.get("parameters", {}), sort_keys=True),
                )
                for index, step in enumerate(snapshot["steps"])
            ],
        )

    def save_choreography_placement(
        self,
        *,
        sequence_id: str,
        fixture_scope: str,
        source: str,
        context: dict[str, Any],
        placement_id: str | None = None,
        song_id: int | None = None,
        recording_id: str | None = None,
        timeline_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        start_beat: float | None = None,
        duration_beats: float | None = None,
        section_label: str | None = None,
        participant_id: str | None = None,
        participant_name: str | None = None,
        client_event_id: str | None = None,
    ) -> str:
        """Place a reusable group routine on a recognized song timeline."""
        if start_ms is None and start_beat is None and section_label is None:
            raise ValueError("a placement needs a time, beat, or section anchor")
        if start_ms is not None and start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if end_ms is not None and (start_ms is None or end_ms <= start_ms):
            raise ValueError("end_ms must be after start_ms")
        if duration_beats is not None and duration_beats <= 0:
            raise ValueError("duration_beats must be positive")
        if not fixture_scope.strip():
            raise ValueError("fixture_scope must not be blank")
        resolved_id = placement_id or f"placement:{uuid.uuid4()}"
        participant = participant_id.strip() if participant_id else None
        participant_label = participant_name.strip() if participant_name else None
        if participant_label is not None and len(participant_label) > 32:
            raise ValueError("participant_name must be 32 characters or fewer")
        client_event = client_event_id.strip() if client_event_id else None
        if client_event_id is not None and not client_event:
            raise ValueError("client_event_id must not be blank")
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if client_event is not None:
                duplicate = connection.execute(
                    """
                    SELECT id FROM choreography_placements
                    WHERE COALESCE(participant_id, '')=COALESCE(?, '')
                      AND client_event_id=?
                    """,
                    (participant, client_event),
                ).fetchone()
                if duplicate is not None:
                    connection.commit()
                    return str(duplicate["id"])
            existing = self._choreography_placement_from_connection(
                connection, resolved_id, include_deleted=True
            )
            revision = 1
            created_ms = now_ms
            if existing is not None:
                revision = int(existing["revision"]) + 1
                created_ms = int(existing["created_unix_ms"])
                connection.execute(
                    """
                    INSERT OR REPLACE INTO choreography_placement_history(
                        placement_id, revision, snapshot_json, operation,
                        created_unix_ms
                    ) VALUES (?, ?, ?, 'revise', ?)
                    """,
                    (
                        resolved_id, int(existing["revision"]),
                        json.dumps(existing, sort_keys=True), now_ms,
                    ),
                )
            connection.execute(
                """
                INSERT INTO choreography_placements(
                    id, sequence_id, song_id, recording_id, timeline_id,
                    fixture_scope, start_ms, end_ms, start_beat,
                    duration_beats, section_label, source, participant_id,
                    participant_name, client_event_id, context_json, revision,
                    created_unix_ms, updated_unix_ms, deleted_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    sequence_id=excluded.sequence_id,
                    song_id=excluded.song_id,
                    recording_id=excluded.recording_id,
                    timeline_id=excluded.timeline_id,
                    fixture_scope=excluded.fixture_scope,
                    start_ms=excluded.start_ms,
                    end_ms=excluded.end_ms,
                    start_beat=excluded.start_beat,
                    duration_beats=excluded.duration_beats,
                    section_label=excluded.section_label,
                    source=excluded.source,
                    participant_id=excluded.participant_id,
                    participant_name=excluded.participant_name,
                    client_event_id=excluded.client_event_id,
                    context_json=excluded.context_json,
                    revision=excluded.revision,
                    updated_unix_ms=excluded.updated_unix_ms,
                    deleted_unix_ms=NULL
                """,
                (
                    resolved_id, sequence_id, song_id, recording_id,
                    timeline_id, fixture_scope, start_ms, end_ms, start_beat,
                    duration_beats, section_label, source, participant,
                    participant_label, client_event,
                    json.dumps(context, sort_keys=True), revision, created_ms,
                    now_ms,
                ),
            )
            connection.commit()
        return resolved_id

    def choreography_placement(
        self, placement_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            return self._choreography_placement_from_connection(
                connection, placement_id, include_deleted=include_deleted
            )

    @staticmethod
    def _choreography_placement_from_connection(
        connection: sqlite3.Connection,
        placement_id: str,
        *,
        include_deleted: bool,
    ) -> dict[str, Any] | None:
        clause = "" if include_deleted else " AND deleted_unix_ms IS NULL"
        row = connection.execute(
            f"SELECT * FROM choreography_placements WHERE id=?{clause}",
            (placement_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["context"] = json.loads(result.pop("context_json"))
        return result

    def list_choreography_placements(
        self,
        *,
        song_id: int | None = None,
        recording_id: str | None = None,
        fixture_scope: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("song_id", song_id),
            ("recording_id", recording_id),
            ("fixture_scope", fixture_scope),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        if not include_deleted:
            clauses.append("deleted_unix_ms IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM choreography_placements{where}
                ORDER BY COALESCE(start_ms, 0), COALESCE(start_beat, 0), id
                """,
                values,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item.pop("context_json"))
            result.append(item)
        return result

    def delete_choreography_placement(self, placement_id: str) -> bool:
        now_ms = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            placement = self._choreography_placement_from_connection(
                connection, placement_id, include_deleted=False
            )
            if placement is None:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT OR REPLACE INTO choreography_placement_history(
                    placement_id, revision, snapshot_json, operation,
                    created_unix_ms
                ) VALUES (?, ?, ?, 'delete', ?)
                """,
                (
                    placement_id, int(placement["revision"]),
                    json.dumps(placement, sort_keys=True), now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE choreography_placements
                SET revision=revision+1, updated_unix_ms=?, deleted_unix_ms=?
                WHERE id=?
                """,
                (now_ms, now_ms, placement_id),
            )
            connection.commit()
            return True

    def undo_choreography_placement(self, placement_id: str) -> bool:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._choreography_placement_from_connection(
                connection, placement_id, include_deleted=True
            )
            if current is None:
                connection.rollback()
                return False
            target_revision = int(current["revision"]) - 1
            history = connection.execute(
                """
                SELECT snapshot_json FROM choreography_placement_history
                WHERE placement_id=? AND revision=?
                """,
                (placement_id, target_revision),
            ).fetchone()
            if history is None:
                connection.rollback()
                return False
            snapshot = json.loads(history["snapshot_json"])
            fields = (
                "sequence_id", "song_id", "recording_id", "timeline_id",
                "fixture_scope", "start_ms", "end_ms", "start_beat",
                "duration_beats", "section_label", "source",
                "participant_id", "participant_name", "client_event_id",
            )
            connection.execute(
                f"""
                UPDATE choreography_placements SET
                    {', '.join(f'{field}=?' for field in fields)},
                    context_json=?, revision=?, created_unix_ms=?,
                    updated_unix_ms=?, deleted_unix_ms=?
                WHERE id=?
                """,
                (
                    *(snapshot.get(field) for field in fields),
                    json.dumps(snapshot.get("context", {}), sort_keys=True),
                    snapshot["revision"], snapshot["created_unix_ms"],
                    snapshot["updated_unix_ms"],
                    snapshot.get("deleted_unix_ms"), placement_id,
                ),
            )
            connection.execute(
                """
                DELETE FROM choreography_placement_history
                WHERE placement_id=? AND revision=?
                """,
                (placement_id, target_revision),
            )
            connection.commit()
            return True

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
                        WHERE deleted_unix_ms IS NULL
                    ) AS choreography_sequences
                    ,(
                        SELECT COUNT(*) FROM choreography_placements
                        WHERE deleted_unix_ms IS NULL
                    ) AS choreography_placements
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
