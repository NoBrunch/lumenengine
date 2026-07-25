"""Private SQLite memory for recordings, analyses, feedback, and routines."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from lumen_engine.models import Feedback, MediaIdentity, PerformanceDecision

SCHEMA_VERSION = 1


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
                """
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
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
                    song_id, position_ms, label, value, note, created_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.song_id,
                    feedback.position_ms,
                    feedback.label,
                    feedback.value,
                    feedback.note,
                    int(time.time() * 1000),
                ),
            )
            return int(cursor.lastrowid)

    def list_feedback(self, song_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, position_ms, label, value, note, created_unix_ms
                FROM feedback WHERE song_id=?
                ORDER BY COALESCE(position_ms, -1), created_unix_ms
                """,
                (song_id,),
            ).fetchall()
        return [dict(row) for row in rows]

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
    ) -> int:
        payload = asdict(decision)
        payload["gesture"] = decision.gesture.value
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
