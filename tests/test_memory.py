from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lumen_engine.memory import SongMemoryStore
from lumen_engine.models import Feedback, MediaIdentity


class MemoryTests(unittest.TestCase):
    def test_song_identity_analysis_feedback_and_routine_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SongMemoryStore(Path(directory) / "memory.sqlite3")
            media = MediaIdentity(
                provider="spotify",
                provider_item_id="spotify:track:abc",
                title="Example",
                artists=("One", "Two"),
                album="Album",
                duration_ms=123_000,
                is_playing=True,
            )
            song_id = store.remember_media(media, count_play=True)
            same_id = store.remember_media(media, count_play=False)
            self.assertEqual(song_id, same_id)
            song = store.get_song(song_id)
            assert song is not None
            self.assertEqual(song["artists"], ("One", "Two"))
            self.assertEqual(song["play_count"], 1)

            store.save_analysis(song_id, 1, {"sections": [{"start": 0, "type": "intro"}]})
            analysis = store.latest_analysis(song_id)
            assert analysis is not None
            self.assertEqual(analysis["analysis_version"], 1)
            self.assertEqual(analysis["payload"]["sections"][0]["type"], "intro")

            store.add_feedback(
                Feedback(
                    song_id=song_id,
                    position_ms=31_000,
                    label="too_busy",
                    value=-1,
                    note="Preserve the vocal.",
                )
            )
            self.assertEqual(store.list_feedback(song_id)[0]["label"], "too_busy")

            store.save_routine(song_id, 2, {"strategy": "adaptive"})
            routine = store.get_routine(song_id)
            assert routine is not None
            self.assertEqual(routine["routine_version"], 2)
            self.assertEqual(routine["payload"]["strategy"], "adaptive")

            store.log_performance_sample(
                "session-1",
                {"observation": {"section": "groove"}, "decision": {"routine": "fan_sweep"}},
                song_id=song_id,
                position_ms=32_000,
            )
            samples = store.latest_performance_session()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["session_id"], "session-1")
            self.assertEqual(samples[0]["payload"]["decision"]["routine"], "fan_sweep")


if __name__ == "__main__":
    unittest.main()
