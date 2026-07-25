from __future__ import annotations

import unittest

from lumen_engine.media import (
    media_identity_from_spotify,
    spotify_playback_summary,
    spotify_track_summary,
)


class SpotifyMappingTests(unittest.TestCase):
    def test_maps_playback_payload_without_leaking_provider_shape(self) -> None:
        media = media_identity_from_spotify(
            {
                "timestamp": 123456,
                "progress_ms": 42_000,
                "is_playing": True,
                "device": {"name": "Garage Chromecast"},
                "context": {"uri": "spotify:playlist:list"},
                "item": {
                    "uri": "spotify:track:track",
                    "name": "A Song",
                    "duration_ms": 200_000,
                    "artists": [{"name": "An Artist"}],
                    "album": {"name": "An Album"},
                },
            }
        )
        assert media is not None
        self.assertEqual(media.provider_item_id, "spotify:track:track")
        self.assertEqual(media.display_name, "An Artist — A Song")
        self.assertEqual(media.device_name, "Garage Chromecast")
        self.assertEqual(media.observed_position_ms, 42_000)
        self.assertTrue(media.is_playing)

    def test_missing_item_means_no_playback(self) -> None:
        self.assertIsNone(media_identity_from_spotify({"item": None}))

    def test_console_summary_keeps_art_and_device_without_raw_payload(self) -> None:
        track = {
            "uri": "spotify:track:abc",
            "id": "abc",
            "name": "Signal",
            "artists": [{"name": "The Inputs"}],
            "duration_ms": 180_000,
            "album": {
                "name": "Line Level",
                "images": [{"url": "https://i.scdn.co/image/test"}],
            },
        }
        summary = spotify_track_summary(track)
        self.assertEqual(summary["artists"], ["The Inputs"])
        self.assertEqual(summary["image_url"], "https://i.scdn.co/image/test")
        playback = spotify_playback_summary(
            {
                "item": track,
                "device": {
                    "id": "speaker",
                    "name": "Garage Chromecast",
                    "type": "speaker",
                    "is_active": True,
                },
                "is_playing": True,
                "progress_ms": 42_000,
            }
        )
        assert playback is not None
        self.assertEqual(playback["device"]["name"], "Garage Chromecast")
        self.assertTrue(playback["is_playing"])


if __name__ == "__main__":
    unittest.main()
