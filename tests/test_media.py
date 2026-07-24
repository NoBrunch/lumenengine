from __future__ import annotations

import unittest

from lumen_engine.media import media_identity_from_spotify


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


if __name__ == "__main__":
    unittest.main()

