from __future__ import annotations

import unittest

from lumen_engine.beat import BeatTracker


def feed_beat(tracker: BeatTracker, beat_time: float):
    tracker.update(0.1, now=beat_time - 0.03)
    return tracker.update(1.0, now=beat_time)


class BeatTrackerTests(unittest.TestCase):
    def test_measures_house_tempo(self) -> None:
        tracker = BeatTracker()
        interval = 60.0 / 132.0
        for index in range(12):
            state = feed_beat(tracker, 1.0 + index * interval)
        self.assertAlmostEqual(state.bpm, 132.0, delta=1.0)
        self.assertTrue(state.beat)

    def test_stays_stable_under_jitter(self) -> None:
        tracker = BeatTracker()
        interval = 60.0 / 128.0
        beat_time = 1.0
        jitter_pattern = (-0.025, 0.012, -0.006, 0.019, 0.0, 0.006, -0.012)
        for index in range(48):
            beat_time += interval + jitter_pattern[index % len(jitter_pattern)]
            state = feed_beat(tracker, beat_time)
        self.assertAlmostEqual(state.bpm, 128.0, delta=1.5)
        for wild_interval in (interval * 1.15, interval * 0.85):
            beat_time += wild_interval
            state = feed_beat(tracker, beat_time)
        self.assertAlmostEqual(state.bpm, 128.0, delta=1.5)

    def test_relocks_after_tempo_change(self) -> None:
        tracker = BeatTracker()
        beat_time = 1.0
        for _ in range(24):
            beat_time += 60.0 / 128.0
            state = feed_beat(tracker, beat_time)
        self.assertAlmostEqual(state.bpm, 128.0, delta=1.0)
        for _ in range(24):
            beat_time += 60.0 / 100.0
            state = feed_beat(tracker, beat_time)
        self.assertAlmostEqual(state.bpm, 100.0, delta=2.0)

    def test_interpolates_bar_progress(self) -> None:
        tracker = BeatTracker()
        feed_beat(tracker, 1.0)
        feed_beat(tracker, 1.5)
        state = tracker.update(0.1, now=1.75)
        self.assertAlmostEqual(state.bpm, 120.0, delta=1.0)
        self.assertAlmostEqual(state.bar_progress, 0.375, delta=0.02)


if __name__ == "__main__":
    unittest.main()
