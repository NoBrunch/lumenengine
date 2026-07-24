from __future__ import annotations

import unittest

from lumen_engine.expression import ExpressionEngine
from lumen_engine.models import Gesture, MusicalObservation


def observation(
    timestamp: float,
    *,
    loudness: float,
    onset: float,
    section: str,
    beat_confidence: float = 0.8,
) -> MusicalObservation:
    return MusicalObservation(
        timestamp_s=timestamp,
        loudness=loudness,
        onset_strength=onset,
        low_energy=loudness,
        mid_energy=loudness * 0.8,
        high_energy=loudness * 0.7,
        beat_phase=0,
        beat_confidence=beat_confidence,
        bpm=120,
        section=section,
        section_confidence=0.9,
        novelty=onset,
    )


class ExpressionTests(unittest.TestCase):
    def test_quiet_music_breathes(self) -> None:
        engine = ExpressionEngine()
        decision = engine.decide(
            observation(0, loudness=0.05, onset=0.02, section="intro")
        )
        self.assertEqual(decision.gesture, Gesture.BREATHE)
        self.assertIn("Low energy", decision.reason)
        held = engine.decide(
            observation(0.75, loudness=0.06, onset=0.01, section="intro")
        )
        self.assertEqual(held.gesture, Gesture.BREATHE)
        self.assertEqual(held.target, decision.target)

    def test_build_converges_and_strong_onset_releases(self) -> None:
        engine = ExpressionEngine()
        for timestamp in (0.0, 2.0, 4.0, 6.0):
            decision = engine.decide(
                observation(
                    timestamp,
                    loudness=0.8,
                    onset=0.65,
                    section="build",
                )
            )
        self.assertEqual(decision.gesture, Gesture.CONVERGE)
        release = engine.decide(
            observation(8.0, loudness=1.0, onset=1.0, section="drop")
        )
        self.assertEqual(release.gesture, Gesture.RELEASE)
        self.assertIn("beat-aligned", release.reason)


if __name__ == "__main__":
    unittest.main()
