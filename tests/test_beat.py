from __future__ import annotations

import math
import unittest

from lumen_engine.beat import (
    BeatState,
    BeatTracker,
    SpectralTempoTracker,
    TempoSourceArbiter,
)


def feed_beat(tracker: BeatTracker, beat_time: float):
    tracker.update(0.1, now=beat_time - 0.03)
    return tracker.update(1.0, now=beat_time)


class BeatTrackerTests(unittest.TestCase):
    def test_arbiter_diagnostics_name_clock_source_and_discontinuities(self) -> None:
        arbiter = TempoSourceArbiter()
        state = BeatState(
            bpm=120.0,
            beat=True,
            beat_count=1,
            bar_progress=0.25,
            confidence=0.9,
        )
        arbiter.select(
            spectral=state,
            fallback=BeatState(0.0, False, 0, 0.0, 0.0),
            now=1.0,
        )
        diagnostics = arbiter.diagnostics
        self.assertEqual(diagnostics["source"], "spectral")
        self.assertEqual(diagnostics["published_bpm"], 120.0)
        before = diagnostics["clock_discontinuities"]
        arbiter.reset()
        self.assertGreater(
            arbiter.diagnostics["clock_discontinuities"], before
        )

    def test_spectral_clock_ignores_syncopation_and_holds_the_grid(self) -> None:
        updates_per_second = 48_000 / 2_048
        tracker = SpectralTempoTracker(updates_per_second)
        emitted_beats = 0
        for index in range(360):
            phase = index % 11
            activation = 1.0 if phase == 0 else 0.42 if phase == 5 else 0.05
            state = tracker.update(
                activation,
                now=index / updates_per_second,
            )
            emitted_beats += int(state.beat)
        self.assertAlmostEqual(state.bpm, 127.8, delta=1.5)
        self.assertGreater(state.confidence, 0.75)
        self.assertGreater(emitted_beats, 12)

    def test_spectral_clock_can_lock_fast_drum_and_bass_meter(self) -> None:
        updates_per_second = 48_000 / 2_048
        tracker = SpectralTempoTracker(updates_per_second)
        # Eight analysis frames is 175.78 BPM. Its sixteen-frame half-time
        # harmonic is equally periodic, so the valid fast lag must be present
        # in the search range instead of forcing an 87.89 BPM answer.
        for index in range(600):
            phase = index % 8
            activation = 1.0 if phase == 0 else 0.20 if phase == 4 else 0.03
            state = tracker.update(
                activation,
                now=index / updates_per_second,
            )

        self.assertAlmostEqual(state.bpm, 175.78, delta=1.5)
        self.assertGreater(state.confidence, 0.40)
        self.assertGreaterEqual(tracker.diagnostics["maximum_bpm"], 175.0)
        self.assertAlmostEqual(
            tracker.diagnostics["candidate_bpm"], 175.78, delta=1.5
        )

    def test_spectral_clock_promotes_supported_147_bpm_over_half_time(self):
        updates_per_second = 48_000 / 2_048
        tracker = SpectralTempoTracker(updates_per_second)
        for index in range(800):
            phase = (index / updates_per_second * 147.0 / 60.0) % 1.0
            distance = min(phase, 1.0 - phase)
            activation = math.exp(-0.5 * (distance / 0.10) ** 2)
            state = tracker.update(
                activation,
                now=index / updates_per_second,
            )

        self.assertAlmostEqual(state.bpm, 147.0, delta=2.0)
        self.assertTrue(tracker.diagnostics["octave_promoted"])

    def test_spectral_clock_keeps_genuine_slow_meter_without_offbeats(self):
        updates_per_second = 48_000 / 2_048
        tracker = SpectralTempoTracker(updates_per_second)
        for index in range(800):
            phase = index % 19
            activation = 1.0 if phase == 0 else 0.02
            state = tracker.update(
                activation,
                now=index / updates_per_second,
            )

        self.assertAlmostEqual(state.bpm, 74.0, delta=2.0)
        self.assertFalse(tracker.diagnostics["octave_promoted"])

    def test_spectral_lock_rejects_short_lived_metrical_relatives(self) -> None:
        cases = (
            (116.6, 156.2),
            (82.0, 164.0),
            (120.0, 80.0),
            (78.0, 117.0),
        )
        for locked_bpm, candidate_bpm in cases:
            with self.subTest(
                locked_bpm=locked_bpm, candidate_bpm=candidate_bpm
            ):
                tracker = SpectralTempoTracker(24.0)
                tracker._bpm = locked_bpm
                tracker._confidence = 0.82
                for _ in range(8):
                    tracker._consider_tempo_candidate(
                        candidate_bpm=candidate_bpm,
                        candidate_score=0.94,
                        candidate_confidence=0.98,
                        locked_score=0.31,
                        locked_confidence=0.56,
                    )
                self.assertAlmostEqual(tracker._bpm, locked_bpm)
                self.assertLessEqual(tracker._confidence, 0.82)
                self.assertIsNone(tracker._pending_bpm)

    def test_supported_fast_family_repairs_stronger_half_time_peak(self) -> None:
        tracker = SpectralTempoTracker(48_000 / 2_048)
        tracker._bpm = 88.0
        tracker._confidence = 0.96
        tracker._latest_octave_promoted = True
        for _ in range(11):
            tracker._consider_tempo_candidate(
                candidate_bpm=176.0,
                candidate_score=0.55,
                candidate_confidence=0.45,
                locked_score=0.70,
                locked_confidence=0.92,
            )
            self.assertAlmostEqual(tracker._bpm, 88.0)
            self.assertLessEqual(tracker._confidence, 0.45)
        tracker._consider_tempo_candidate(
            candidate_bpm=176.0,
            candidate_score=0.55,
            candidate_confidence=0.45,
            locked_score=0.70,
            locked_confidence=0.92,
        )
        self.assertAlmostEqual(tracker._bpm, 176.0)
        self.assertAlmostEqual(tracker._confidence, 0.45)

    def test_challenger_confidence_does_not_inflate_published_lock(self) -> None:
        tracker = SpectralTempoTracker(24.0)
        tracker._bpm = 116.6
        tracker._confidence = 0.64

        tracker._consider_tempo_candidate(
            candidate_bpm=156.2,
            candidate_score=0.97,
            candidate_confidence=1.0,
            locked_score=0.24,
            locked_confidence=0.30,
        )

        self.assertEqual(tracker._bpm, 116.6)
        self.assertLess(tracker._confidence, 0.64)

    def test_low_confidence_spectral_startup_is_not_published(self) -> None:
        tracker = SpectralTempoTracker(24.0)
        for _ in range(10):
            tracker._consider_tempo_candidate(
                candidate_bpm=128.0,
                candidate_score=0.18,
                candidate_confidence=0.17,
                locked_score=0.0,
                locked_confidence=0.0,
            )
        self.assertEqual(tracker._bpm, 0.0)
        self.assertEqual(tracker._confidence, 0.0)

    def test_sustained_non_harmonic_change_relocks_after_confirmation(self) -> None:
        tracker = SpectralTempoTracker(24.0)
        tracker._bpm = 117.0
        tracker._confidence = 0.80
        for _ in range(11):
            tracker._consider_tempo_candidate(
                candidate_bpm=128.0,
                candidate_score=0.86,
                candidate_confidence=0.82,
                locked_score=0.34,
                locked_confidence=0.46,
            )
        self.assertAlmostEqual(tracker._bpm, 117.0)
        tracker._consider_tempo_candidate(
            candidate_bpm=128.0,
            candidate_score=0.86,
            candidate_confidence=0.82,
            locked_score=0.34,
            locked_confidence=0.46,
        )
        self.assertAlmostEqual(tracker._bpm, 128.0, delta=0.1)
        self.assertGreaterEqual(tracker._confidence, 0.45)

    def test_source_arbiter_holds_lock_against_metrical_rival(self) -> None:
        arbiter = TempoSourceArbiter()
        fallback = BeatState(116.6, False, 4, 0.25, 0.82)
        spectral_rival = BeatState(156.2, False, 8, 0.50, 0.98)
        selected = arbiter.select(
            spectral=BeatState(0.0, False, 0, 0.0, 0.0),
            fallback=fallback,
        )
        self.assertEqual(arbiter.source, "fallback")
        self.assertEqual(selected.bpm, 116.6)
        for _ in range(20):
            selected = arbiter.select(
                spectral=spectral_rival,
                fallback=fallback,
            )
        self.assertEqual(arbiter.source, "fallback")
        self.assertEqual(selected.bpm, 116.6)

    def test_nonclose_fallback_cannot_displace_healthy_spectral_lock(self):
        arbiter = TempoSourceArbiter()
        spectral = BeatState(175.8, False, 4, 0.25, 0.45)
        fallback = BeatState(140.6, False, 4, 0.25, 0.95)
        arbiter.select(
            spectral=spectral,
            fallback=BeatState(0.0, False, 0, 0.0, 0.0),
            now=1.0,
        )

        for index in range(1, 60):
            selected = arbiter.select(
                spectral=spectral,
                fallback=fallback,
                now=1.0 + index / 24.0,
            )

        self.assertEqual(arbiter.source, "spectral")
        self.assertAlmostEqual(selected.bpm, 175.8)

    def test_source_arbiter_adopts_confirmed_close_spectral_clock(self) -> None:
        arbiter = TempoSourceArbiter()
        fallback = BeatState(117.0, False, 4, 0.25, 0.90)
        spectral = BeatState(117.4, False, 4, 0.25, 0.74)
        arbiter.select(
            spectral=BeatState(0.0, False, 0, 0.0, 0.0),
            fallback=fallback,
        )
        for _ in range(3):
            selected = arbiter.select(
                spectral=spectral,
                fallback=fallback,
            )
            self.assertEqual(selected.bpm, fallback.bpm)
        selected = arbiter.select(spectral=spectral, fallback=fallback)
        self.assertEqual(arbiter.source, "spectral")
        self.assertEqual(selected.bpm, spectral.bpm)

    def test_spectral_double_repairs_confirmed_fallback_half_time(self) -> None:
        arbiter = TempoSourceArbiter()
        fallback = BeatState(87.0, False, 4, 0.25, 0.98)
        spectral = BeatState(174.0, False, 8, 0.50, 0.52)
        arbiter.select(
            spectral=BeatState(0.0, False, 0, 0.0, 0.0),
            fallback=fallback,
            now=1.0,
        )
        for index in range(1, 12):
            selected = arbiter.select(
                spectral=spectral,
                fallback=fallback,
                now=1.0 + index / 24.0,
            )
            self.assertEqual(selected.bpm, fallback.bpm)
        selected = arbiter.select(
            spectral=spectral,
            fallback=fallback,
            now=1.5,
        )
        self.assertEqual(arbiter.source, "spectral")
        self.assertEqual(selected.bpm, spectral.bpm)

    def test_confirmed_fallback_double_repairs_spectral_half_time(self) -> None:
        arbiter = TempoSourceArbiter()
        spectral = BeatState(88.0, False, 4, 0.25, 1.0)
        fallback = BeatState(176.0, False, 8, 0.50, 0.84)
        arbiter.select(
            spectral=spectral,
            fallback=BeatState(0.0, False, 0, 0.0, 0.0),
            now=1.0,
        )
        for index in range(1, 24):
            selected = arbiter.select(
                spectral=spectral,
                fallback=fallback,
                now=1.0 + index / 24.0,
            )
            self.assertEqual(selected.bpm, spectral.bpm)
        selected = arbiter.select(
            spectral=spectral,
            fallback=fallback,
            now=2.0,
        )
        self.assertEqual(arbiter.source, "fallback")
        self.assertEqual(selected.bpm, fallback.bpm)

    def test_fallback_internal_tempo_jump_cannot_bypass_clock_continuity(self):
        arbiter = TempoSourceArbiter()
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        established = BeatState(175.8, False, 4, 0.25, 0.85)
        arbiter.select(spectral=invalid, fallback=established, now=1.0)

        for index in range(1, 60):
            selected = arbiter.select(
                spectral=invalid,
                fallback=BeatState(140.6, False, 5, 0.30, 0.92),
                now=1.0 + index / 24.0,
            )

        self.assertAlmostEqual(selected.bpm, 175.8)
        self.assertEqual(arbiter.source, "fallback")

    def test_source_handoff_preserves_published_bar_phase(self) -> None:
        arbiter = TempoSourceArbiter()
        fallback = BeatState(117.0, False, 4, 0.50, 0.90)
        spectral = BeatState(117.4, False, 99, 0.10, 0.82)
        selected = arbiter.select(
            spectral=BeatState(0.0, False, 0, 0.0, 0.0),
            fallback=fallback,
            now=10.0,
        )
        self.assertEqual(selected.bar_progress, 0.50)
        for index in range(1, 5):
            selected = arbiter.select(
                spectral=spectral,
                fallback=fallback,
                now=10.0 + index * 0.04,
            )
        self.assertEqual(arbiter.source, "spectral")
        self.assertGreater(selected.bar_progress, 0.50)
        self.assertLess(selected.bar_progress, 0.65)

    def test_source_dropout_holds_clock_until_confirmed_handoff(self) -> None:
        arbiter = TempoSourceArbiter()
        fallback = BeatState(120.0, False, 8, 0.25, 0.80)
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        arbiter.select(spectral=invalid, fallback=fallback, now=2.0)
        spectral = BeatState(121.0, False, 12, 0.90, 0.88)
        first = arbiter.select(
            spectral=spectral, fallback=invalid, now=2.04
        )
        self.assertEqual(first.bpm, 120.0)
        self.assertGreater(first.confidence, 0.0)
        second = arbiter.select(
            spectral=spectral, fallback=invalid, now=2.08
        )
        self.assertEqual(second.bpm, 121.0)
        self.assertEqual(arbiter.source, "spectral")

    def test_candidate_pulse_does_not_bypass_published_phase(self) -> None:
        arbiter = TempoSourceArbiter()
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        arbiter.select(
            spectral=invalid,
            fallback=BeatState(120.0, False, 4, 0.25, 0.9),
            now=10.0,
        )
        selected = arbiter.select(
            spectral=invalid,
            # This independent pulse is nowhere near a crossing on the
            # continuous published clock.
            fallback=BeatState(120.0, True, 5, 0.90, 0.9),
            now=10.04,
        )
        self.assertFalse(selected.beat)
        self.assertLess(selected.bar_progress, 0.30)

    def test_published_phase_crossing_emits_beat_without_candidate_pulse(self):
        arbiter = TempoSourceArbiter()
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        arbiter.select(
            spectral=invalid,
            fallback=BeatState(120.0, False, 4, 0.24, 0.9),
            now=20.0,
        )
        selected = arbiter.select(
            spectral=invalid,
            fallback=BeatState(120.0, False, 4, 0.26, 0.9),
            now=20.04,
        )
        self.assertTrue(selected.beat)
        self.assertEqual(selected.beat_count, 5)

    def test_phase_correction_never_reverses_or_moves_without_elapsed_audio(self):
        arbiter = TempoSourceArbiter()
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        initial = arbiter.select(
            spectral=invalid,
            fallback=BeatState(60.0, False, 4, 0.10, 0.9),
            now=25.0,
        )
        duplicate_time = arbiter.select(
            spectral=invalid,
            fallback=BeatState(60.0, False, 4, 0.90, 0.9),
            now=25.0,
        )
        self.assertEqual(duplicate_time.bar_progress, initial.bar_progress)
        advanced = arbiter.select(
            spectral=invalid,
            fallback=BeatState(60.0, False, 4, 0.90, 0.9),
            now=25.04,
        )
        self.assertGreater(advanced.bar_progress, duplicate_time.bar_progress)

    def test_clock_loss_increments_discontinuity_serial(self) -> None:
        arbiter = TempoSourceArbiter()
        invalid = BeatState(0.0, False, 0, 0.0, 0.0)
        arbiter.select(
            spectral=invalid,
            fallback=BeatState(120.0, False, 4, 0.9, 0.9),
            now=30.0,
        )
        initial_serial = arbiter.clock_discontinuities
        for index in range(1, 10):
            selected = arbiter.select(
                spectral=invalid,
                fallback=invalid,
                now=30.0 + index * 0.04,
            )
        self.assertEqual(selected.bpm, 0.0)
        self.assertEqual(
            arbiter.clock_discontinuities, initial_serial + 1
        )

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
        early = feed_beat(tracker, 1.5)
        self.assertEqual(early.bpm, 0.0)
        for beat_time in (2.0, 2.5, 3.0, 3.5):
            feed_beat(tracker, beat_time)
        state = tracker.update(0.1, now=3.75)
        self.assertAlmostEqual(state.bpm, 120.0, delta=1.0)
        self.assertAlmostEqual(state.bar_progress, 0.375, delta=0.02)


if __name__ == "__main__":
    unittest.main()
