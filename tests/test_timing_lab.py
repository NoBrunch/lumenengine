from __future__ import annotations

from array import array
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from lumen_engine.audio import AudioInputMetrics
from lumen_engine.beat import BeatState
from lumen_engine.config import load_rig
from lumen_engine.control import LumenApplication
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.models import MediaIdentity
from lumen_engine.profiles import party_parrot_profile
from lumen_engine.timing_lab import (
    TimingLabAnalysis,
    TimingLabAnalyzer,
    TimingLabControls,
    RetainedBassClock,
    TimingLabRuntime,
)


ROOT = Path(__file__).resolve().parents[1]


def _pcm_chunk(
    start_frame: int,
    *,
    sample_rate: int = 48_000,
    frames: int = 1_024,
    bpm: float = 126.0,
    frequency: float = 72.0,
    burst_seconds: float = 0.065,
    amplitude: float = 0.82,
) -> bytes:
    period = 60.0 / bpm
    values = array("h")
    for offset in range(frames):
        timestamp = (start_frame + offset) / sample_rate
        beat_phase = timestamp % period
        envelope = (
            math.exp(-beat_phase / 0.022)
            if beat_phase < burst_seconds
            else 0.0
        )
        sample = round(
            math.sin(timestamp * math.tau * frequency)
            * envelope
            * amplitude
            * 32767
        )
        values.extend((sample, sample))
    return values.tobytes()


def _analysis(
    timestamp_s: float,
    *,
    beat_event: bool = False,
    broad_event: bool = False,
    beat_count: int = 0,
) -> TimingLabAnalysis:
    return TimingLabAnalysis(
        timestamp_s=timestamp_s,
        metrics=AudioInputMetrics.silence(timestamp_s),
        bass_level=0.1,
        bass_onset=0.8 if beat_event else 0.0,
        bass_threshold=0.2,
        broadband_onset=0.8 if broad_event else 0.0,
        broadband_threshold=0.2,
        bass_transient=beat_event,
        broadband_transient=broad_event,
        beat_event=beat_event,
        predicted_beat=False,
        bpm=126.0,
        confidence=0.8,
        beat_phase=0.0,
        bar_phase=0.0,
        beat_count=beat_count,
        clock_state="locked",
        candidate_bpm=126.0,
        rejected_candidate_bpm=None,
        last_bass_age_s=0.0,
        family_anchor_bpm=126.0,
        alternate_candidate_bpm=None,
        alternate_evidence_s=0.0,
        tempo_switch_count=0,
        last_pulse_interval_s=None,
        raw_candidate_bpm=126.0,
        candidate_harmonic_factor=1.0,
        bass_interval_bpm=126.0,
        phase_error_ms=0.0,
        phase_error_rms_ms=0.0,
        phase_correction_count=1,
        rejected_phase_error_ms=None,
        phase_rejection_count=0,
    )


class TimingLabAnalyzerTests(unittest.TestCase):
    def test_half_time_candidate_is_promoted_only_when_bass_intervals_support_it(
        self,
    ) -> None:
        fast = TimingLabAnalyzer()
        fast._bass_transient_times.extend(  # noqa: SLF001 - focused clock test
            index * (60.0 / 155.0) for index in range(18)
        )
        candidate = BeatState(77.5, False, 0, 0.0, 0.8)
        normalized, factor, interval_bpm = fast._normalize_candidate(  # noqa: SLF001
            candidate, fast._bass_transient_times[-1]
        )
        self.assertAlmostEqual(normalized.bpm, 155.0)
        self.assertEqual(factor, 2.0)
        self.assertAlmostEqual(interval_bpm or 0.0, 155.0)

        slow = TimingLabAnalyzer()
        slow._bass_transient_times.extend(  # noqa: SLF001 - focused clock test
            index * (60.0 / 72.0) for index in range(10)
        )
        normalized, factor, interval_bpm = slow._normalize_candidate(  # noqa: SLF001
            BeatState(72.0, False, 0, 0.0, 0.8),
            slow._bass_transient_times[-1],
        )
        self.assertAlmostEqual(normalized.bpm, 72.0)
        self.assertEqual(factor, 1.0)
        self.assertAlmostEqual(interval_bpm or 0.0, 72.0)

    def test_bass_clock_acquires_and_holds_tempo_without_silent_prediction(
        self,
    ) -> None:
        analyzer = TimingLabAnalyzer()
        frame = 0
        locked = None
        events = 0
        for _ in range(round(42.0 * 48_000 / 1_024)):
            pcm = _pcm_chunk(frame)
            timestamp = (frame + 512) / 48_000
            result = analyzer.analyze_pcm16(pcm, timestamp_s=timestamp)
            frame += 1_024
            events += int(result.beat_event)
            if result.bpm is not None:
                locked = result
        self.assertIsNotNone(locked)
        assert locked is not None
        self.assertGreater(events, 30)
        self.assertAlmostEqual(locked.bpm or 0.0, 126.0, delta=4.0)
        self.assertIn(locked.clock_state, {"locked", "held"})

        silent_events = []
        silence = bytes(1_024 * 2 * 2)
        for _ in range(round(3.0 * 48_000 / 1_024)):
            timestamp = (frame + 512) / 48_000
            result = analyzer.analyze_pcm16(silence, timestamp_s=timestamp)
            frame += 1_024
            silent_events.append(result.beat_event)
        self.assertFalse(any(silent_events[-round(1.5 * 48_000 / 1_024) :]))
        self.assertIsNotNone(result.bpm)
        self.assertEqual(result.clock_state, "held")

    def test_high_frequency_transient_is_broad_evidence_not_bass_clock(self) -> None:
        analyzer = TimingLabAnalyzer()
        frame = 0
        bass_events = 0
        broad_events = 0
        for _ in range(round(8.0 * 48_000 / 1_024)):
            pcm = _pcm_chunk(
                frame,
                bpm=120.0,
                frequency=2_400.0,
                burst_seconds=0.035,
            )
            timestamp = (frame + 512) / 48_000
            result = analyzer.analyze_pcm16(pcm, timestamp_s=timestamp)
            frame += 1_024
            bass_events += int(result.bass_transient)
            broad_events += int(result.broadband_transient)
        self.assertGreater(broad_events, 5)
        self.assertLess(bass_events, broad_events)


class RetainedBassClockTests(unittest.TestCase):
    @staticmethod
    def _seed_clock(bpm: float) -> RetainedBassClock:
        clock = RetainedBassClock()
        clock.bpm = bpm
        clock.confidence = 0.8
        clock.origin_s = 0.0
        clock._family_anchor_bpm = bpm  # noqa: SLF001 - seeded clock state
        clock._last_emitted_grid = 0  # noqa: SLF001 - seeded clock state
        clock._last_bass_s = 0.0  # noqa: SLF001 - seeded clock state
        clock._last_retune_s = 0.0  # noqa: SLF001 - seeded clock state
        return clock

    def test_same_family_retune_preserves_phase_and_regular_pulses(self) -> None:
        clock = self._seed_clock(132.0)
        events: list[float] = []
        for step in range(1, 2_001):
            now = step / 100.0
            target = 132.0 + 4.0 * now / 20.0
            bass_transient = step % 43 == 0
            state = clock.update(
                BeatState(
                    target,
                    bass_transient,
                    0,
                    ((now * target / 60.0) % 4.0) / 4.0,
                    0.85,
                ),
                bass_transient=bass_transient,
                bass_level=0.1,
                timestamp_s=now,
            )
            if state["beat_event"]:
                events.append(now)
        intervals = [right - left for left, right in zip(events, events[1:])]
        self.assertGreater(len(events), 40)
        self.assertLess(max(intervals), 0.65)
        self.assertGreater(min(intervals), 0.31)
        self.assertAlmostEqual(clock.bpm or 0.0, 136.0, delta=0.35)
        self.assertEqual(state["tempo_switch_count"], 0)

    def test_acquisition_transients_do_not_flash_without_a_proven_grid(self) -> None:
        clock = RetainedBassClock()
        events = 0
        for step in range(1, 101):
            now = step * 0.02
            state = clock.update(
                BeatState(124.0, step % 9 == 0, 0, 0.0, 0.8),
                bass_transient=step % 9 == 0,
                bass_level=0.1,
                timestamp_s=now,
            )
            events += int(state["beat_event"])
        self.assertEqual(events, 0)
        self.assertIsNone(clock.bpm)

    def test_syncopated_transients_cannot_fire_outside_the_audio_grid(
        self,
    ) -> None:
        clock = self._seed_clock(120.0)
        events: list[float] = []
        for step in range(1, 3_001):
            now = step / 100.0
            # Deliberately jitter low-band transients around the half-second
            # grid. They must never become another lighting trigger; the
            # spectral PCM grid remains the single phase authority.
            phase = now % 0.5
            bass_transient = phase < 0.011 and int(now * 2) % 3 != 1
            state = clock.update(
                BeatState(
                    120.0,
                    bass_transient,
                    0,
                    (now * 2.0 % 4.0) / 4.0,
                    0.85,
                ),
                bass_transient=bass_transient,
                bass_level=0.08,
                timestamp_s=now,
            )
            if state["beat_event"]:
                events.append(now)
        intervals = [right - left for left, right in zip(events, events[1:])]
        self.assertGreaterEqual(len(events), 59)
        self.assertLessEqual(max(intervals), 0.52)
        self.assertGreaterEqual(min(intervals), 0.48)
        self.assertGreater(state["phase_correction_count"], 0)
        self.assertIsNotNone(state["phase_error_rms_ms"])

    def test_half_beat_candidate_phase_flip_cannot_drag_the_output_grid(
        self,
    ) -> None:
        clock = self._seed_clock(120.0)
        events: list[float] = []
        for step in range(1, 1_001):
            now = step / 100.0
            phase = (now * 2.0) % 4.0
            if 3.0 <= now <= 5.0:
                phase = (phase + 0.5) % 4.0
            state = clock.update(
                BeatState(120.0, False, 0, phase / 4.0, 0.85),
                bass_transient=False,
                bass_level=0.08,
                timestamp_s=now,
            )
            if state["beat_event"]:
                events.append(now)
        intervals = [right - left for left, right in zip(events, events[1:])]
        self.assertGreaterEqual(len(events), 19)
        self.assertGreater(state["phase_rejection_count"], 0)
        self.assertAlmostEqual(min(intervals), 0.5, delta=0.02)
        self.assertAlmostEqual(max(intervals), 0.5, delta=0.02)

    def test_stable_alternate_family_replaces_stale_track_clock(self) -> None:
        clock = self._seed_clock(127.0)
        state = {}
        for step in range(1, 241):
            now = step * 0.025
            bass_transient = step % 22 == 0
            state = clock.update(
                BeatState(107.0, bass_transient, 0, 0.0, 0.82),
                bass_transient=bass_transient,
                bass_level=0.1,
                timestamp_s=now,
                allow_family_switch=True,
            )
        self.assertAlmostEqual(clock.bpm or 0.0, 107.0, delta=0.1)
        self.assertAlmostEqual(state["family_anchor_bpm"], 107.0)
        self.assertEqual(state["tempo_switch_count"], 1)
        self.assertIsNone(state["rejected_candidate_bpm"])

    def test_family_anchor_prevents_incremental_candidate_ratcheting(self) -> None:
        clock = self._seed_clock(127.0)
        for second, candidate_bpm in enumerate(
            (128.0, 131.0, 134.0, 137.0, 140.0, 143.0),
            start=1,
        ):
            for offset in range(5):
                now = second + offset * 0.21
                clock.update(
                    BeatState(candidate_bpm, False, 0, 0.0, 0.8),
                    bass_transient=False,
                    bass_level=0.1,
                    timestamp_s=now,
                )
        self.assertLess(clock.bpm or 0.0, 133.0)
        self.assertAlmostEqual(clock._family_anchor_bpm or 0.0, 127.0)  # noqa: SLF001
        self.assertEqual(clock._tempo_switch_count, 0)  # noqa: SLF001

    def test_proven_grid_continues_on_bass_signal_but_stops_on_silence(self) -> None:
        clock = self._seed_clock(120.0)
        active_events = 0
        for step in range(1, 301):
            now = step / 100.0
            state = clock.update(
                BeatState(
                    120.0, False, 0, (now * 2.0 % 4.0) / 4.0, 0.8
                ),
                bass_transient=False,
                bass_level=0.08,
                timestamp_s=now,
            )
            active_events += int(state["beat_event"])
        self.assertGreaterEqual(active_events, 5)
        self.assertEqual(state["clock_state"], "locked")

        silent_events = 0
        for step in range(301, 501):
            now = step / 100.0
            state = clock.update(
                BeatState(
                    120.0, False, 0, (now * 2.0 % 4.0) / 4.0, 0.8
                ),
                bass_transient=False,
                bass_level=0.0,
                timestamp_s=now,
            )
            silent_events += int(state["beat_event"])
        self.assertEqual(silent_events, 0)
        self.assertEqual(state["clock_state"], "held")


class TimingLabRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = load_rig(ROOT / "config" / "party-parrot-active.json")
        self.output = VirtualDMXOutput()
        self.runtime = TimingLabRuntime(self.rig, self.output)
        self.controls = TimingLabControls()

    def test_controls_clamp_and_validate_without_persistence(self) -> None:
        controls = TimingLabControls()
        controls.patch({
            "base_intensity": 2,
            "flash_intensity": -1,
            "pulse_ms": 900,
            "movers_motion_period_s": 1,
            "center_motion_period_s": 80,
            "color_bars": 6,
        })
        self.assertEqual(controls.base_intensity, 0.35)
        self.assertEqual(controls.flash_intensity, 0.35)
        self.assertEqual(controls.pulse_ms, 250)
        self.assertEqual(controls.movers_motion_period_s, 4.0)
        self.assertEqual(controls.center_motion_period_s, 30.0)
        self.assertEqual(controls.color_bars, 6)
        with self.assertRaises(ValueError):
            controls.patch({"movers_source": "spotify_beat"})

    def test_beat_flash_uses_dimmers_and_every_internal_strobe_stays_zero(
        self,
    ) -> None:
        _, resting = self.runtime.step(_analysis(10.0), self.controls)
        _, flash = self.runtime.step(
            _analysis(10.2, beat_event=True, broad_event=True, beat_count=8),
            self.controls,
        )
        for fixture in (*self.rig.fixtures, *self.rig.auxiliary_fixtures):
            profile = party_parrot_profile(fixture.profile_key)
            assert profile is not None
            strobe = profile.channels.get("strobe")
            if strobe is not None:
                channel = fixture.address + strobe - 1
                self.assertEqual(flash.dmx.get_channel(fixture.universe, channel), 0)

        for fixture in self.rig.fixtures:
            profile = party_parrot_profile(fixture.profile_key)
            assert profile is not None
            dimmer = fixture.address + profile.channels["dimmer"] - 1
            self.assertGreater(
                flash.dmx.get_channel(fixture.universe, dimmer),
                resting.dmx.get_channel(fixture.universe, dimmer),
            )
        center = self.rig.auxiliary_fixtures[0]
        center_profile = party_parrot_profile(center.profile_key)
        assert center_profile is not None
        center_dimmer = center.address + center_profile.channels["master_dimmer"] - 1
        self.assertGreater(
            flash.dmx.get_channel(center.universe, center_dimmer),
            resting.dmx.get_channel(center.universe, center_dimmer),
        )
        self.assertFalse(self.runtime.invariants()["learning_writes"])
        self.assertFalse(self.runtime.invariants()["performance_runtime"])

    def test_lane_sources_are_independent(self) -> None:
        controls = replace(
            self.controls,
            movers_source="bass_clock",
            center_source="broadband_onset",
        )
        self.runtime.step(
            _analysis(20.0, beat_event=True, broad_event=False), controls
        )
        snapshot = self.runtime.snapshot()
        self.assertTrue(snapshot["lanes"]["movers"]["flash_active"])
        self.assertFalse(snapshot["lanes"]["center"]["flash_active"])


class TimingLabApplicationTests(unittest.TestCase):
    def test_spotify_identity_resets_clock_without_persisting_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            application = LumenApplication(
                rig_path=ROOT / "config" / "party-parrot-active.json",
                memory_path=root / "memory.sqlite3",
                settings_path=root / "settings.json",
            )
            try:
                application.engine_mode = "timing_lab"
                identities = [
                    MediaIdentity("spotify", "first", "First"),
                    MediaIdentity("spotify", "second", "Second"),
                ]
                with (
                    patch.object(
                        application,
                        "_spotify_valid_token",
                        return_value=object(),
                    ),
                    patch("lumen_engine.control.SpotifyWebAPI") as api,
                    patch(
                        "lumen_engine.control.media_identity_from_spotify",
                        side_effect=identities,
                    ),
                    patch.object(application.memory, "remember_media") as persist,
                ):
                    api.return_value.playback.return_value = {}
                    application._poll_timing_lab_identity()  # noqa: SLF001
                    self.assertEqual(application._timing_lab_track_generation, 0)  # noqa: SLF001
                    application._poll_timing_lab_identity()  # noqa: SLF001
                    self.assertEqual(application._timing_lab_track_generation, 1)  # noqa: SLF001
                    self.assertEqual(application._timing_lab_track_resets, 1)  # noqa: SLF001
                    persist.assert_not_called()
            finally:
                application.engine_mode = "standby"
                application.close()

    def test_default_mode_is_virtual_and_output_switch_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            application = LumenApplication(
                rig_path=ROOT / "config" / "party-parrot-active.json",
                memory_path=root / "memory.sqlite3",
                settings_path=root / "settings.json",
            )
            try:
                with (
                    patch(
                        "lumen_engine.control.OpenDmxUsbOutput.open"
                    ) as physical_open,
                    patch.object(
                        application,
                        "_run_timing_lab",
                        side_effect=lambda _runtime: application._stop.wait(2.0),
                    ),
                ):
                    application.start("timing_lab")
                    deadline = time.monotonic() + 1.0
                    status = application.snapshot()
                    while (
                        status["output"] is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                        status = application.snapshot()
                    self.assertEqual(status["engine"]["mode"], "timing_lab")
                    self.assertEqual(status["output"]["backend"], "Virtual DMX")
                    self.assertFalse(
                        status["timing_lab"]["invariants"]["learning_writes"]
                    )
                    physical_open.assert_not_called()
                    status = application.patch_timing_lab({"reset_clock": True})
                    self.assertEqual(
                        status["timing_lab"]["track_boundary"]["reset_count"],
                        1,
                    )
                    with self.assertRaises(RuntimeError):
                        application.patch_timing_lab({"output": "live"})
                    self.assertEqual(
                        application.snapshot()["timing_lab"]["output"],
                        "virtual",
                    )
                    application.stop()
            finally:
                application.close()


if __name__ == "__main__":
    unittest.main()
