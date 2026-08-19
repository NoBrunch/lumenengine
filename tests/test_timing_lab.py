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
from lumen_engine.config import load_rig
from lumen_engine.control import LumenApplication
from lumen_engine.dmx import VirtualDMXOutput
from lumen_engine.profiles import party_parrot_profile
from lumen_engine.timing_lab import (
    TimingLabAnalysis,
    TimingLabAnalyzer,
    TimingLabControls,
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
    )


class TimingLabAnalyzerTests(unittest.TestCase):
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
