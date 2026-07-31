from __future__ import annotations

from array import array
import math
import unittest

from lumen_engine.audio import RealtimeAudioAnalyzer


def sine_pcm(frequency: float, sample_rate: int, frames: int, amplitude: int) -> bytes:
    samples = array(
        "h",
        (
            round(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            for i in range(frames)
        ),
    )
    return samples.tobytes()


class AudioAnalyzerTests(unittest.TestCase):
    def test_section_tracker_rejects_packet_scale_label_flicker(self) -> None:
        analyzer = RealtimeAudioAnalyzer(sample_rate=48_000, channels=1)
        sections = []
        for index in range(240):
            timestamp = index / 24.0
            onset = 0.64 if index % 12 == 0 else 0.12
            section, _confidence = analyzer._classify_section(
                timestamp, 0.62, onset, 0.42, onset * 0.72
            )
            sections.append(section)
        self.assertEqual(set(sections), {"groove"})

    def test_section_tracker_holds_build_and_release(self) -> None:
        analyzer = RealtimeAudioAnalyzer(sample_rate=48_000, channels=1)
        timestamp = 0.0
        for _ in range(96):
            analyzer._classify_section(timestamp, 0.35, 0.18, 0.35, 0.18)
            timestamp += 1 / 24.0
        seen_build = False
        for index in range(72):
            level = 0.35 + index / 72.0 * 0.45
            section, _ = analyzer._classify_section(
                timestamp, level, 0.55, 0.45, 0.50
            )
            timestamp += 1 / 24.0
            seen_build |= section == "build"
        self.assertTrue(seen_build)
        section, _ = analyzer._classify_section(
            timestamp, 0.92, 0.95, 0.62, 0.90
        )
        self.assertEqual(section, "release")
        for index in range(60):
            timestamp += 1 / 24.0
            section, _ = analyzer._classify_section(
                timestamp,
                0.76,
                0.62 if index % 8 == 0 else 0.16,
                0.46,
                0.48 if index % 8 == 0 else 0.20,
            )
            if index < 36:
                self.assertEqual(section, "release")
        self.assertEqual(section, "groove")

    def test_silence_is_bounded(self) -> None:
        analyzer = RealtimeAudioAnalyzer(sample_rate=48_000, channels=1)
        observation = analyzer.analyze_pcm16(bytes(4096), timestamp_s=0.0)
        self.assertEqual(observation.loudness, 0.0)
        self.assertEqual(observation.onset_strength, 0.0)
        self.assertEqual(observation.low_energy, 0.0)
        self.assertEqual(analyzer.last_metrics.dbfs, -120.0)
        self.assertEqual(analyzer.last_metrics.frame_count, 2048)
        self.assertEqual(analyzer.last_metrics.clipped_samples, 0)

    def test_low_sine_has_more_low_than_high_energy(self) -> None:
        analyzer = RealtimeAudioAnalyzer(sample_rate=8_000, channels=1)
        pcm = sine_pcm(100, 8_000, 2048, 12_000)
        observation = analyzer.analyze_pcm16(pcm, timestamp_s=1.0)
        self.assertGreater(observation.loudness, 0.2)
        self.assertGreater(observation.low_energy, observation.high_energy)
        self.assertAlmostEqual(
            observation.low_energy
            + observation.mid_energy
            + observation.high_energy,
            1.0,
            places=6,
        )
        self.assertGreater(analyzer.last_metrics.dbfs, -20.0)
        self.assertGreater(analyzer.last_metrics.peak, 0.3)
        self.assertEqual(len(analyzer.last_metrics.waveform), 128)

    def test_stereo_metrics_report_each_channel_and_clipping(self) -> None:
        analyzer = RealtimeAudioAnalyzer(sample_rate=48_000, channels=2)
        samples = array("h", [32767, 1000, -32768, -1000] * 512)
        analyzer.analyze_pcm16(samples.tobytes(), timestamp_s=2.0)
        metrics = analyzer.last_metrics
        self.assertEqual(metrics.frame_count, 1024)
        self.assertGreater(metrics.channel_rms[0], metrics.channel_rms[1])
        self.assertGreaterEqual(metrics.clipped_samples, 1024)

    def test_periodic_kicks_create_tempo_clock_and_visible_pulse(self) -> None:
        sample_rate = 48_000
        frames = 2_048
        analyzer = RealtimeAudioAnalyzer(sample_rate=sample_rate, channels=1)
        pulse_count = 0
        observation = None
        for chunk in range(180):
            amplitude = 18_000 if chunk % 12 == 0 else 500
            start = chunk * frames
            pcm = array(
                "h",
                (
                    round(
                        amplitude
                        * math.sin(
                            2
                            * math.pi
                            * 100
                            * (start + index)
                            / sample_rate
                        )
                    )
                    for index in range(frames)
                ),
            ).tobytes()
            observation = analyzer.analyze_pcm16(
                pcm,
                timestamp_s=chunk * frames / sample_rate,
            )
            if observation.beat_pulse >= 0.9:
                pulse_count += 1
        assert observation is not None
        self.assertGreaterEqual(pulse_count, 12)
        self.assertAlmostEqual(observation.bpm or 0.0, 117.2, delta=2.0)
        self.assertGreater(observation.beat_confidence, 0.8)
        self.assertGreaterEqual(observation.bar_phase, 0.0)
        self.assertLessEqual(observation.bar_phase, 1.0)

    def test_silence_clears_tempo_confidence_and_reset_clears_song_state(self) -> None:
        sample_rate = 48_000
        frames = 2_048
        analyzer = RealtimeAudioAnalyzer(sample_rate=sample_rate, channels=1)
        for chunk in range(180):
            amplitude = 18_000 if chunk % 12 == 0 else 500
            pcm = sine_pcm(100, sample_rate, frames, amplitude)
            observation = analyzer.analyze_pcm16(
                pcm, timestamp_s=chunk * frames / sample_rate
            )
        self.assertIsNotNone(observation.bpm)
        for offset in range(30):
            observation = analyzer.analyze_pcm16(
                bytes(frames * 2),
                timestamp_s=(180 + offset) * frames / sample_rate,
            )
        self.assertIsNone(observation.bpm)
        self.assertLess(observation.beat_confidence, 0.05)
        analyzer.reset()
        self.assertEqual(analyzer._section, "groove")
        self.assertIsNone(analyzer._tempo_tracker)


if __name__ == "__main__":
    unittest.main()
