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


if __name__ == "__main__":
    unittest.main()
