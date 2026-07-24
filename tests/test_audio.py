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


if __name__ == "__main__":
    unittest.main()

