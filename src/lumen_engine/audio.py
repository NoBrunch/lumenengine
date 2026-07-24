"""Dependency-free PCM analysis and an ALSA line-in capture adapter."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import subprocess
import sys
import time
from typing import Iterator

from lumen_engine.models import MusicalObservation, clamp
from lumen_engine.beat import BeatTracker


@dataclass(frozen=True, slots=True)
class AudioCaptureConfig:
    device: str = "default"
    sample_rate: int = 48_000
    channels: int = 2
    chunk_frames: int = 2_048


class RealtimeAudioAnalyzer:
    """Create useful first-pass musical observations from signed PCM16 audio."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self._previous_loudness = 0.0
        self._beat_tracker = BeatTracker()
        self._noise_floor = 0.005

    def analyze_pcm16(
        self, pcm: bytes, timestamp_s: float | None = None
    ) -> MusicalObservation:
        timestamp = time.monotonic() if timestamp_s is None else timestamp_s
        if len(pcm) % 2:
            raise ValueError("PCM16 input must contain an even number of bytes")
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return self._silence_observation(timestamp)

        mono = self._downmix(samples)
        rms = math.sqrt(sum(sample * sample for sample in mono) / len(mono)) / 32768.0
        # A gentle logarithmic mapping makes normal line levels readable.
        loudness = clamp(math.log10(1.0 + 24.0 * rms), 0.0, 1.0)
        self._noise_floor = 0.995 * self._noise_floor + 0.005 * min(rms, 0.08)
        rise = max(0.0, loudness - self._previous_loudness)
        onset = clamp(rise * 5.0 + max(0.0, rms - self._noise_floor * 2.5) * 1.8, 0, 1)
        self._previous_loudness = loudness

        low, mid, high = self._spectral_proportions(mono)
        beat_state = self._beat_tracker.update(low * loudness, now=timestamp)
        bpm = beat_state.bpm or None
        beat_confidence = beat_state.confidence
        beat_phase = (beat_state.bar_progress * 4.0) % 1.0
        novelty = clamp(0.65 * onset + 0.35 * abs(high - low), 0.0, 1.0)
        return MusicalObservation(
            timestamp_s=timestamp,
            loudness=loudness,
            onset_strength=onset,
            low_energy=low,
            mid_energy=mid,
            high_energy=high,
            beat_phase=beat_phase,
            beat_confidence=beat_confidence,
            bpm=bpm,
            novelty=novelty,
        )

    def _downmix(self, samples: array[int]) -> list[float]:
        if self.channels == 1:
            return [float(sample) for sample in samples]
        complete = len(samples) - len(samples) % self.channels
        return [
            sum(samples[index : index + self.channels]) / self.channels
            for index in range(0, complete, self.channels)
        ]

    def _spectral_proportions(
        self, samples: list[float]
    ) -> tuple[float, float, float]:
        if len(samples) < 8:
            return 0.0, 0.0, 0.0
        # Goertzel bins are cheaper than a general FFT and sufficient for the
        # initial low/mid/high character estimate.
        low_power = sum(self._goertzel(samples, f) for f in (63, 100, 160, 250))
        mid_power = sum(self._goertzel(samples, f) for f in (400, 800, 1600, 2500))
        high_power = sum(self._goertzel(samples, f) for f in (4000, 6300, 9000))
        total = low_power + mid_power + high_power
        if total <= 1e-9:
            return 0.0, 0.0, 0.0
        return low_power / total, mid_power / total, high_power / total

    def _goertzel(self, samples: list[float], frequency: float) -> float:
        omega = 2.0 * math.pi * frequency / self.sample_rate
        coefficient = 2.0 * math.cos(omega)
        previous = 0.0
        previous_two = 0.0
        for sample in samples:
            current = sample + coefficient * previous - previous_two
            previous_two = previous
            previous = current
        return max(
            0.0,
            previous_two * previous_two
            + previous * previous
            - coefficient * previous * previous_two,
        )

    @staticmethod
    def _silence_observation(timestamp: float) -> MusicalObservation:
        return MusicalObservation(
            timestamp_s=timestamp,
            loudness=0.0,
            onset_strength=0.0,
            low_energy=0.0,
            mid_energy=0.0,
            high_energy=0.0,
        )


class AlsaLineIn:
    """Read raw PCM from the system `arecord` command."""

    def __init__(self, config: AudioCaptureConfig | None = None) -> None:
        self.config = config or AudioCaptureConfig()
        self._process: subprocess.Popen[bytes] | None = None

    def chunks(self) -> Iterator[bytes]:
        if self._process is not None:
            raise RuntimeError("capture is already active")
        config = self.config
        command = [
            "arecord",
            "-q",
            "-D",
            config.device,
            "-f",
            "S16_LE",
            "-r",
            str(config.sample_rate),
            "-c",
            str(config.channels),
            "-t",
            "raw",
        ]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE)
        assert self._process.stdout is not None
        chunk_bytes = config.chunk_frames * config.channels * 2
        try:
            while True:
                data = self._process.stdout.read(chunk_bytes)
                if not data:
                    code = self._process.poll()
                    if code not in (None, 0):
                        raise RuntimeError(f"arecord stopped with exit code {code}")
                    break
                yield data
        finally:
            self.close()

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process = None

    def __enter__(self) -> "AlsaLineIn":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
