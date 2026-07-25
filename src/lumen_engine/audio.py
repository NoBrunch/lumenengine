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


@dataclass(frozen=True, slots=True)
class AudioInputMetrics:
    """Measurements taken directly from one captured PCM packet."""

    timestamp_s: float
    frame_count: int
    rms: float
    dbfs: float
    peak: float
    channel_rms: tuple[float, ...]
    channel_peak: tuple[float, ...]
    clipped_samples: int
    waveform: tuple[float, ...]

    @classmethod
    def silence(
        cls,
        timestamp_s: float = 0.0,
        channels: int = 2,
    ) -> "AudioInputMetrics":
        return cls(
            timestamp_s=timestamp_s,
            frame_count=0,
            rms=0.0,
            dbfs=-120.0,
            peak=0.0,
            channel_rms=tuple(0.0 for _ in range(channels)),
            channel_peak=tuple(0.0 for _ in range(channels)),
            clipped_samples=0,
            waveform=tuple(0.0 for _ in range(128)),
        )


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
        self._noise_floor = 0.0005
        self._level_envelope = 0.005
        self._beat_envelope = 0.0
        self._last_timestamp: float | None = None
        self.last_metrics = AudioInputMetrics.silence(channels=channels)

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
            self.last_metrics = AudioInputMetrics.silence(timestamp, self.channels)
            return self._silence_observation(timestamp)

        mono = self._downmix(samples)
        rms = math.sqrt(sum(sample * sample for sample in mono) / len(mono)) / 32768.0
        self.last_metrics = self._input_metrics(samples, mono, timestamp, rms)
        # Track the actual interface noise without allowing sustained music to
        # drag the floor upward. This prevents electrical hiss from becoming a
        # fictional low-frequency "performance" during silence.
        noise_alpha = 0.97 if rms <= self._noise_floor * 1.5 else 0.9995
        self._noise_floor = clamp(
            noise_alpha * self._noise_floor
            + (1.0 - noise_alpha) * min(rms, 0.02),
            0.00005,
            0.02,
        )
        signal_present = rms >= max(0.001, self._noise_floor * 2.2)
        if signal_present:
            # A gentle logarithmic mapping makes normal line levels readable.
            loudness = clamp(math.log10(1.0 + 24.0 * rms), 0.0, 1.0)
            rise = max(0.0, loudness - self._previous_loudness)
            transient = max(
                0.0,
                (rms - self._level_envelope)
                / max(0.01, self._level_envelope),
            )
            onset = clamp(rise * 3.8 + transient * 0.85, 0.0, 1.0)
            self._level_envelope += 0.08 * (rms - self._level_envelope)
            low, mid, high = self._spectral_proportions(mono)
        else:
            loudness = 0.0
            onset = 0.0
            low = mid = high = 0.0
            self._level_envelope += 0.02 * (
                max(self._noise_floor, rms) - self._level_envelope
            )
        self._previous_loudness = loudness

        beat_state = self._beat_tracker.update(
            low * loudness if signal_present else 0.0,
            now=timestamp,
        )
        elapsed = (
            1.0 / 24.0
            if self._last_timestamp is None
            else max(0.0, min(0.25, timestamp - self._last_timestamp))
        )
        self._last_timestamp = timestamp
        self._beat_envelope *= math.exp(-elapsed / 0.14)
        if beat_state.beat:
            self._beat_envelope = 1.0
        elif onset >= 0.72:
            # Strong transients remain useful while the tempo tracker is
            # collecting enough beats to establish a confident clock.
            self._beat_envelope = max(self._beat_envelope, 0.58)

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
            bar_phase=beat_state.bar_progress,
            beat_pulse=clamp(self._beat_envelope, 0.0, 1.0),
            beat_confidence=beat_confidence,
            bpm=bpm,
            novelty=novelty,
        )

    def _input_metrics(
        self,
        samples: array[int],
        mono: list[float],
        timestamp: float,
        rms: float,
    ) -> AudioInputMetrics:
        channel_rms: list[float] = []
        channel_peak: list[float] = []
        for channel in range(self.channels):
            values = samples[channel::self.channels]
            if not values:
                channel_rms.append(0.0)
                channel_peak.append(0.0)
                continue
            channel_rms.append(
                math.sqrt(sum(sample * sample for sample in values) / len(values))
                / 32768.0
            )
            channel_peak.append(max(abs(sample) for sample in values) / 32768.0)

        point_count = min(128, len(mono))
        if point_count:
            waveform = tuple(
                clamp(
                    mono[
                        min(
                            len(mono) - 1,
                            round(index * (len(mono) - 1) / max(1, point_count - 1)),
                        )
                    ]
                    / 32768.0,
                    -1.0,
                    1.0,
                )
                for index in range(point_count)
            )
        else:
            waveform = ()
        return AudioInputMetrics(
            timestamp_s=timestamp,
            frame_count=len(mono),
            rms=rms,
            dbfs=max(-120.0, 20.0 * math.log10(max(rms, 1e-6))),
            peak=max(abs(sample) for sample in samples) / 32768.0,
            channel_rms=tuple(channel_rms),
            channel_peak=tuple(channel_peak),
            clipped_samples=sum(1 for sample in samples if abs(sample) >= 32760),
            waveform=waveform,
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
        # A raw, rectangular PCM window leaks DC and low-frequency power into
        # every Goertzel bin.  On the garage line input that made ordinary
        # full-range music appear to be roughly 99% bass.  Remove the packet's
        # DC offset and taper both ends before measuring the bands.
        mean = sum(samples) / len(samples)
        last_index = len(samples) - 1
        windowed = [
            (sample - mean)
            * (0.5 - 0.5 * math.cos(math.tau * index / last_index))
            for index, sample in enumerate(samples)
        ]
        # Goertzel bins are cheaper than a general FFT and sufficient for the
        # initial low/mid/high character estimate.
        low_power = sum(self._goertzel(windowed, f) for f in (63, 100, 160, 250))
        mid_power = sum(
            self._goertzel(windowed, f) for f in (400, 800, 1600, 2500)
        )
        high_power = sum(
            self._goertzel(windowed, f) for f in (4000, 6300, 9000)
        )
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
