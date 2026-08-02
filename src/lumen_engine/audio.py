"""Dependency-free PCM analysis and an ALSA line-in capture adapter."""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import math
import subprocess
import sys
import time
from typing import Iterator

import numpy as np

from lumen_engine.models import MusicalObservation, clamp
from lumen_engine.beat import BeatTracker, SpectralTempoTracker


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
        self._tempo_tracker: SpectralTempoTracker | None = None
        self._tempo_update_rate = 0.0
        self._noise_floor = 0.0005
        self._level_envelope = 0.005
        self._beat_envelope = 0.0
        self._previous_spectrum: np.ndarray | None = None
        self._previous_chroma: np.ndarray | None = None
        self._previous_bands: tuple[float, float, float] | None = None
        self._rhythm_density = 0.0
        self._arrangement_change = 0.0
        self._previous_bass_level = 0.0
        self._flux_history: deque[float] = deque(maxlen=420)
        self._bass_rise_history: deque[float] = deque(maxlen=420)
        self._last_timestamp: float | None = None
        self._section = "groove"
        self._section_started_at: float | None = None
        self._section_candidate: str | None = None
        self._section_candidate_since: float | None = None
        self._section_fast_level = 0.0
        self._section_slow_level = 0.0
        self._section_last_timestamp: float | None = None
        self._silence_started_at: float | None = None
        self._tempo_reset_for_silence = False
        self._release_refractory_until = float("-inf")
        self.last_metrics = AudioInputMetrics.silence(channels=channels)

    def reset(self) -> None:
        """Reset temporal analysis at a recording/seek boundary."""
        self.__init__(self.sample_rate, self.channels)

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
        # Learn the floor only from genuinely quiet packets. Sustained music
        # must not ratchet the floor upward until soft passages disappear.
        if rms <= max(0.003, self._noise_floor * 1.5):
            self._noise_floor = clamp(
                0.97 * self._noise_floor + 0.03 * min(rms, 0.003),
                0.00005,
                0.003,
            )
        signal_present = rms >= max(0.001, self._noise_floor * 2.2)
        if signal_present:
            # A gentle logarithmic mapping makes normal line levels readable.
            loudness = clamp(math.log10(1.0 + 24.0 * rms), 0.0, 1.0)
            rise = max(0.0, loudness - self._previous_loudness)
            amplitude_transient = max(
                0.0,
                (rms - self._level_envelope)
                / max(0.01, self._level_envelope),
            )
            (
                low,
                mid,
                high,
                spectral_onset,
                spectral_flux,
                spectral_brightness,
                harmonic_change,
            ) = self._spectral_features(mono)
            onset = clamp(
                0.76 * spectral_onset
                + 0.16 * rise * 3.0
                + 0.08 * amplitude_transient,
                0.0,
                1.0,
            )
            self._level_envelope += 0.08 * (rms - self._level_envelope)
        else:
            loudness = 0.0
            onset = 0.0
            spectral_onset = 0.0
            spectral_flux = 0.0
            spectral_brightness = 0.0
            harmonic_change = 0.0
            low = mid = high = 0.0
            self._level_envelope += 0.02 * (
                max(self._noise_floor, rms) - self._level_envelope
            )
        self._previous_loudness = loudness

        beat_drive = spectral_onset * loudness if signal_present else 0.0
        fallback_beat_state = self._beat_tracker.update(beat_drive, now=timestamp)
        update_rate = self.sample_rate / max(1, len(mono))
        if (
            self._tempo_tracker is None
            or abs(update_rate - self._tempo_update_rate) > 0.01
        ):
            self._tempo_update_rate = update_rate
            self._tempo_tracker = SpectralTempoTracker(update_rate)
        spectral_beat_state = self._tempo_tracker.update(beat_drive, timestamp)
        beat_state = (
            spectral_beat_state
            if spectral_beat_state.bpm > 0.0
            else fallback_beat_state
        )
        if signal_present:
            self._silence_started_at = None
            self._tempo_reset_for_silence = False
            quiet_for = 0.0
        else:
            if self._silence_started_at is None:
                self._silence_started_at = timestamp
            quiet_for = max(0.0, timestamp - self._silence_started_at)
        elapsed = (
            1.0 / 24.0
            if self._last_timestamp is None
            else max(0.0, min(0.25, timestamp - self._last_timestamp))
        )
        self._last_timestamp = timestamp
        rhythm_alpha = 1.0 - math.exp(-max(0.001, elapsed) / 2.0)
        self._rhythm_density += rhythm_alpha * (
            clamp(0.65 * onset + 0.35 * beat_state.confidence, 0.0, 1.0)
            - self._rhythm_density
        )
        band_change = 0.0
        if self._previous_bands is not None:
            band_change = min(
                1.0,
                sum(
                    abs(value - previous)
                    for value, previous in zip(
                        (low, mid, high), self._previous_bands
                    )
                )
                * 1.8,
            )
        self._previous_bands = (low, mid, high)
        arrangement_evidence = clamp(
            0.35 * spectral_flux
            + 0.35 * harmonic_change
            + 0.20 * band_change
            + 0.10 * abs(onset - self._rhythm_density),
            0.0,
            1.0,
        )
        arrangement_alpha = 1.0 - math.exp(-max(0.001, elapsed) / 0.8)
        self._arrangement_change += arrangement_alpha * (
            arrangement_evidence - self._arrangement_change
        )
        self._beat_envelope *= math.exp(-elapsed / 0.14)
        if beat_state.beat:
            self._beat_envelope = 1.0
        elif onset >= 0.72 and beat_state.confidence < 0.15:
            # Strong transients remain useful while the tempo tracker is
            # collecting enough beats to establish a confident clock.
            self._beat_envelope = max(self._beat_envelope, 0.58)

        bpm = beat_state.bpm or None
        beat_confidence = beat_state.confidence
        if not signal_present:
            beat_confidence *= math.exp(-quiet_for / 0.35)
            if quiet_for >= 0.75:
                bpm = None
            if quiet_for >= 1.5 and not self._tempo_reset_for_silence:
                self._beat_tracker = BeatTracker()
                self._tempo_tracker = SpectralTempoTracker(self._tempo_update_rate)
                self._tempo_reset_for_silence = True
        beat_phase = (beat_state.bar_progress * 4.0) % 1.0
        novelty = clamp(0.65 * onset + 0.35 * abs(high - low), 0.0, 1.0)
        section, section_confidence = self._classify_section(
            timestamp, loudness, onset, low, novelty
        )
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
            section=section,
            section_confidence=section_confidence,
            novelty=novelty,
            spectral_flux=spectral_flux,
            spectral_brightness=spectral_brightness,
            rhythm_density=clamp(self._rhythm_density, 0.0, 1.0),
            harmonic_change=harmonic_change,
            arrangement_change=clamp(self._arrangement_change, 0.0, 1.0),
        )

    def _classify_section(
        self,
        timestamp: float,
        loudness: float,
        onset: float,
        low_energy: float,
        novelty: float,
    ) -> tuple[str, float]:
        """Track stable musical regions with hysteresis and transition memory.

        Spectral features arrive every ~43 ms. Treating each packet as a whole
        song section caused the dashboard to alternate groove/build on normal
        notes. This tracker compares short- and long-term level, requires a
        candidate to persist, and holds a release long enough to be meaningful.
        """
        if self._section_last_timestamp is None:
            elapsed = 1.0 / 24.0
            self._section_fast_level = loudness
            self._section_slow_level = loudness
            self._section_started_at = timestamp
        else:
            elapsed = clamp(timestamp - self._section_last_timestamp, 0.005, 0.25)
        self._section_last_timestamp = timestamp
        fast_alpha = 1.0 - math.exp(-elapsed / 0.70)
        slow_alpha = 1.0 - math.exp(-elapsed / 5.5)
        self._section_fast_level += fast_alpha * (loudness - self._section_fast_level)
        self._section_slow_level += slow_alpha * (loudness - self._section_slow_level)
        trend = self._section_fast_level - self._section_slow_level
        section_age = timestamp - (
            timestamp if self._section_started_at is None else self._section_started_at
        )

        if self._section == "release" and section_age < 1.6:
            return "release", clamp(0.72 + 0.18 * onset, 0.0, 1.0)
        if self._section == "release":
            # Release is a bounded transition event, not a persistent region.
            # Return to groove unconditionally so ordinary post-drop
            # transients cannot trap the classifier in release forever.
            self._section = "groove"
            self._section_started_at = timestamp
            self._section_candidate = None
            self._section_candidate_since = None
            section_age = 0.0

        strong_release = (
            onset >= 0.82
            and low_energy >= 0.32
            and novelty >= 0.58
            and loudness >= max(0.34, self._section_slow_level * 0.82)
        )
        if (
            self._section != "release"
            and timestamp >= self._release_refractory_until
            and strong_release
            and (self._section == "build" or trend >= 0.045)
        ):
            candidate = "release"
            required = 0.0
        elif loudness <= max(0.10, self._section_slow_level * 0.48):
            candidate = "breakdown"
            required = 0.75
        elif trend >= 0.055 and novelty >= 0.30 and loudness >= 0.26:
            candidate = "build"
            required = 1.10
        else:
            candidate = "groove"
            required = 1.35

        if candidate != self._section_candidate:
            self._section_candidate = candidate
            self._section_candidate_since = timestamp
        candidate_age = timestamp - (
            timestamp
            if self._section_candidate_since is None
            else self._section_candidate_since
        )
        minimum_hold = 1.0 if self._section == "breakdown" else 2.4
        if (
            candidate != self._section
            and candidate_age >= required
            and (section_age >= minimum_hold or candidate == "release")
        ):
            self._section = candidate
            self._section_started_at = timestamp
            if candidate == "release":
                self._release_refractory_until = timestamp + 3.5
            self._section_candidate = None
            self._section_candidate_since = None
            section_age = 0.0

        stability = clamp(section_age / 3.0, 0.0, 1.0)
        evidence = {
            "breakdown": clamp((0.24 - loudness) / 0.20, 0.0, 1.0),
            "build": clamp(trend / 0.14 + novelty * 0.35, 0.0, 1.0),
            "release": clamp(onset * 0.70 + novelty * 0.30, 0.0, 1.0),
            "groove": clamp(0.45 + 0.35 * (1.0 - novelty), 0.0, 1.0),
        }[self._section]
        return self._section, clamp(0.35 + 0.35 * stability + 0.30 * evidence, 0.0, 1.0)

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
            clipped_samples=sum(1 for sample in samples if abs(sample) >= 32767),
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
        low, mid, high, *_rest = self._spectral_features(samples)
        return low, mid, high

    def _spectral_features(
        self, samples: list[float]
    ) -> tuple[float, float, float, float, float, float, float]:
        if len(samples) < 8:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        values = np.asarray(samples, dtype=np.float64) / 32768.0
        values -= float(np.mean(values))
        magnitude = np.abs(np.fft.rfft(values * np.hanning(len(values))))
        frequencies = np.fft.rfftfreq(len(values), 1.0 / self.sample_rate)
        power = magnitude * magnitude

        def band_level(low_hz: float, high_hz: float) -> float:
            mask = (frequencies >= low_hz) & (frequencies < high_hz)
            if not np.any(mask):
                return 0.0
            return float(np.sqrt(np.mean(power[mask])))

        # Compare average power density rather than a few isolated bins. This
        # avoids both DC leakage and the unequal bandwidth bias that previously
        # labeled a full-range mix as almost entirely bass.
        low_level = band_level(35.0, 250.0)
        mid_level = band_level(250.0, 3_000.0)
        high_level = band_level(3_000.0, 12_000.0)
        total = low_level + mid_level + high_level
        if total <= 1e-12:
            low = mid = high = 0.0
        else:
            low = low_level / total
            mid = mid_level / total
            high = high_level / total

        log_spectrum = np.log1p(magnitude * 20.0)
        onset_mask = (frequencies >= 40.0) & (frequencies <= 4_000.0)
        if (
            self._previous_spectrum is None
            or self._previous_spectrum.shape != log_spectrum.shape
        ):
            raw_flux = 0.0
        else:
            raw_flux = float(
                np.mean(
                    np.maximum(
                        0.0,
                        log_spectrum[onset_mask]
                        - self._previous_spectrum[onset_mask],
                    )
                )
            )
        self._previous_spectrum = log_spectrum
        bass_level = math.log1p(low_level * 20.0)
        bass_rise = max(0.0, bass_level - self._previous_bass_level)
        self._previous_bass_level = bass_level
        self._flux_history.append(raw_flux)
        self._bass_rise_history.append(bass_rise)
        flux = self._normalize_feature(raw_flux, self._flux_history)
        bass = self._normalize_feature(bass_rise, self._bass_rise_history)
        onset = clamp(0.68 * flux + 0.32 * bass, 0.0, 1.0)
        audible = (frequencies >= 40.0) & (frequencies <= 12_000.0)
        audible_power = power[audible]
        brightness = (
            clamp(
                float(np.sum(frequencies[audible] * audible_power))
                / max(1e-12, float(np.sum(audible_power)))
                / 8_000.0,
                0.0,
                1.0,
            )
            if np.any(audible)
            else 0.0
        )
        chroma = np.zeros(12, dtype=np.float64)
        harmonic_mask = (frequencies >= 55.0) & (frequencies <= 5_000.0)
        harmonic_frequencies = frequencies[harmonic_mask]
        harmonic_power = power[harmonic_mask]
        for frequency, value in zip(harmonic_frequencies, harmonic_power):
            note = int(round(69.0 + 12.0 * math.log2(float(frequency) / 440.0)))
            chroma[note % 12] += float(value)
        chroma_norm = float(np.linalg.norm(chroma))
        if chroma_norm > 1e-12:
            chroma /= chroma_norm
        harmonic_change = 0.0
        if self._previous_chroma is not None and chroma_norm > 1e-12:
            harmonic_change = clamp(
                1.0 - float(np.dot(chroma, self._previous_chroma)),
                0.0,
                1.0,
            )
        if chroma_norm > 1e-12:
            self._previous_chroma = chroma
        return low, mid, high, onset, flux, brightness, harmonic_change

    @staticmethod
    def _normalize_feature(value: float, history: deque[float]) -> float:
        if len(history) < 8:
            return clamp(value, 0.0, 1.0)
        values = np.asarray(history, dtype=np.float64)
        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 95))
        return clamp((value - low) / max(1e-9, high - low), 0.0, 1.0)

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
