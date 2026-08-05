"""Dependency-free PCM analysis and an ALSA line-in capture adapter."""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import math
import queue
import subprocess
import threading
import time
import sys
from typing import Any, Iterator

import numpy as np

from lumen_engine.models import MusicalObservation, clamp
from lumen_engine.structure import transition_event_for
from lumen_engine.beat import (
    BeatState,
    BeatTracker,
    SpectralTempoTracker,
    TempoSourceArbiter,
)


@dataclass(frozen=True, slots=True)
class AudioCaptureConfig:
    device: str = "default"
    sample_rate: int = 48_000
    channels: int = 2
    chunk_frames: int = 2_048


class SourcePcm(bytes):
    """PCM bytes tagged with their position on the capture sample clock."""

    source_start_frame: int
    frame_count: int
    timestamp_s: float

    def __new__(
        cls,
        pcm: bytes,
        *,
        source_start_frame: int,
        frame_count: int,
        timestamp_s: float,
    ) -> "SourcePcm":
        value = super().__new__(cls, pcm)
        value.source_start_frame = int(source_start_frame)
        value.frame_count = int(frame_count)
        value.timestamp_s = float(timestamp_s)
        return value


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
        self._tempo_arbiter = TempoSourceArbiter()
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
        self._section_reference_level = 0.0
        self._section_fast_rhythm = 0.0
        self._section_slow_rhythm = 0.0
        self._section_fast_brightness = 0.0
        self._section_slow_brightness = 0.0
        self._section_fast_bass = 0.0
        self._section_slow_bass = 0.0
        self._section_last_timestamp: float | None = None
        self._silence_started_at: float | None = None
        self._tempo_reset_for_silence = False
        self._tempo_discarded_for_silence = False
        self._drop_refractory_until = float("-inf")
        self._last_transition_event: str | None = None
        self._last_spectral_beat_state = BeatState(0.0, False, 0, 0.0, 0.0)
        self._last_fallback_beat_state = BeatState(0.0, False, 0, 0.0, 0.0)
        self.last_metrics = AudioInputMetrics.silence(channels=channels)

    @property
    def tempo_diagnostics(self) -> dict[str, object]:
        diagnostics = dict(self._tempo_arbiter.diagnostics)
        diagnostics["spectral"] = (
            self._tempo_tracker.diagnostics
            if self._tempo_tracker is not None
            else {}
        )
        diagnostics["spectral_state"] = {
            "bpm": self._last_spectral_beat_state.bpm,
            "confidence": self._last_spectral_beat_state.confidence,
        }
        diagnostics["fallback_state"] = {
            "bpm": self._last_fallback_beat_state.bpm,
            "confidence": self._last_fallback_beat_state.confidence,
        }
        return diagnostics

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
        self._last_spectral_beat_state = spectral_beat_state
        self._last_fallback_beat_state = fallback_beat_state
        beat_state = self._tempo_arbiter.select(
            spectral=spectral_beat_state,
            fallback=fallback_beat_state,
            now=timestamp,
        )
        if signal_present:
            self._silence_started_at = None
            self._tempo_reset_for_silence = False
            self._tempo_discarded_for_silence = False
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
                # Stop publishing the clock, but preserve its private tempo
                # hypothesis across a musical breakdown. Spotify/seek
                # boundaries reset the whole analyzer separately. This avoids
                # reacquiring the half-time octave when the same song returns.
                self._tempo_arbiter.reset()
                self._tempo_reset_for_silence = True
            if quiet_for >= 30.0 and not self._tempo_discarded_for_silence:
                self._beat_tracker = BeatTracker()
                self._tempo_tracker = SpectralTempoTracker(
                    self._tempo_update_rate
                )
                self._tempo_discarded_for_silence = True
        beat_phase = (beat_state.bar_progress * 4.0) % 1.0
        novelty = clamp(0.65 * onset + 0.35 * abs(high - low), 0.0, 1.0)
        section, section_confidence = self._classify_section(
            timestamp,
            loudness,
            onset,
            low,
            novelty,
            rhythm_density=self._rhythm_density,
            spectral_brightness=spectral_brightness,
            harmonic_change=harmonic_change,
            arrangement_change=self._arrangement_change,
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
            transition_event=self._last_transition_event,
        )

    def _classify_section(
        self,
        timestamp: float,
        loudness: float,
        onset: float,
        low_energy: float,
        novelty: float,
        *,
        rhythm_density: float | None = None,
        spectral_brightness: float | None = None,
        harmonic_change: float = 0.0,
        arrangement_change: float | None = None,
    ) -> tuple[str, float]:
        """Track causal techno energy regions from arrangement trajectories.

        The first implementation calculated rhythm, timbre, harmony, and
        arrangement change but classified sections almost entirely from
        loudness.  This tracker uses independent short/long trajectories and a
        slowly decaying song reference.  It remains causal: no future audio or
        metadata timing is used.
        """
        self._last_transition_event = None
        rhythm = clamp(
            self._rhythm_density if rhythm_density is None else rhythm_density,
            0.0,
            1.0,
        )
        brightness = clamp(
            0.0 if spectral_brightness is None else spectral_brightness,
            0.0,
            1.0,
        )
        arrangement = clamp(
            self._arrangement_change
            if arrangement_change is None
            else arrangement_change,
            0.0,
            1.0,
        )
        if self._section_last_timestamp is None:
            elapsed = 1.0 / 24.0
            self._section_fast_level = loudness
            self._section_slow_level = loudness
            self._section_reference_level = loudness
            self._section_fast_rhythm = self._section_slow_rhythm = rhythm
            self._section_fast_brightness = self._section_slow_brightness = brightness
            self._section_fast_bass = self._section_slow_bass = low_energy
            self._section_started_at = timestamp
        else:
            elapsed = clamp(timestamp - self._section_last_timestamp, 0.005, 0.25)
        self._section_last_timestamp = timestamp
        fast_alpha = 1.0 - math.exp(-elapsed / 0.70)
        slow_alpha = 1.0 - math.exp(-elapsed / 5.5)
        self._section_fast_level += fast_alpha * (loudness - self._section_fast_level)
        self._section_slow_level += slow_alpha * (loudness - self._section_slow_level)
        self._section_fast_rhythm += fast_alpha * (rhythm - self._section_fast_rhythm)
        self._section_slow_rhythm += slow_alpha * (rhythm - self._section_slow_rhythm)
        self._section_fast_brightness += fast_alpha * (
            brightness - self._section_fast_brightness
        )
        self._section_slow_brightness += slow_alpha * (
            brightness - self._section_slow_brightness
        )
        self._section_fast_bass += fast_alpha * (low_energy - self._section_fast_bass)
        self._section_slow_bass += slow_alpha * (low_energy - self._section_slow_bass)
        # Remember meaningful program level for long enough that a sustained
        # breakdown remains visibly soft after the five-second average adapts.
        reference_decay = math.exp(-elapsed / 45.0)
        self._section_reference_level = max(
            loudness,
            self._section_reference_level * reference_decay,
        )
        level_trend = self._section_fast_level - self._section_slow_level
        rhythm_trend = self._section_fast_rhythm - self._section_slow_rhythm
        brightness_trend = (
            self._section_fast_brightness - self._section_slow_brightness
        )
        bass_trend = self._section_fast_bass - self._section_slow_bass
        relative_level = loudness / max(0.08, self._section_reference_level)
        section_age = timestamp - (
            timestamp if self._section_started_at is None else self._section_started_at
        )

        rise_score = clamp(
            0.34 * clamp(level_trend / 0.12, 0.0, 1.0)
            + 0.24 * clamp(rhythm_trend / 0.16, 0.0, 1.0)
            + 0.16 * clamp(brightness_trend / 0.16, 0.0, 1.0)
            + 0.12 * clamp(bass_trend / 0.16, 0.0, 1.0)
            + 0.14 * max(novelty, arrangement),
            0.0,
            1.0,
        )
        withdrawal_score = clamp(
            0.46 * clamp((0.72 - relative_level) / 0.45, 0.0, 1.0)
            + 0.22 * clamp(-rhythm_trend / 0.14, 0.0, 1.0)
            + 0.14 * clamp(-brightness_trend / 0.14, 0.0, 1.0)
            + 0.10 * clamp(-bass_trend / 0.14, 0.0, 1.0)
            + 0.08 * arrangement,
            0.0,
            1.0,
        )
        high_state = bool(
            relative_level >= 0.78
            and loudness >= 0.34
            and (rhythm >= 0.30 or low_energy >= 0.30)
        )
        drop_onset = bool(
            timestamp >= self._drop_refractory_until
            and high_state
            and (onset >= 0.72 or arrangement >= 0.42 or novelty >= 0.58)
            and (
                (self._section == "build" and section_age >= 2.0)
                or (self._section == "breakdown" and relative_level >= 0.88)
            )
        )
        if drop_onset:
            candidate = "drop"
            required = 0.0
        elif withdrawal_score >= 0.46 and relative_level <= 0.72:
            candidate = "breakdown"
            required = 0.65
        elif self._section == "drop" and high_state and withdrawal_score < 0.40:
            # A drop is a sustained state. Residual upward trajectories after
            # the onset must not immediately relabel it as another build.
            candidate = "drop"
            required = 0.0
        elif rise_score >= 0.42 and loudness >= 0.22:
            candidate = "build"
            required = 0.90
        else:
            candidate = "groove"
            required = 1.10

        if candidate != self._section_candidate:
            self._section_candidate = candidate
            self._section_candidate_since = timestamp
        candidate_age = timestamp - (
            timestamp
            if self._section_candidate_since is None
            else self._section_candidate_since
        )
        minimum_hold = 0.8 if self._section == "breakdown" else 1.8
        if (
            candidate != self._section
            and candidate_age >= required
            and (section_age >= minimum_hold or candidate == "drop")
        ):
            previous = self._section
            self._section = candidate
            self._section_started_at = timestamp
            if candidate == "drop":
                self._drop_refractory_until = timestamp + 3.5
            self._last_transition_event = transition_event_for(
                previous, candidate
            ).value
            if previous == candidate:
                self._last_transition_event = None
            self._section_candidate = None
            self._section_candidate_since = None
            section_age = 0.0

        stability = clamp(section_age / 3.0, 0.0, 1.0)
        evidence = {
            "breakdown": withdrawal_score,
            "build": rise_score,
            "drop": clamp(
                0.45 * relative_level + 0.30 * rhythm + 0.25 * low_energy,
                0.0,
                1.0,
            ),
            "groove": clamp(
                0.42 + 0.24 * (1.0 - arrangement) + 0.18 * rhythm,
                0.0,
                1.0,
            ),
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
        self._process_lock = threading.RLock()

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
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        with self._process_lock:
            self._process = process
        assert process.stdout is not None
        chunk_bytes = config.chunk_frames * config.channels * 2
        try:
            while True:
                data = process.stdout.read(chunk_bytes)
                if not data:
                    code = process.poll()
                    if code not in (None, 0):
                        raise RuntimeError(f"arecord stopped with exit code {code}")
                    break
                yield data
        finally:
            self.close()

    def close(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def __enter__(self) -> "AlsaLineIn":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class CapturedPcmChunk:
    """One PCM block located on the authoritative ALSA sample clock."""

    pcm: SourcePcm
    captured_monotonic_s: float
    source_start_frame: int
    frame_count: int
    timestamp_s: float


class ContinuouslyDrainedAudio:
    """Drain an audio source on its own thread and expose an ordered queue.

    The analyzer and HTTP interface may pause without back-pressuring
    ``arecord``. A minute of PCM costs only tens of megabytes on the target
    machine, so ordinary filesystem and database pauses are absorbed instead
    of becoming holes in the authoritative audio timeline. Exceptional
    overflow still collapses stale packets and remains explicit in diagnostics.
    """

    def __init__(
        self,
        source: AlsaLineIn,
        *,
        max_chunks: int | None = None,
        overflow_low_water_chunks: int | None = None,
        buffer_seconds: float = 60.0,
    ) -> None:
        if buffer_seconds <= 0.0:
            raise ValueError("buffer_seconds must be positive")
        if max_chunks is None:
            source_config = getattr(source, "config", AudioCaptureConfig())
            max_chunks = max(
                8,
                round(
                    buffer_seconds
                    * source_config.sample_rate
                    / source_config.chunk_frames
                ),
            )
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        if overflow_low_water_chunks is None:
            overflow_low_water_chunks = min(
                max(1, max_chunks // 4), max_chunks - 1
            )
        if not 0 <= overflow_low_water_chunks < max_chunks:
            raise ValueError(
                "overflow_low_water_chunks must be in [0, max_chunks)"
            )
        self.source = source
        self._queue: queue.Queue[CapturedPcmChunk] = queue.Queue(
            maxsize=max_chunks
        )
        self._stop = threading.Event()
        self._overflow_low_water_chunks = overflow_low_water_chunks
        source_config = getattr(source, "config", AudioCaptureConfig())
        self._buffer_seconds = max_chunks * source_config.chunk_frames / (
            source_config.sample_rate
        )
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._maximum_depth = 0
        self._sample_clock_origin_s: float | None = None
        self._source_frames = 0
        self._packets_read = 0
        self._last_packet_monotonic_s: float | None = None
        self._dropped_packets = 0
        self._dropped_frames = 0
        self._dropped_ranges: list[dict[str, int | str]] = []
        self._state_lock = threading.Lock()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def maximum_queue_depth(self) -> int:
        with self._state_lock:
            return self._maximum_depth

    @property
    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            thread = self._thread
            return {
                "reader_alive": bool(thread is not None and thread.is_alive()),
                "packets_read": self._packets_read,
                "source_frames": self._source_frames,
                "last_packet_monotonic_s": self._last_packet_monotonic_s,
                "last_packet_age_ms": (
                    None
                    if self._last_packet_monotonic_s is None
                    else max(
                        0.0,
                        (now - self._last_packet_monotonic_s) * 1000.0,
                    )
                ),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "buffer_seconds": self._buffer_seconds,
                "overflow_low_water_chunks": self._overflow_low_water_chunks,
                "maximum_queue_depth": self._maximum_depth,
                "dropped_packets": self._dropped_packets,
                "dropped_frames": self._dropped_frames,
                "dropped_ranges": [
                    dict(item) for item in self._dropped_ranges
                ],
                "sample_clock_origin_s": self._sample_clock_origin_s,
                "error": None if self._error is None else str(self._error),
            }

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("audio drain is already started")
        self._thread = threading.Thread(
            target=self._drain,
            name="lumen-alsa-drain",
            daemon=True,
        )
        self._thread.start()

    def chunks(
        self, *, stop_event: threading.Event | None = None
    ) -> Iterator[CapturedPcmChunk]:
        if self._thread is None:
            self.start()
        while True:
            if stop_event is not None and stop_event.is_set():
                self.close()
                return
            try:
                item = self._queue.get(timeout=0.10)
            except queue.Empty:
                if not self._done.is_set():
                    continue
                if self._error is not None and not self._stop.is_set():
                    raise RuntimeError(
                        f"audio capture stopped: {self._error}"
                    ) from self._error
                return
            try:
                yield item
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 3.0) -> None:
        if timeout < 0.0:
            raise ValueError("timeout must not be negative")
        self._stop.set()
        close_source = getattr(self.source, "close", None)
        if callable(close_source):
            close_source()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError(
                "audio capture reader did not stop after its source was closed"
            )

    def _drain(self) -> None:
        try:
            origin: float | None = None
            for pcm in self.source.chunks():
                if self._stop.is_set():
                    break
                bytes_per_frame = self.source.config.channels * 2
                if len(pcm) % bytes_per_frame:
                    raise RuntimeError(
                        "ALSA returned PCM that is not sample-frame aligned"
                    )
                frame_count = len(pcm) // bytes_per_frame
                read_completed_s = time.monotonic()
                if origin is None:
                    # The generator starts arecord lazily. Anchor the first
                    # packet to when it was actually read, not to process
                    # startup time before ALSA opened.
                    origin = (
                        read_completed_s
                        - frame_count / self.source.config.sample_rate
                    )
                    with self._state_lock:
                        self._sample_clock_origin_s = origin
                with self._state_lock:
                    start_frame = self._source_frames
                    self._source_frames += frame_count
                    self._packets_read += 1
                    self._last_packet_monotonic_s = read_completed_s
                timestamp_s = (
                    origin
                    + (start_frame + frame_count / 2.0)
                    / self.source.config.sample_rate
                )
                source_pcm = SourcePcm(
                    pcm,
                    source_start_frame=start_frame,
                    frame_count=frame_count,
                    timestamp_s=timestamp_s,
                )
                item = CapturedPcmChunk(
                    pcm=source_pcm,
                    captured_monotonic_s=time.monotonic(),
                    source_start_frame=start_frame,
                    frame_count=frame_count,
                    timestamp_s=timestamp_s,
                )
                while not self._stop.is_set():
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        # A one-packet eviction leaves a mildly overloaded live
                        # consumer permanently riding at the queue ceiling. Drop
                        # back to a low-water mark in one operation so DMX catches
                        # up to the room instead of remaining a third of a second
                        # behind it. Every discarded source-frame range remains
                        # explicit for diagnostics and the training recorder.
                        self._collapse_overflow_backlog()
                        continue
                    with self._state_lock:
                        self._maximum_depth = max(
                            self._maximum_depth, self._queue.qsize()
                        )
                    break
        except BaseException as error:
            if not self._stop.is_set():
                self._error = error
        finally:
            self._done.set()

    def _collapse_overflow_backlog(self) -> None:
        while self._queue.qsize() > self._overflow_low_water_chunks:
            try:
                dropped = self._queue.get_nowait()
            except queue.Empty:
                return
            self._queue.task_done()
            with self._state_lock:
                self._dropped_packets += 1
                self._dropped_frames += dropped.frame_count
                self._append_dropped_range(dropped)

    def _append_dropped_range(self, item: CapturedPcmChunk) -> None:
        if (
            self._dropped_ranges
            and int(self._dropped_ranges[-1]["start_frame"])
            + int(self._dropped_ranges[-1]["frame_count"])
            == item.source_start_frame
        ):
            self._dropped_ranges[-1]["frame_count"] = (
                int(self._dropped_ranges[-1]["frame_count"])
                + item.frame_count
            )
            return
        self._dropped_ranges.append(
            {
                "start_frame": item.source_start_frame,
                "frame_count": item.frame_count,
                "reason": "capture_queue_overflow",
            }
        )

    def __enter__(self) -> "ContinuouslyDrainedAudio":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
