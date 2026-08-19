"""Isolated audio-to-light timing experiments.

Timing Lab deliberately does not import the expression, choreography, memory,
student, or media subsystems.  It turns physical PCM into a retained bass
clock and a second broad-transient lane, then renders a small deterministic
fixture exercise.  Proved behavior can be promoted into Live later without
making an unfinished detector part of the performance path.
"""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import asdict, dataclass
import math
import statistics
import sys
from typing import Any

from lumen_engine.audio import AudioInputMetrics
from lumen_engine.beat import BeatState, SpectralTempoTracker
from lumen_engine.config import RigConfig
from lumen_engine.dmx import DMXFrame, DMXOutput, apply_moving_head_solution
from lumen_engine.fixture_output import rgb_to_rgbw
from lumen_engine.models import (
    ExpressionState,
    Gesture,
    MusicalObservation,
    PerformanceDecision,
    Vec3,
    clamp,
)
from lumen_engine.profiles import party_parrot_profile
from lumen_engine.runtime import RuntimeFrame
from lumen_engine.spatial import TargetingSolution


TIMING_LAB_PULSE_SOURCES = frozenset(
    {"bass_clock", "broadband_onset", "off"}
)


@dataclass(slots=True)
class TimingLabControls:
    """Ephemeral controls used only by the isolated Timing Lab engine mode."""

    output: str = "virtual"
    movers_source: str = "bass_clock"
    center_source: str = "broadband_onset"
    base_intensity: float = 0.04
    flash_intensity: float = 0.92
    pulse_ms: int = 105
    color_bars: int = 4
    movers_motion_period_s: float = 9.0
    center_motion_period_s: float = 11.0

    def patch(self, values: dict[str, Any]) -> None:
        if "output" in values:
            output = str(values["output"]).strip().casefold()
            if output not in {"virtual", "live"}:
                raise ValueError("Timing Lab output must be virtual or live")
            self.output = output
        for name in ("movers_source", "center_source"):
            if name in values:
                source = str(values[name]).strip().casefold()
                if source not in TIMING_LAB_PULSE_SOURCES:
                    raise ValueError(
                        "Timing Lab pulse source must be bass_clock, "
                        "broadband_onset, or off"
                    )
                setattr(self, name, source)
        if "base_intensity" in values:
            self.base_intensity = clamp(
                float(values["base_intensity"]), 0.0, 0.35
            )
        if "flash_intensity" in values:
            self.flash_intensity = clamp(
                float(values["flash_intensity"]), 0.05, 1.0
            )
        if self.flash_intensity < self.base_intensity:
            self.flash_intensity = self.base_intensity
        if "pulse_ms" in values:
            self.pulse_ms = round(
                clamp(float(values["pulse_ms"]), 45.0, 250.0)
            )
        if "color_bars" in values:
            bars = int(values["color_bars"])
            if bars not in {2, 4, 6, 8, 16}:
                raise ValueError("Timing Lab color bars must be 2, 4, 6, 8, or 16")
            self.color_bars = bars
        for name in ("movers_motion_period_s", "center_motion_period_s"):
            if name in values:
                setattr(
                    self,
                    name,
                    clamp(float(values[name]), 4.0, 30.0),
                )


@dataclass(frozen=True, slots=True)
class TimingLabAnalysis:
    timestamp_s: float
    metrics: AudioInputMetrics
    bass_level: float
    bass_onset: float
    bass_threshold: float
    broadband_onset: float
    broadband_threshold: float
    bass_transient: bool
    broadband_transient: bool
    beat_event: bool
    predicted_beat: bool
    bpm: float | None
    confidence: float
    beat_phase: float
    bar_phase: float
    beat_count: int
    clock_state: str
    candidate_bpm: float | None
    rejected_candidate_bpm: float | None
    last_bass_age_s: float | None
    family_anchor_bpm: float | None
    alternate_candidate_bpm: float | None
    alternate_evidence_s: float
    tempo_switch_count: int
    last_pulse_interval_s: float | None
    raw_candidate_bpm: float | None
    candidate_harmonic_factor: float
    bass_interval_bpm: float | None
    phase_error_ms: float | None
    phase_error_rms_ms: float | None
    phase_correction_count: int
    rejected_phase_error_ms: float | None
    phase_rejection_count: int

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("metrics", None)
        payload["signal_dbfs"] = self.metrics.dbfs
        return payload


class RetainedBassClock:
    """Hold a proven tempo family through sparse sections.

    The underlying spectral estimator is intentionally fed only a bass-band
    onset envelope.  Once several seconds agree on one tempo, a contradictory
    half/double-time candidate is displayed but cannot silently replace the
    clock. The spectral bass-onset tracker establishes phase; the retained
    layer follows that one audio-derived grid instead of treating every
    low-band transient as another clock. Prediction fills an occasional
    missed kick but is gated off when bass activity disappears.
    """

    def __init__(self) -> None:
        self.bpm: float | None = None
        self.confidence = 0.0
        self.origin_s: float | None = None
        self.beat_count = 0
        self._family_anchor_bpm: float | None = None
        self._candidate_history: deque[tuple[float, float, float]] = deque()
        self._alternate_history: deque[tuple[float, float, float]] = deque()
        self._last_candidate_sample_s = float("-inf")
        self._last_retune_s: float | None = None
        self._last_emitted_grid: int | None = None
        self._last_pulse_s: float | None = None
        self._last_pulse_interval_s: float | None = None
        self._last_bass_s: float | None = None
        self._bass_signal_active = False
        self._rejected_candidate_bpm: float | None = None
        self._tempo_switch_count = 0
        self._last_phase_error_s: float | None = None
        self._phase_errors_s: deque[float] = deque(maxlen=64)
        self._phase_correction_count = 0
        self._last_rejected_phase_error_s: float | None = None
        self._phase_rejection_count = 0

    def update(
        self,
        candidate: BeatState,
        *,
        bass_transient: bool,
        bass_level: float,
        timestamp_s: float,
        allow_family_switch: bool = False,
    ) -> dict[str, Any]:
        if bass_transient:
            self._last_bass_s = timestamp_s
        self._bass_signal_active = bass_level >= 0.003
        candidate_bpm = candidate.bpm if candidate.bpm > 0.0 else None
        sampled_candidate = bool(
            candidate_bpm is not None
            and candidate.confidence >= 0.35
            and timestamp_s - self._last_candidate_sample_s >= 0.20
        )
        if sampled_candidate:
            assert candidate_bpm is not None
            self._candidate_history.append(
                (timestamp_s, candidate_bpm, candidate.confidence)
            )
            self._last_candidate_sample_s = timestamp_s
        cutoff = timestamp_s - 7.0
        while self._candidate_history and self._candidate_history[0][0] < cutoff:
            self._candidate_history.popleft()
        while self._alternate_history and self._alternate_history[0][0] < cutoff:
            self._alternate_history.popleft()

        if self.bpm is None:
            self._try_acquire(timestamp_s, candidate)
        elif candidate_bpm is not None and candidate.confidence >= 0.35:
            family_anchor = self._family_anchor_bpm or self.bpm
            difference = abs(candidate_bpm - family_anchor) / family_anchor
            if difference <= 0.055:
                # Stay inside the acquired family. Retuning preserves the
                # continuous grid coordinate, so last-emitted indices never
                # become stranded ahead of a newly scaled clock.
                self._alternate_history.clear()
                self._retune_preserving_phase(candidate_bpm, timestamp_s)
                if sampled_candidate:
                    self._follow_candidate_phase(candidate, timestamp_s)
                self.confidence = clamp(
                    0.97 * self.confidence + 0.03 * candidate.confidence,
                    0.0,
                    1.0,
                )
                self._rejected_candidate_bpm = None
            else:
                self._rejected_candidate_bpm = candidate_bpm
                if sampled_candidate:
                    self._alternate_history.append(
                        (timestamp_s, candidate_bpm, candidate.confidence)
                    )
                if allow_family_switch:
                    self._try_switch_family(timestamp_s, candidate)

        beat_event = False
        predicted = False
        phase = 0.0
        bar_phase = 0.0
        if self.bpm is None or self.origin_s is None:
            # Raw onset detections are evidence used to acquire a clock, not
            # lighting commands.  Flashing them made the first few seconds of
            # every recording look random and could emit several pulses per
            # musical beat before a tempo had been proved.
            return self._state(
                beat_event=False,
                predicted=False,
                beat_phase=phase,
                bar_phase=bar_phase,
                candidate_bpm=candidate_bpm,
                timestamp_s=timestamp_s,
            )

        period = 60.0 / self.bpm
        position = (timestamp_s - self.origin_s) / period
        grid_index = math.floor(position + 0.04)
        bass_recent = (
            self._last_bass_s is not None
            and (
                timestamp_s - self._last_bass_s <= 0.80
                or self._bass_signal_active
            )
        )
        if (
            bass_recent
            and self._last_emitted_grid is not None
            and grid_index > self._last_emitted_grid
        ):
            beat_event = self._record_pulse(timestamp_s, period)
            if beat_event:
                predicted = not bass_transient
                self._last_emitted_grid = grid_index
        elif self._last_emitted_grid is None:
            self._last_emitted_grid = grid_index

        exact_position = (timestamp_s - self.origin_s) / period
        phase = exact_position % 1.0
        bar_phase = exact_position % 4.0 / 4.0
        return self._state(
            beat_event=beat_event,
            predicted=predicted,
            beat_phase=phase,
            bar_phase=bar_phase,
            candidate_bpm=candidate_bpm,
            timestamp_s=timestamp_s,
        )

    def _try_acquire(self, now: float, candidate: BeatState) -> None:
        if len(self._candidate_history) < 12:
            return
        duration = self._candidate_history[-1][0] - self._candidate_history[0][0]
        if duration < 3.0:
            return
        bpms = [item[1] for item in self._candidate_history]
        median_bpm = statistics.median(bpms)
        relative_spread = statistics.median(
            abs(value - median_bpm) for value in bpms
        ) / max(median_bpm, 1e-6)
        mean_confidence = statistics.fmean(item[2] for item in self._candidate_history)
        if relative_spread > 0.025 or mean_confidence < 0.42:
            return
        self.bpm = median_bpm
        self._family_anchor_bpm = median_bpm
        self.confidence = clamp(mean_confidence, 0.0, 1.0)
        # Bar-counted color changes begin only after a stable grid exists;
        # raw acquisition transients are not bars.
        self.beat_count = 0
        period = 60.0 / self.bpm
        beat_phase = (candidate.bar_progress * 4.0) % 1.0
        self.origin_s = now - beat_phase * period
        self._last_emitted_grid = math.floor(
            (now - self.origin_s) / period
        )
        self._last_retune_s = now
        self._rejected_candidate_bpm = None

    def _retune_preserving_phase(self, target_bpm: float, now: float) -> None:
        if self.bpm is None or self.origin_s is None:
            return
        elapsed = (
            0.20
            if self._last_retune_s is None
            else clamp(now - self._last_retune_s, 0.0, 1.0)
        )
        # At most 0.35% per second. The former 0.2%-per-audio-packet rule
        # could compound by several percent in one second.
        maximum_change = self.bpm * 0.0035 * elapsed
        new_bpm = self.bpm + clamp(
            target_bpm - self.bpm,
            -maximum_change,
            maximum_change,
        )
        old_period = 60.0 / self.bpm
        grid_position = (now - self.origin_s) / old_period
        new_period = 60.0 / new_bpm
        self.origin_s = now - grid_position * new_period
        self.bpm = new_bpm
        self._last_retune_s = now

    def _follow_candidate_phase(self, candidate: BeatState, now: float) -> None:
        """Gently follow the one spectral PCM grid without a second PLL."""

        if self.bpm is None or self.origin_s is None:
            return
        period = 60.0 / self.bpm
        retained_phase = ((now - self.origin_s) / period) % 1.0
        candidate_phase = (candidate.bar_progress * 4.0) % 1.0
        phase_error_beats = (
            (candidate_phase - retained_phase + 0.5) % 1.0 - 0.5
        )
        phase_error_s = phase_error_beats * period
        if abs(phase_error_beats) > 0.16:
            # Spectral grids can briefly choose the other half-beat in
            # syncopated material.  The established PCM clock must not chase
            # that metrical flip; a real recording boundary resets the clock.
            self._last_rejected_phase_error_s = phase_error_s
            self._phase_rejection_count += 1
            return
        self._last_phase_error_s = phase_error_s
        self._phase_errors_s.append(phase_error_s)
        self._phase_correction_count += 1
        # At the 5 Hz candidate-sampling rate this settles quickly while one
        # estimator update can move the output grid by at most about 12 ms.
        correction_beats = clamp(phase_error_beats * 0.35, -0.025, 0.025)
        self.origin_s -= correction_beats * period

    def _try_switch_family(self, now: float, candidate: BeatState) -> None:
        if len(self._alternate_history) < 20:
            return
        duration = self._alternate_history[-1][0] - self._alternate_history[0][0]
        if duration < 4.5:
            return
        bpms = [item[1] for item in self._alternate_history]
        median_bpm = statistics.median(bpms)
        relative_spread = statistics.median(
            abs(value - median_bpm) for value in bpms
        ) / max(median_bpm, 1e-6)
        mean_confidence = statistics.fmean(item[2] for item in self._alternate_history)
        if relative_spread > 0.025 or mean_confidence < 0.42:
            return
        self.bpm = median_bpm
        self._family_anchor_bpm = median_bpm
        self.confidence = clamp(mean_confidence, 0.0, 1.0)
        period = 60.0 / median_bpm
        beat_phase = (candidate.bar_progress * 4.0) % 1.0
        self.origin_s = now - beat_phase * period
        self._last_emitted_grid = math.floor(
            (now - self.origin_s) / period
        )
        self._last_retune_s = now
        self._alternate_history.clear()
        self._candidate_history.clear()
        self._rejected_candidate_bpm = None
        self._tempo_switch_count += 1

    def _record_pulse(self, now: float, period: float) -> bool:
        if (
            self._last_pulse_s is not None
            and now - self._last_pulse_s < period * 0.72
        ):
            return False
        if self._last_pulse_s is not None:
            self._last_pulse_interval_s = now - self._last_pulse_s
        self._last_pulse_s = now
        self.beat_count += 1
        return True

    def _state(
        self,
        *,
        beat_event: bool,
        predicted: bool,
        beat_phase: float,
        bar_phase: float,
        candidate_bpm: float | None,
        timestamp_s: float,
    ) -> dict[str, Any]:
        bass_age = (
            None
            if self._last_bass_s is None
            else max(0.0, timestamp_s - self._last_bass_s)
        )
        clock_state = "acquiring"
        if self.bpm is not None:
            clock_state = (
                "held"
                if (
                    bass_age is None
                    or (bass_age > 0.80 and not self._bass_signal_active)
                )
                else "locked"
            )
        return {
            "beat_event": beat_event,
            "predicted_beat": predicted,
            "bpm": self.bpm,
            "confidence": self.confidence,
            "beat_phase": beat_phase,
            "bar_phase": bar_phase,
            "beat_count": self.beat_count,
            "clock_state": clock_state,
            "candidate_bpm": candidate_bpm,
            "rejected_candidate_bpm": self._rejected_candidate_bpm,
            "last_bass_age_s": bass_age,
            "family_anchor_bpm": self._family_anchor_bpm,
            "alternate_candidate_bpm": (
                None
                if not self._alternate_history
                else statistics.median(
                    item[1] for item in self._alternate_history
                )
            ),
            "alternate_evidence_s": (
                0.0
                if len(self._alternate_history) < 2
                else self._alternate_history[-1][0]
                - self._alternate_history[0][0]
            ),
            "tempo_switch_count": self._tempo_switch_count,
            "last_pulse_interval_s": self._last_pulse_interval_s,
            "phase_error_ms": (
                None
                if self._last_phase_error_s is None
                else self._last_phase_error_s * 1000.0
            ),
            "phase_error_rms_ms": (
                None
                if not self._phase_errors_s
                else math.sqrt(
                    statistics.fmean(
                        error * error for error in self._phase_errors_s
                    )
                )
                * 1000.0
            ),
            "phase_correction_count": self._phase_correction_count,
            "rejected_phase_error_ms": (
                None
                if self._last_rejected_phase_error_s is None
                else self._last_rejected_phase_error_s * 1000.0
            ),
            "phase_rejection_count": self._phase_rejection_count,
        }


class TimingLabAnalyzer:
    """Extract independent bass and broadband events from authoritative PCM."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2) -> None:
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample rate and channel count must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self._low_pass_40 = 0.0
        self._low_pass_180 = 0.0
        self._alpha_40 = 1.0 - math.exp(-2.0 * math.pi * 40.0 / sample_rate)
        self._alpha_180 = 1.0 - math.exp(-2.0 * math.pi * 180.0 / sample_rate)
        self._bass_baseline: float | None = None
        self._broad_baseline: float | None = None
        self._bass_history: deque[float] = deque(maxlen=420)
        self._broad_history: deque[float] = deque(maxlen=420)
        self._previous_bass_onset = 0.0
        self._previous_broad_onset = 0.0
        self._last_bass_event_s = float("-inf")
        self._last_broad_event_s = float("-inf")
        self._bass_transient_times: deque[float] = deque(maxlen=96)
        self._tempo = SpectralTempoTracker(sample_rate / 1024.0)
        self._clock = RetainedBassClock()

    def analyze_pcm16(
        self, pcm: bytes, *, timestamp_s: float
    ) -> TimingLabAnalysis:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if len(samples) % self.channels:
            raise ValueError("PCM16 input is not aligned to complete sample frames")
        mono = [
            sum(samples[offset : offset + self.channels])
            / self.channels
            / 32768.0
            for offset in range(0, len(samples), self.channels)
        ]
        metrics = self._metrics(samples, mono, timestamp_s)
        bass_energy = 0.0
        for value in mono:
            self._low_pass_40 += self._alpha_40 * (
                value - self._low_pass_40
            )
            self._low_pass_180 += self._alpha_180 * (
                value - self._low_pass_180
            )
            band = self._low_pass_180 - self._low_pass_40
            bass_energy += band * band
        bass_level = math.sqrt(bass_energy / max(1, len(mono)))
        elapsed = len(mono) / self.sample_rate
        bass_onset, self._bass_baseline = self._onset(
            bass_level, self._bass_baseline, elapsed, scale=80.0
        )
        broad_onset, self._broad_baseline = self._onset(
            metrics.rms, self._broad_baseline, elapsed, scale=30.0
        )
        bass_threshold = self._threshold(self._bass_history)
        broad_threshold = self._threshold(self._broad_history)
        bass_share = bass_level / max(metrics.rms, 1e-6)
        bass_transient = bool(
            bass_level >= 0.001
            and bass_share >= 0.12
            and bass_onset >= bass_threshold
            and self._previous_bass_onset < bass_threshold * 0.72
            and timestamp_s - self._last_bass_event_s >= 0.18
        )
        broadband_transient = bool(
            metrics.rms >= 0.002
            and broad_onset >= broad_threshold
            and self._previous_broad_onset < broad_threshold * 0.72
            and timestamp_s - self._last_broad_event_s >= 0.12
        )
        if bass_transient:
            self._last_bass_event_s = timestamp_s
            self._bass_transient_times.append(timestamp_s)
        if broadband_transient:
            self._last_broad_event_s = timestamp_s
        self._bass_history.append(bass_onset)
        self._broad_history.append(broad_onset)
        self._previous_bass_onset = bass_onset
        self._previous_broad_onset = broad_onset

        raw_candidate = self._tempo.update(
            clamp(bass_onset, 0.0, 1.0) if bass_share >= 0.12 else 0.0,
            timestamp_s,
        )
        candidate, harmonic_factor, interval_bpm = self._normalize_candidate(
            raw_candidate, timestamp_s
        )
        clock = self._clock.update(
            candidate,
            bass_transient=bass_transient,
            bass_level=bass_level,
            timestamp_s=timestamp_s,
        )
        return TimingLabAnalysis(
            timestamp_s=timestamp_s,
            metrics=metrics,
            bass_level=bass_level,
            bass_onset=bass_onset,
            bass_threshold=bass_threshold,
            broadband_onset=broad_onset,
            broadband_threshold=broad_threshold,
            bass_transient=bass_transient,
            broadband_transient=broadband_transient,
            raw_candidate_bpm=(
                raw_candidate.bpm if raw_candidate.bpm > 0.0 else None
            ),
            candidate_harmonic_factor=harmonic_factor,
            bass_interval_bpm=interval_bpm,
            **clock,
        )

    def _normalize_candidate(
        self, candidate: BeatState, now: float
    ) -> tuple[BeatState, float, float | None]:
        """Resolve a half/double-time candidate from physical bass intervals."""

        cutoff = now - 8.0
        while self._bass_transient_times and self._bass_transient_times[0] < cutoff:
            self._bass_transient_times.popleft()
        intervals = [
            right - left
            for left, right in zip(
                self._bass_transient_times,
                list(self._bass_transient_times)[1:],
            )
            if 0.28 <= right - left <= 1.25
        ]
        if candidate.bpm <= 0.0 or len(intervals) < 6:
            return candidate, 1.0, None
        options = [
            (factor, candidate.bpm * factor)
            for factor in (0.5, 1.0, 2.0)
            if 72.0 <= candidate.bpm * factor <= 200.0
        ]

        def support(option_bpm: float) -> float:
            period = 60.0 / option_bpm
            total = 0.0
            for interval in intervals:
                direct_error = abs(interval - period) / period
                skipped_error = abs(interval - 2.0 * period) / (2.0 * period)
                direct = max(0.0, 1.0 - direct_error / 0.14)
                skipped = 0.35 * max(0.0, 1.0 - skipped_error / 0.10)
                total += max(direct, skipped)
            return total

        scores = [(support(bpm), factor, bpm) for factor, bpm in options]
        best_score, factor, normalized_bpm = max(
            scores,
            key=lambda item: (item[0], -abs(math.log2(item[1]))),
        )
        if best_score < max(3.0, len(intervals) * 0.34):
            return candidate, 1.0, None
        period = 60.0 / normalized_bpm
        matching = [
            interval
            for interval in intervals
            if min(
                abs(interval - period) / period,
                abs(interval - 2.0 * period) / (2.0 * period),
            )
            <= 0.14
        ]
        interval_bpm = (
            None
            if not matching
            else 60.0 / statistics.median(
                interval if interval <= period * 1.4 else interval / 2.0
                for interval in matching
            )
        )
        if factor == 1.0:
            return candidate, factor, interval_bpm
        raw_bar_position = candidate.bar_progress * 4.0
        normalized_bar_progress = (raw_bar_position * factor) % 4.0 / 4.0
        return (
            BeatState(
                bpm=normalized_bpm,
                beat=candidate.beat,
                beat_count=candidate.beat_count,
                bar_progress=normalized_bar_progress,
                confidence=candidate.confidence,
            ),
            factor,
            interval_bpm,
        )

    @staticmethod
    def _onset(
        level: float,
        baseline: float | None,
        elapsed: float,
        *,
        scale: float,
    ) -> tuple[float, float]:
        value = math.log1p(max(0.0, level) * scale)
        if baseline is None:
            return 0.0, value
        onset = clamp((value - baseline) * 3.5, 0.0, 1.0)
        alpha = 1.0 - math.exp(-max(0.001, elapsed) / 0.65)
        return onset, baseline + alpha * (value - baseline)

    @staticmethod
    def _threshold(history: deque[float]) -> float:
        if len(history) < 12:
            return 0.12
        median = statistics.median(history)
        deviation = statistics.median(
            abs(value - median) for value in history
        )
        return clamp(median + 3.0 * max(0.012, deviation), 0.10, 0.72)

    def _metrics(
        self,
        samples: array[int],
        mono: list[float],
        timestamp_s: float,
    ) -> AudioInputMetrics:
        frame_count = len(mono)
        rms = math.sqrt(sum(value * value for value in mono) / max(1, frame_count))
        peak = max((abs(value) for value in mono), default=0.0)
        channel_rms: list[float] = []
        channel_peak: list[float] = []
        for channel in range(self.channels):
            values = samples[channel :: self.channels]
            normalized = [value / 32768.0 for value in values]
            channel_rms.append(
                math.sqrt(
                    sum(value * value for value in normalized)
                    / max(1, len(normalized))
                )
            )
            channel_peak.append(
                max((abs(value) for value in normalized), default=0.0)
            )
        waveform = tuple(
            mono[
                min(
                    frame_count - 1,
                    round(index * (frame_count - 1) / 127),
                )
            ]
            if frame_count
            else 0.0
            for index in range(128)
        )
        return AudioInputMetrics(
            timestamp_s=timestamp_s,
            frame_count=frame_count,
            rms=rms,
            dbfs=20.0 * math.log10(max(rms, 1e-6)),
            peak=peak,
            channel_rms=tuple(channel_rms),
            channel_peak=tuple(channel_peak),
            clipped_samples=sum(abs(value) >= 32767 for value in samples),
            waveform=waveform,
        )


class TimingLabRuntime:
    """Render the timing experiment without invoking the performance runtime."""

    def __init__(self, rig: RigConfig, output: DMXOutput) -> None:
        self.rig = rig
        self.output = output
        self._started_s: float | None = None
        self._movers_flash_until_s = float("-inf")
        self._center_flash_until_s = float("-inf")
        self._last_mover_angles: dict[str, tuple[float, float]] = {}
        self._snapshot: dict[str, Any] = {
            "state": "waiting",
            "invariants": self.invariants(),
            "lanes": {},
        }

    @staticmethod
    def invariants() -> dict[str, Any]:
        return {
            "internal_strobe_dmx": 0,
            "beat_authority": "physical_audio_sample_clock",
            "metadata_timing": False,
            "learning_writes": False,
            "song_memory_reads": False,
            "performance_runtime": False,
        }

    def step(
        self,
        analysis: TimingLabAnalysis,
        controls: TimingLabControls,
    ) -> tuple[MusicalObservation, RuntimeFrame]:
        if self._started_s is None:
            self._started_s = analysis.timestamp_s
        elapsed = max(0.0, analysis.timestamp_s - self._started_s)
        pulse_seconds = controls.pulse_ms / 1000.0
        if self._source_event(controls.movers_source, analysis):
            self._movers_flash_until_s = analysis.timestamp_s + pulse_seconds
        if self._source_event(controls.center_source, analysis):
            self._center_flash_until_s = analysis.timestamp_s + pulse_seconds
        movers_flash = analysis.timestamp_s <= self._movers_flash_until_s
        center_flash = analysis.timestamp_s <= self._center_flash_until_s
        movers_brightness = (
            controls.flash_intensity if movers_flash else controls.base_intensity
        )
        center_brightness = (
            controls.flash_intensity if center_flash else controls.base_intensity
        )
        normalized_level = clamp(analysis.metrics.rms * 9.0, 0.0, 1.0)
        observation = MusicalObservation(
            timestamp_s=analysis.timestamp_s,
            loudness=normalized_level,
            onset_strength=analysis.broadband_onset,
            low_energy=clamp(analysis.bass_level * 18.0, 0.0, 1.0),
            mid_energy=normalized_level,
            high_energy=clamp(analysis.broadband_onset, 0.0, 1.0),
            beat_phase=analysis.beat_phase,
            bar_phase=analysis.bar_phase,
            beat_pulse=1.0 if analysis.beat_event else 0.0,
            beat_confidence=analysis.confidence,
            bpm=analysis.bpm,
            section="timing_lab",
            section_confidence=1.0,
            novelty=analysis.broadband_onset,
        )
        color_index = analysis.beat_count // max(1, controls.color_bars * 4)
        colors = ((0.03, 0.10, 1.0), (1.0, 0.0, 0.34))
        mover_rgb = colors[color_index % 2]
        center_rgb = colors[(color_index + 1) % 2]
        decision = PerformanceDecision(
            timestamp_s=analysis.timestamp_s,
            gesture=Gesture.HOLD,
            expression=ExpressionState(
                energy=normalized_level,
                tension=analysis.broadband_onset,
                motion=0.32,
                intimacy=0.5,
                confidence=max(analysis.confidence, 0.25),
            ),
            target=Vec3(0.0, 0.0, min(1.3, self.rig.room.height_m * 0.45)),
            brightness=max(movers_brightness, center_brightness),
            reason=(
                "Isolated Timing Lab: bass-clock mover flashes and independent "
                "broad-transient center flashes; motion is free-running"
            ),
            confidence=max(analysis.confidence, 0.25),
            palette_hint="solid:001aff",
            routine="timing_lab",
        )
        frame = DMXFrame()
        solutions = self._render_movers(
            frame,
            elapsed=elapsed,
            period_s=controls.movers_motion_period_s,
            brightness=movers_brightness,
            rgb=mover_rgb,
            decision=decision,
        )
        self._render_center(
            frame,
            elapsed=elapsed,
            period_s=controls.center_motion_period_s,
            brightness=center_brightness,
            rgb=center_rgb,
        )
        self._assert_internal_strobes_off(frame)
        self.output.send(frame)
        self._snapshot = {
            "state": "running",
            "analysis": analysis.snapshot(),
            "lanes": {
                "movers": {
                    "pulse_source": controls.movers_source,
                    "flash_active": movers_flash,
                    "brightness": movers_brightness,
                    "motion_clock": "independent_free_running",
                    "motion_period_s": controls.movers_motion_period_s,
                },
                "center": {
                    "pulse_source": controls.center_source,
                    "flash_active": center_flash,
                    "brightness": center_brightness,
                    "motion_clock": "independent_free_running",
                    "motion_period_s": controls.center_motion_period_s,
                },
            },
            "color": {
                "changes_every_bars": controls.color_bars,
                "index": color_index,
            },
            "invariants": self.invariants(),
        }
        return observation, RuntimeFrame(
            decision=decision,
            solutions=tuple(solutions),
            dmx=frame,
            warnings=(),
        )

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot

    @staticmethod
    def _source_event(source: str, analysis: TimingLabAnalysis) -> bool:
        if source == "bass_clock":
            return analysis.beat_event
        if source == "broadband_onset":
            return analysis.broadband_transient
        return False

    def _render_movers(
        self,
        frame: DMXFrame,
        *,
        elapsed: float,
        period_s: float,
        brightness: float,
        rgb: tuple[float, float, float],
        decision: PerformanceDecision,
    ) -> list[TargetingSolution]:
        solutions: list[TargetingSolution] = []
        count = max(1, len(self.rig.fixtures))
        for index, fixture in enumerate(self.rig.fixtures):
            phase = elapsed / period_s * math.tau + index * math.tau / count
            calibration = fixture.calibration
            pan_mid = (calibration.pan_min_deg + calibration.pan_max_deg) / 2.0
            tilt_mid = (calibration.tilt_min_deg + calibration.tilt_max_deg) / 2.0
            pan = pan_mid + (
                calibration.pan_max_deg - calibration.pan_min_deg
            ) * 0.18 * math.sin(phase)
            tilt = tilt_mid + (
                calibration.tilt_max_deg - calibration.tilt_min_deg
            ) * 0.12 * math.sin(phase * 2.0 + math.pi / 3.0)
            previous = self._last_mover_angles.get(fixture.fixture_id)
            movement_cost = 0.0 if previous is None else abs(pan - previous[0]) + abs(tilt - previous[1])
            solution = TargetingSolution(
                fixture_id=fixture.fixture_id,
                target=decision.target,
                pan_deg=pan,
                tilt_deg=tilt,
                distance_m=fixture.position_m.distance_to(decision.target),
                movement_cost_deg=movement_cost,
                aim_error_deg=0.0,
                branch="timing-lab/calibrated-envelope",
            )
            self._last_mover_angles[fixture.fixture_id] = (pan, tilt)
            apply_moving_head_solution(frame, fixture, solution, brightness)
            profile = party_parrot_profile(fixture.profile_key)
            if profile is not None:
                channels = profile.channels
                self._set_relative(frame, fixture, channels.get("movement_speed"), 190)
                self._set_relative(frame, fixture, channels.get("dimmer"), round(brightness * 255))
                self._set_relative(frame, fixture, channels.get("strobe"), 0)
                for name, value in zip(
                    ("red", "green", "blue", "white"), rgb_to_rgbw(rgb)
                ):
                    self._set_relative(frame, fixture, channels.get(name), value)
            solutions.append(solution)
        return solutions

    def _render_center(
        self,
        frame: DMXFrame,
        *,
        elapsed: float,
        period_s: float,
        brightness: float,
        rgb: tuple[float, float, float],
    ) -> None:
        for fixture in self.rig.auxiliary_fixtures:
            profile = party_parrot_profile(fixture.profile_key)
            if profile is None or fixture.profile_key != "generic_multi_effect_19ch":
                continue
            channels = profile.channels
            phase = elapsed / period_s * math.tau
            values = {
                "body_rotation": round(128.0 + 42.0 * math.sin(phase)),
                "body_rotation_speed": 220,
                "arm_1_motor": round(128.0 + 58.0 * math.sin(phase * 1.35)),
                "arm_2_motor": round(128.0 - 58.0 * math.cos(phase * 1.10)),
                "master_dimmer": round(brightness * 255),
                "strobe": 0,
                "red_laser": 0,
                "green_laser": 0,
                "strip_program": 0,
                "strip_speed": 0,
                "macro": 0,
            }
            rgbw = rgb_to_rgbw(rgb)
            for prefix in ("magic_ball", "arm_beams"):
                for name, value in zip(("red", "green", "blue", "white"), rgbw):
                    values[f"{prefix}_{name}"] = value
            for name, value in values.items():
                self._set_relative(frame, fixture, channels.get(name), value)

    def _assert_internal_strobes_off(self, frame: DMXFrame) -> None:
        for fixture in (*self.rig.fixtures, *self.rig.auxiliary_fixtures):
            profile = party_parrot_profile(fixture.profile_key)
            relative = None if profile is None else profile.channels.get("strobe")
            if relative is None:
                continue
            channel = fixture.address + relative - 1
            if frame.get_channel(fixture.universe, channel) != 0:
                raise RuntimeError(
                    f"Timing Lab invariant failed: {fixture.fixture_id} internal strobe is not zero"
                )

    @staticmethod
    def _set_relative(
        frame: DMXFrame,
        fixture: Any,
        relative: int | None,
        value: int,
    ) -> None:
        if relative is None:
            return
        frame.set_channel(
            fixture.universe,
            fixture.address + relative - 1,
            round(clamp(float(value), 0.0, 255.0)),
        )
