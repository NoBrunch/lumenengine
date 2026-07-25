"""Transient and spectrum-onset tempo tracking for musical synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time

import numpy as np

from lumen_engine.models import clamp


@dataclass(frozen=True, slots=True)
class BeatState:
    bpm: float
    beat: bool
    beat_count: int
    bar_progress: float
    confidence: float = 0.0


class BeatTracker:
    """Track low-frequency energy spikes with tempo smoothing and re-locking."""

    def __init__(
        self,
        min_bpm: float = 60.0,
        max_bpm: float = 160.0,
        default_bpm: float = 120.0,
        history_seconds: float = 6.0,
        interval_history_seconds: float = 24.0,
        smoothing_beats: float = 16.0,
        retune_ratio: float = 0.06,
        retune_beats: int = 8,
    ) -> None:
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.default_bpm = default_bpm
        self.history_seconds = history_seconds
        self.interval_history_seconds = interval_history_seconds
        self.smoothing_beats = smoothing_beats
        self.retune_ratio = retune_ratio
        self.retune_beats = retune_beats
        self._min_interval = 60.0 / max_bpm
        self._max_interval = 60.0 / min_bpm
        self._refractory_seconds = self._min_interval * 0.72
        self._energy_history: list[tuple[float, float]] = []
        self._interval_history: list[tuple[float, float]] = []
        self._pending_intervals: list[tuple[float, float]] = []
        self._last_energy: float | None = None
        self._last_beat_time: float | None = None
        self._beat_count = -1
        self._bpm = 0.0

    def update(self, low_energy: float, now: float | None = None) -> BeatState:
        now = time.perf_counter() if now is None else now
        energy = clamp(float(low_energy), 0.0, 1.0)
        threshold = self._current_threshold()
        beat = False
        previous = self._last_energy
        if previous is not None:
            crossed_threshold = previous < threshold <= energy
            sharp_rise = energy >= threshold and energy - previous >= 0.10
            enough_time_elapsed = (
                self._last_beat_time is None
                or now - self._last_beat_time >= self._refractory_seconds
            )
            beat = enough_time_elapsed and (crossed_threshold or sharp_rise)
        self._remember_energy(now, energy)
        self._last_energy = energy
        if beat:
            self._record_beat(now)
        return BeatState(
            bpm=self._bpm,
            beat=beat,
            beat_count=max(self._beat_count, 0),
            bar_progress=self._bar_progress(now),
            confidence=self._tempo_confidence(),
        )

    def _tempo_confidence(self) -> float:
        if self._bpm <= 0.0 or len(self._interval_history) < 3:
            return 0.0
        period = 60.0 / self._bpm
        folded = [
            self._fold_to_period(interval, period)
            for _, interval in self._interval_history
        ]
        relative_error = statistics.median(
            abs(interval - period) for interval in folded
        ) / max(period, 1e-6)
        consistency = clamp(1.0 - relative_error / 0.12, 0.0, 1.0)
        evidence = clamp(len(folded) / 12.0, 0.0, 1.0)
        pending_penalty = clamp(
            len(self._pending_intervals) / self.retune_beats, 0.0, 1.0
        )
        return clamp(
            consistency * evidence * (1.0 - 0.7 * pending_penalty), 0.0, 1.0
        )

    def _remember_energy(self, now: float, energy: float) -> None:
        self._energy_history.append((now, energy))
        cutoff = now - self.history_seconds
        self._energy_history = [
            sample for sample in self._energy_history if sample[0] >= cutoff
        ]

    def _current_threshold(self) -> float:
        if len(self._energy_history) < 8:
            return 0.45
        energies = sorted(energy for _, energy in self._energy_history)
        floor = _percentile(energies, 50)
        peak = _percentile(energies, 95)
        return clamp(floor + (peak - floor) * 0.58, 0.18, 0.85)

    def _record_beat(self, now: float) -> None:
        if self._last_beat_time is not None:
            interval = now - self._last_beat_time
            if interval > self._max_interval * 2.0:
                self._interval_history = []
                self._pending_intervals = []
            elif self._min_interval * 0.8 <= interval <= self._max_interval * 1.2:
                self._ingest_interval(now, interval)
        self._last_beat_time = now
        self._beat_count = (self._beat_count + 1) % 64

    def _ingest_interval(self, now: float, interval: float) -> None:
        locked_period = 60.0 / self._bpm if self._bpm > 0.0 else 0.0
        if locked_period > 0.0 and len(self._interval_history) >= 4:
            folded = self._fold_to_period(interval, locked_period)
            if abs(folded - locked_period) / locked_period > self.retune_ratio:
                self._pending_intervals.append((now, interval))
                if len(self._pending_intervals) >= self.retune_beats:
                    self._interval_history = self._pending_intervals
                    self._pending_intervals = []
                    self._bpm = 0.0
                    self._set_bpm_from_history(jump=True)
                return
            interval = folded
            self._pending_intervals = []
        self._interval_history.append((now, interval))
        cutoff = now - self.interval_history_seconds
        self._interval_history = [
            sample for sample in self._interval_history if sample[0] >= cutoff
        ]
        self._set_bpm_from_history(jump=len(self._interval_history) < 4)

    def _set_bpm_from_history(self, *, jump: bool) -> None:
        if not self._interval_history:
            return
        median_interval = statistics.median(
            interval for _, interval in self._interval_history
        )
        measured_bpm = 60.0 / median_interval
        while measured_bpm < self.min_bpm:
            measured_bpm *= 2.0
        while measured_bpm > self.max_bpm:
            measured_bpm *= 0.5
        target = clamp(measured_bpm, self.min_bpm, self.max_bpm)
        if jump or self._bpm <= 0.0:
            self._bpm = target
        else:
            self._bpm += (target - self._bpm) / self.smoothing_beats

    @staticmethod
    def _fold_to_period(interval: float, period: float) -> float:
        return min(
            (interval, interval * 0.5, interval * 2.0),
            key=lambda candidate: abs(candidate - period),
        )

    def _bar_progress(self, now: float) -> float:
        if self._last_beat_time is None or self._beat_count < 0:
            return 0.0
        bpm = self._bpm if self._bpm > 0.0 else self.default_bpm
        beat_period = 60.0 / bpm
        elapsed = max(0.0, now - self._last_beat_time)
        beat_position = (self._beat_count % 4) + elapsed / beat_period
        return clamp(beat_position / 4.0, 0.0, 1.0)


class SpectralTempoTracker:
    """Recover a stable beat grid from a fixed-rate spectral-onset envelope.

    The original interval tracker is useful for clean, isolated triggers but
    can mistake hi-hats or syncopated notes for every beat in a full mix.
    This tracker instead compares several seconds of onset history against
    every plausible musical period.  A phase-locked grid then continues
    between individual transients, so movement does not wander when a kick is
    briefly absent.
    """

    def __init__(
        self,
        updates_per_second: float,
        min_bpm: float = 72.0,
        max_bpm: float = 165.0,
        history_seconds: float = 18.0,
        minimum_history_seconds: float = 5.0,
    ) -> None:
        if updates_per_second <= 0:
            raise ValueError("updates_per_second must be positive")
        self.updates_per_second = float(updates_per_second)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self._maximum_frames = round(history_seconds * updates_per_second)
        self._minimum_frames = round(
            minimum_history_seconds * updates_per_second
        )
        self._activation: list[float] = []
        self._bpm = 0.0
        self._confidence = 0.0
        self._last_estimate_at = float("-inf")
        self._previous_activation = 0.0
        self._last_onset_time: float | None = None
        self._recent_peak_time: float | None = None
        self._origin_time: float | None = None
        self._last_grid_index: int | None = None
        self._pending_bpm: float | None = None
        self._pending_count = 0

    def update(self, activation: float, now: float) -> BeatState:
        value = clamp(float(activation), 0.0, 1.0)
        onset = self._is_onset(value, now)
        self._activation.append(value)
        if len(self._activation) > self._maximum_frames:
            del self._activation[: len(self._activation) - self._maximum_frames]
        if (
            len(self._activation) >= self._minimum_frames
            and now - self._last_estimate_at >= 0.45
        ):
            self._last_estimate_at = now
            self._estimate_tempo()

        if self._bpm <= 0.0:
            self._previous_activation = value
            return BeatState(
                bpm=0.0,
                beat=onset,
                beat_count=0,
                bar_progress=0.0,
                confidence=0.0,
            )

        period = 60.0 / self._bpm
        if self._origin_time is None:
            self._origin_time = self._recent_peak_time or now
        if onset:
            nearest = round((now - self._origin_time) / period)
            predicted = self._origin_time + nearest * period
            error = now - predicted
            if abs(error) <= period * 0.24:
                # A restrained phase correction keeps the grid attached to
                # real kicks without allowing every syncopation to reset it.
                self._origin_time += error * (
                    0.24 + 0.22 * (1.0 - self._confidence)
                )

        position = (now - self._origin_time) / period
        grid_index = math.floor(position + 0.10)
        beat = False
        if self._last_grid_index is None:
            self._last_grid_index = grid_index
        elif grid_index > self._last_grid_index:
            beat = True
            self._last_grid_index = grid_index
        elif grid_index < self._last_grid_index - 1:
            self._last_grid_index = grid_index

        phase_position = (now - self._origin_time) / period
        beat_index = math.floor(phase_position)
        beat_phase = phase_position - beat_index
        bar_progress = ((beat_index % 4) + beat_phase) / 4.0
        self._previous_activation = value
        return BeatState(
            bpm=self._bpm,
            beat=beat,
            beat_count=beat_index % 64,
            bar_progress=bar_progress % 1.0,
            confidence=self._confidence,
        )

    def _is_onset(self, value: float, now: float) -> bool:
        if len(self._activation) < 8:
            threshold = 0.52
        else:
            recent = np.asarray(
                self._activation[-round(self.updates_per_second * 3.0) :],
                dtype=np.float64,
            )
            median = float(np.median(recent))
            high = float(np.percentile(recent, 90))
            threshold = clamp(median + 0.52 * (high - median), 0.28, 0.78)
        refractory = 60.0 / self.max_bpm * 0.68
        onset = (
            value >= threshold
            and value - self._previous_activation >= 0.09
            and (
                self._last_onset_time is None
                or now - self._last_onset_time >= refractory
            )
        )
        if onset:
            self._last_onset_time = now
            self._recent_peak_time = now
        return onset

    def _estimate_tempo(self) -> None:
        values = np.asarray(self._activation, dtype=np.float64)
        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 95))
        if high - low <= 0.04:
            self._confidence *= 0.85
            return
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        floor = float(np.percentile(normalized, 35))
        novelty = np.maximum(0.0, normalized - floor)
        novelty -= float(np.mean(novelty))

        minimum_lag = max(
            2, round(self.updates_per_second * 60.0 / self.max_bpm)
        )
        maximum_lag = round(
            self.updates_per_second * 60.0 / self.min_bpm
        )
        lags = list(range(minimum_lag, maximum_lag + 1))
        correlations: list[float] = []
        for lag in lags:
            current = novelty[lag:]
            delayed = novelty[:-lag]
            denominator = float(
                np.linalg.norm(current) * np.linalg.norm(delayed)
            )
            correlations.append(
                0.0
                if denominator <= 1e-9
                else float(np.dot(current, delayed) / denominator)
            )
        best_offset = max(
            range(len(correlations)),
            key=correlations.__getitem__,
        )
        best_lag = float(lags[best_offset])
        best_score = correlations[best_offset]
        if 0 < best_offset < len(correlations) - 1:
            left = correlations[best_offset - 1]
            center = correlations[best_offset]
            right = correlations[best_offset + 1]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-9:
                best_lag += clamp(
                    0.5 * (left - right) / denominator,
                    -0.5,
                    0.5,
                )
        candidate_bpm = 60.0 * self.updates_per_second / best_lag

        competitors = [
            score
            for index, score in enumerate(correlations)
            if abs(index - best_offset) > 1
        ]
        second_score = max(competitors, default=0.0)
        prominence = best_score - second_score
        evidence = clamp(
            (len(values) / self.updates_per_second - 3.0) / 3.0,
            0.0,
            1.0,
        )
        quality = clamp((best_score - 0.04) / 0.34, 0.0, 1.0)
        # A strong half-time harmonic is normal in four-on-the-floor music;
        # it should modestly temper confidence rather than prevent lock.
        distinctness = 0.82 + 0.18 * clamp(prominence / 0.14, 0.0, 1.0)
        confidence = clamp(evidence * quality * distinctness, 0.0, 1.0)
        self._confidence = 0.45 * self._confidence + 0.55 * confidence

        if self._bpm <= 0.0:
            if confidence >= 0.12:
                self._bpm = candidate_bpm
            return
        difference = abs(candidate_bpm - self._bpm) / self._bpm
        if difference <= 0.08:
            self._bpm += (candidate_bpm - self._bpm) * 0.24
            self._pending_bpm = None
            self._pending_count = 0
            return
        if (
            self._pending_bpm is not None
            and abs(candidate_bpm - self._pending_bpm)
            / max(self._pending_bpm, 1e-6)
            <= 0.05
        ):
            self._pending_bpm += (candidate_bpm - self._pending_bpm) * 0.35
            self._pending_count += 1
        else:
            self._pending_bpm = candidate_bpm
            self._pending_count = 1
        if self._pending_count >= 4 and confidence >= 0.25:
            self._bpm = self._pending_bpm
            self._pending_bpm = None
            self._pending_count = 0


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * fraction
