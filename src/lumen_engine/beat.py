"""Party Parrot's stable beat tracker, ported to the dependency-free core."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

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
