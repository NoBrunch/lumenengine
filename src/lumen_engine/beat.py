"""Transient and spectrum-onset tempo tracking for musical synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time

import numpy as np

from lumen_engine.models import clamp


MIN_PUBLISHED_TEMPO_CONFIDENCE = 0.18


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
        max_bpm: float = 200.0,
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
        confidence = self._tempo_confidence()
        published_bpm = (
            self._bpm
            if confidence >= MIN_PUBLISHED_TEMPO_CONFIDENCE
            else 0.0
        )
        return BeatState(
            bpm=published_bpm,
            beat=beat,
            beat_count=max(self._beat_count, 0),
            bar_progress=(
                self._bar_progress(now) if published_bpm > 0.0 else 0.0
            ),
            confidence=(confidence if published_bpm > 0.0 else 0.0),
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
        max_bpm: float = 200.0,
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
        self._pending_confidence = 0.0
        self._lock_score = 0.0
        self._challenger_score = 0.0
        self._challenger_confidence = 0.0
        self._latest_candidate_bpm = 0.0
        self._latest_candidate_score = 0.0
        self._latest_candidate_confidence = 0.0
        self._latest_octave_promoted = False
        self._latest_family_ambiguity = 0.0
        self._latest_prior_selected = False
        self._latest_tempos = np.asarray([], dtype=np.float64)
        self._latest_correlations = np.asarray([], dtype=np.float64)

    @property
    def diagnostics(self) -> dict[str, float]:
        return {
            "locked_bpm": self._bpm,
            "locked_confidence": self._confidence,
            "candidate_bpm": self._latest_candidate_bpm,
            "candidate_score": self._latest_candidate_score,
            "candidate_confidence": self._latest_candidate_confidence,
            "octave_promoted": self._latest_octave_promoted,
            "family_ambiguity": self._latest_family_ambiguity,
            "prior_selected": self._latest_prior_selected,
            "minimum_bpm": self.min_bpm,
            "maximum_bpm": self.max_bpm,
        }

    def tempo_support(self, bpm: float) -> float:
        """Return current onset-envelope support for an external tempo.

        The interval clock can sometimes identify the correct metrical layer
        while this tracker has selected a stronger half-time or 3:2 peak.  The
        arbiter may use this value to verify that the spectral envelope itself
        supports the interval clock before accepting a handoff.
        """

        if bpm <= 0.0 or not len(self._latest_tempos):
            return 0.0
        offset = int(np.argmin(np.abs(self._latest_tempos - float(bpm))))
        return max(0.0, float(self._latest_correlations[offset]))

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

        if (
            self._bpm <= 0.0
            or self._confidence < MIN_PUBLISHED_TEMPO_CONFIDENCE
        ):
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

        # Search musical tempo directly at fractional lags. At Lumen's normal
        # 23.4375-Hz packet rate, whole-frame lags can represent 140.625 and
        # 156.25 BPM but not the common tempos between them. That quantization
        # was enough to report a real 147-BPM pulse as either neighbor. Linear
        # delay interpolation retains the dependency-free live path while
        # giving the clock quarter-BPM resolution.
        tempos = np.arange(
            self.min_bpm,
            self.max_bpm + 0.125,
            0.25,
            dtype=np.float64,
        )
        lags = self.updates_per_second * 60.0 / tempos
        correlations = [
            _fractional_correlation(novelty, float(lag)) for lag in lags
        ]
        self._latest_tempos = tempos
        self._latest_correlations = np.asarray(correlations, dtype=np.float64)
        # Autocorrelation alone frequently selects a sparse 1/2- or 2/3-rate
        # layer even when the actual beat layer is also strongly present.  A
        # broad log-tempo prior (the standard approach in established tempo
        # trackers) resolves that ambiguity without importing an offline audio
        # stack into Live.  It is deliberately broad enough to retain slow
        # genuine pulses and fast drum-and-bass when their evidence is real.
        positive_correlations = np.maximum(
            0.0, np.asarray(correlations, dtype=np.float64)
        )
        log_tempo_prior = -0.5 * (
            np.log2(tempos / 128.0) / 0.60
        ) ** 2
        weighted_scores = (
            np.log1p(1_000_000.0 * positive_correlations)
            + log_tempo_prior
        )
        raw_best_offset = int(np.argmax(positive_correlations))
        weighted_offset = int(np.argmax(weighted_scores))
        weighted_bpm = float(tempos[weighted_offset])
        family_offsets = np.flatnonzero(
            np.abs(tempos - weighted_bpm)
            / max(weighted_bpm, 1e-9)
            <= 0.08
        )
        # The prior chooses a metrical family, not an exact BPM. Recover the
        # strongest unweighted peak inside that family so a clean 175.8 pulse
        # is not dragged toward the 128-BPM prior.
        best_offset = int(
            family_offsets[
                np.argmax(positive_correlations[family_offsets])
            ]
        )
        raw_best_bpm = float(tempos[raw_best_offset])
        selected_bpm = float(tempos[best_offset])
        prior_octave_promoted = bool(
            raw_best_bpm < 105.0
            and 1.90 <= selected_bpm / max(raw_best_bpm, 1e-9) <= 2.10
        )
        self._latest_octave_promoted = prior_octave_promoted
        self._latest_prior_selected = bool(
            abs(selected_bpm - raw_best_bpm)
            / max(raw_best_bpm, 1e-9)
            > 0.08
            and _same_metrical_family(selected_bpm, raw_best_bpm)
        )
        self._latest_family_ambiguity = 0.0
        coarse_bpm = float(tempos[best_offset])
        # Autocorrelation often gives the bar-accent/half-time lag a slightly
        # higher score even when clear transients also support every beat. If
        # the best result is slow, inspect the neighborhood of its half lag.
        # Promote that octave only when it retains most of the slow peak's
        # correlation. A genuinely slow pulse has no such intervening evidence
        # and therefore remains slow.
        ambiguous_slow_octave = False
        if coarse_bpm < 105.0 and coarse_bpm * 2.0 <= self.max_bpm:
            octave_offset = min(
                range(len(tempos)),
                key=lambda index: abs(
                    float(tempos[index]) - coarse_bpm * 2.0
                ),
            )
            octave_score = correlations[octave_offset]
            slow_score = correlations[best_offset]
            octave_ratio = octave_score / max(slow_score, 1e-9)
            self._latest_family_ambiguity = clamp(
                octave_ratio, 0.0, 1.0
            )
            ambiguous_slow_octave = bool(
                octave_score >= 0.12 and octave_ratio >= 0.62
            )
            # Alternating strong/weak beats normally make the half-time peak
            # larger. The faster family is accepted when intervening pulses
            # still retain substantial independent correlation. If support is
            # only suggestive, confidence is suppressed below publication
            # rather than confidently announcing the slow answer.
            if (
                octave_score >= 0.14
                and octave_ratio >= 0.72
                and octave_score >= slow_score - 0.20
            ):
                best_offset = octave_offset
                self._latest_octave_promoted = True
        best_score = correlations[best_offset]
        candidate_bpm = float(tempos[best_offset])

        competitors = [
            score
            for index, score in enumerate(correlations)
            if abs(float(tempos[index]) - candidate_bpm)
            > max(4.0, candidate_bpm * 0.04)
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
        distinctness = 0.45 + 0.55 * clamp(prominence / 0.14, 0.0, 1.0)
        candidate_confidence = clamp(
            evidence * quality * distinctness, 0.0, 1.0
        )
        if ambiguous_slow_octave and not self._latest_octave_promoted:
            candidate_confidence = min(
                candidate_confidence,
                MIN_PUBLISHED_TEMPO_CONFIDENCE * 0.80,
            )
        self._latest_candidate_bpm = candidate_bpm
        self._latest_candidate_score = best_score
        self._latest_candidate_confidence = candidate_confidence

        locked_score = 0.0
        locked_confidence = 0.0
        if self._bpm > 0.0:
            current_offset = min(
                range(len(tempos)),
                key=lambda index: abs(float(tempos[index]) - self._bpm),
            )
            locked_score = correlations[current_offset]
            locked_competitors = [
                score
                for index, score in enumerate(correlations)
                if abs(float(tempos[index]) - self._bpm)
                > max(4.0, self._bpm * 0.04)
            ]
            locked_prominence = locked_score - max(
                locked_competitors, default=0.0
            )
            locked_confidence = self._peak_confidence(
                score=locked_score,
                prominence=locked_prominence,
                evidence=evidence,
            )
        self._consider_tempo_candidate(
            candidate_bpm=candidate_bpm,
            candidate_score=best_score,
            candidate_confidence=candidate_confidence,
            locked_score=locked_score,
            locked_confidence=locked_confidence,
        )

    @staticmethod
    def _peak_confidence(
        *, score: float, prominence: float, evidence: float
    ) -> float:
        quality = clamp((score - 0.04) / 0.34, 0.0, 1.0)
        distinctness = 0.45 + 0.55 * clamp(
            prominence / 0.14, 0.0, 1.0
        )
        return clamp(evidence * quality * distinctness, 0.0, 1.0)

    def _consider_tempo_candidate(
        self,
        *,
        candidate_bpm: float,
        candidate_score: float,
        candidate_confidence: float,
        locked_score: float,
        locked_confidence: float,
    ) -> None:
        """Arbitrate a correlation peak without contaminating the lock.

        Candidate confidence describes the candidate only. Until a challenger
        is committed, the public lock is smoothed exclusively from evidence at
        the locked tempo. This prevents a strong 4:3 or double-time peak from
        making the old BPM appear confidently supported while it is not.
        """
        candidate_bpm = clamp(candidate_bpm, self.min_bpm, self.max_bpm)
        candidate_confidence = clamp(candidate_confidence, 0.0, 1.0)
        if self._bpm <= 0.0:
            self._remember_challenger(
                candidate_bpm,
                candidate_score,
                candidate_confidence,
            )
            if (
                self._pending_count >= 3
                and self._pending_confidence
                >= MIN_PUBLISHED_TEMPO_CONFIDENCE
            ):
                self._commit_challenger()
            return

        difference = abs(candidate_bpm - self._bpm) / self._bpm
        if difference <= 0.08:
            self._bpm += (candidate_bpm - self._bpm) * 0.24
            self._lock_score = candidate_score
            self._confidence = (
                0.45 * self._confidence + 0.55 * candidate_confidence
            )
            self._clear_challenger()
            return

        self._lock_score = locked_score
        self._confidence = (
            0.82 * self._confidence + 0.18 * locked_confidence
        )
        self._challenger_score = candidate_score
        self._challenger_confidence = candidate_confidence

        # Triplets and 3:4/4:3 subdivisions remain alternate readings of the
        # established meter. An octave-promoted candidate is different: it
        # means intervening transients explicitly support every fast beat, so
        # allow sustained evidence to repair an earlier half-time lock.
        if _same_metrical_family(candidate_bpm, self._bpm):
            ratio = candidate_bpm / self._bpm
            is_supported_double = bool(
                1.90 <= ratio <= 2.10
                and candidate_confidence >= 0.35
                and self._latest_octave_promoted
                and candidate_score >= locked_score - 0.20
            )
            is_supported_prior_relative = bool(
                self._latest_prior_selected
                and candidate_score >= locked_score - 0.20
                and candidate_confidence >= 0.35
            )
            if not is_supported_double and not is_supported_prior_relative:
                self._clear_challenger()
                return
            self._remember_challenger(
                candidate_bpm,
                candidate_score,
                candidate_confidence,
            )
            # While the fast family is being confirmed, do not continue to
            # advertise near-perfect certainty in the contradicted half-time
            # lock. The clock remains continuous, but the public confidence
            # now honestly represents the unresolved family.
            self._confidence = min(
                self._confidence,
                max(
                    MIN_PUBLISHED_TEMPO_CONFIDENCE,
                    candidate_confidence * 0.72,
                ),
            )
            if self._pending_count >= 12:
                self._commit_challenger()
            return

        if (
            candidate_score - locked_score < 0.065
            or candidate_confidence < 0.35
        ):
            self._clear_challenger()
            return
        self._remember_challenger(
            candidate_bpm,
            candidate_score,
            candidate_confidence,
        )
        # A real non-harmonic tempo change remains possible, but it must be
        # sustained for several independent 0.45-second estimates.
        if self._pending_count >= 12 and self._pending_confidence >= 0.45:
            self._commit_challenger()

    def _remember_challenger(
        self, bpm: float, score: float, confidence: float
    ) -> None:
        if (
            self._pending_bpm is not None
            and abs(bpm - self._pending_bpm)
            / max(self._pending_bpm, 1e-6)
            <= 0.05
        ):
            self._pending_bpm += (bpm - self._pending_bpm) * 0.35
            self._pending_confidence += (
                confidence - self._pending_confidence
            ) * 0.35
            self._pending_count += 1
        else:
            self._pending_bpm = bpm
            self._pending_confidence = confidence
            self._pending_count = 1
        self._challenger_score = score
        self._challenger_confidence = confidence

    def _commit_challenger(self) -> None:
        if self._pending_bpm is None:
            return
        previous_bpm = self._bpm
        self._bpm = self._pending_bpm
        self._confidence = self._pending_confidence
        self._lock_score = self._challenger_score
        self._clear_challenger()
        if previous_bpm > 0.0:
            # Do not reinterpret the old grid under a new period. Anchor the
            # replacement to a measured onset and start its beat counter clean.
            self._origin_time = (
                self._recent_peak_time
                if self._recent_peak_time is not None
                else self._last_estimate_at
            )
            self._last_grid_index = None

    def _clear_challenger(self) -> None:
        self._pending_bpm = None
        self._pending_count = 0
        self._pending_confidence = 0.0
        self._challenger_score = 0.0
        self._challenger_confidence = 0.0


class TempoSourceArbiter:
    """Hold a stable tracker source while a second clock warms or disagrees."""

    def __init__(self) -> None:
        self.source = "none"
        self._pending_source: str | None = None
        self._pending_count = 0
        self._published: BeatState | None = None
        self._published_phase_beats = 0.0
        self._published_beat_count = 0
        self._last_now: float | None = None
        self._dropout_packets = 0
        self.clock_discontinuities = 0

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "source": self.source,
            "pending_source": self._pending_source,
            "pending_packets": self._pending_count,
            "dropout_packets": self._dropout_packets,
            "clock_discontinuities": self.clock_discontinuities,
            "published_bpm": (
                None if self._published is None else self._published.bpm
            ),
            "published_confidence": (
                0.0 if self._published is None else self._published.confidence
            ),
            "published_bar_phase": (
                None if self._published is None else self._published.bar_progress
            ),
        }

    def select(
        self,
        *,
        spectral: BeatState,
        fallback: BeatState,
        now: float | None = None,
        spectral_support_for_fallback: float = 0.0,
        spectral_peak_score: float = 0.0,
    ) -> BeatState:
        candidates = {"spectral": spectral, "fallback": fallback}
        valid = {
            name: state
            for name, state in candidates.items()
            if state.bpm > 0.0
            and state.confidence >= MIN_PUBLISHED_TEMPO_CONFIDENCE
        }
        if self.source == "none":
            if not valid:
                return self._hold_or_unlock(spectral, fallback, now)
            self.source = max(
                valid,
                key=lambda name: (
                    valid[name].confidence,
                    name == "spectral",
                ),
            )
            return self._publish(valid[self.source], now)

        current = valid.get(self.source)
        other_source = "fallback" if self.source == "spectral" else "spectral"
        challenger = valid.get(other_source)
        if current is not None and self._published is not None:
            current_jump = (
                abs(current.bpm - self._published.bpm)
                / max(self._published.bpm, 1e-6)
            )
            if current_jump > 0.08:
                # A tracker may internally retune even though the alternative
                # clock still supports the established tempo. Do not let that
                # internal state change bypass source arbitration.
                if (
                    challenger is not None
                    and abs(challenger.bpm - self._published.bpm)
                    / max(self._published.bpm, 1e-6)
                    <= 0.08
                ):
                    self.source = other_source
                    self._clear_pending_source()
                    return self._publish(challenger, now)
                tempo_token = f"{self.source}:tempo-change"
                if self._pending_source == tempo_token:
                    self._pending_count += 1
                else:
                    self._pending_source = tempo_token
                    self._pending_count = 1
                required_tempo_packets = (
                    12 if self.source == "spectral" else 120
                )
                if self._pending_count < required_tempo_packets:
                    return self._hold_or_unlock(
                        spectral,
                        fallback,
                        now,
                        maximum_hold_packets=required_tempo_packets,
                    )
                self._clear_pending_source()
        if challenger is None:
            self._clear_pending_source()
            if current is not None:
                return self._publish(current, now)
            return self._hold_or_unlock(spectral, fallback, now)
        if current is None:
            published_bpm = (
                self._published.bpm if self._published is not None else 0.0
            )
            close_to_published = bool(
                published_bpm > 0.0
                and abs(challenger.bpm - published_bpm) / published_bpm
                <= 0.08
            )
            # A close replacement can take over immediately. A non-close
            # fallback must persist for several seconds before replacing a
            # spectral clock; otherwise one sparse breakdown can publish a
            # brief 140 BPM interval inside a stable 176 BPM song.
            required = (
                2
                if close_to_published
                else 8 if other_source == "spectral" else 120
            )
        else:
            close = abs(challenger.bpm - current.bpm) / current.bpm <= 0.08
            # Once the preferred spectral clock owns a matching tempo family,
            # a slightly more confident interval clock must not take it back.
            # That caused dozens of source handoffs inside a single steady
            # track even though the two clocks differed by only a few BPM.
            if close and self.source == "spectral":
                self._clear_pending_source()
                return self._publish(current, now)
            ratio = challenger.bpm / current.bpm
            supported_spectral_double = bool(
                other_source == "spectral"
                and 1.90 <= ratio <= 2.10
                and challenger.confidence >= 0.35
            )
            supported_fallback_double = bool(
                other_source == "fallback"
                and 1.90 <= ratio <= 2.10
                and challenger.confidence >= 0.78
            )
            supported_fallback_relative = bool(
                other_source == "fallback"
                and _same_metrical_family(challenger.bpm, current.bpm)
                and challenger.confidence >= 0.72
                and spectral_peak_score >= 0.16
                and spectral_support_for_fallback
                >= max(0.16, spectral_peak_score * 0.68)
            )
            if (
                not close
                and _same_metrical_family(challenger.bpm, current.bpm)
                and not supported_spectral_double
                and not supported_fallback_double
                and not supported_fallback_relative
            ):
                self._clear_pending_source()
                return self._publish(current, now)
            preferred_spectral = other_source == "spectral" and close
            confidence_margin = (
                -0.55
                if supported_spectral_double
                else -0.25
                if supported_fallback_double
                else -0.20
                if supported_fallback_relative
                else -0.20
                if preferred_spectral
                else 0.12
            )
            if challenger.confidence < current.confidence + confidence_margin:
                self._clear_pending_source()
                return self._publish(current, now)
            required = (
                4
                if close
                else 12
                if supported_spectral_double
                else 24
                if supported_fallback_double
                else 36
                if supported_fallback_relative
                else 12 if other_source == "spectral" else 120
            )
        if self._pending_source == other_source:
            self._pending_count += 1
        else:
            self._pending_source = other_source
            self._pending_count = 1
        if self._pending_count >= required:
            self.source = other_source
            self._clear_pending_source()
            return self._publish(challenger, now)
        if current is not None:
            return self._publish(current, now)
        return self._hold_or_unlock(
            spectral,
            fallback,
            now,
            maximum_hold_packets=max(8, required),
        )

    def reset(self) -> None:
        self.source = "none"
        self._clear_pending_source()
        self._published = None
        self._published_phase_beats = 0.0
        self._published_beat_count = 0
        self._last_now = None
        self._dropout_packets = 0
        self.clock_discontinuities += 1

    def _clear_pending_source(self) -> None:
        self._pending_source = None
        self._pending_count = 0

    def _hold_or_unlock(
        self,
        spectral: BeatState,
        fallback: BeatState,
        now: float | None,
        *,
        maximum_hold_packets: int = 8,
    ) -> BeatState:
        if (
            self._published is not None
            and self._dropout_packets < maximum_hold_packets
        ):
            dropout_packets = self._dropout_packets + 1
            held = BeatState(
                bpm=self._published.bpm,
                beat=False,
                beat_count=self._published.beat_count,
                bar_progress=self._published.bar_progress,
                confidence=self._published.confidence * 0.86,
            )
            published = self._publish(held, now)
            self._dropout_packets = dropout_packets
            return published
        self.source = "none"
        self._published = None
        self._last_now = now
        self.clock_discontinuities += 1
        return _unlocked_beat_state(spectral, fallback)

    def _publish(
        self, candidate: BeatState, now: float | None
    ) -> BeatState:
        self._dropout_packets = 0
        if self._published is None or now is None:
            self._published = candidate
            self._published_phase_beats = candidate.bar_progress * 4.0
            self._published_beat_count = candidate.beat_count
            self._last_now = now
            return candidate
        elapsed = (
            0.0
            if self._last_now is None
            else max(0.0, min(0.25, now - self._last_now))
        )
        self._last_now = now
        previous_phase = self._published_phase_beats
        phase_step = elapsed * candidate.bpm / 60.0
        predicted_phase = self._published_phase_beats + phase_step
        # Tracker sources have independent bar origins, so correcting the full
        # four-beat phase would introduce a bar jump at handoff. Correct only
        # the within-beat phase, gently, toward measured onset timing. This
        # keeps one continuous published bar clock while preventing it from
        # drifting away from the audio-derived tracker.
        measured_phase = (candidate.bar_progress * 4.0) % 1.0
        predicted_beat_phase = predicted_phase % 1.0
        phase_error = (
            (measured_phase - predicted_beat_phase + 0.5) % 1.0 - 0.5
        )
        correction_gain = 0.04 + 0.08 * clamp(
            candidate.confidence, 0.0, 1.0
        )
        # Limit correction to a fraction of this packet's forward movement.
        # A low BPM or duplicate timestamp must never make the public clock run
        # backward or jump forward without elapsed audio.
        maximum_correction = min(0.04, phase_step * 0.35)
        phase_correction = clamp(
            phase_error * correction_gain,
            -maximum_correction,
            maximum_correction,
        )
        self._published_phase_beats = predicted_phase + phase_correction
        crossed_beats = max(
            0,
            math.floor(self._published_phase_beats + 1e-9)
            - math.floor(previous_phase + 1e-9),
        )
        self._published_beat_count += crossed_beats
        published = BeatState(
            bpm=candidate.bpm,
            # Candidate pulses belong to the candidate's independent phase.
            # Emit pulses only when the continuous published clock crosses a
            # beat, after measured phase correction has been applied.
            beat=bool(crossed_beats),
            beat_count=self._published_beat_count,
            bar_progress=(self._published_phase_beats % 4.0) / 4.0,
            confidence=candidate.confidence,
        )
        self._published = published
        return published


def _same_metrical_family(first_bpm: float, second_bpm: float) -> bool:
    if first_bpm <= 0.0 or second_bpm <= 0.0:
        return False
    ratio = first_bpm / second_bpm
    relatives = (0.5, 2.0 / 3.0, 0.75, 4.0 / 3.0, 1.5, 2.0)
    return any(
        abs(ratio - relative) / relative <= 0.04
        for relative in relatives
    )


def _fractional_correlation(values: np.ndarray, lag: float) -> float:
    """Normalized correlation at a non-integral delay in analysis frames."""

    start = max(1, int(math.ceil(lag)))
    if len(values) - start < 16:
        return 0.0
    positions = np.arange(start, len(values), dtype=np.float64)
    current = values[start:]
    delayed = np.interp(
        positions - lag,
        np.arange(len(values), dtype=np.float64),
        values,
    )
    denominator = float(np.linalg.norm(current) * np.linalg.norm(delayed))
    if denominator <= 1e-9:
        return 0.0
    return float(np.dot(current, delayed) / denominator)


def _unlocked_beat_state(
    spectral: BeatState, fallback: BeatState
) -> BeatState:
    return BeatState(
        bpm=0.0,
        beat=spectral.beat or fallback.beat,
        beat_count=0,
        bar_progress=0.0,
        confidence=0.0,
    )


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
