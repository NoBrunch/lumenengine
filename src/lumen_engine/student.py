"""Small causal neural student for CPU music-structure inference.

Large teachers run offline. This compact NumPy network consumes only present
and accumulated past features, so it can eventually run beside Lumen's live
analyzer without looking ahead or importing PyTorch into the DMX process.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from lumen_engine.structure import (
    ContentRole,
    EnergySection,
    FunctionalSection,
)


FEATURE_NAMES = (
    "loudness",
    "onset_strength",
    "low_energy",
    "mid_energy",
    "high_energy",
    "spectral_flux",
    "spectral_brightness",
    "bpm_normalized",
    "beat_confidence",
    "tempo_confidence",
    "silence_confidence",
    "clipping",
    "rhythm_density",
    "harmonic_change",
    "arrangement_change",
)

CONTEXT_SECONDS = (0.5, 2.0, 8.0, 30.0, 60.0)

LABELS = {
    "functional": tuple(value.value for value in FunctionalSection),
    "energy": tuple(value.value for value in EnergySection),
    "content": tuple(value.value for value in ContentRole),
}

_LEGACY_LABEL_ALIASES = {
    "functional": {"instrumental_section": "instrumental"},
    "energy": {"restrained": "low"},
    "content": {"vocal_focus": "vocal"},
}


@dataclass(frozen=True, slots=True)
class StudentPrediction:
    functional: str
    energy: str
    content: str
    confidence: dict[str, float]
    probabilities: dict[str, dict[str, float]]
    boundary_probability: float


class StableStructureDecoder:
    """Turn frame predictions into musical regions with causal hysteresis."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._selected: dict[str, str | None] = {
            "functional": None,
            "energy": None,
            "content": None,
        }
        self._confidence = {axis: 0.0 for axis in self._selected}
        self._changed_at = {axis: float("-inf") for axis in self._selected}
        self._candidate: dict[str, str | None] = {
            axis: None for axis in self._selected
        }
        self._candidate_since = {axis: 0.0 for axis in self._selected}

    def update(
        self, prediction: StudentPrediction, timestamp_s: float
    ) -> dict[str, Any]:
        boundary = prediction.boundary_probability
        for axis in self._selected:
            proposed = str(getattr(prediction, axis))
            confidence = float(prediction.confidence[axis])
            if proposed == "unknown" or confidence < 0.34:
                continue
            current = self._selected[axis]
            if current is None:
                self._accept(axis, proposed, confidence, timestamp_s)
                continue
            if proposed == current:
                self._confidence[axis] += 0.18 * (
                    confidence - self._confidence[axis]
                )
                self._candidate[axis] = None
                continue
            if self._candidate[axis] != proposed:
                self._candidate[axis] = proposed
                self._candidate_since[axis] = timestamp_s
                continue
            candidate_age = timestamp_s - self._candidate_since[axis]
            hold_age = timestamp_s - self._changed_at[axis]
            persistence = (
                1.6 if axis == "energy" else 3.0 if axis == "functional" else 2.2
            )
            minimum_hold = (
                3.5 if axis == "energy" else 7.0 if axis == "functional" else 4.0
            )
            if boundary >= 0.62:
                persistence *= 0.35
                minimum_hold *= 0.45
            if (
                candidate_age >= persistence
                and hold_age >= minimum_hold
                and confidence >= 0.42
            ):
                self._accept(axis, proposed, confidence, timestamp_s)
        return {
            **self._selected,
            "confidence": dict(self._confidence),
            "boundary_probability": boundary,
        }

    def _accept(
        self,
        axis: str,
        label: str,
        confidence: float,
        timestamp_s: float,
    ) -> None:
        self._selected[axis] = label
        self._confidence[axis] = confidence
        self._changed_at[axis] = timestamp_s
        self._candidate[axis] = None


class StreamingStructureStudent:
    """A compact MLP over causal multi-timescale musical state."""

    format_version = 2

    def __init__(self, hidden_size: int = 32, *, seed: int = 20260730) -> None:
        if hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        self.hidden_size = int(hidden_size)
        self.feature_size = len(FEATURE_NAMES)
        # Current frame, five causal exponential memories, and frame delta.
        self.context_size = self.feature_size * (2 + len(CONTEXT_SECONDS))
        generator = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (self.context_size + self.hidden_size))
        self.input_weights = generator.normal(
            0.0, scale, (self.context_size, self.hidden_size)
        ).astype(np.float64)
        self.input_bias = np.zeros(self.hidden_size, dtype=np.float64)
        self.head_weights = {
            axis: generator.normal(
                0.0,
                np.sqrt(2.0 / (self.hidden_size + len(labels))),
                (self.hidden_size, len(labels)),
            ).astype(np.float64)
            for axis, labels in LABELS.items()
        }
        self.head_bias = {
            axis: np.zeros(len(labels), dtype=np.float64)
            for axis, labels in LABELS.items()
        }
        self.boundary_weights = generator.normal(
            0.0,
            np.sqrt(2.0 / (self.hidden_size + 1)),
            self.hidden_size,
        ).astype(np.float64)
        self.boundary_bias = 0.0
        self.training_examples = 0
        self.reset()

    def reset(self) -> None:
        self._memories = [
            np.zeros(self.feature_size, dtype=np.float64)
            for _ in CONTEXT_SECONDS
        ]
        self._previous = np.zeros(self.feature_size, dtype=np.float64)
        self._started = False
        self._last_timestamp_s: float | None = None

    def predict(
        self,
        features: Iterable[float],
        *,
        timestamp_s: float | None = None,
    ) -> StudentPrediction:
        context = self._causal_context(features, timestamp_s=timestamp_s)
        hidden = np.tanh(context @ self.input_weights + self.input_bias)
        probabilities: dict[str, dict[str, float]] = {}
        selected: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for axis, labels in LABELS.items():
            values = _softmax(hidden @ self.head_weights[axis] + self.head_bias[axis])
            index = int(np.argmax(values))
            selected[axis] = labels[index]
            confidence[axis] = float(values[index])
            probabilities[axis] = {
                label: float(values[label_index])
                for label_index, label in enumerate(labels)
            }
        boundary_probability = _sigmoid(
            float(hidden @ self.boundary_weights + self.boundary_bias)
        )
        return StudentPrediction(
            functional=selected["functional"],
            energy=selected["energy"],
            content=selected["content"],
            confidence=confidence,
            probabilities=probabilities,
            boundary_probability=boundary_probability,
        )

    def train(
        self,
        examples: Iterable[dict[str, Any]],
        *,
        epochs: int = 20,
        learning_rate: float = 0.025,
        l2: float = 1e-5,
        cancel_check: Callable[[], bool | None] | None = None,
    ) -> dict[str, Any]:
        rows = list(examples)
        if not rows:
            raise ValueError("student training requires examples")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        losses: list[float] = []
        labeled = 0
        class_weights = _class_weights(rows)
        boundary_weights = _binary_class_weights(
            [float(row["boundary"]) for row in rows if "boundary" in row]
        )
        for _ in range(epochs):
            _check_cancel(cancel_check)
            self.reset()
            previous_group: str | None = None
            previous_offset_ms: int | None = None
            total_loss = 0.0
            total_labels = 0
            for row_index, row in enumerate(rows):
                if row_index % 64 == 0:
                    _check_cancel(cancel_check)
                group = _context_sequence_key(row)
                offset_ms = _context_offset_ms(row)
                if previous_group is not None and (
                    group != previous_group
                    or (
                        offset_ms is not None
                        and previous_offset_ms is not None
                        and offset_ms <= previous_offset_ms
                    )
                ):
                    self.reset()
                previous_group = group
                previous_offset_ms = offset_ms
                context = self._causal_context(
                    row["features"],
                    timestamp_s=(
                        offset_ms / 1000.0
                        if offset_ms is not None
                        else None
                    ),
                )
                preactivation = (
                    context @ self.input_weights + self.input_bias
                )
                hidden = np.tanh(preactivation)
                hidden_gradient = np.zeros_like(hidden)
                head_updates: list[
                    tuple[str, np.ndarray, np.ndarray, np.ndarray]
                ] = []
                for axis, labels in LABELS.items():
                    label = row.get(axis)
                    if label is None:
                        continue
                    label = _canonical_label(axis, label)
                    if label == "unknown":
                        continue
                    if label not in labels:
                        raise ValueError(f"unknown {axis} label {label!r}")
                    probabilities = _softmax(
                        hidden @ self.head_weights[axis]
                        + self.head_bias[axis]
                    )
                    target_index = labels.index(label)
                    example_weight = class_weights[axis].get(label, 1.0)
                    total_loss -= example_weight * float(
                        np.log(max(probabilities[target_index], 1e-12))
                    )
                    total_labels += example_weight
                    logits_gradient = probabilities.copy()
                    logits_gradient[target_index] -= 1.0
                    logits_gradient *= example_weight
                    hidden_gradient += (
                        self.head_weights[axis] @ logits_gradient
                    )
                    head_updates.append(
                        (
                            axis,
                            np.outer(hidden, logits_gradient),
                            logits_gradient,
                            probabilities,
                        )
                    )
                boundary_update: tuple[np.ndarray, float] | None = None
                if "boundary" in row:
                    boundary_target = 1.0 if float(row["boundary"]) >= 0.5 else 0.0
                    boundary_probability = _sigmoid(
                        float(hidden @ self.boundary_weights + self.boundary_bias)
                    )
                    boundary_weight = boundary_weights[int(boundary_target)]
                    total_loss -= boundary_weight * (
                        boundary_target
                        * math.log(max(boundary_probability, 1e-12))
                        + (1.0 - boundary_target)
                        * math.log(max(1.0 - boundary_probability, 1e-12))
                    )
                    total_labels += boundary_weight
                    boundary_gradient = boundary_weight * (
                        boundary_probability - boundary_target
                    )
                    hidden_gradient += self.boundary_weights * boundary_gradient
                    boundary_update = (
                        hidden * boundary_gradient,
                        boundary_gradient,
                    )
                if not head_updates and boundary_update is None:
                    continue
                preactivation_gradient = (
                    hidden_gradient * (1.0 - hidden * hidden)
                )
                input_gradient = np.outer(
                    context, preactivation_gradient
                )
                for axis, weight_gradient, bias_gradient, _ in head_updates:
                    self.head_weights[axis] -= learning_rate * (
                        weight_gradient + l2 * self.head_weights[axis]
                    )
                    self.head_bias[axis] -= learning_rate * bias_gradient
                if boundary_update is not None:
                    boundary_weight_gradient, boundary_bias_gradient = boundary_update
                    self.boundary_weights -= learning_rate * (
                        boundary_weight_gradient + l2 * self.boundary_weights
                    )
                    self.boundary_bias -= learning_rate * boundary_bias_gradient
                self.input_weights -= learning_rate * (
                    input_gradient + l2 * self.input_weights
                )
                self.input_bias -= learning_rate * preactivation_gradient
            if total_labels == 0:
                raise ValueError("student examples contain no recognized labels")
            losses.append(total_loss / total_labels)
            labeled = total_labels
            _check_cancel(cancel_check)
        self.training_examples += len(rows)
        return {
            "examples": len(rows),
            "labels_per_epoch": labeled,
            "epochs": epochs,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "losses": losses,
            "class_weights": class_weights,
            "boundary_class_weights": boundary_weights,
        }

    def evaluate(self, examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        counts = {axis: 0 for axis in LABELS}
        correct = {axis: 0 for axis in LABELS}
        target_counts = {axis: {} for axis in LABELS}
        boundary_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.reset()
        previous_group: str | None = None
        previous_offset_ms: int | None = None
        for row in examples:
            group = _context_sequence_key(row)
            offset_ms = _context_offset_ms(row)
            if previous_group is not None and (
                group != previous_group
                or (
                    offset_ms is not None
                    and previous_offset_ms is not None
                    and offset_ms <= previous_offset_ms
                )
            ):
                self.reset()
            previous_group = group
            previous_offset_ms = offset_ms
            prediction = self.predict(
                row["features"],
                timestamp_s=(
                    offset_ms / 1000.0 if offset_ms is not None else None
                ),
            )
            for axis in LABELS:
                target = row.get(axis)
                if target is None:
                    continue
                target = _canonical_label(axis, target)
                if target == "unknown":
                    continue
                counts[axis] += 1
                target_counts[axis][target] = target_counts[axis].get(target, 0) + 1
                correct[axis] += int(getattr(prediction, axis) == target)
            if "boundary" in row:
                target_boundary = float(row["boundary"]) >= 0.5
                predicted_boundary = prediction.boundary_probability >= 0.5
                key = (
                    "tp" if target_boundary and predicted_boundary
                    else "fn" if target_boundary
                    else "fp" if predicted_boundary
                    else "tn"
                )
                boundary_counts[key] += 1
        result = {
            axis: {
                "examples": counts[axis],
                "accuracy": (
                    correct[axis] / counts[axis]
                    if counts[axis]
                    else None
                ),
                "majority_baseline": (
                    max(target_counts[axis].values()) / counts[axis]
                    if counts[axis] and target_counts[axis]
                    else None
                ),
            }
            for axis in LABELS
        }
        precision = _safe_ratio(
            boundary_counts["tp"], boundary_counts["tp"] + boundary_counts["fp"]
        )
        recall = _safe_ratio(
            boundary_counts["tp"], boundary_counts["tp"] + boundary_counts["fn"]
        )
        result["boundary"] = {
            **boundary_counts,
            "examples": sum(boundary_counts.values()),
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        }
        return result

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        metadata = {
            "format": "lumen_streaming_structure_student",
            "format_version": self.format_version,
            "hidden_size": self.hidden_size,
            "feature_names": FEATURE_NAMES,
            "labels": LABELS,
            "training_examples": self.training_examples,
            "context_seconds": CONTEXT_SECONDS,
        }
        arrays: dict[str, Any] = {
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
            "input_weights": self.input_weights,
            "input_bias": self.input_bias,
            "boundary_weights": self.boundary_weights,
            "boundary_bias": np.asarray(self.boundary_bias),
        }
        for axis in LABELS:
            arrays[f"{axis}_weights"] = self.head_weights[axis]
            arrays[f"{axis}_bias"] = self.head_bias[axis]
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "StreamingStructureStudent":
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                if (
                    metadata.get("format")
                    != "lumen_streaming_structure_student"
                    or int(metadata.get("format_version", -1))
                    != cls.format_version
                ):
                    raise ValueError("unsupported Lumen student model")
                if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
                    raise ValueError(
                        "student model feature contract does not match"
                    )
                if tuple(metadata.get("context_seconds", ())) != CONTEXT_SECONDS:
                    raise ValueError(
                        "student model causal-context contract does not match"
                    )
                model = cls(hidden_size=int(metadata["hidden_size"]))
                arrays = {
                    "input_weights": archive["input_weights"].astype(
                        np.float64, copy=True
                    ),
                    "input_bias": archive["input_bias"].astype(
                        np.float64, copy=True
                    ),
                    "boundary_weights": archive["boundary_weights"].astype(
                        np.float64, copy=True
                    ),
                    "boundary_bias": np.asarray(
                        archive["boundary_bias"], dtype=np.float64
                    ),
                }
                expected_shapes = {
                    "input_weights": (model.context_size, model.hidden_size),
                    "input_bias": (model.hidden_size,),
                    "boundary_weights": (model.hidden_size,),
                    "boundary_bias": (),
                }
                for name, expected_shape in expected_shapes.items():
                    if arrays[name].shape != expected_shape:
                        raise ValueError(
                            f"student model {name} shape does not match"
                        )
                    if not np.all(np.isfinite(arrays[name])):
                        raise ValueError(
                            f"student model {name} contains non-finite values"
                        )
                model.input_weights = arrays["input_weights"]
                model.input_bias = arrays["input_bias"]
                model.boundary_weights = arrays["boundary_weights"]
                model.boundary_bias = float(arrays["boundary_bias"])
                for axis, labels in LABELS.items():
                    if tuple(metadata["labels"].get(axis, ())) != labels:
                        raise ValueError(
                            f"student model {axis} labels do not match"
                        )
                    weights = archive[f"{axis}_weights"].astype(
                        np.float64, copy=True
                    )
                    bias = archive[f"{axis}_bias"].astype(
                        np.float64, copy=True
                    )
                    if weights.shape != (model.hidden_size, len(labels)):
                        raise ValueError(
                            f"student model {axis} weights shape does not match"
                        )
                    if bias.shape != (len(labels),):
                        raise ValueError(
                            f"student model {axis} bias shape does not match"
                        )
                    if not np.all(np.isfinite(weights)) or not np.all(
                        np.isfinite(bias)
                    ):
                        raise ValueError(
                            f"student model {axis} head contains non-finite values"
                        )
                    model.head_weights[axis] = weights
                    model.head_bias[axis] = bias
                model.training_examples = int(
                    metadata.get("training_examples", 0)
                )
                return model
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid Lumen student model archive") from error

    def _causal_context(
        self,
        features: Iterable[float],
        *,
        timestamp_s: float | None = None,
    ) -> np.ndarray:
        current = np.asarray(tuple(features), dtype=np.float64)
        if current.shape != (self.feature_size,):
            raise ValueError(
                f"student expects {self.feature_size} input features"
            )
        current = np.nan_to_num(
            current, nan=0.0, posinf=1.0, neginf=0.0
        )
        current = np.clip(current, -1.0, 1.0)
        if timestamp_s is None:
            elapsed_s = 0.1
        else:
            resolved_timestamp = float(timestamp_s)
            if not math.isfinite(resolved_timestamp):
                raise ValueError("student timestamp must be finite")
            if self._last_timestamp_s is None:
                elapsed_s = 0.1
            else:
                elapsed_s = resolved_timestamp - self._last_timestamp_s
                if elapsed_s <= 0.0:
                    raise ValueError(
                        "student timestamps must be strictly increasing"
                    )
            self._last_timestamp_s = resolved_timestamp
        if not self._started:
            self._memories = [current.copy() for _ in CONTEXT_SECONDS]
            self._previous = current.copy()
            self._started = True
        delta = current - self._previous
        for index, seconds in enumerate(CONTEXT_SECONDS):
            alpha = 1.0 - math.exp(-elapsed_s / seconds)
            self._memories[index] += alpha * (
                current - self._memories[index]
            )
        self._previous = current.copy()
        return np.concatenate((current, *self._memories, delta))


def semantic_frame_features(payload: dict[str, Any]) -> np.ndarray:
    """Convert a recorded Lumen semantic frame to the student contract."""
    observation = payload.get("observation") or {}
    audio = payload.get("audio") or payload.get("audio_metrics") or {}
    bpm = float(observation.get("bpm") or 0.0)
    clipping = audio.get("clipping")
    if clipping is None:
        clipping = max(
            float(audio.get("left_clipping", 0.0)),
            float(audio.get("right_clipping", 0.0)),
        )
    return np.asarray(
        (
            _unit(observation.get("loudness")),
            _unit(observation.get("onset_strength")),
            _unit(observation.get("low_energy")),
            _unit(observation.get("mid_energy")),
            _unit(observation.get("high_energy")),
            _unit(observation.get("spectral_flux")),
            _unit(
                observation.get(
                    "spectral_brightness",
                    observation.get("brightness"),
                )
            ),
            max(0.0, min(1.0, bpm / 240.0)),
            _unit(observation.get("beat_confidence")),
            _unit(observation.get("tempo_confidence")),
            _unit(observation.get("silence_confidence")),
            _unit(clipping),
            _unit(
                observation.get(
                    "rhythm_density",
                    0.55 * _unit(observation.get("onset_strength"))
                    + 0.45 * _unit(observation.get("beat_confidence")),
                )
            ),
            _unit(observation.get("harmonic_change")),
            _unit(
                observation.get(
                    "arrangement_change",
                    max(
                        _unit(observation.get("novelty")),
                        _unit(observation.get("spectral_flux")),
                    ),
                )
            ),
        ),
        dtype=np.float64,
    )


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _canonical_label(axis: str, value: Any) -> str:
    label = str(value)
    return _LEGACY_LABEL_ALIASES.get(axis, {}).get(label, label)


def _context_sequence_key(row: dict[str, Any]) -> str:
    recording = str(row.get("recording_id") or "")
    capture = str(row.get("capture_session_id") or "")
    if recording or capture:
        return f"{recording}\x1f{capture}"
    return str(row.get("split_group_id") or "")


def _context_offset_ms(row: dict[str, Any]) -> int | None:
    value = row.get("recording_offset_ms")
    return int(value) if value is not None else None


def _check_cancel(
    cancel_check: Callable[[], bool | None] | None,
) -> None:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("student training canceled")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_weights(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for axis, labels in LABELS.items():
        counts = {label: 0 for label in labels}
        for row in rows:
            value = row.get(axis)
            if value is not None:
                label = _canonical_label(axis, value)
                if label in counts and label != "unknown":
                    counts[label] += 1
        present = [count for count in counts.values() if count]
        mean = sum(present) / len(present) if present else 1.0
        result[axis] = {
            label: max(0.5, min(5.0, math.sqrt(mean / count)))
            if count
            else 1.0
            for label, count in counts.items()
        }
    return result


def _binary_class_weights(values: list[float]) -> dict[int, float]:
    positives = sum(value >= 0.5 for value in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return {0: 1.0, 1: 1.0}
    total = len(values)
    return {
        0: min(8.0, total / (2.0 * negatives)),
        1: min(8.0, total / (2.0 * positives)),
    }
