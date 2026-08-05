from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lumen_engine.student import (
    CONTEXT_SECONDS,
    FEATURE_NAMES,
    StableStructureDecoder,
    StudentPrediction,
    StreamingStructureStudent,
    semantic_frame_features,
    _binary_class_weights,
    _causal_sequences,
    _class_weights,
    _student_targets,
)


class StreamingStudentTests(unittest.TestCase):
    def test_training_precomputes_each_causal_sequence_before_shuffling(self) -> None:
        class TrackingStudent(StreamingStructureStudent):
            def __init__(self) -> None:
                self.visited_sequences: list[list[tuple[float, float]]] = []
                super().__init__(hidden_size=8, seed=41)

            def reset(self) -> None:
                super().reset()
                self.visited_sequences.append([])

            def _causal_context(self, features, *, timestamp_s=None):
                values = tuple(float(value) for value in features)
                self.visited_sequences[-1].append(
                    (values[0], float(timestamp_s or 0.0))
                )
                return super()._causal_context(
                    values, timestamp_s=timestamp_s
                )

        examples = []
        for group_index in range(3):
            for frame_index in range(2):
                features = np.zeros(len(FEATURE_NAMES))
                features[0] = (group_index + 1) / 10.0
                examples.append(
                    {
                        "split_group_id": f"song-{group_index}",
                        "recording_offset_ms": frame_index * 100,
                        "features": features,
                        "energy": "build",
                    }
                )

        model = TrackingStudent()
        report = model.train(examples, epochs=8, learning_rate=0.005)
        visited = [sequence for sequence in model.visited_sequences if sequence]

        self.assertEqual(report["causal_sequences"], 3)
        self.assertEqual(
            report["epoch_ordering"],
            "precomputed_causal_context_minibatch_shuffle",
        )
        self.assertEqual(len(visited), 3)
        for sequence in visited:
            self.assertEqual([timestamp for _, timestamp in sequence], [0.0, 0.1])
            self.assertEqual(sequence[0][0], sequence[1][0])

    def test_causal_memories_follow_wall_clock_at_different_call_rates(self):
        def memories(rate_hz: float, duration_s: float) -> list[np.ndarray]:
            model = StreamingStructureStudent(hidden_size=8, seed=11)
            zero = np.zeros(len(FEATURE_NAMES))
            one = np.ones(len(FEATURE_NAMES))
            model.predict(zero, timestamp_s=0.0)
            steps = round(rate_hz * duration_s)
            for index in range(1, steps + 1):
                model.predict(one, timestamp_s=index / rate_hz)
            return [value.copy() for value in model._memories]

        ten_hz = memories(10.0, 16.0)
        live_chunk_rate = memories(48_000 / 2_048, 16.0)
        self.assertEqual(len(ten_hz), len(CONTEXT_SECONDS))
        for expected, observed in zip(ten_hz, live_chunk_rate):
            np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_default_student_step_remains_ten_hertz(self):
        zero = np.zeros(len(FEATURE_NAMES))
        one = np.ones(len(FEATURE_NAMES))
        default = StreamingStructureStudent(hidden_size=8, seed=12)
        timed = StreamingStructureStudent(hidden_size=8, seed=12)
        default.predict(zero)
        timed.predict(zero, timestamp_s=0.0)
        default.predict(one)
        timed.predict(one, timestamp_s=0.1)
        for expected, observed in zip(default._memories, timed._memories):
            np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_stable_decoder_rejects_frame_to_frame_section_flicker(self) -> None:
        decoder = StableStructureDecoder()

        def prediction(energy: str, confidence: float, boundary: float = 0.0):
            return StudentPrediction(
                functional="chorus",
                energy=energy,
                content="instrumental",
                confidence={
                    "functional": 0.8,
                    "energy": confidence,
                    "content": 0.8,
                },
                probabilities={"functional": {}, "energy": {}, "content": {}},
                boundary_probability=boundary,
            )

        self.assertEqual(decoder.update(prediction("groove", 0.8), 0.0)["energy"], "groove")
        for index in range(1, 30):
            label = "build" if index % 2 else "groove"
            state = decoder.update(prediction(label, 0.75), index * 0.1)
            self.assertEqual(state["energy"], "groove")
        for index in range(30, 50):
            state = decoder.update(
                prediction("build", 0.8, boundary=0.8), index * 0.1
            )
        self.assertEqual(state["energy"], "build")

    def test_synthetic_teacher_labels_train_and_round_trip(self) -> None:
        restrained = np.zeros(len(FEATURE_NAMES))
        restrained[:12] = [
            0.15, 0.05, 0.18, 0.12, 0.08, 0.05,
            0.10, 0.34, 0.4, 0.8, 0.0, 0.0,
        ]
        build = np.zeros(len(FEATURE_NAMES))
        build[:12] = [
            0.65, 0.72, 0.70, 0.58, 0.62, 0.70,
            0.75, 0.53, 0.9, 0.9, 0.0, 0.0,
        ]
        examples = []
        for _ in range(30):
            examples.append(
                {
                    "features": restrained,
                    "functional": "intro",
                    "energy": "restrained",
                    "content": "instrumental",
                    "split_group_id": "restrained-song",
                }
            )
        for _ in range(30):
            examples.append(
                {
                    "features": build,
                    "functional": "transition",
                    "energy": "build",
                    "content": "transition",
                    "split_group_id": "build-song",
                }
            )
        model = StreamingStructureStudent(hidden_size=20, seed=7)
        report = model.train(examples, epochs=35, learning_rate=0.003)
        self.assertLess(report["final_loss"], report["initial_loss"])
        metrics = model.evaluate(examples)
        self.assertGreaterEqual(metrics["energy"]["accuracy"], 0.95)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.npz"
            model.save(path)
            restored = StreamingStructureStudent.load(path)
            restored.reset()
            prediction = restored.predict(build)
            self.assertEqual(prediction.energy, "build")
            self.assertEqual(prediction.functional, "transition")

    def test_semantic_frame_mapping_is_bounded_and_complete(self) -> None:
        features = semantic_frame_features(
            {
                "observation": {
                    "loudness": 1.4,
                    "onset_strength": 0.5,
                    "low_energy": 0.7,
                    "mid_energy": 0.4,
                    "high_energy": 0.2,
                    "spectral_flux": 0.3,
                    "brightness": 0.8,
                    "bpm": 120,
                    "beat_confidence": 0.9,
                    "tempo_confidence": 0.8,
                    "silence_confidence": -1,
                },
                "audio": {"clipping": 0.1},
            }
        )
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(features >= 0.0))
        self.assertTrue(np.all(features <= 1.0))
        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[7], 0.5)

    def test_unknown_targets_are_not_counted_as_learned_labels(self) -> None:
        row = {
            "features": [0.2] * len(FEATURE_NAMES),
            "functional": "unknown",
            "energy": "groove",
            "content": "unknown",
            "boundary": 0,
        }
        model = StreamingStructureStudent(hidden_size=8, seed=17)
        report = model.train([row] * 4, epochs=1)
        self.assertEqual(report["class_weights"]["functional"]["unknown"], 1.0)
        metrics = model.evaluate([row])
        self.assertEqual(metrics["functional"]["examples"], 0)
        self.assertEqual(metrics["content"]["examples"], 0)
        self.assertEqual(metrics["energy"]["examples"], 1)

    def test_reported_final_loss_is_the_restored_frozen_model_loss(self) -> None:
        examples = []
        for group_index, energy in enumerate(("build", "release")):
            for frame_index in range(24):
                features = np.zeros(len(FEATURE_NAMES))
                features[0] = 0.2 + 0.6 * group_index
                examples.append(
                    {
                        "split_group_id": f"song-{group_index}",
                        "recording_offset_ms": frame_index * 100,
                        "features": features,
                        "energy": energy,
                        "boundary": int(frame_index < 3),
                    }
                )
        model = StreamingStructureStudent(hidden_size=8, seed=23)
        report = model.train(examples, epochs=8, learning_rate=0.002)
        raw_contexts = model._prepare_causal_contexts(
            _causal_sequences(examples), None
        )
        contexts = model._normalize_context(raw_contexts)
        frozen_loss = model._frozen_loss(
            contexts,
            _student_targets(examples),
            _class_weights(examples),
            _binary_class_weights(
                [float(row["boundary"]) for row in examples]
            ),
        )
        self.assertAlmostEqual(report["final_loss"], frozen_loss, places=12)

    def test_approved_axes_survive_model_round_trip(self) -> None:
        model = StreamingStructureStudent(hidden_size=8, seed=24)
        model.approved_axes = {"energy", "boundary"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.npz"
            model.save(path)
            restored = StreamingStructureStudent.load(path)
        self.assertEqual(restored.approved_axes, {"energy", "boundary"})

    def test_training_cancel_check_runs_periodically_through_rows(self) -> None:
        row = {
            "features": [0.2] * len(FEATURE_NAMES),
            "energy": "groove",
        }
        checks = 0

        def cancel_check() -> bool:
            nonlocal checks
            checks += 1
            return checks == 3

        model = StreamingStructureStudent(hidden_size=8, seed=18)
        with self.assertRaisesRegex(InterruptedError, "canceled"):
            model.train([row] * 130, epochs=1, cancel_check=cancel_check)
        self.assertEqual(checks, 3)

    def test_model_loader_rejects_nonfinite_weights(self) -> None:
        model = StreamingStructureStudent(hidden_size=8, seed=19)
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.npz"
            invalid_path = Path(directory) / "invalid.npz"
            model.save(valid_path)
            with np.load(valid_path, allow_pickle=False) as source:
                arrays = {name: source[name].copy() for name in source.files}
            arrays["input_weights"][0, 0] = np.nan
            np.savez_compressed(invalid_path, **arrays)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                StreamingStructureStudent.load(invalid_path)


if __name__ == "__main__":
    unittest.main()
