import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "edmformer-cpu-runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lumen_edmformer_cpu_runner", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EDMFormerCPURunnerTests(unittest.TestCase):
    def test_bounded_attention_is_installed_only_for_worker_model_lifetime(self):
        runner = _runner_module()

        class FakeAttention:
            position_embeddings_type = "rotary"

            def forward(self, *_args, **_kwargs):
                return "eager"

        def fake_model():
            attention = FakeAttention()
            model = SimpleNamespace(
                conformer=SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=attention)]
                )
            )
            return model, attention

        class FakePipeline:
            def _create_muq_model(self):
                return fake_model()[0]

            def _create_musicfm_model(self):
                return fake_model()[0]

        original_muq = FakePipeline._create_muq_model
        original_musicfm = FakePipeline._create_musicfm_model
        upstream = SimpleNamespace(InferencePipeline=FakePipeline)

        with runner._bounded_foundation_model_attention(upstream):
            model = FakePipeline()._create_muq_model()
            attention = model.conformer.layers[0].self_attn
            self.assertTrue(attention._lumen_sdpa_installed)
            self.assertIs(attention.forward.__self__, attention)
            self.assertIs(
                attention.forward.__func__,
                runner._sdpa_rotary_attention_forward,
            )

        self.assertIs(FakePipeline._create_muq_model, original_muq)
        self.assertIs(FakePipeline._create_musicfm_model, original_musicfm)

    def test_validation_accepts_published_context_and_rejects_long_song(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = []
            for name in ("audio.wav", "model.pt", "config.yaml", "stat.json", "music.pt"):
                path = root / name
                path.touch()
                files.append(path)
            with wave.open(str(files[0]), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(8_000)
                target.writeframes(b"\0\0" * 8_000)
            base = SimpleNamespace(
                audio=files[0],
                checkpoint=files[1],
                config=files[2],
                musicfm_stat=files[3],
                musicfm_model=files[4],
                threads=4,
            )
            base.musicfm_source = root / "musicfm"
            (base.musicfm_source / "model").mkdir(parents=True)
            (base.musicfm_source / "model" / "musicfm_25hz.py").touch()
            runner._validate_arguments(base)

            with wave.open(str(files[0]), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(8)
                target.setnframes(0)
                target.writeframes(b"\0\0" * 3_361)
            with self.assertRaisesRegex(ValueError, "published 420-second"):
                runner._validate_arguments(base)

    def test_segment_timeline_is_contiguous_merged_and_song_length_exact(self):
        runner = _runner_module()
        result = runner._merge_and_clamp_segments(
            [
                {"label": "intro", "start": 0.0001, "end": 30.0},
                {"label": "intro", "start": 30.0001, "end": 42.0},
                {"label": "buildup", "start": 42.0, "end": 60.0},
                {"label": "drop", "start": 60.0, "end": 91.0},
            ],
            90.25,
        )
        self.assertEqual(
            result,
            [
                {"label": "intro", "start": 0.0, "end": 42.0},
                {"label": "buildup", "start": 42.0, "end": 60.0},
                {"label": "drop", "start": 60.0, "end": 90.25},
            ],
        )

    def test_pcm_duration_and_atomic_result_output(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(8_000)
                target.writeframes(b"\0\0" * 20_000)
            self.assertEqual(runner._audio_duration_seconds(audio), 2.5)

            output = root / "nested" / "result.json"
            segments = [{"label": "intro", "start": 0.0, "end": 2.5}]
            runner._write_atomic(output, segments)
            self.assertEqual(json.loads(output.read_text()), segments)
            self.assertFalse(output.with_suffix(".json.partial").exists())

    def test_command_defaults_to_four_threads_without_context_override(self):
        runner = _runner_module()
        args = runner._arguments(
            [
                "audio.wav",
                "--checkpoint",
                "model.pt",
                "--config",
                "config.yaml",
                "--musicfm-stat",
                "stats.json",
                "--musicfm-model",
                "music.pt",
                "--musicfm-source",
                "musicfm",
                "--hf-cache-dir",
                "cache",
                "--output",
                "output.json",
            ]
        )
        self.assertFalse(hasattr(args, "window_seconds"))
        self.assertEqual(args.threads, 4)

    def test_predict_preserves_full_context_and_requires_low_memory(self):
        runner = _runner_module()

        class FakeTorch:
            thread_count = None
            interop_count = None

            @classmethod
            def set_num_threads(cls, value):
                cls.thread_count = value

            @classmethod
            def set_num_interop_threads(cls, value):
                cls.interop_count = value

        class FakeNumpy:
            current = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}

            @classmethod
            def seterr(cls, **values):
                previous = dict(cls.current)
                cls.current.update(values)
                return previous

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(8_000)
                target.writeframes(b"\0\0" * 360_000)
            captured = {}

            def fake_predict(path, **kwargs):
                captured["context_during_call"] = upstream.TIME_DUR
                captured["path"] = path
                captured.update(kwargs)
                return [{"label": "intro", "start": 0.0, "end": 45.0}]

            upstream = SimpleNamespace(
                TIME_DUR=420,
                torch=FakeTorch,
                np=FakeNumpy,
                predict_file=fake_predict,
            )
            args = SimpleNamespace(
                audio=audio,
                checkpoint=root / "model.pt",
                config=root / "config.yaml",
                musicfm_stat=root / "stats.json",
                musicfm_model=root / "music.pt",
                musicfm_source=root / "musicfm",
                hf_cache_dir=root / "cache",
                threads=3,
            )
            result = runner._predict(args, upstream=upstream)

        self.assertEqual(captured["context_during_call"], 420)
        self.assertEqual(upstream.TIME_DUR, 420)
        self.assertEqual(FakeNumpy.current["invalid"], "warn")
        self.assertEqual(FakeNumpy.current["divide"], "warn")
        self.assertEqual(FakeTorch.thread_count, 3)
        self.assertEqual(FakeTorch.interop_count, 1)
        self.assertEqual(captured["device"], "cpu")
        self.assertTrue(captured["low_memory"])
        self.assertFalse(captured["persistent_models"])
        self.assertTrue(captured["offline"])
        self.assertEqual(result[-1]["end"], 45.0)


if __name__ == "__main__":
    unittest.main()
