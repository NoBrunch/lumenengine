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
    def test_window_is_strictly_bounded_to_target_pc_range(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = []
            for name in ("audio.wav", "model.pt", "config.yaml", "stat.json", "music.pt"):
                path = root / name
                path.touch()
                files.append(path)
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
            for value in (30, 45, 60):
                base.window_seconds = value
                runner._validate_arguments(base)
            for value in (29, 61, 420):
                base.window_seconds = value
                with self.assertRaisesRegex(ValueError, "between 30 and 60"):
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

    def test_command_defaults_to_sixty_seconds_and_four_threads(self):
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
        self.assertEqual(args.window_seconds, 60)
        self.assertEqual(args.threads, 4)

    def test_predict_overrides_only_child_context_and_requires_low_memory(self):
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
                window_seconds=45,
                threads=3,
            )
            result = runner._predict(args, upstream=upstream)

        self.assertEqual(captured["context_during_call"], 45)
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
