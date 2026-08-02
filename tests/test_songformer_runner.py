import importlib.util
from pathlib import Path
import unittest


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "songformer-cpu-runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lumen_songformer_cpu_runner", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SongFormerRunnerTests(unittest.TestCase):
    def test_adjacent_identical_predictions_form_one_stable_segment(self):
        runner = _runner_module()
        merged = runner._merge_adjacent_segments(
            [
                {"label": "intro", "start": 0.0, "end": 4.0},
                {"label": "intro", "start": 4.0, "end": 8.0},
                {"label": "chorus", "start": 8.0, "end": 12.0},
            ]
        )
        self.assertEqual(
            merged,
            [
                {"label": "intro", "start": 0.0, "end": 8.0},
                {"label": "chorus", "start": 8.0, "end": 12.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
