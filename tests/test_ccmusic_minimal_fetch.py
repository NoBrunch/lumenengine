import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _fetch_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ccmusic-minimal-fetch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lumen_ccmusic_minimal_fetch", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CCMusicMinimalFetchTests(unittest.TestCase):
    def test_label_cell_accepts_both_viewer_arrow_shapes(self):
        fetch = _fetch_module()
        list_shape = [
            {"onset_time": 0, "offset_time": 100, "structure": "Intro"},
            {"onset_time": 100, "offset_time": 200, "structure": "Verse"},
        ]
        struct_shape = {
            "onset_time": [0, 100],
            "offset_time": [100, 200],
            "structure": ["Intro", "Verse"],
        }
        expected = [(0, 100, "Intro"), (100, 200, "Verse")]
        self.assertEqual(fetch.parse_label_cell(list_shape), expected)
        self.assertEqual(fetch.parse_label_cell(struct_shape), expected)

    def test_export_retains_only_labels_and_verifies_all_rows(self):
        fetch = _fetch_module()
        rows = []
        for index in range(fetch.EXPECTED_ROWS):
            rows.append(
                {
                    "row_idx": index,
                    "row": {
                        "audio": {
                            "src": "https://audio.invalid/must-not-persist.mp3"
                        },
                        "mel": {
                            "src": "https://image.invalid/must-not-persist.jpg"
                        },
                        "label": [
                            {
                                "onset_time": 0,
                                "offset_time": 100,
                                "structure": "Intro A",
                            },
                            {
                                "onset_time": 100,
                                "offset_time": 200,
                                "structure": "Verse A",
                            },
                        ],
                    },
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authorized-annotations"
            manifest = fetch.export_rows(rows, output)
            verified = fetch.verify_export(output)
            self.assertEqual(verified, {"rows": 300, "segments": 600})
            self.assertFalse(manifest["contains_audio_or_links"])
            self.assertEqual(manifest["excluded_fields"], ["audio", "mel"])
            serialized = "".join(
                path.read_text(encoding="utf-8")
                for path in output.iterdir()
            )
            self.assertNotIn("audio.invalid", serialized)
            self.assertNotIn("image.invalid", serialized)
            self.assertIn("Intro A", serialized)
            stored = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["revision"], fetch.LOCKED_REVISION)
            self.assertEqual(stored["time_unit"], "centiseconds")

    def test_known_embedded_tsv_record_is_recovered_and_audited(self):
        fetch = _fetch_module()
        segments, repairs, discontinuities = fetch._parse_label_cell_with_repairs(
            [
                {
                    "onset_time": 0,
                    "offset_time": 7102,
                    "structure": "Pre-chorus A\n7102\t9107\tChorus A\"",
                },
                {
                    "onset_time": 9107,
                    "offset_time": 10000,
                    "structure": "Outro",
                },
            ]
        )
        self.assertEqual(
            repairs,
            [
                {
                    "source_segment_index": 0,
                    "kind": "embedded_tsv_record",
                    "onset_time": 7102,
                    "offset_time": 9107,
                }
            ],
        )
        self.assertEqual(discontinuities, [])
        self.assertEqual(
            segments,
            [
                (0, 7102, "Pre-chorus A"),
                (7102, 9107, "Chorus A"),
                (9107, 10000, "Outro"),
            ],
        )

    def test_arbitrary_multiline_label_is_rejected(self):
        fetch = _fetch_module()
        with self.assertRaisesRegex(ValueError, "malformed embedded"):
            fetch.parse_label_cell(
                [
                    {
                        "onset_time": 0,
                        "offset_time": 100,
                        "structure": "Intro\nnot a structured record",
                    }
                ]
            )

    def test_upstream_discontinuity_is_preserved_and_audited(self):
        fetch = _fetch_module()
        cell = [
            {
                "onset_time": 0,
                "offset_time": 100,
                "structure": "Intro",
            },
            {
                "onset_time": 101,
                "offset_time": 200,
                "structure": "Verse",
            },
        ]
        segments, repairs, discontinuities = (
            fetch._parse_label_cell_with_repairs(cell)
        )
        self.assertEqual(
            segments, [(0, 100, "Intro"), (101, 200, "Verse")]
        )
        self.assertEqual(repairs, [])
        self.assertEqual(
            discontinuities,
            [
                {
                    "previous_segment_index": 0,
                    "previous_offset": 100,
                    "next_onset": 101,
                    "delta_centiseconds": 1,
                    "kind": "gap",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
