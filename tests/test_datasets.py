from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lumen_engine.datasets import (
    deterministic_track_split,
    normalize_structure_label,
    normalize_techno_structure_label,
    parse_ccmusic,
    parse_ccmusic_label_records,
    parse_edm98,
    parse_harmonix,
    parse_salami,
)
from lumen_engine.structure import (
    CANONICAL_TECHNO_SECTIONS,
    ContentRole,
    EnergySection,
    FunctionalSection,
    StructureValidationError,
    TechnoSection,
    TransitionEvent,
    validate_dataset,
)


class StructureDatasetTests(unittest.TestCase):
    def test_normalization_keeps_function_energy_and_content_independent(self) -> None:
        combined = normalize_structure_label("Instrumental pre-chorus buildup")
        self.assertEqual(combined.functional, FunctionalSection.PRE_CHORUS)
        self.assertEqual(combined.energy, EnergySection.BUILD)
        self.assertEqual(combined.content, ContentRole.INSTRUMENTAL)

        self.assertEqual(
            normalize_structure_label("Chorus 2").functional,
            FunctionalSection.CHORUS,
        )
        self.assertEqual(
            normalize_structure_label("Re-intro B").functional,
            FunctionalSection.INTRO,
        )
        self.assertEqual(
            normalize_structure_label("Breakdown").energy,
            EnergySection.BREAKDOWN,
        )
        self.assertEqual(
            normalize_structure_label("Drop").energy,
            EnergySection.DROP,
        )
        self.assertEqual(
            normalize_structure_label("Release").energy,
            EnergySection.DROP,
        )
        self.assertNotIn("release", CANONICAL_TECHNO_SECTIONS)
        self.assertEqual(
            CANONICAL_TECHNO_SECTIONS,
            tuple(section.value for section in TechnoSection),
        )
        self.assertEqual(
            tuple(event.value for event in TransitionEvent),
            (
                "section_start",
                "energy_rise",
                "energy_fall",
                "build_start",
                "drop_onset",
                "breakdown_onset",
                "groove_return",
                "outro_start",
                "track_end",
            ),
        )
        techno_intro = normalize_techno_structure_label("intro")
        self.assertEqual(techno_intro.energy, EnergySection.INTRO)
        self.assertEqual(techno_intro.functional, FunctionalSection.UNKNOWN)

    def test_edm98_jsonl_adapter_preserves_deezer_identity_and_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.jsonl"
            payload = {
                "id": "1060564312",
                "file_path": "01 - Oak - Airwalk.mp3",
                "labels": [
                    [0.054, "intro"],
                    [35.942, "buildup"],
                    [58.38, "silence"],
                    [62.866, "drop"],
                    [118.941, "end"],
                ],
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            tracks = parse_edm98(path)

            self.assertEqual(len(tracks), 1)
            track = tracks[0]
            self.assertEqual(track.identity.external_ids["deezer"], "1060564312")
            self.assertEqual(track.identity.audio_filename, payload["file_path"])
            self.assertEqual(track.segments[1].label.energy, EnergySection.BUILD)
            self.assertEqual(track.segments[3].label.energy, EnergySection.DROP)
            self.assertTrue(track.boundaries[-1].terminal)
            self.assertEqual(
                track.boundaries[3].event, TransitionEvent.DROP_ONSET
            )
            self.assertEqual(
                track.boundaries[-1].event, TransitionEvent.TRACK_END
            )
            self.assertEqual(track.segments[0].provenance.source, "edm98")

    def test_edm98_rejects_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.jsonl"
            row = {"id": "one", "labels": [[0, "intro"], [1, "end"]]}
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                StructureValidationError, "duplicate track id"
            ):
                parse_edm98(path)

    def test_harmonix_adapter_reads_metadata_boundaries_and_beats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = root / "segments"
            beats = root / "beats_and_downbeats"
            segments.mkdir()
            beats.mkdir()
            (root / "metadata.csv").write_text(
                "File,Title,Artist,Release,Duration,BPM,Ratio Bars in 4,"
                "Time Signature,Genre,MusicBrainz Id,Acoustid Id\n"
                "0001_track,Track,Artist,Album,30.0,120,100,4|4,Pop,"
                "mbid-1,acoustid-1\n",
                encoding="utf-8",
            )
            (segments / "0001_track.txt").write_text(
                "0.0 intro\n10.0 verse1\n20.0 chorus2\n30.0 end\n",
                encoding="utf-8",
            )
            (beats / "0001_track.txt").write_text(
                "0.5\t1\t1\n1.0\t2\t1\n1.5\t3\t1\n2.0\t4\t1\n",
                encoding="utf-8",
            )

            track = parse_harmonix(
                root / "metadata.csv", segments, beats
            )[0]

            self.assertEqual(track.identity.title, "Track")
            self.assertEqual(track.identity.recording_id, "mbid-1")
            self.assertEqual(track.segments[1].label.functional, FunctionalSection.VERSE)
            self.assertEqual(track.segments[2].label.functional, FunctionalSection.CHORUS)
            self.assertTrue(track.beats[0].downbeat)
            self.assertFalse(track.beats[1].downbeat)
            self.assertEqual(track.metadata["bpm"], 120.0)

    def test_ccmusic_adapter_reads_centisecond_start_end_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Britney Spears - Toxic.txt"
            path.write_text(
                "Start time\tEnd time\tStructure Annotation\n"
                '0000\t4241\t"Intro"\n'
                '4241\t6924\t"Verse A"\n'
                '6924\t8606\t"Pre-chorus A"\n'
                '8606\t11289\t"Chorus A"\n'
                '11289\t12631\t"Re-intro A"\n',
                encoding="utf-8",
            )

            track = parse_ccmusic(path)[0]

            self.assertAlmostEqual(track.segments[0].end_s, 42.41)
            self.assertEqual(
                [segment.label.functional for segment in track.segments],
                [
                    FunctionalSection.INTRO,
                    FunctionalSection.VERSE,
                    FunctionalSection.PRE_CHORUS,
                    FunctionalSection.CHORUS,
                    FunctionalSection.INTRO,
                ],
            )
            self.assertEqual(track.identity.dataset, "ccmusic")

    def test_ccmusic_arrow_label_projection_discards_audio_and_mel(self) -> None:
        revision = "be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3"
        records = [
            {
                "audio": {"path": "must-not-be-retained.mp3", "bytes": b"x"},
                "mel": {"path": "must-not-be-retained.jpg", "bytes": b"y"},
                "label": [
                    {
                        "onset_time": 0,
                        "offset_time": 1000,
                        "structure": "Intro",
                    },
                    {
                        "onset_time": 1000,
                        "offset_time": 2000,
                        "structure": "Pre-chorus A",
                    },
                    {
                        "onset_time": 2000,
                        "offset_time": 3000,
                        "structure": "Chorus A",
                    },
                ],
            }
        ]

        track = parse_ccmusic_label_records(
            records, source_version=revision
        )[0]

        self.assertEqual(track.identity.source_track_id, "row-000000")
        self.assertIsNone(track.identity.audio_filename)
        self.assertEqual(track.identity.external_ids, {})
        self.assertEqual(track.duration_s, 30.0)
        self.assertEqual(
            [segment.label.functional for segment in track.segments],
            [
                FunctionalSection.INTRO,
                FunctionalSection.PRE_CHORUS,
                FunctionalSection.CHORUS,
            ],
        )
        provenance = track.segments[0].provenance
        self.assertEqual(provenance.source_version, revision)
        self.assertFalse(provenance.details["contains_audio_or_links"])
        self.assertEqual(
            set(provenance.details["excluded_fields"]), {"audio", "mel"}
        )

    def test_ccmusic_preserves_discontinuities_without_fabricated_labels(self) -> None:
        records = [
            {
                "label": [
                    {
                        "onset_time": 0,
                        "offset_time": 1000,
                        "structure": "Intro",
                    },
                    {
                        "onset_time": 1200,
                        "offset_time": 2000,
                        "structure": "Verse",
                    },
                    {
                        "onset_time": 1990,
                        "offset_time": 3000,
                        "structure": "Chorus",
                    },
                ]
            }
        ]

        track = parse_ccmusic_label_records(
            records, source_version="locked"
        )[0]

        self.assertEqual(
            [segment.label.raw for segment in track.segments],
            ["Intro", "unannotated gap", "Verse", "Chorus"],
        )
        gap = track.segments[1]
        self.assertEqual((gap.start_s, gap.end_s), (10.0, 12.0))
        self.assertEqual(gap.provenance.annotation_type, "derived_gap")
        self.assertEqual(gap.provenance.confidence, 0.0)
        self.assertEqual(
            track.metadata["timeline_discontinuities"],
            [
                {
                    "previous_segment_index": 0,
                    "previous_offset": 1000,
                    "next_onset": 1200,
                    "delta_centiseconds": 200,
                    "kind": "gap",
                },
                {
                    "previous_segment_index": 1,
                    "previous_offset": 2000,
                    "next_onset": 1990,
                    "delta_centiseconds": -10,
                    "kind": "overlap",
                },
            ],
        )
        self.assertAlmostEqual(track.segments[2].end_s, 19.9)

    def test_salami_prefers_parsed_functions_and_preserves_annotators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations"
            parsed = annotations / "10" / "parsed"
            parsed.mkdir(parents=True)
            metadata = root / "metadata"
            metadata.mkdir()
            (metadata / "metadata.csv").write_text(
                "SONG_ID,SONG_DURATION,SONG_TITLE,ARTIST,ANNOTATOR1,"
                "ANNOTATOR2,CLASS,GENRE\n"
                "10,30,Example,Example Artist,5,8,popular,Jazz\n",
                encoding="utf-8",
            )
            (parsed / "textfile1_functions.txt").write_text(
                "0.0\tSilence\n"
                "0.2\tIntro\n"
                "10.0\tVerse\n"
                "20.0\tOutro\n"
                "30.0\tEnd\n",
                encoding="utf-8",
            )
            (parsed / "textfile2_functions.txt").write_text(
                "0.0\tSilence\n"
                "0.3\tIntro\n"
                "9.5\tVerse\n"
                "21.0\tOutro\n"
                "30.0\tEnd\n",
                encoding="utf-8",
            )

            tracks = parse_salami(annotations)

            self.assertEqual(len(tracks), 2)
            self.assertEqual({track.identity.source_track_id for track in tracks}, {"10"})
            self.assertEqual(
                {track.segments[0].provenance.annotator for track in tracks},
                {"5", "8"},
            )
            self.assertEqual(tracks[0].identity.title, "Example")
            self.assertEqual(
                tracks[0].segments[1].label.functional,
                FunctionalSection.INTRO,
            )

    def test_salami_raw_format_inherits_function_across_fine_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "44" / "textfile1.txt"
            path.parent.mkdir()
            path.write_text(
                "0.000\tSilence\n"
                "0.200\tA, a, Intro, (guitar\n"
                "4.000\ta'\n"
                "8.000\tB, b, Verse, (voice\n"
                "12.000\tb'\n"
                "16.000\tEnd\n",
                encoding="utf-8",
            )

            track = parse_salami(path)[0]

            self.assertEqual(
                track.segments[2].label.functional, FunctionalSection.INTRO
            )
            self.assertEqual(
                track.segments[4].label.functional, FunctionalSection.VERSE
            )
            self.assertEqual(
                track.segments[3].label.content, ContentRole.VOCAL
            )

    def test_deterministic_split_keeps_salami_annotators_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks = []
            for track_id in range(10):
                directory = root / str(track_id)
                directory.mkdir()
                for annotator in (1, 2):
                    path = directory / f"textfile{annotator}.txt"
                    path.write_text(
                        "0\tIntro\n10\tOutro\n20\tEnd\n",
                        encoding="utf-8",
                    )
                    tracks.extend(parse_salami(path))

            first = deterministic_track_split(tracks, seed="repeatable")
            second = deterministic_track_split(reversed(tracks), seed="repeatable")

            keys_by_split = {
                name: {track.identity.group_key for track in values}
                for name, values in first.items()
            }
            self.assertFalse(keys_by_split["train"] & keys_by_split["validation"])
            self.assertFalse(keys_by_split["train"] & keys_by_split["test"])
            self.assertFalse(keys_by_split["validation"] & keys_by_split["test"])
            self.assertEqual(
                {
                    name: [track.identity.group_key for track in values]
                    for name, values in first.items()
                },
                {
                    name: [track.identity.group_key for track in values]
                    for name, values in second.items()
                },
            )
            self.assertEqual(
                {name: len(values) for name, values in first.items()},
                {"train": 16, "validation": 2, "test": 2},
            )
            self.assertTrue(validate_dataset(tracks).valid)


if __name__ == "__main__":
    unittest.main()
