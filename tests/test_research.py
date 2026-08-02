from __future__ import annotations

from contextlib import closing, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lumen_engine.cli import main
from lumen_engine.memory import SongMemoryStore
from lumen_engine.research import RESEARCH_COMPONENTS, ResearchManager


class ResearchManagerTests(unittest.TestCase):
    def test_status_is_dependency_free_and_reports_every_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(root / "research", store=store)
            status = manager.status()
            self.assertEqual(
                {item["component_id"] for item in status["components"]},
                {item.component_id for item in RESEARCH_COMPONENTS},
            )
            self.assertTrue(
                all(
                    item["source"]["state"] == "missing"
                    for item in status["components"]
                )
            )
            self.assertEqual(status["database"]["dataset_sources"], 0)

    def test_existing_checkout_is_verified_and_registered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "research" / "sources" / "edm98"
            source.mkdir(parents=True)
            subprocess.run(
                ["git", "init", "-q", str(source)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "config", "user.email", "test@lumen"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "config", "user.name", "Lumen Test"],
                check=True,
            )
            (source / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(source), "add", "README.md"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(root / "research", store=store)
            result = manager.provision_sources(["edm98"])
            self.assertEqual(result["results"][0]["state"], "present")
            edm98 = next(
                item
                for item in result["status"]["components"]
                if item["component_id"] == "edm98"
            )
            self.assertEqual(edm98["source"]["state"], "ready")
            sources = store.list_dataset_sources()
            self.assertEqual(sources[0]["source_id"], "edm98")
            self.assertEqual(sources[0]["status"], "ready")

    def test_edm98_annotations_import_into_normalized_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                root
                / "research"
                / "sources"
                / "edm98"
                / "src"
                / "edm98"
                / "resources"
            )
            source.mkdir(parents=True)
            (source / "dataset.jsonl").write_text(
                json.dumps(
                    {
                        "id": "12345",
                        "file_path": "example.mp3",
                        "labels": [
                            [0.0, "intro"],
                            [10.0, "buildup"],
                            [10.0004, "drop"],
                            [30.0, "end"],
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            splits = source / "splits"
            splits.mkdir()
            (splits / "train.txt").write_text("12345\n", encoding="utf-8")
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(root / "research", store=store)

            result = manager.import_annotations(["edm98"])

            self.assertEqual(result["results"][0]["state"], "imported")
            self.assertEqual(result["results"][0]["tracks"], 1)
            self.assertEqual(result["results"][0]["timelines"], 1)
            self.assertEqual(result["results"][0]["segments"], 3)
            self.assertEqual(
                store.list_dataset_sources()[0]["status"], "imported"
            )
            self.assertEqual(result["database"]["dataset_tracks"], 1)
            self.assertEqual(result["database"]["timelines"], 1)

            repeated = manager.import_annotations(["edm98"])
            self.assertTrue(repeated["results"][0]["unchanged"])
            self.assertEqual(repeated["database"], result["database"])
            with closing(sqlite3.connect(store.path)) as connection:
                self.assertEqual(
                    connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM structure_segments"
                    ).fetchone()[0],
                    3,
                )

    def test_ccmusic_import_ignores_non_annotation_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "research" / "sources" / "ccmusic"
            source.mkdir(parents=True)
            (source / "README.txt").write_text(
                "CCMusic access and license instructions\n",
                encoding="utf-8",
            )
            (source / "lfs-pointer.txt").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:fixture\nsize 123\n",
                encoding="utf-8",
            )
            (source / "Song One.txt").write_text(
                "Start time\tEnd time\tStructure Annotation\n"
                "0000\t1000\tIntro\n"
                "1000\t2000\tVerse A\n"
                "2000\t3000\tChorus A\n",
                encoding="utf-8",
            )
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(root / "research", store=store)

            result = manager.import_annotations(["ccmusic"])

            self.assertEqual(result["results"][0]["state"], "imported")
            self.assertEqual(result["results"][0]["tracks"], 1)
            self.assertEqual(result["results"][0]["segments"], 3)
            self.assertEqual(result["database"]["dataset_tracks"], 1)
            self.assertEqual(result["database"]["timelines"], 1)

    def test_authorized_ccmusic_manifest_is_label_only_provenanced_and_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            research_root = root / "research"
            authorized = (
                research_root
                / "sources"
                / "ccmusic"
                / "authorized-annotations"
            )
            authorized.mkdir(parents=True)
            rows = []
            for row_index in range(300):
                filename = f"row-{row_index:06d}.txt"
                payload = (
                    "0\t1000\tIntro\n"
                    "1000\t2000\tVerse A\n"
                    "2000\t3000\tChorus A\n"
                ).encode("utf-8")
                (authorized / filename).write_bytes(payload)
                rows.append(
                    {
                        "row_index": row_index,
                        "filename": filename,
                        "segments": 3,
                        "embedded_records_repaired": 0,
                        "source_repairs": [],
                        "timeline_discontinuities": [],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            revision = "be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3"
            manifest = {
                "schema": "lumen.ccmusic.authorized_annotations.v1",
                "dataset_id": "ccmusic-database/song_structure",
                "revision": revision,
                "config": "default",
                "split": "train",
                "time_unit": "centiseconds",
                "row_count": 300,
                "rows": rows,
                "retained_fields": [
                    "label.onset_time",
                    "label.offset_time",
                    "label.structure",
                ],
                "excluded_fields": ["audio", "mel"],
                "contains_audio_or_links": False,
                "source_repair_count": 0,
                "source_repairs": [],
                "timeline_discontinuity_count": 0,
                "timeline_discontinuities": [],
                "transport": (
                    "DuckDB HTTP range projection of pinned Hugging Face "
                    "Parquet"
                ),
                "parquet_revision": (
                    "6ac1a082ca649072518d9fcd7fbf448a1e844266"
                ),
                "projected_label_compressed_bytes": 42_000,
            }
            (authorized / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            metadata = (
                research_root
                / "sources"
                / "ccmusic-gated-metadata"
                / "default"
                / "train"
            )
            metadata.mkdir(parents=True)
            (metadata / "dataset_info.json").write_text(
                json.dumps(
                    {
                        "features": {
                            "label": {
                                "feature": {
                                    "onset_time": {"dtype": "uint32"},
                                    "offset_time": {"dtype": "uint32"},
                                    "structure": {"dtype": "string"},
                                }
                            }
                        },
                        "splits": {"train": {"num_examples": 300}},
                    }
                ),
                encoding="utf-8",
            )
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(research_root, store=store)

            result = manager.import_annotations(["ccmusic"])

            item = result["results"][0]
            self.assertEqual(item["state"], "imported")
            self.assertEqual(item["tracks"], 300)
            self.assertEqual(item["segments"], 900)
            source = store.list_dataset_sources()[0]
            self.assertEqual(source["revision"], revision)
            self.assertTrue(source["metadata"]["labels_only"])
            self.assertFalse(
                source["metadata"]["contains_audio_or_links"]
            )
            self.assertEqual(
                set(source["metadata"]["excluded_fields"]),
                {"audio", "mel"},
            )
            with closing(sqlite3.connect(store.path)) as connection:
                split_counts = dict(
                    connection.execute(
                        "SELECT split, COUNT(*) FROM dataset_tracks "
                        "WHERE source_id='ccmusic' GROUP BY split"
                    ).fetchall()
                )
            self.assertEqual(
                split_counts,
                {"train": 240, "validation": 30, "test": 30},
            )
            with closing(sqlite3.connect(store.path)) as connection:
                provenance = json.loads(
                    connection.execute(
                        "SELECT provenance_json FROM structure_segments "
                        "ORDER BY timeline_id, segment_index LIMIT 1"
                    ).fetchone()[0]
                )
            self.assertEqual(provenance["source"], "ccmusic")
            self.assertEqual(provenance["source_version"], revision)
            self.assertTrue(provenance["details"]["labels_only"])
            self.assertFalse(
                provenance["details"]["contains_audio_or_links"]
            )
            # Export timestamps and transport notes are non-semantic; changing
            # them must not force 300 identical timelines through SQLite again.
            manifest["created_at"] = "2099-01-01T00:00:00Z"
            (authorized / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            repeated = manager.import_annotations(["ccmusic"])
            self.assertTrue(repeated["results"][0]["unchanged"])
            self.assertEqual(
                repeated["database"]["timelines"],
                result["database"]["timelines"],
            )

    def test_partial_ccmusic_export_is_never_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            research_root = root / "research"
            authorized = (
                research_root
                / "sources"
                / "ccmusic"
                / "authorized-annotations"
            )
            authorized.mkdir(parents=True)
            (authorized / "row-000000.txt").write_text(
                "0\t1000\tIntro\n1000\t2000\tChorus\n",
                encoding="utf-8",
            )
            store = SongMemoryStore(root / "memory.sqlite3")
            manager = ResearchManager(research_root, store=store)

            status = manager.status()
            ccmusic = next(
                item
                for item in status["components"]
                if item["component_id"] == "ccmusic"
            )
            self.assertEqual(
                ccmusic["annotations"]["state"],
                "authorized_export_incomplete",
            )
            self.assertEqual(
                ccmusic["annotations"]["reason_code"],
                "ccmusic_manifest_not_finalized",
            )
            result = manager.import_annotations(["ccmusic"])
            self.assertEqual(
                result["results"][0]["state"],
                "authorized_export_incomplete",
            )
            self.assertEqual(result["database"]["dataset_tracks"], 0)
            self.assertEqual(result["database"]["timelines"], 0)

    def test_ccmusic_gate_is_explicit_in_status_import_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            research_root = root / "research"
            (research_root / "sources" / "ccmusic").mkdir(parents=True)
            metadata = (
                research_root
                / "sources"
                / "ccmusic-gated-metadata"
                / "default"
                / "train"
            )
            metadata.mkdir(parents=True)
            (metadata / "dataset_info.json").write_text(
                json.dumps(
                    {
                        "features": {
                            "label": {
                                "feature": {
                                    "onset_time": {"dtype": "uint32"},
                                    "offset_time": {"dtype": "uint32"},
                                    "structure": {"dtype": "string"},
                                }
                            }
                        },
                        "splits": {"train": {"num_examples": 300}},
                    }
                ),
                encoding="utf-8",
            )
            memory = root / "memory.sqlite3"
            manager = ResearchManager(
                research_root,
                store=SongMemoryStore(memory),
            )

            with patch.object(
                ResearchManager,
                "_huggingface_credential_status",
                return_value={"present": False, "source": None},
            ):
                status = manager.status()
            ccmusic = next(
                item
                for item in status["components"]
                if item["component_id"] == "ccmusic"
            )
            self.assertEqual(
                ccmusic["annotations"]["state"],
                "awaiting_user_credentials",
            )
            self.assertEqual(
                ccmusic["annotations"]["reason_code"],
                "ccmusic_hf_token_missing",
            )
            self.assertFalse(
                ccmusic["annotations"]["automatic_download_attempted"]
            )

            output = io.StringIO()
            with patch.object(
                ResearchManager,
                "_huggingface_credential_status",
                return_value={"present": False, "source": None},
            ), redirect_stdout(output):
                exit_code = main(
                    [
                        "research-import-annotations",
                        "ccmusic",
                        "--root",
                        str(research_root),
                        "--memory",
                        str(memory),
                    ]
                )
            payload = json.loads(output.getvalue())
            gate = payload["results"][0]
            self.assertEqual(exit_code, 1)
            self.assertEqual(
                gate["state"], "awaiting_user_credentials"
            )
            self.assertEqual(
                gate["reason_code"], "ccmusic_hf_token_missing"
            )
            self.assertTrue(gate["requires_user_credentials"])
            self.assertFalse(gate["credential_present"])
            self.assertTrue(gate["labels_only_fetch"])
            self.assertFalse(gate["retains_audio"])
            self.assertIn("cached login", gate["required_action"])


if __name__ == "__main__":
    unittest.main()
