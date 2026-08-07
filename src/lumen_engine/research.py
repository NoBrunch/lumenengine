"""Local management for structure datasets and offline teacher models.

The live Lumen process remains dependency-light. Research sources, model
environments, checkpoints, and caches are kept beneath Lumen's ignored local
state and inspected through this module without importing heavyweight packages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable

from lumen_engine.memory import SongMemoryStore


@dataclass(frozen=True, slots=True)
class ResearchComponent:
    component_id: str
    display_name: str
    kind: str
    role: str
    source_url: str
    checkout: str
    license_name: str
    audio_policy: str
    environment: str | None = None
    required_modules: tuple[str, ...] = ()


RESEARCH_COMPONENTS: tuple[ResearchComponent, ...] = (
    ResearchComponent(
        component_id="edm98",
        display_name="EDM-98 / EDMFormer",
        kind="dataset_and_teacher",
        role="edm_energy_structure",
        source_url="https://github.com/25ohms/EDM-98.git",
        checkout="main",
        license_name="MIT metadata; CC BY 4.0 code/model materials",
        audio_policy="external_by_deezer_id",
        environment="edmformer",
        required_modules=("torch", "edm98"),
    ),
    ResearchComponent(
        component_id="harmonix",
        display_name="The Harmonix Set",
        kind="dataset",
        role="beats_downbeats_functional_structure",
        source_url="https://github.com/urinieto/harmonixset.git",
        checkout="main",
        license_name="MIT repository; separate mel-spectrogram license",
        audio_policy="annotations_and_optional_licensed_melspectrograms",
    ),
    ResearchComponent(
        component_id="ccmusic",
        display_name="CCMusic Song Structure",
        kind="dataset",
        role="pop_functional_structure",
        source_url=(
            "https://huggingface.co/datasets/"
            "ccmusic-database/song_structure"
        ),
        checkout="main",
        license_name=(
            "CC-BY-NC-ND-4.0 gated wrapper; underlying audio third-party"
        ),
        audio_policy="authorized_labels_only_no_audio_retention",
    ),
    ResearchComponent(
        component_id="salami",
        display_name="SALAMI",
        kind="dataset",
        role="hierarchical_structure",
        source_url="https://github.com/DDMAL/salami-data-public.git",
        checkout="master",
        license_name="CC0 annotations",
        audio_policy="external_with_public_internet_archive_subset",
    ),
    ResearchComponent(
        component_id="songformer",
        display_name="SongFormer",
        kind="teacher",
        role="general_structure_pseudo_labels",
        source_url="https://github.com/ASLP-lab/SongFormer.git",
        checkout="main",
        license_name="Upstream repository/model terms",
        audio_policy="analyzes_local_audio",
        environment="songformer",
        required_modules=(
            "torch",
            "transformers",
            "huggingface_hub",
            "librosa",
        ),
    ),
)


CCMUSIC_DATASET_ID = "ccmusic-database/song_structure"
CCMUSIC_DATASET_REVISION = "be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3"
CCMUSIC_PARQUET_REVISION = "6ac1a082ca649072518d9fcd7fbf448a1e844266"
CCMUSIC_MANIFEST_SCHEMA = "lumen.ccmusic.authorized_annotations.v1"
CCMUSIC_EXPECTED_ROWS = 300


class ResearchManager:
    """Provision and diagnose isolated, local research components."""

    def __init__(
        self,
        root: str | Path,
        *,
        store: SongMemoryStore | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.sources = self.root / "sources"
        self.environments = self.root / "environments"
        self.models = self.root / "models"
        self.cache = self.root / "cache"
        self.audio = self.root / "audio"
        self.exports = self.root / "exports"

    def ensure_layout(self) -> None:
        for directory in (
            self.root,
            self.sources,
            self.environments,
            self.models,
            self.cache,
            self.audio,
            self.exports,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def status(self, *, include_training: bool = True) -> dict[str, Any]:
        """Return a complete diagnosis without importing ML into this process."""
        self.ensure_layout()
        components = [
            self._component_status(component)
            for component in RESEARCH_COMPONENTS
        ]
        environments = {
            name: self._environment_status(name)
            for name in ("edmformer", "songformer")
        }
        ffmpeg = shutil.which("ffmpeg")
        bundled_ffmpeg = self.root / "tools" / "ffmpeg"
        if ffmpeg is None and bundled_ffmpeg.is_file():
            ffmpeg = str(bundled_ffmpeg)
        ready_sources = sum(
            item["source"]["state"] == "ready" for item in components
        )
        result = {
            "schema": "lumen_research_status_v1",
            "root": str(self.root),
            "components": components,
            "environments": environments,
            "tools": {
                "git": shutil.which("git"),
                "ffmpeg": ffmpeg,
                "python_3_12": self._find_python_312(),
            },
            "summary": {
                "components": len(components),
                "sources_ready": ready_sources,
                "environments_ready": sum(
                    item["state"] == "ready"
                    for item in environments.values()
                ),
                "fully_ready": (
                    ready_sources == len(components)
                    and all(
                        item["state"] == "ready"
                        for item in environments.values()
                    )
                ),
            },
            "database": (
                self.store.research_summary()
                if self.store is not None
                else None
            ),
        }
        if self.store is not None and include_training:
            from lumen_engine.offline import training_readiness

            result["training"] = training_readiness(
                self.store, research_root=self.root
            )
        return result

    def provision_sources(
        self,
        component_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Clone annotation/code repositories without downloading song audio."""
        self.ensure_layout()
        selected = set(component_ids or ())
        known = {component.component_id for component in RESEARCH_COMPONENTS}
        unknown = selected - known
        if unknown:
            raise ValueError(
                "unknown research components: " + ", ".join(sorted(unknown))
            )
        results: list[dict[str, Any]] = []
        for component in RESEARCH_COMPONENTS:
            if selected and component.component_id not in selected:
                continue
            destination = self.sources / component.component_id
            started = time.monotonic()
            if (destination / ".git").is_dir():
                state = "present"
                message = "existing checkout preserved"
            else:
                environment = dict(os.environ)
                environment["GIT_LFS_SKIP_SMUDGE"] = "1"
                command = [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    component.checkout,
                    component.source_url,
                    str(destination),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=environment,
                )
                if completed.returncode != 0:
                    results.append(
                        {
                            "component_id": component.component_id,
                            "state": "error",
                            "message": (
                                completed.stderr.strip()
                                or completed.stdout.strip()
                                or f"git exited {completed.returncode}"
                            ),
                            "elapsed_s": time.monotonic() - started,
                        }
                    )
                    continue
                state = "cloned"
                message = "source checkout created"
            component_status = self._component_status(component)
            results.append(
                {
                    "component_id": component.component_id,
                    "state": state,
                    "message": message,
                    "revision": component_status["source"]["revision"],
                    "elapsed_s": time.monotonic() - started,
                }
            )
            self._register_component(component, component_status)
        return {
            "root": str(self.root),
            "results": results,
            "status": self.status(),
        }

    def import_annotations(
        self,
        component_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Normalize available public annotations into Lumen's database."""
        if self.store is None:
            raise RuntimeError("annotation import requires a Lumen memory store")
        from lumen_engine.datasets import (
            deterministic_track_split,
            parse_ccmusic,
            parse_edm98,
            parse_harmonix,
            parse_salami,
        )

        self.ensure_layout()
        selected = set(component_ids or ("edm98", "harmonix", "ccmusic", "salami"))
        importable = {"edm98", "harmonix", "ccmusic", "salami"}
        unknown = selected - importable
        if unknown:
            raise ValueError(
                "unknown annotation components: " + ", ".join(sorted(unknown))
            )
        results: list[dict[str, Any]] = []
        registered_sources = {
            item["source_id"]: item
            for item in self.store.list_dataset_sources()
        }
        for component_id in ("edm98", "harmonix", "ccmusic", "salami"):
            if component_id not in selected:
                continue
            source_root = self.sources / component_id
            started = time.monotonic()
            if not source_root.exists():
                results.append(
                    {
                        "component_id": component_id,
                        "state": "missing_source",
                        "tracks": 0,
                        "timelines": 0,
                    }
                )
                continue
            component = next(
                item
                for item in RESEARCH_COMPONENTS
                if item.component_id == component_id
            )
            component_status = self._component_status(component)
            ccmusic_package = (
                self._ccmusic_authorized_package(source_root)
                if component_id == "ccmusic"
                else None
            )
            import_revision = (
                ccmusic_package["revision"]
                if ccmusic_package is not None
                else component_status["source"]["revision"]
            )
            registered = registered_sources.get(component_id)
            registered_metadata = (
                registered.get("metadata", {})
                if registered is not None
                else {}
            )
            if (
                registered is not None
                and registered.get("status") == "imported"
                and registered.get("revision")
                == import_revision
                and int(registered_metadata.get("timelines", 0)) > 0
                and (
                    ccmusic_package is None
                    or not ccmusic_package["labels_only"]
                    or registered_metadata.get("annotation_fingerprint")
                    == ccmusic_package["annotation_fingerprint"]
                )
            ):
                results.append(
                    {
                        "component_id": component_id,
                        "state": "imported",
                        "unchanged": True,
                        "tracks": int(
                            registered_metadata.get("tracks", 0)
                        ),
                        "timelines": int(
                            registered_metadata.get("timelines", 0)
                        ),
                        "segments": int(
                            registered_metadata.get("segments", 0)
                        ),
                        "beats": int(
                            registered_metadata.get("beats", 0)
                        ),
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                continue
            if component_id == "edm98":
                tracks = parse_edm98(source_root)
                split_by_group = self._edm98_split_map(source_root)
            elif component_id == "harmonix":
                dataset_root = source_root / "dataset"
                tracks = parse_harmonix(
                    dataset_root / "metadata.csv",
                    dataset_root / "segments",
                    dataset_root / "beats_and_downbeats",
                )
                split_by_group = self._deterministic_split_map(
                    tracks, deterministic_track_split
                )
            elif component_id == "salami":
                tracks = parse_salami(
                    source_root / "annotations",
                    source_root / "metadata" / "metadata.csv",
                )
                split_by_group = self._deterministic_split_map(
                    tracks, deterministic_track_split
                )
            else:
                assert ccmusic_package is not None
                annotation_files = ccmusic_package["files"]
                if not annotation_files:
                    access = self._ccmusic_access_details(source_root)
                    self.store.register_dataset_source(
                        source_id=component_id,
                        display_name=component.display_name,
                        role=component.role,
                        status=access["state"],
                        revision=import_revision,
                        license_name=component.license_name,
                        root_path=component_status["source"]["path"],
                        metadata={
                            "audio_policy": component.audio_policy,
                            **access,
                        },
                    )
                    results.append(
                        {
                            "component_id": component_id,
                            "state": access["state"],
                            **access,
                            "tracks": 0,
                            "timelines": 0,
                            "elapsed_s": time.monotonic() - started,
                        }
                    )
                    continue
                # Only pass files which actually contain CCMusic timeline
                # rows. Authorized bundles can also include README, license,
                # and Git-LFS pointer .txt files which are not annotations.
                tracks = []
                for annotation_path in annotation_files:
                    parsed = parse_ccmusic(
                        annotation_path,
                        time_unit=ccmusic_package["time_unit"],
                        source_version=(
                            ccmusic_package["revision"]
                            or "song_structure"
                        ),
                        provenance_details={
                            "dataset_id": CCMUSIC_DATASET_ID,
                            "revision": ccmusic_package["revision"],
                            "labels_only": ccmusic_package["labels_only"],
                            "contains_audio_or_links": False,
                            "manifest_sha256": ccmusic_package[
                                "manifest_sha256"
                            ],
                            "annotation_fingerprint": ccmusic_package[
                                "annotation_fingerprint"
                            ],
                            "source_repair_count": ccmusic_package[
                                "source_repair_count"
                            ],
                            "timeline_discontinuity_count": ccmusic_package[
                                "timeline_discontinuity_count"
                            ],
                        },
                    )
                    expected_segments = ccmusic_package["row_segments"].get(
                        annotation_path.name
                    )
                    expected_discontinuities = ccmusic_package[
                        "row_discontinuities"
                    ].get(annotation_path.name, [])
                    actual_discontinuities = (
                        parsed[0].metadata.get("timeline_discontinuities", [])
                        if len(parsed) == 1
                        else []
                    )
                    if actual_discontinuities != expected_discontinuities:
                        raise ValueError(
                            "CCMusic parsed discontinuity audit does not "
                            "match manifest: "
                            + annotation_path.name
                        )
                    normalized_segments = (
                        expected_segments
                        + sum(
                            1
                            for item in expected_discontinuities
                            if item.get("kind") == "gap"
                        )
                        if expected_segments is not None
                        else None
                    )
                    if (
                        normalized_segments is not None
                        and (
                            len(parsed) != 1
                            or len(parsed[0].segments) != normalized_segments
                        )
                    ):
                        raise ValueError(
                            "CCMusic normalized segment count does not match: "
                            + annotation_path.name
                        )
                    tracks.extend(parsed)
                split_by_group = self._deterministic_split_map(
                    tracks, deterministic_track_split
                )
            component = next(
                item
                for item in RESEARCH_COMPONENTS
                if item.component_id == component_id
            )
            status = self._component_status(component)
            # dataset_tracks has a foreign key to dataset_sources. Register the
            # source before inserting its tracks, then replace this provisional
            # status with complete import counts below.
            self.store.register_dataset_source(
                source_id=component_id,
                display_name=component.display_name,
                role=component.role,
                status="importing",
                revision=import_revision,
                license_name=component.license_name,
                root_path=status["source"]["path"],
                metadata={"audio_policy": component.audio_policy},
            )
            timeline_count = 0
            segment_count = 0
            beat_count = 0
            for track in tracks:
                split = split_by_group.get(track.identity.group_key, "train")
                self.store.upsert_dataset_track(
                    source_id=component_id,
                    source_track_id=track.identity.source_track_id,
                    title=track.identity.title,
                    artist=", ".join(track.identity.artists) or None,
                    duration_ms=(
                        round(track.duration_s * 1000)
                        if track.duration_s is not None
                        else None
                    ),
                    split=split,
                    audio_path=None,
                    metadata={
                        "recording_id": track.identity.recording_id,
                        "audio_filename": track.identity.audio_filename,
                        "external_ids": track.identity.external_ids,
                        "source_metadata": track.metadata,
                    },
                )
                source_file = (
                    track.segments[0].provenance.source_file
                    if track.segments
                    else None
                )
                annotator = (
                    track.segments[0].provenance.annotator
                    if track.segments
                    else None
                )
                timeline_material = "\x1f".join(
                    (
                        track.identity.group_key,
                        source_file or "",
                        annotator or "",
                    )
                )
                timeline_id = (
                    f"dataset:{component_id}:"
                    + hashlib.sha256(
                        timeline_material.encode("utf-8")
                    ).hexdigest()[:32]
                )
                segments: list[dict[str, Any]] = []
                previous_end_ms: int | None = None
                for index, segment in enumerate(track.segments):
                    start_ms = (
                        previous_end_ms
                        if previous_end_ms is not None
                        else round(segment.start_s * 1000)
                    )
                    # A few upstream boundaries are less than half a
                    # millisecond apart. SQLite stores integer milliseconds, so
                    # quantize the shared boundary once and retain a non-empty,
                    # contiguous segment instead of rounding it to zero length.
                    end_ms = max(
                        start_ms + 1,
                        round(segment.end_s * 1000),
                    )
                    segments.append(
                        {
                            "segment_index": index,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "functional_label": (
                                segment.label.functional.value
                            ),
                            "energy_label": segment.label.energy.value,
                            "content_label": segment.label.content.value,
                            "boundary_confidence": segment.confidence,
                            "label_confidence": segment.confidence,
                            "raw_label": segment.label.raw,
                            "provenance": asdict(segment.provenance),
                        }
                    )
                    previous_end_ms = end_ms
                beats = [
                    {
                        "time_ms": round(beat.time_s * 1000),
                        "position_in_bar": beat.position_in_bar,
                        "bar_number": beat.bar_number,
                        "confidence": beat.confidence,
                        "provenance": (
                            asdict(beat.provenance)
                            if beat.provenance is not None
                            else {"source": component_id}
                        ),
                    }
                    for beat in track.beats
                ]
                confidence = (
                    sum(segment.confidence for segment in track.segments)
                    / len(track.segments)
                    if track.segments
                    else 0.0
                )
                self.store.save_structure_timeline(
                    timeline_id=timeline_id,
                    dataset_source_id=component_id,
                    provenance=f"{component_id}_ground_truth",
                    timeline_version="lumen_normalized_structure_v1",
                    confidence=confidence,
                    segments=segments,
                    beats=beats,
                    metadata={
                        "source_track_id": track.identity.source_track_id,
                        "track_group": track.identity.group_key,
                        "split": split,
                        "title": track.identity.title,
                        "artists": track.identity.artists,
                        "duration_s": track.duration_s,
                        "source_file": source_file,
                        "annotator": annotator,
                    },
                )
                timeline_count += 1
                segment_count += len(segments)
                beat_count += len(beats)
            self.store.register_dataset_source(
                source_id=component_id,
                display_name=component.display_name,
                role=component.role,
                status="imported",
                revision=import_revision,
                license_name=component.license_name,
                root_path=status["source"]["path"],
                metadata={
                    "audio_policy": component.audio_policy,
                    "tracks": len(
                        {track.identity.group_key for track in tracks}
                    ),
                    "timelines": timeline_count,
                    "segments": segment_count,
                    "beats": beat_count,
                    **(
                        {
                            "labels_only": True,
                            "contains_audio_or_links": False,
                            "dataset_revision": ccmusic_package["revision"],
                            "manifest_sha256": ccmusic_package[
                                "manifest_sha256"
                            ],
                            "annotation_fingerprint": ccmusic_package[
                                "annotation_fingerprint"
                            ],
                            "retained_fields": ccmusic_package[
                                "retained_fields"
                            ],
                            "excluded_fields": ccmusic_package[
                                "excluded_fields"
                            ],
                            "source_repair_count": ccmusic_package[
                                "source_repair_count"
                            ],
                            "timeline_discontinuity_count": ccmusic_package[
                                "timeline_discontinuity_count"
                            ],
                        }
                        if ccmusic_package is not None
                        and ccmusic_package["labels_only"]
                        else {}
                    ),
                },
            )
            results.append(
                {
                    "component_id": component_id,
                    "state": "imported",
                    "tracks": len(
                        {track.identity.group_key for track in tracks}
                    ),
                    "timelines": timeline_count,
                    "segments": segment_count,
                    "beats": beat_count,
                    "elapsed_s": time.monotonic() - started,
                }
            )
        return {
            "schema": "lumen_annotation_import_v1",
            "results": results,
            "database": self.store.research_summary(),
        }

    def write_status_report(self, destination: str | Path) -> dict[str, Any]:
        report = self.status()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return report

    def _component_status(
        self, component: ResearchComponent
    ) -> dict[str, Any]:
        destination = self.sources / component.component_id
        revision: str | None = None
        error: str | None = None
        if (destination / ".git").is_dir():
            completed = subprocess.run(
                ["git", "-C", str(destination), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.returncode == 0:
                revision = completed.stdout.strip()
            else:
                error = completed.stderr.strip() or "cannot read revision"
        if not destination.exists():
            source_state = "missing"
        elif revision:
            source_state = "ready"
        else:
            source_state = "invalid"
        environment = (
            self._environment_status(component.environment)
            if component.environment
            else None
        )
        return {
            **asdict(component),
            "source": {
                "path": str(destination),
                "state": source_state,
                "revision": revision,
                "error": error,
            },
            "annotations": self._annotation_status(
                component,
                destination,
                source_state,
            ),
            "environment_status": environment,
        }

    def _annotation_status(
        self,
        component: ResearchComponent,
        destination: Path,
        source_state: str,
    ) -> dict[str, Any]:
        if component.kind == "teacher":
            return {
                "state": "not_applicable",
                "message": "This component is a teacher model, not an annotation dataset.",
            }
        if component.component_id == "ccmusic":
            package = self._ccmusic_authorized_package(destination)
            if not package["files"]:
                access = self._ccmusic_access_details(destination)
                return {
                    "state": access["state"],
                    **access,
                }
            return {
                "state": "ready",
                "annotation_files": len(package["files"]),
                "labels_only": package["labels_only"],
                "contains_audio_or_links": False,
                "revision": package["revision"],
                "manifest_sha256": package["manifest_sha256"],
                "annotation_fingerprint": package[
                    "annotation_fingerprint"
                ],
                "metadata": self._ccmusic_metadata_status(),
                "message": (
                    "Authorized label-only CCMusic annotations are available "
                    "locally; audio and mel fields were not retained."
                ),
            }
        if source_state == "missing":
            return {
                "state": "missing_source",
                "message": "Provision the source checkout before importing annotations.",
            }
        expected_paths = {
            "edm98": destination
            / "src"
            / "edm98"
            / "resources"
            / "dataset.jsonl",
            "harmonix": destination / "dataset" / "segments",
            "salami": destination / "annotations",
        }
        expected = expected_paths.get(component.component_id)
        if expected is None:
            return {
                "state": "not_applicable",
                "message": "No annotation importer is configured for this component.",
            }
        return {
            "state": "ready" if expected.exists() else "missing_annotations",
            "path": str(expected),
            "message": (
                "Annotations are available for import."
                if expected.exists()
                else "The checkout does not contain the expected annotation data."
            ),
        }

    @staticmethod
    def _ccmusic_annotation_files(source_root: Path) -> list[Path]:
        """Find real CCMusic timelines while ignoring README/LFS pointer text."""

        result: list[Path] = []
        row_pattern = re.compile(
            r"^\s*\"?\d+(?:\.\d+)?\"?"
            r"(?:\s*[\t,]\s*|\s+)"
            r"\"?\d+(?:\.\d+)?\"?"
            r"(?:\s*[\t,]\s*|\s+).+"
        )
        for path in sorted(source_root.rglob("*.txt")):
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    if any(
                        row_pattern.match(line)
                        for _, line in zip(range(32), handle)
                    ):
                        result.append(path)
            except (OSError, UnicodeError):
                continue
        return result

    def _ccmusic_authorized_package(
        self, source_root: Path
    ) -> dict[str, Any]:
        authorized_root = source_root / "authorized-annotations"
        manifest_path = authorized_root / "manifest.json"
        if not manifest_path.is_file():
            partial_rows = list(authorized_root.glob("row-*.txt"))
            return {
                "files": (
                    []
                    if partial_rows
                    else self._ccmusic_annotation_files(source_root)
                ),
                "revision": None,
                "time_unit": "centiseconds",
                "labels_only": False,
                "manifest_sha256": None,
                "annotation_fingerprint": None,
                "retained_fields": [],
                "excluded_fields": [],
                "row_segments": {},
                "row_discontinuities": {},
                "source_repair_count": 0,
                "timeline_discontinuity_count": 0,
                "partial_rows": len(partial_rows),
            }
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid CCMusic authorized manifest: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError("CCMusic authorized manifest must be an object")
        required_values = {
            "schema": CCMUSIC_MANIFEST_SCHEMA,
            "dataset_id": CCMUSIC_DATASET_ID,
            "revision": CCMUSIC_DATASET_REVISION,
            "config": "default",
            "split": "train",
            "time_unit": "centiseconds",
            "row_count": CCMUSIC_EXPECTED_ROWS,
            "contains_audio_or_links": False,
            "transport": (
                "DuckDB HTTP range projection of pinned Hugging Face Parquet"
            ),
            "parquet_revision": CCMUSIC_PARQUET_REVISION,
        }
        for field, expected in required_values.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"CCMusic manifest {field!r} must equal {expected!r}"
                )
        retained = manifest.get("retained_fields")
        excluded = manifest.get("excluded_fields")
        required_labels = {
            "label.onset_time",
            "label.offset_time",
            "label.structure",
        }
        if not isinstance(retained, list) or set(retained) != required_labels:
            raise ValueError(
                "CCMusic manifest must retain only the three label fields"
            )
        if not isinstance(excluded, list) or not {"audio", "mel"}.issubset(
            set(excluded)
        ):
            raise ValueError(
                "CCMusic manifest must explicitly exclude audio and mel"
            )
        rows = manifest.get("rows")
        if not isinstance(rows, list) or len(rows) != CCMUSIC_EXPECTED_ROWS:
            raise ValueError("CCMusic manifest must inventory all 300 rows")
        source_repairs = manifest.get("source_repairs")
        source_repair_count = manifest.get("source_repair_count")
        timeline_discontinuities = manifest.get("timeline_discontinuities")
        timeline_discontinuity_count = manifest.get(
            "timeline_discontinuity_count"
        )
        if (
            not isinstance(source_repair_count, int)
            or source_repair_count < 0
            or not isinstance(source_repairs, list)
            or len(source_repairs) != source_repair_count
        ):
            raise ValueError("CCMusic manifest source repair audit is invalid")
        if (
            not isinstance(timeline_discontinuity_count, int)
            or timeline_discontinuity_count < 0
            or not isinstance(timeline_discontinuities, list)
            or len(timeline_discontinuities)
            != timeline_discontinuity_count
        ):
            raise ValueError(
                "CCMusic manifest discontinuity audit is invalid"
            )

        def validate_repair(item: object) -> dict[str, object]:
            if not isinstance(item, dict):
                raise ValueError("CCMusic source repair must be an object")
            try:
                row_index = int(item["row_index"])
                source_index = int(item["source_segment_index"])
                onset = int(item["onset_time"])
                offset = int(item["offset_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("CCMusic source repair is malformed") from error
            if (
                row_index not in range(CCMUSIC_EXPECTED_ROWS)
                or source_index < 0
                or onset < 0
                or offset <= onset
                or item.get("kind") != "embedded_tsv_record"
            ):
                raise ValueError("CCMusic source repair values are invalid")
            return {
                "row_index": row_index,
                "source_segment_index": source_index,
                "kind": "embedded_tsv_record",
                "onset_time": onset,
                "offset_time": offset,
            }

        def validate_discontinuity(item: object) -> dict[str, object]:
            if not isinstance(item, dict):
                raise ValueError(
                    "CCMusic timeline discontinuity must be an object"
                )
            try:
                row_index = int(item["row_index"])
                previous_index = int(item["previous_segment_index"])
                previous_offset = int(item["previous_offset"])
                next_onset = int(item["next_onset"])
                delta = int(item["delta_centiseconds"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "CCMusic timeline discontinuity is malformed"
                ) from error
            kind = "gap" if delta > 0 else "overlap"
            if (
                row_index not in range(CCMUSIC_EXPECTED_ROWS)
                or previous_index < 0
                or previous_offset < 0
                or next_onset < 0
                or delta == 0
                or delta != next_onset - previous_offset
                or item.get("kind") != kind
            ):
                raise ValueError(
                    "CCMusic timeline discontinuity values are invalid"
                )
            return {
                "row_index": row_index,
                "previous_segment_index": previous_index,
                "previous_offset": previous_offset,
                "next_onset": next_onset,
                "delta_centiseconds": delta,
                "kind": kind,
            }

        source_repairs = [validate_repair(item) for item in source_repairs]
        timeline_discontinuities = [
            validate_discontinuity(item)
            for item in timeline_discontinuities
        ]
        projected_bytes = manifest.get("projected_label_compressed_bytes")
        if (
            not isinstance(projected_bytes, int)
            or projected_bytes < 1
            or projected_bytes > 1_000_000
        ):
            raise ValueError(
                "CCMusic projected label columns exceed the fail-closed limit"
            )
        files: list[Path] = []
        row_indexes: set[int] = set()
        filenames: set[str] = set()
        row_segments: dict[str, int] = {}
        row_discontinuities: dict[str, list[dict[str, object]]] = {}
        resolved_root = authorized_root.resolve()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("CCMusic manifest row must be an object")
            row_index = int(row.get("row_index", -1))
            filename = str(row.get("filename", ""))
            if row_index < 0 or row_index in row_indexes:
                raise ValueError("CCMusic manifest row indexes must be unique")
            if not filename or filename in filenames:
                raise ValueError("CCMusic manifest filenames must be unique")
            if filename != f"row-{row_index:06d}.txt":
                raise ValueError(
                    "CCMusic manifest filename must match its row index"
                )
            segment_count = int(row.get("segments", -1))
            if segment_count < 1:
                raise ValueError(
                    f"CCMusic manifest segment count is invalid: {filename}"
                )
            expected_repairs = [
                {key: value for key, value in item.items() if key != "row_index"}
                for item in source_repairs
                if item["row_index"] == row_index
            ]
            expected_discontinuities = [
                {key: value for key, value in item.items() if key != "row_index"}
                for item in timeline_discontinuities
                if item["row_index"] == row_index
            ]
            if (
                row.get("source_repairs") != expected_repairs
                or row.get("embedded_records_repaired")
                != len(expected_repairs)
            ):
                raise ValueError(
                    f"CCMusic row repair audit does not match: {filename}"
                )
            if row.get("timeline_discontinuities") != expected_discontinuities:
                raise ValueError(
                    "CCMusic row discontinuity audit does not match: "
                    + filename
                )
            path = (authorized_root / filename).resolve()
            if path.parent != resolved_root or path.suffix.casefold() != ".txt":
                raise ValueError("CCMusic manifest contains an unsafe filename")
            if not path.is_file():
                raise ValueError(f"CCMusic annotation file is missing: {filename}")
            expected_hash = str(row.get("sha256", "")).casefold()
            actual_hash = self._file_sha256(path)
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise ValueError(f"CCMusic manifest hash is invalid: {filename}")
            if actual_hash != expected_hash:
                raise ValueError(
                    f"CCMusic annotation hash does not match: {filename}"
                )
            row_indexes.add(row_index)
            filenames.add(filename)
            row_segments[filename] = segment_count
            row_discontinuities[filename] = expected_discontinuities
            files.append(path)
        if row_indexes != set(range(CCMUSIC_EXPECTED_ROWS)):
            raise ValueError("CCMusic manifest row indexes must cover 0 through 299")
        files.sort(key=lambda path: path.name)
        fingerprint_payload = {
            **required_values,
            "retained_fields": sorted(retained),
            "excluded_fields": sorted(excluded),
            "source_repairs": source_repairs,
            "timeline_discontinuities": timeline_discontinuities,
            "rows": [
                {
                    "row_index": row["row_index"],
                    "filename": row["filename"],
                    "segments": row["segments"],
                    "embedded_records_repaired": row[
                        "embedded_records_repaired"
                    ],
                    "source_repairs": row["source_repairs"],
                    "timeline_discontinuities": row[
                        "timeline_discontinuities"
                    ],
                    "sha256": row["sha256"],
                }
                for row in sorted(rows, key=lambda item: item["row_index"])
            ],
        }
        return {
            "files": files,
            "revision": CCMUSIC_DATASET_REVISION,
            "time_unit": "centiseconds",
            "labels_only": True,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "annotation_fingerprint": hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "retained_fields": list(retained),
            "excluded_fields": list(excluded),
            "row_segments": row_segments,
            "row_discontinuities": row_discontinuities,
            "source_repair_count": source_repair_count,
            "timeline_discontinuity_count": timeline_discontinuity_count,
            "partial_rows": 0,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _huggingface_credential_status() -> dict[str, Any]:
        if os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        ):
            return {"present": True, "source": "environment"}
        token_paths = []
        if os.environ.get("HF_TOKEN_PATH"):
            token_paths.append(Path(os.environ["HF_TOKEN_PATH"]))
        hf_home = Path(
            os.environ.get(
                "HF_HOME", str(Path.home() / ".cache" / "huggingface")
            )
        )
        token_paths.extend(
            (
                hf_home / "token",
                Path.home() / ".huggingface" / "token",
            )
        )
        if any(path.is_file() and path.stat().st_size > 0 for path in token_paths):
            return {"present": True, "source": "cached_login"}
        return {"present": False, "source": None}

    def _ccmusic_metadata_status(self) -> dict[str, Any]:
        path = (
            self.sources
            / "ccmusic-gated-metadata"
            / "default"
            / "train"
            / "dataset_info.json"
        )
        if not path.is_file():
            return {"state": "missing", "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = payload["features"]["label"]["feature"]
            examples = int(payload["splits"]["train"]["num_examples"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"state": "invalid", "path": str(path)}
        fields = {
            "onset_time": label.get("onset_time", {}).get("dtype"),
            "offset_time": label.get("offset_time", {}).get("dtype"),
            "structure": label.get("structure", {}).get("dtype"),
        }
        expected = {
            "onset_time": "uint32",
            "offset_time": "uint32",
            "structure": "string",
        }
        return {
            "state": "ready" if fields == expected else "invalid",
            "path": str(path),
            "examples": examples,
            "label_fields": fields,
        }

    def _ccmusic_access_details(self, source_root: Path) -> dict[str, Any]:
        credential = self._huggingface_credential_status()
        metadata = self._ccmusic_metadata_status()
        authorized_root = source_root / "authorized-annotations"
        partial_rows = (
            len(list(authorized_root.glob("row-*.txt")))
            if not (authorized_root / "manifest.json").is_file()
            else 0
        )
        if partial_rows:
            state = "authorized_export_incomplete"
            reason_code = "ccmusic_manifest_not_finalized"
            message = (
                f"The CCMusic label-only exporter has written {partial_rows} "
                "of 300 rows but has not finalized its verified manifest."
            )
            required_action = (
                "Allow the active export to finish, or rerun "
                "scripts/setup-research ccmusic-annotations. Lumen will not "
                "import partial rows."
            )
        elif metadata["state"] != "ready":
            state = "awaiting_authorized_metadata"
            reason_code = "ccmusic_gated_metadata_not_ready"
            message = (
                "The local CCMusic source is empty and the audited gated "
                "dataset metadata is not ready."
            )
            required_action = (
                "After rotating/login, run scripts/setup-research "
                "ccmusic-metadata, then ccmusic-annotations."
            )
        elif not credential["present"]:
            state = "awaiting_user_credentials"
            reason_code = "ccmusic_hf_token_missing"
            message = (
                "The CCMusic gate has been accepted, but no standard "
                "Hugging Face login or token is currently available to the "
                "research process."
            )
            required_action = (
                "Rotate the previously exposed credential, then use a "
                "standard Hugging Face cached login or HF_TOKEN and run "
                "scripts/setup-research ccmusic-annotations."
            )
        else:
            state = "ready_to_fetch_authorized_labels"
            reason_code = "ccmusic_authorized_labels_not_fetched"
            message = (
                "Authorization and audited metadata are available; fetch the "
                "label-only CCMusic export before importing."
            )
            required_action = (
                "Run scripts/setup-research ccmusic-annotations. The exporter "
                "must retain labels only and exclude audio and mel."
            )
        return {
            "state": state,
            "reason_code": reason_code,
            "message": message,
            "required_action": required_action,
            "provider_url": (
                "https://huggingface.co/datasets/"
                "ccmusic-database/song_structure"
            ),
            "source_path": str(source_root),
            "authorized_output_path": str(
                source_root / "authorized-annotations"
            ),
            "requires_user_credentials": True,
            "credential_present": credential["present"],
            "credential_source": credential["source"],
            "automatic_download_attempted": False,
            "metadata": metadata,
            "labels_only_fetch": True,
            "retains_audio": False,
        }

    def _environment_status(self, name: str) -> dict[str, Any]:
        environment_root = self.environments / name
        python = environment_root / "bin" / "python"
        component = next(
            (
                item
                for item in RESEARCH_COMPONENTS
                if item.environment == name
            ),
            None,
        )
        required = component.required_modules if component else ()
        if not python.is_file():
            return {
                "name": name,
                "path": str(environment_root),
                "python": None,
                "state": "missing",
                "modules": {module: "unknown" for module in required},
                "error": None,
            }
        probe = (
            "import importlib.util,json,sys;"
            f"mods={list(required)!r};"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'modules':{m:bool(importlib.util.find_spec(m)) for m in mods}}))"
        )
        completed = subprocess.run(
            [str(python), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            return {
                "name": name,
                "path": str(environment_root),
                "python": str(python),
                "state": "error",
                "modules": {},
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
        payload = json.loads(completed.stdout)
        modules = {
            module: ("ready" if present else "missing")
            for module, present in payload["modules"].items()
        }
        return {
            "name": name,
            "path": str(environment_root),
            "python": payload["python"],
            "state": (
                "ready"
                if all(value == "ready" for value in modules.values())
                else "incomplete"
            ),
            "modules": modules,
            "error": None,
        }

    def _register_component(
        self,
        component: ResearchComponent,
        status: dict[str, Any],
    ) -> None:
        if self.store is None:
            return
        source = status["source"]
        self.store.register_dataset_source(
            source_id=component.component_id,
            display_name=component.display_name,
            role=component.role,
            status=source["state"],
            revision=source["revision"],
            license_name=component.license_name,
            root_path=source["path"],
            metadata={
                "kind": component.kind,
                "source_url": component.source_url,
                "checkout": component.checkout,
                "audio_policy": component.audio_policy,
                "environment": component.environment,
            },
        )

    @staticmethod
    def _deterministic_split_map(
        tracks: list[Any],
        splitter: Any,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for split, items in splitter(tracks).items():
            for track in items:
                result[track.identity.group_key] = split
        return result

    @staticmethod
    def _edm98_split_map(source_root: Path) -> dict[str, str]:
        split_root = source_root / "src" / "edm98" / "resources" / "splits"
        result: dict[str, str] = {}
        for filename, split in (
            ("train.txt", "train"),
            ("val.txt", "validation"),
            ("test.txt", "test"),
        ):
            path = split_root / filename
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                track_id = line.strip()
                if track_id:
                    result[f"edm98:{track_id}"] = split
        return result

    @staticmethod
    def _find_python_312() -> str | None:
        candidate = shutil.which("python3.12")
        if candidate:
            return candidate
        pyenv = shutil.which("pyenv")
        if pyenv:
            completed = subprocess.run(
                [pyenv, "which", "python3.12"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
        return None
