#!/usr/bin/env python3
"""Fetch only authorized CCMusic structure labels, never Arrow/audio payloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET_ID = "ccmusic-database/song_structure"
LOCKED_REVISION = "be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3"
CONFIG = "default"
SPLIT = "train"
EXPECTED_ROWS = 300
DATASET_SERVER = "https://datasets-server.huggingface.co"
PARQUET_REVISION = "6ac1a082ca649072518d9fcd7fbf448a1e844266"
PARQUET_FILES = {
    "0000.parquet": 894_708_606,
    "0001.parquet": 505_166_036,
    "0002.parquet": 915_092_957,
}
DUCKDB_VERSION = "1.5.5"
DUCKDB_PLATFORM = "linux_amd64"
HTTPFS_EXTENSION_SIZE = 21_570_542
HTTPFS_EXTENSION_SHA256 = (
    "887c392b1e49128d11667c81e3698d8b00dfdeb456771acf66d05a0f74f7b7d8"
)
SCHEMA = "lumen.ccmusic.authorized_annotations.v1"


class AccessGateError(RuntimeError):
    """The local credential or provider gate is not ready."""


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("status", "fetch", "verify"),
        help="inspect access, fetch minimal labels, or verify an existing export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="authorized-annotations output directory (required for fetch/verify)",
    )
    parser.add_argument(
        "--duckdb-extension-dir",
        type=Path,
        help="isolated directory for DuckDB's pinned httpfs extension",
    )
    return parser.parse_args()


def _credential():
    from huggingface_hub import get_token

    token = get_token()
    if os.environ.get("HF_TOKEN"):
        source = "HF_TOKEN environment"
    elif token:
        source = "standard cached login"
    else:
        source = "none"
    return token, source


def _dataset_info(token):
    from huggingface_hub import HfApi

    try:
        return HfApi().dataset_info(
            DATASET_ID,
            revision=LOCKED_REVISION,
            files_metadata=True,
            token=token if token else False,
        )
    except Exception as error:
        raise AccessGateError(
            "Unable to inspect the CCMusic repository metadata"
        ) from error


def _validate_token(token: str) -> None:
    from huggingface_hub import HfApi

    try:
        HfApi().whoami(token=token)
    except Exception as error:
        raise AccessGateError(
            "The resolved Hugging Face credential is not valid"
        ) from error


def _assert_locked_revision(token) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    locked = _dataset_info(token)
    if locked.sha != LOCKED_REVISION:
        raise RuntimeError(
            f"CCMusic locked revision resolved to {locked.sha}, "
            f"expected {LOCKED_REVISION}"
        )
    try:
        main = api.dataset_info(
            DATASET_ID, revision="main", token=token or False
        )
    except Exception as error:
        raise AccessGateError(
            "Unable to verify the current CCMusic revision"
        ) from error
    if main.sha != LOCKED_REVISION:
        raise RuntimeError(
            "CCMusic main moved away from Lumen's locked revision; the "
            "repository and pinned Parquet conversion must be re-audited "
            "before importing a different version"
        )


def _dataset_server_json(endpoint: str, token: str, parameters: dict) -> dict:
    query = urlencode(parameters)
    request = Request(
        f"{DATASET_SERVER}/{endpoint}?{query}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "lumen-ccmusic-minimal-fetch/1",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code in {401, 403}:
            raise AccessGateError(
                "CCMusic access was denied. Confirm the dataset agreement is "
                "accepted for the account used by `huggingface-cli login`."
            ) from error
        raise RuntimeError(
            f"CCMusic dataset-server {endpoint} returned HTTP {error.code}"
        ) from error
    except URLError as error:
        raise RuntimeError("CCMusic dataset server is unreachable") from error


def _parquet_inventory(token: str) -> list[str]:
    from huggingface_hub import HfApi

    try:
        converted = HfApi().dataset_info(
            DATASET_ID,
            revision=PARQUET_REVISION,
            files_metadata=True,
            token=token,
        )
    except Exception as error:
        raise AccessGateError(
            "Unable to verify CCMusic's pinned Parquet conversion"
        ) from error
    if converted.sha != PARQUET_REVISION:
        raise RuntimeError("CCMusic Parquet conversion revision mismatch")
    observed = {
        Path(item.rfilename).name: item.size
        for item in converted.siblings
        if item.rfilename.endswith(".parquet")
    }
    if observed != PARQUET_FILES:
        raise RuntimeError(
            "CCMusic Parquet conversion layout changed; refusing projection"
        )
    server = _dataset_server_json(
        "parquet", token, {"dataset": DATASET_ID}
    )
    advertised = {
        item.get("filename"): item.get("size")
        for item in server.get("parquet_files", [])
        if item.get("config") == CONFIG and item.get("split") == SPLIT
    }
    if advertised != PARQUET_FILES:
        raise RuntimeError(
            "CCMusic dataset-server Parquet inventory changed"
        )
    return [
        (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{PARQUET_REVISION}/default/train/{filename}"
        )
        for filename in sorted(PARQUET_FILES)
    ]


def _project_label_rows(token: str, extension_dir: Path):
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError(
            "CCMusic label projection requires the locked duckdb package"
        ) from error

    if duckdb.__version__ != DUCKDB_VERSION:
        raise RuntimeError(
            f"CCMusic requires duckdb {DUCKDB_VERSION}, found "
            f"{duckdb.__version__}"
        )
    urls = _parquet_inventory(token)
    extension_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "SET extension_directory=?", [str(extension_dir.resolve())]
        )
        try:
            connection.execute("LOAD httpfs")
        except duckdb.Error:
            connection.execute("INSTALL httpfs")
            connection.execute("LOAD httpfs")
        extension_path = (
            extension_dir
            / f"v{DUCKDB_VERSION}"
            / DUCKDB_PLATFORM
            / "httpfs.duckdb_extension"
        )
        if (
            not extension_path.is_file()
            or extension_path.stat().st_size != HTTPFS_EXTENSION_SIZE
            or _sha256_bytes(extension_path.read_bytes())
            != HTTPFS_EXTENSION_SHA256
        ):
            raise RuntimeError(
                "DuckDB httpfs extension differs from Lumen's audited artifact"
            )
        # The explicitly temporary secret exists only in this in-memory
        # connection. Parameter binding prevents the credential from entering
        # SQL strings or logs.
        connection.execute(
            "CREATE TEMPORARY SECRET lumen_hf (TYPE http, BEARER_TOKEN ?)",
            [token],
        )
        rows = []
        projected_compressed_bytes = 0
        for url in urls:
            metadata = connection.execute(
                """
                SELECT path_in_schema, sum(total_compressed_size)
                FROM parquet_metadata(?)
                WHERE path_in_schema LIKE 'label,%'
                GROUP BY path_in_schema
                """,
                [url],
            ).fetchall()
            projected_compressed_bytes += sum(
                int(item[1]) for item in metadata
            )
            for (label,) in connection.execute(
                "SELECT label FROM read_parquet(?)", [url]
            ).fetchall():
                row_index = len(rows)
                rows.append({"row_idx": row_index, "row": {"label": label}})
    finally:
        connection.close()
    # The three label columns are tiny. A larger projection means the pinned
    # Parquet schema changed and risks pulling media columns.
    if projected_compressed_bytes > 1_000_000:
        raise RuntimeError(
            "CCMusic projected label columns exceed the 1 MB fail-closed limit"
        )
    return rows, projected_compressed_bytes


def _clean_label(value) -> str:
    label = str(value).strip()
    if any(character in label for character in "\t\r\n"):
        raise ValueError("CCMusic structure label is not a single line")
    if not label:
        raise ValueError("CCMusic segment has an empty structure label")
    return label


_EMBEDDED_RECORD = re.compile(
    r"^\s*(\d+)\s*\t\s*(\d+)\s*\t\s*(.*?)\s*$"
)


def _parse_label_cell_with_repairs(
    cell,
) -> tuple[list[tuple[int, int, str]], list[dict], list[dict]]:
    """Normalize Arrow shapes and recover two known embedded source rows.

    The pinned CCMusic revision contains two structure strings where a missing
    TSV annotation row was embedded after a newline. Only an exact numeric
    tab-separated record is accepted; all other multiline content fails
    closed instead of being flattened into a misleading label.
    """
    if isinstance(cell, dict):
        starts = cell.get("onset_time")
        ends = cell.get("offset_time")
        labels = cell.get("structure")
        if not all(isinstance(item, list) for item in (starts, ends, labels)):
            raise ValueError("CCMusic label object has an unknown shape")
        if not len(starts) == len(ends) == len(labels):
            raise ValueError("CCMusic label columns have different lengths")
        items = [
            {"onset_time": start, "offset_time": end, "structure": label}
            for start, end, label in zip(starts, ends, labels)
        ]
    elif isinstance(cell, list):
        items = cell
    else:
        raise ValueError("CCMusic label cell is not a sequence")

    segments: list[tuple[int, int, str]] = []
    repairs = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"CCMusic segment {index} is not an object")
        try:
            start = int(item["onset_time"])
            end = int(item["offset_time"])
            raw_label = str(item["structure"]).replace("\r\n", "\n").replace(
                "\r", "\n"
            )
            label_lines = raw_label.split("\n")
            label = _clean_label(label_lines[0])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"CCMusic segment {index} is malformed") from error
        if start < 0 or end <= start:
            raise ValueError(f"CCMusic segment {index} has an invalid range")
        segments.append((start, end, label))
        for embedded_line in label_lines[1:]:
            match = _EMBEDDED_RECORD.fullmatch(embedded_line)
            if match is None:
                raise ValueError(
                    f"CCMusic segment {index} has malformed embedded content"
                )
            embedded_start = int(match.group(1))
            embedded_end = int(match.group(2))
            embedded_label = match.group(3).strip()
            if embedded_label.endswith('"'):
                embedded_label = embedded_label[:-1].rstrip()
            embedded_label = _clean_label(embedded_label)
            if embedded_start < 0 or embedded_end <= embedded_start:
                raise ValueError(
                    f"CCMusic segment {index} has an invalid embedded range"
                )
            segments.append(
                (embedded_start, embedded_end, embedded_label)
            )
            repairs.append(
                {
                    "source_segment_index": index,
                    "kind": "embedded_tsv_record",
                    "onset_time": embedded_start,
                    "offset_time": embedded_end,
                }
            )
    if not segments:
        raise ValueError("CCMusic row has no structure labels")
    discontinuities = []
    for segment_index, (previous, current) in enumerate(
        zip(segments, segments[1:])
    ):
        if current[0] != previous[1]:
            discontinuities.append(
                {
                    "previous_segment_index": segment_index,
                    "previous_offset": previous[1],
                    "next_onset": current[0],
                    "delta_centiseconds": current[0] - previous[1],
                    "kind": (
                        "gap" if current[0] > previous[1] else "overlap"
                    ),
                }
            )
    return segments, repairs, discontinuities


def parse_label_cell(cell) -> list[tuple[int, int, str]]:
    """Normalize list-of-struct or struct-of-lists representations."""
    segments, _, _ = _parse_label_cell_with_repairs(cell)
    return segments


def _annotation_text(segments: list[tuple[int, int, str]]) -> str:
    return "".join(
        f"{start}\t{end}\t{label}\n" for start, end, label in segments
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def export_rows(
    rows: list[dict],
    output: Path,
    *,
    transport: str = "Hugging Face dataset-viewer rows API",
    parquet_revision: str | None = None,
    projected_compressed_bytes: int | None = None,
) -> dict:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"CCMusic expected {EXPECTED_ROWS} rows, received {len(rows)}"
        )
    expected_names = {f"row-{index:06d}.txt" for index in range(EXPECTED_ROWS)}
    existing_names = {path.name for path in output.glob("row-*.txt")}
    unexpected = sorted(existing_names - expected_names)
    if unexpected:
        raise RuntimeError(
            "CCMusic output contains unmanaged row files; refusing to delete: "
            + ", ".join(unexpected[:5])
        )

    manifest_rows = []
    source_repairs = []
    timeline_discontinuities = []
    for expected_index, wrapper in enumerate(rows):
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
            raise ValueError(f"CCMusic row {expected_index} has an unknown wrapper")
        row_index = int(wrapper.get("row_idx", expected_index))
        if row_index != expected_index:
            raise ValueError(
                f"CCMusic row order changed: expected {expected_index}, got {row_index}"
            )
        segments, repairs, discontinuities = _parse_label_cell_with_repairs(
            wrapper["row"].get("label")
        )
        source_repairs.extend(
            {"row_index": row_index, **item} for item in repairs
        )
        timeline_discontinuities.extend(
            {"row_index": row_index, **item} for item in discontinuities
        )
        filename = f"row-{row_index:06d}.txt"
        encoded = _annotation_text(segments).encode("utf-8")
        _atomic_text(output / filename, encoded.decode("utf-8"))
        manifest_rows.append(
            {
                "row_index": row_index,
                "filename": filename,
                "segments": len(segments),
                "embedded_records_repaired": len(repairs),
                "source_repairs": repairs,
                "timeline_discontinuities": discontinuities,
                "sha256": _sha256_bytes(encoded),
            }
        )

    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "revision": LOCKED_REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "time_unit": "centiseconds",
        "row_count": EXPECTED_ROWS,
        "retained_fields": [
            "label.onset_time",
            "label.offset_time",
            "label.structure",
        ],
        "excluded_fields": ["audio", "mel"],
        "contains_audio_or_links": False,
        "source_repair_count": len(source_repairs),
        "source_repairs": source_repairs,
        "timeline_discontinuity_count": len(timeline_discontinuities),
        "timeline_discontinuities": timeline_discontinuities,
        "transport": transport,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": manifest_rows,
    }
    if parquet_revision is not None:
        manifest["parquet_revision"] = parquet_revision
    if projected_compressed_bytes is not None:
        manifest["projected_label_compressed_bytes"] = int(
            projected_compressed_bytes
        )
    _atomic_text(
        output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def verify_export(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"CCMusic manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "revision": LOCKED_REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "time_unit": "centiseconds",
        "row_count": EXPECTED_ROWS,
        "contains_audio_or_links": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"CCMusic manifest {key} is {manifest.get(key)!r}, expected {value!r}"
            )
    if manifest.get("excluded_fields") != ["audio", "mel"]:
        raise RuntimeError("CCMusic manifest does not exclude audio and mel")
    repair_count = manifest.get("source_repair_count")
    source_repairs = manifest.get("source_repairs")
    discontinuity_count = manifest.get("timeline_discontinuity_count")
    timeline_discontinuities = manifest.get("timeline_discontinuities")
    if not isinstance(repair_count, int) or repair_count < 0:
        raise RuntimeError("CCMusic embedded repair audit is invalid")
    if not isinstance(source_repairs, list) or len(source_repairs) != repair_count:
        raise RuntimeError("CCMusic source repair inventory is invalid")
    if not isinstance(discontinuity_count, int) or discontinuity_count < 0:
        raise RuntimeError("CCMusic discontinuity audit is invalid")
    if (
        not isinstance(timeline_discontinuities, list)
        or len(timeline_discontinuities) != discontinuity_count
    ):
        raise RuntimeError("CCMusic discontinuity inventory is invalid")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("CCMusic manifest row inventory is incomplete")
    segments_total = 0
    observed_repairs = []
    observed_discontinuities = []
    for expected_index, item in enumerate(rows):
        filename = f"row-{expected_index:06d}.txt"
        if item.get("row_index") != expected_index or item.get("filename") != filename:
            raise RuntimeError("CCMusic manifest row order is invalid")
        path = output / filename
        value = path.read_bytes()
        if _sha256_bytes(value) != item.get("sha256"):
            raise RuntimeError(f"CCMusic annotation hash mismatch: {filename}")
        segments, _, discontinuities = _parse_label_cell_with_repairs(
            [
                {
                    "onset_time": parts[0],
                    "offset_time": parts[1],
                    "structure": parts[2],
                }
                for line in value.decode("utf-8").splitlines()
                if line.strip()
                for parts in [line.split("\t", 2)]
            ]
        )
        if len(segments) != item.get("segments"):
            raise RuntimeError(f"CCMusic segment count mismatch: {filename}")
        if item.get("timeline_discontinuities") != discontinuities:
            raise RuntimeError(
                f"CCMusic discontinuity audit mismatch: {filename}"
            )
        row_repairs = item.get("source_repairs")
        repaired = item.get("embedded_records_repaired")
        if (
            not isinstance(row_repairs, list)
            or not isinstance(repaired, int)
            or repaired != len(row_repairs)
        ):
            raise RuntimeError(f"CCMusic repair audit is invalid: {filename}")
        observed_repairs.extend(
            {"row_index": expected_index, **repair} for repair in row_repairs
        )
        observed_discontinuities.extend(
            {"row_index": expected_index, **entry}
            for entry in discontinuities
        )
        segments_total += len(segments)
    if observed_repairs != source_repairs:
        raise RuntimeError("CCMusic embedded repair total does not match")
    if observed_discontinuities != timeline_discontinuities:
        raise RuntimeError("CCMusic discontinuity total does not match")
    return {"rows": len(rows), "segments": segments_total}


def status() -> int:
    token, source = _credential()
    info = _dataset_info(token)
    print(f"CCMusic credential source: {source}")
    print(f"CCMusic locked revision: {info.sha}")
    print(f"CCMusic gate type: {info.gated}")
    print(f"CCMusic credential resolved: {bool(token)}")
    if not token:
        print(
            "CCMusic access: login required with the research environment's "
            "`huggingface-cli login` command"
        )
        return 0
    try:
        _validate_token(token)
        _assert_locked_revision(token)
        page = _dataset_server_json(
            "first-rows",
            token,
            {"dataset": DATASET_ID, "config": CONFIG, "split": SPLIT},
        )
        if not page.get("rows"):
            raise RuntimeError("CCMusic gate opened but returned no rows")
        _parquet_inventory(token)
    except AccessGateError as error:
        print(f"CCMusic access: denied ({error})")
        return 0
    print("CCMusic access: authorized; minimal annotation export is available")
    return 0


def fetch(output: Path, extension_dir: Path) -> int:
    token, source = _credential()
    if not token:
        raise AccessGateError(
            "No Hugging Face credential was found. Run `huggingface-cli login` "
            "inside the EDMFormer research environment or set HF_TOKEN."
        )
    print(f"Using CCMusic credential source: {source}")
    _validate_token(token)
    _assert_locked_revision(token)
    rows, projected_bytes = _project_label_rows(token, extension_dir)
    print(f"Projected CCMusic label rows: {len(rows)}/{EXPECTED_ROWS}")
    _assert_locked_revision(token)
    manifest = export_rows(
        rows,
        output,
        transport="DuckDB HTTP range projection of pinned Hugging Face Parquet",
        parquet_revision=PARQUET_REVISION,
        projected_compressed_bytes=projected_bytes,
    )
    verified = verify_export(output)
    print(
        f"CCMusic minimal export ready: {verified['rows']} rows, "
        f"{verified['segments']} segments, revision {manifest['revision']}"
    )
    print("CCMusic audio/mel/Arrow payloads downloaded: no")
    return 0


def main() -> int:
    args = _arguments()
    if args.command in {"fetch", "verify"} and args.output is None:
        raise SystemExit("--output is required for fetch and verify")
    if args.command == "fetch" and args.duckdb_extension_dir is None:
        raise SystemExit("--duckdb-extension-dir is required for fetch")
    try:
        if args.command == "status":
            return status()
        output = args.output.expanduser().resolve()
        if args.command == "fetch":
            return fetch(
                output, args.duckdb_extension_dir.expanduser().resolve()
            )
        verified = verify_export(output)
        print(
            f"CCMusic export verified: {verified['rows']} rows, "
            f"{verified['segments']} segments"
        )
        return 0
    except (AccessGateError, RuntimeError, ValueError) as error:
        print(f"CCMusic: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
