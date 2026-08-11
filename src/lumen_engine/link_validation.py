"""Memory-isolated local qualification for returned student candidates."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterator

from lumen_engine.link import _atomic_json, _student_gate_assessment
from lumen_engine.student import StreamingStructureStudent


_MISSING = object()


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"student example row {line_number} is not an object"
                )
            yield value


def validate_student_import(specification: dict[str, Any]) -> dict[str, Any]:
    """Verify supervision parity and reproduce the held-out local gate."""

    if specification.get("schema") != "lumen.link.student-validation.v1":
        raise ValueError("unsupported student-validation specification")
    original_path = Path(str(specification["original_path"])).resolve()
    prepared_path = Path(str(specification["prepared_path"])).resolve()
    expected_feature_version = str(
        specification["student_audio_feature_version"]
    )
    prepared_rows: list[dict[str, Any]] = []
    pairs = zip_longest(
        _jsonl_rows(original_path),
        _jsonl_rows(prepared_path),
        fillvalue=_MISSING,
    )
    for row_number, (original, prepared) in enumerate(pairs, start=1):
        if original is _MISSING or prepared is _MISSING:
            raise ValueError("prepared student row count changed")
        assert isinstance(original, dict) and isinstance(prepared, dict)
        original_target = {
            key: value
            for key, value in original.items()
            if key not in {"features", "feature_preprocessing_version"}
        }
        prepared_target = {
            key: value
            for key, value in prepared.items()
            if key not in {"features", "feature_preprocessing_version"}
        }
        if original_target != prepared_target:
            raise ValueError(
                "remote feature preparation changed student supervision "
                f"at row {row_number}"
            )
        if (
            prepared.get("feature_preprocessing_version")
            != expected_feature_version
        ):
            raise ValueError(
                "prepared student feature contract is invalid at row "
                f"{row_number}"
            )
        prepared_rows.append(prepared)

    model = StreamingStructureStudent.load(
        Path(str(specification["candidate_path"])).resolve()
    )
    local_gate = _student_gate_assessment(
        model,
        prepared_rows,
        dict(specification["payload"]),
    )
    if sorted(model.approved_axes) != local_gate["approved_axes"]:
        raise ValueError(
            "candidate approved axes fail local held-out validation"
        )
    return local_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specification")
    args = parser.parse_args()
    specification_path = Path(args.specification).resolve()
    specification = json.loads(
        specification_path.read_text(encoding="utf-8")
    )
    result_path = Path(str(specification["result_path"])).resolve()
    _atomic_json(
        result_path,
        {
            "schema": "lumen.link.student-validation.result.v1",
            "local_gate": validate_student_import(specification),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
