#!/usr/bin/env python3
"""Fixed, isolated Lumen Link student-training executor.

The node service creates this specification from a validated signed manifest.
This program accepts no command or plugin field and never opens the operator's
database.  It materializes causal features from immutable WAV objects, trains
with Lumen's canonical trainer/gate, and leaves checksummed artifacts for the
node spool to publish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumen_engine.link import _atomic_json, _hash_file
from lumen_engine.memory import SongMemoryStore
from lumen_engine.offline import (
    OfflineResearchWorker,
    _load_jsonl,
    _refresh_student_audio_features,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specification")
    args = parser.parse_args()
    spec_path = Path(args.specification).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "lumen.link.student-runner.v1":
        raise ValueError("unsupported fixed student-runner specification")
    work = Path(str(spec["work_root"])).resolve()
    progress_path = Path(str(spec["progress_path"])).resolve()
    work.mkdir(parents=True, exist_ok=True)

    stage_history: list[str] = []

    def progress(stage: str, **details: object) -> None:
        if not stage_history or stage_history[-1] != stage:
            stage_history.append(stage)
        _atomic_json(
            progress_path,
            {"stage": stage, "stage_history": stage_history, **details},
        )

    store = SongMemoryStore(work / "isolated.sqlite3")
    jobs: list[dict[str, object]] = []
    for item in spec["recordings"]:
        path = Path(str(item["audio_path"])).resolve()
        digest = str(item["sha256"])
        if _hash_file(path) != digest:
            raise ValueError("immutable student audio checksum changed")
        jobs.append(
            {
                "payload": {
                    "recording_id": str(item["recording_id"]),
                    "audio_path": str(path),
                    "content_sha256": digest,
                }
            }
        )
    examples_path = Path(str(spec["examples_path"])).resolve()
    if _hash_file(examples_path) != str(spec["examples_sha256"]):
        raise ValueError("immutable student examples checksum changed")
    examples = _load_jsonl(examples_path)
    feature_workers = max(1, int(spec.get("feature_workers") or 1))
    progress(
        "student_feature_preparation",
        progress=0.0,
        recordings_complete=0,
        recordings_total=len(spec["recordings"]),
        feature_workers=feature_workers,
    )

    def feature_progress(details: dict[str, object]) -> None:
        progress(
            str(details.get("stage") or "student_feature_preparation"),
            **{
                name: value
                for name, value in details.items()
                if name != "stage"
            },
            feature_workers=feature_workers,
        )

    feature_report = _refresh_student_audio_features(
        examples,
        jobs=jobs,
        research_root=Path(
            str(spec.get("feature_cache_root") or work / "research")
        ).resolve(),
        maximum_workers=feature_workers,
        progress_callback=feature_progress,
    )
    prepared = Path(str(spec["prepared_examples_path"])).resolve()
    prepared_partial = prepared.with_suffix(prepared.suffix + ".partial")
    with prepared_partial.open("w", encoding="utf-8") as output:
        for row in examples:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    prepared_partial.replace(prepared)

    payload = {
        **dict(spec["training"]),
        "examples_path": str(prepared),
        "examples_sha256": _hash_file(prepared),
        "output_path": str(Path(str(spec["output_path"])).resolve()),
        "refresh_audio_features": False,
        "require_activation_gate": True,
        "remote_feature_preprocessing": feature_report,
    }
    worker = OfflineResearchWorker(store, research_root=work / "research")
    result = worker._train_student(
        {"payload": payload}, progress_callback=progress
    )
    evaluation_path = Path(str(result["evaluation_path"])).resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["feature_preprocessing"] = feature_report
    _atomic_json(evaluation_path, evaluation)
    result["feature_preprocessing"] = feature_report
    result_path = Path(str(spec["result_path"])).resolve()
    _atomic_json(
        result_path,
        {
            "schema": "lumen.link.student-runner.result.v1",
            "result": result,
            "feature_preprocessing": feature_report,
        },
    )
    progress("student_artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
