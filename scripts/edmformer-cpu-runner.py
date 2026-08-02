#!/usr/bin/env python3
"""Run EDMFormer locally with a hard-bounded CPU context window.

The upstream EDM-98 inference pipeline evaluates as much as 420 seconds of
audio in one MuQ/MusicFM transformer call.  That is appropriate for its
original GPU environment, but it can exhaust the target Lumen PC's 16 GiB of
RAM.  This isolated adapter retains the upstream feature extraction, model,
and whole-song post-processing while limiting every foundation-model call to
30--60 seconds.

No model or source checkout is modified.  The context override exists only in
this short-lived worker process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import wave


MIN_WINDOW_SECONDS = 30
MAX_WINDOW_SECONDS = 60
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_THREADS = 4


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--musicfm-stat", required=True, type=Path)
    parser.add_argument("--musicfm-model", required=True, type=Path)
    parser.add_argument("--musicfm-source", required=True, type=Path)
    parser.add_argument("--hf-cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help="Maximum transformer context; must be between 30 and 60 seconds.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU threads available to PyTorch (default: 4).",
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    if not MIN_WINDOW_SECONDS <= args.window_seconds <= MAX_WINDOW_SECONDS:
        raise ValueError(
            "--window-seconds must be between "
            f"{MIN_WINDOW_SECONDS} and {MAX_WINDOW_SECONDS}"
        )
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads must be between 1 and 8")
    required = {
        "audio": args.audio,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "MusicFM statistics": args.musicfm_stat,
        "MusicFM model": args.musicfm_model,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("EDMFormer runner assets are missing: " + ", ".join(missing))
    if not (args.musicfm_source / "model" / "musicfm_25hz.py").is_file():
        raise FileNotFoundError(
            "MusicFM source checkout is missing model/musicfm_25hz.py: "
            f"{args.musicfm_source}"
        )


def _configure_process(args: argparse.Namespace) -> None:
    # Set the native-library limits before importing torch through EDM-98.
    thread_count = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MUSICFMPATH"] = str(args.musicfm_source)


def _audio_duration_seconds(path: Path) -> float:
    """Read the duration of Lumen's verified PCM WAV without decoding it."""

    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            frames = source.getnframes()
    except (wave.Error, EOFError) as error:
        raise ValueError(f"EDMFormer input is not a readable PCM WAV: {path}") from error
    if rate <= 0 or frames <= 0:
        raise ValueError(f"EDMFormer input has no audio frames: {path}")
    return frames / rate


def _merge_and_clamp_segments(
    raw_segments: list[dict[str, object]], duration_s: float
) -> list[dict[str, float | str]]:
    """Make the upstream result contiguous and exact at song boundaries."""

    normalized: list[dict[str, float | str]] = []
    for row in raw_segments:
        label = str(row.get("label") or "").strip()
        if not label:
            raise ValueError("EDMFormer returned a segment without a label")
        start = max(0.0, min(duration_s, float(row["start"])))
        end = max(0.0, min(duration_s, float(row["end"])))
        if end <= start:
            continue
        if normalized:
            # Upstream boundaries are generated from one concatenated logit
            # timeline.  Eliminate only floating-point seams here.
            start = float(normalized[-1]["end"])
        else:
            start = 0.0
        if end <= start:
            continue
        if normalized and normalized[-1]["label"] == label:
            normalized[-1]["end"] = end
        else:
            normalized.append({"label": label, "start": start, "end": end})
    if not normalized:
        raise RuntimeError("EDMFormer returned no positive-duration segments")
    normalized[-1]["end"] = duration_s
    return normalized


def _predict(
    args: argparse.Namespace, upstream=None
) -> list[dict[str, float | str]]:
    # Import only after setting thread and offline-cache limits.
    if upstream is None:
        from edm98.inference import pipeline as upstream

    original_window = int(upstream.TIME_DUR)
    original_numpy_errors = upstream.np.seterr(invalid="ignore", divide="ignore")
    upstream.TIME_DUR = int(args.window_seconds)
    try:
        upstream.torch.set_num_threads(args.threads)
        if hasattr(upstream.torch, "set_num_interop_threads"):
            try:
                upstream.torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits setting this only before parallel work starts.
                pass
        began = time.monotonic()
        print(
            "EDMFormer bounded CPU inference: "
            f"upstream_context={original_window}s, "
            f"maximum_context={args.window_seconds}s, threads={args.threads}",
            file=sys.stderr,
            flush=True,
        )
        raw = upstream.predict_file(
            args.audio,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            musicfm_stat_path=args.musicfm_stat,
            musicfm_model_path=args.musicfm_model,
            device="cpu",
            low_memory=True,
            persistent_models=False,
            hf_cache_dir=args.hf_cache_dir,
            offline=True,
            no_cache=False,
        )
        result = _merge_and_clamp_segments(
            raw,
            _audio_duration_seconds(args.audio),
        )
        print(
            f"EDMFormer bounded CPU inference complete in "
            f"{time.monotonic() - began:.2f}s; segments={len(result)}",
            file=sys.stderr,
            flush=True,
        )
        return result
    finally:
        upstream.TIME_DUR = original_window
        upstream.np.seterr(**original_numpy_errors)


def _write_atomic(output: Path, segments: list[dict[str, float | str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    partial.replace(output)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    args.audio = args.audio.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.musicfm_stat = args.musicfm_stat.expanduser().resolve()
    args.musicfm_model = args.musicfm_model.expanduser().resolve()
    args.musicfm_source = args.musicfm_source.expanduser().resolve()
    args.hf_cache_dir = args.hf_cache_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    _validate_arguments(args)
    _configure_process(args)
    _write_atomic(args.output, _predict(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
