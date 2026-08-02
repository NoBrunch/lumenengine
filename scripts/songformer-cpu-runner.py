#!/usr/bin/env python3
"""CPU adaptation of SongFormer's official offline audio inference path.

This process is intentionally isolated from Lumen's live runtime. It loads the
provisioned upstream sources and checkpoints without modifying them, emits the
same ``[{label, start, end}]`` shape as EDMFormer, and performs no downloads.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time


SAMPLE_RATE = 24_000
WRAPPED_WINDOW_SECONDS = 30
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_ID = 5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=60,
        help=(
            "CPU-bounded whole-context window. Upstream uses 420 seconds, "
            "which is not viable within this machine's 16 GiB RAM."
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _configure_imports(root: Path) -> tuple[Path, Path]:
    sources = root / "sources"
    songformer = sources / "songformer" / "src" / "SongFormer"
    musicfm = sources / "musicfm"
    for path in (sources, songformer):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    cache = root / "cache" / "huggingface"
    os.environ["MUSICFMPATH"] = str(musicfm)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return songformer, cache


def _require_files(
    audio_path: Path, research_root: Path, songformer: Path
) -> None:
    paths = {
        "audio": audio_path,
        "config": songformer / "configs" / "SongFormer.yaml",
        "head": (
            research_root
            / "models"
            / "songformer"
            / "SongFormer.safetensors"
        ),
        "musicfm_stats": (
            research_root
            / "sources"
            / "edm98"
            / "data"
            / "checkpoints"
            / "msd_stats.json"
        ),
        "musicfm_model": (
            research_root
            / "sources"
            / "edm98"
            / "data"
            / "checkpoints"
            / "pretrained_msd.pt"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("SongFormer runner assets are missing: " + ", ".join(missing))


def _masked_label_tensor(torch, label2id_module):
    mask = torch.ones(128, dtype=torch.bool)
    allowed = label2id_module.DATASET_ID_ALLOWED_LABEL_IDS[DATASET_ID]
    mask[allowed] = False
    return mask.unsqueeze(0).unsqueeze(0)


def _load_models(root: Path, source: Path, cache: Path, torch):
    import numpy as np
    import scipy
    from ema_pytorch import EMA
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    # The official entry point applies this before importing MSAF.
    scipy.inf = np.inf
    config = OmegaConf.load(source / "configs" / "SongFormer.yaml")
    model_type = getattr(importlib.import_module("models.SongFormer"), "Model")
    model = model_type(config)
    state = load_file(
        root / "models" / "songformer" / "SongFormer.safetensors",
        device="cpu",
    )
    ema = EMA(model, include_online_model=False)
    ema.load_state_dict(state)
    model.load_state_dict(ema.ema_model.state_dict(), strict=True)
    model.eval()

    muq = MuQ.from_pretrained(
        "OpenMuQ/MuQ-large-msd-iter",
        cache_dir=str(cache),
        local_files_only=True,
    ).eval()

    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load
    try:
        musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=str(
                root
                / "sources"
                / "edm98"
                / "data"
                / "checkpoints"
                / "msd_stats.json"
            ),
            model_path=str(
                root
                / "sources"
                / "edm98"
                / "data"
                / "checkpoints"
                / "pretrained_msd.pt"
            ),
        ).eval()
    finally:
        torch.load = original_torch_load
    return model, muq, musicfm, config


def _feature_block(model, audio, torch, *, muq: bool):
    if muq:
        return model(audio.unsqueeze(0), output_hidden_states=True).hidden_states[10]
    return model.get_predictions(audio.unsqueeze(0))[1][10]


def _wrapped_features(model, audio, torch, *, muq: bool):
    block_samples = WRAPPED_WINDOW_SECONDS * SAMPLE_RATE
    blocks = []
    for start in range(0, int(audio.shape[-1]), block_samples):
        segment = audio[start : start + block_samples]
        if segment.numel() <= 1024:
            continue
        blocks.append(_feature_block(model, segment, torch, muq=muq))
    if not blocks:
        raise RuntimeError("audio is too short for SongFormer feature extraction")
    return torch.cat(blocks, dim=1)


def _merge_adjacent_segments(segments):
    merged = []
    for segment in segments:
        if (
            merged
            and merged[-1]["label"] == segment["label"]
            and abs(float(merged[-1]["end"]) - float(segment["start"])) < 1e-6
        ):
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def _predict(audio_path: Path, research_root: Path, window_seconds: int, threads: int):
    source, cache = _configure_imports(research_root)
    _require_files(audio_path, research_root, source)

    import librosa
    import numpy as np
    import torch
    from dataset import label2id
    from postprocessing.functional import postprocess_functional_structure

    if not WRAPPED_WINDOW_SECONDS <= window_seconds <= 60:
        raise ValueError(
            f"--window-seconds must be between "
            f"{WRAPPED_WINDOW_SECONDS} and 60 on this CPU"
        )
    torch.manual_seed(0)
    torch.set_num_threads(max(1, threads))
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if waveform.size <= 1024:
        raise ValueError("audio is too short for SongFormer")
    audio = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    duration_s = float(audio.shape[-1]) / SAMPLE_RATE
    model, muq, musicfm, config = _load_models(
        research_root, source, cache, torch
    )
    label_mask = _masked_label_tensor(torch, label2id)
    dataset_ids = torch.tensor([DATASET_ID], dtype=torch.long)
    chunk_samples = window_seconds * SAMPLE_RATE
    function_chunks = []
    boundary_chunks = []

    with torch.inference_mode():
        for chunk_index, start in enumerate(
            range(0, int(audio.shape[-1]), chunk_samples), 1
        ):
            chunk = audio[start : start + chunk_samples]
            if chunk.numel() <= 1024:
                continue
            began = time.monotonic()
            muq_whole = _feature_block(muq, chunk, torch, muq=True)
            muq_wrapped = _wrapped_features(muq, chunk, torch, muq=True)
            musicfm_whole = _feature_block(musicfm, chunk, torch, muq=False)
            musicfm_wrapped = _wrapped_features(
                musicfm, chunk, torch, muq=False
            )
            features = [
                musicfm_wrapped,
                muq_wrapped,
                musicfm_whole,
                muq_whole,
            ]
            lengths = [int(item.shape[1]) for item in features]
            if max(lengths) - min(lengths) > 4:
                raise RuntimeError(
                    "SongFormer feature clocks diverged: "
                    + ", ".join(str(value) for value in lengths)
                )
            common = min(lengths)
            fused = torch.cat(
                [item[:, :common, :] for item in features], dim=-1
            )
            _sections, logits = model.infer(
                input_embeddings=fused,
                dataset_ids=dataset_ids,
                label_id_masks=label_mask,
                with_logits=True,
            )
            function_chunks.append(logits["function_logits"].cpu())
            boundary_chunks.append(logits["boundary_logits"].cpu())
            elapsed = time.monotonic() - began
            print(
                f"SongFormer CPU chunk {chunk_index}: "
                f"{chunk.numel() / SAMPLE_RATE:.1f}s audio in {elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )

    if not function_chunks:
        raise RuntimeError("SongFormer produced no inference chunks")
    logits = {
        "function_logits": torch.cat(function_chunks, dim=1),
        "boundary_logits": torch.cat(boundary_chunks, dim=1),
    }
    boundaries = postprocess_functional_structure(logits, config)
    segments = []
    for index in range(len(boundaries) - 1):
        start_s = max(0.0, float(boundaries[index][0]))
        end_s = min(duration_s, float(boundaries[index + 1][0]))
        if end_s <= start_s:
            continue
        segments.append(
            {
                "label": str(boundaries[index][1]),
                "start": start_s,
                "end": end_s,
            }
        )
    if segments:
        segments[-1]["end"] = duration_s
    if not segments:
        raise RuntimeError("SongFormer returned no positive-duration segments")
    return _merge_adjacent_segments(segments)


def main() -> int:
    args = _arguments()
    root = args.research_root.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    output = args.output.expanduser().resolve()
    segments = _predict(
        audio,
        root,
        window_seconds=args.window_seconds,
        threads=args.threads,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    partial.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
