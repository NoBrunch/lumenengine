#!/usr/bin/env python3
"""Run one full-song EDMFormer inference in an isolated CPU worker.

EDMFormer's 30-second inputs are local feature-extraction chunks.  They are
combined with a global representation of the complete song (up to the
published 420-second limit) before one structural prediction is decoded.  A
shorter ``TIME_DUR`` changes the model's musical context and is therefore not
a valid memory-saving adaptation.

No model or source checkout is modified.  Heavy foundation models are loaded
sequentially through EDM-98's low-memory path and released with the worker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
import wave


PUBLISHED_CONTEXT_SECONDS = 420
LOCAL_CHUNK_SECONDS = 30
DEFAULT_THREADS = 4


def _sdpa_rotary_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    relative_position_embeddings=None,
    output_attentions=False,
):
    """Equivalent rotary attention without materializing the score square."""

    if output_attentions:
        raise RuntimeError(
            "Lumen's bounded-memory EDMFormer path does not expose attention "
            "probability matrices"
        )
    if self.position_embeddings_type != "rotary":
        raise RuntimeError(
            "Lumen's bounded-memory attention adapter requires rotary "
            "position embeddings"
        )
    if relative_position_embeddings is None:
        raise ValueError("rotary position embeddings are required")

    import torch

    batch_size, sequence_length, _hidden_size = hidden_states.size()
    query_key_states = self._apply_rotary_embedding(
        hidden_states, relative_position_embeddings
    )
    query = self.linear_q(query_key_states).view(
        batch_size, sequence_length, self.num_heads, self.head_size
    )
    key = self.linear_k(query_key_states).view(
        batch_size, sequence_length, self.num_heads, self.head_size
    )
    value = self.linear_v(hidden_states).view(
        batch_size, sequence_length, self.num_heads, self.head_size
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attended = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=float(self.dropout.p) if self.training else 0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).reshape(
        batch_size, sequence_length, self.num_heads * self.head_size
    )
    return self.linear_out(attended), None


def _install_memory_efficient_attention(model) -> None:
    """Install bounded rotary attention on every foundation-model layer."""

    conformer = getattr(getattr(model, "model", model), "conformer", None)
    layers = getattr(conformer, "layers", None)
    if layers is None:
        raise RuntimeError("foundation model does not expose conformer layers")
    for layer in layers:
        attention = layer.self_attn
        if getattr(attention, "_lumen_sdpa_installed", False):
            continue
        if attention.position_embeddings_type != "rotary":
            raise RuntimeError(
                "foundation model does not use the validated rotary attention"
            )
        attention.forward = MethodType(
            _sdpa_rotary_attention_forward, attention
        )
        attention._lumen_sdpa_installed = True


@contextmanager
def _bounded_foundation_model_attention(upstream):
    """Patch only models created during this isolated worker inference."""

    pipeline_class = getattr(upstream, "InferencePipeline", None)
    if pipeline_class is None:
        # Test doubles verify the surrounding inference contract without
        # importing the heavyweight model implementation.
        yield
        return
    original_muq = pipeline_class._create_muq_model
    original_musicfm = pipeline_class._create_musicfm_model

    def create_muq(pipeline):
        model = original_muq(pipeline)
        _install_memory_efficient_attention(model)
        return model

    def create_musicfm(pipeline):
        model = original_musicfm(pipeline)
        _install_memory_efficient_attention(model)
        return model

    pipeline_class._create_muq_model = create_muq
    pipeline_class._create_musicfm_model = create_musicfm
    try:
        yield
    finally:
        pipeline_class._create_muq_model = original_muq
        pipeline_class._create_musicfm_model = original_musicfm


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
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU threads available to PyTorch (default: 4).",
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
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
    duration_s = _audio_duration_seconds(args.audio)
    if duration_s > PUBLISHED_CONTEXT_SECONDS:
        raise ValueError(
            "EDMFormer full-song inference is currently limited to the "
            f"published {PUBLISHED_CONTEXT_SECONDS}-second context; got "
            f"{duration_s:.3f}s. Overlapping long-context inference must "
            "merge frame probabilities before boundary decoding and is not "
            "yet validated."
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
    # If the host reaches a genuine kernel OOM after Lumen's own RSS monitor
    # and swap have both been exhausted, select this disposable offline worker
    # before the graphical session, audio services, or Lumen's control plane.
    try:
        Path("/proc/self/oom_score_adj").write_text("750\n", encoding="ascii")
    except OSError:
        pass


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

    upstream_context = int(upstream.TIME_DUR)
    if upstream_context != PUBLISHED_CONTEXT_SECONDS:
        raise RuntimeError(
            "EDMFormer upstream context does not match the published "
            f"{PUBLISHED_CONTEXT_SECONDS}s configuration: "
            f"{upstream_context}s"
        )
    original_numpy_errors = upstream.np.seterr(invalid="ignore", divide="ignore")
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
            "EDMFormer full-song CPU inference: "
            f"local_chunks={LOCAL_CHUNK_SECONDS}s, "
            f"global_context={PUBLISHED_CONTEXT_SECONDS}s, "
            f"song_duration={_audio_duration_seconds(args.audio):.3f}s, "
            f"threads={args.threads}",
            file=sys.stderr,
            flush=True,
        )
        with _bounded_foundation_model_attention(upstream):
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
            f"EDMFormer full-song CPU inference complete in "
            f"{time.monotonic() - began:.2f}s; segments={len(result)}",
            file=sys.stderr,
            flush=True,
        )
        return result
    finally:
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
