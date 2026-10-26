#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph benchmark: code predictor vs codebook-MTP, across dtypes.

OVERVIEW
========
This test compares the residual-codebook decoder of Qwen3-TTS under CUDA
Graphs and eager execution, for float32, float16 and bfloat16:

  * original ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``
  * ``Qwen3TTSCodebookMTPForConditionalGeneration`` loaded from
    ``finetuning/output_notime_dist_1023/mtp_stage1`` (or an override)

Only the two decoder modules are benchmarked.  The talker, text frontend,
speech tokenizer and real sampling are intentionally excluded, so the numbers
isolate the per-frame residual-codebook computation.

WHY A CUSTOM DECODE LOOP
========================
Hugging Face ``GenerationMixin.generate`` is not CUDA-Graph capture safe: its
Python loop performs host-side state checks and allocates state in ways that
invalidate stream capture.  The test therefore captures the equivalent
fixed-shape, greedy decode core:

  * Code predictor: a 15-step loop using ``forward(..., generation_steps=i)``
    with a fresh KV cache.  This is exactly what ``generate`` does for a frame
    with ``do_sample=False`` and ``max_new_tokens=num_code_groups-1``.
  * MTP: ``_decode_frame(..., do_sample=False, num_steps=15)``, the core used
    by the public ``generate``.

The captured graph contains only GPU work.  This is the right comparison for
CUDA Graphs because graph replay removes Python and kernel-launch overhead;
the remaining time is the module's actual GPU computation.

WHAT IS CAPTURED
================
One complete audio frame:
  * input 1: talker hidden state                   ``[B, 1, talker_hidden]``
  * input 2: codebook-0 embedding                  ``[B, 1, embed_dim]``
  * output : residual codebooks 1..15              ``[B, num_code_groups-1]``
  * batch size: ``--batch_size`` (default 1)
  * decoding: greedy argmax, no sampling, no EOS

Inputs are fixed static buffers.  Before replays they are not changed, so the
replay time reflects fixed-shape inference.  Both modules receive identical
random tensors per dtype/seed.

For a frame-local MTP (``use_time_attention=False``, the checkpoint used here),
one captured frame is representative of every frame.  For a time-attention MTP
the graph captures one frame with an empty KV cache (position 0); that is
still a valid per-frame compute measurement, but it does not include a
growing cross-frame cache.

DTYPES
======
``--dtypes fp32,fp16,bf16`` (default: all three).

  * fp32: the 0.6B checkpoint is loaded/upcast to float32.
  * fp16: cast in place; supported on all CUDA GPUs.
  * bf16: requires Ampere or newer.  The test skips it with a warning when
    ``torch.cuda.is_bf16_supported()`` is false.
  * Modules are cast in place between dtypes; graphs are destroyed and the
    CUDA cache is emptied before the next dtype.

METHOD
======
For each dtype:
  1. Cast both decoder modules to the dtype.
  2. Build identical synthetic inputs.
  3. Eager warmup + timed module-core loop.
  4. Warm up CUDA Graph capture on a side stream (required by PyTorch).
  5. Capture one frame, replay once, and verify the argmax codes equal eager.
  6. Time ``--graph_replays`` graph replays.

USAGE
=====
Run from the repository root:

    python test_cudagraph_predictor_vs_mtp.py --device cuda:0

Pick dtypes:

    python test_cudagraph_predictor_vs_mtp.py \
        --dtypes fp32,fp16,bf16

Pick an MTP epoch or an explicit checkpoint:

    python test_cudagraph_predictor_vs_mtp.py --mtp_epoch 2
    python test_cudagraph_predictor_vs_mtp.py \
        --mtp_checkpoint finetuning/output_notime_dist_1023/mtp_stage1/checkpoint-epoch-5/codebook_mtp

Batch or timing controls:

    python test_cudagraph_predictor_vs_mtp.py \
        --batch_size 4 \
        --eager_frames 20 \
        --graph_replays 500

Save JSON:

    python test_cudagraph_predictor_vs_mtp.py --json_out cudagraph.json

INTERPRETING THE OUTPUT
=======================
* ``eager ms/frame``: module core with normal CUDA launches and Python loops.
* ``graph ms/frame``: CUDA Graph replay per frame.  This is the primary
  result for the "CUDA Graphs" comparison.
* ``graph speedup``: eager / graph.  It shows how much launch/Python overhead
  CUDA Graphs removes for that module; it is not a dtype comparison by itself.
* ``MTP speedup``: code-predictor latency / MTP latency.  Compare this within
  the same dtype and execution mode.
* ``verified``: the graph replay produces the same argmax codes as eager.
  A ``no`` value means the capture is not trustworthy for that configuration.

NOTES / LIMITATIONS
===================
* CUDA is required.  The test exits early on CPU-only machines.
* CUDA Graph capture is done with ``torch.no_grad()``; modules are in eval mode.
* ``--attn_implementation flash_attention_2`` is not CUDA Graph capture-safe
  in transformers 4.57: its per-layer ``_is_packed_sequence`` check performs a
  GPU->CPU synchronization at batch 1.  The test detects this backend and
  automatically falls back to ``--eager_only`` so the eager comparison still
  runs.  Use ``--eager_only`` to request the same mode for other backends.
* This is a latency microbenchmark, not an audio-quality test.
* At batch 1 the modules are latency-bound.  fp16/bf16 can be slightly slower
  than fp32 because ``Qwen3TTSRMSNorm`` casts activations to float32 and back
  on every norm, adding conversion work; fp32 needs no conversion.  Dtype
  benefits become visible when the workload becomes compute/bandwidth bound
  (for example with a larger ``--batch_size``).
* The MTP is loaded through ``install_codebook_mtp`` and does not modify any
  source file.
* ``_decode_frame`` is a private MTP method.  It is used deliberately because
  it is the capture-safe core; ``generate`` only wraps it with Python logic.
* The code predictor's 15-step loop uses ``forward`` directly, matching the
  public ``generate`` path for a single frame with greedy decoding.
* Keep other GPU jobs off the selected device.  Use ``--repeat``-style
  repeated replays (``--graph_replays``) for stable numbers.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "finetuning")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from qwen_tts.core.models.codebook_mtp import install_codebook_mtp  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

DEFAULT_MTP_STAGE1_DIR = os.path.join(
    _REPO_ROOT, "finetuning", "output_notime_dist_1023", "mtp_stage1"
)
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = """\
examples:
  # All three dtypes, batch 1, 200 graph replays per module/dtype
  python test_cudagraph_predictor_vs_mtp.py --device cuda:0

  # Only fp16 and bf16, more replays
  python test_cudagraph_predictor_vs_mtp.py \\
      --dtypes fp16,bf16 --graph_replays 500

  # Specific MTP epoch
  python test_cudagraph_predictor_vs_mtp.py --mtp_epoch 2

  # Explicit checkpoint
  python test_cudagraph_predictor_vs_mtp.py \\
      --mtp_checkpoint finetuning/output_notime_dist_1023/mtp_stage1/checkpoint-epoch-5/codebook_mtp

  # Save all metrics as JSON
  python test_cudagraph_predictor_vs_mtp.py --json_out cudagraph.json

what is measured:
  * One complete frame: talker hidden + codebook-0 embedding -> 15 residual
    codebooks, greedy argmax, batch from --batch_size.
  * Eager: normal module core with Python loops and CUDA launches.
  * Graph: torch.cuda.CUDAGraph replay of the same fixed-shape frame.
  * Output is verified against eager argmax codes by default.

notes:
  * CUDA is required.
  * HF GenerationMixin.generate() is not CUDA-Graph capture safe, so the test
    captures the equivalent fixed 15-step greedy decode core instead.
  * bf16 is skipped when the GPU does not support it.
  * This is a module latency test, not an audio-quality test.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the original Qwen3-TTS code predictor and the "
            "codebook-MTP decoder under CUDA Graphs for different dtypes."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model_path",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help=(
            "Base voice-clone model (HF repo ID or local directory). Only used "
            "to obtain the original code predictor and the MTP dimensions. "
            "Default: Qwen/Qwen3-TTS-12Hz-0.6B-Base."
        ),
    )
    model_group.add_argument(
        "--mtp_stage1_dir",
        default=DEFAULT_MTP_STAGE1_DIR,
        help=(
            "Stage-1 output directory. It may be the codebook_mtp checkpoint "
            "itself or a parent containing checkpoint-epoch-*/codebook_mtp. "
            f"Default: {DEFAULT_MTP_STAGE1_DIR}."
        ),
    )
    model_group.add_argument(
        "--mtp_checkpoint",
        default=None,
        help=(
            "Exact codebook_mtp directory containing config.json and "
            "model.safetensors. Overrides --mtp_stage1_dir and --mtp_epoch."
        ),
    )
    model_group.add_argument(
        "--mtp_epoch",
        type=int,
        default=None,
        help=(
            "Use <mtp_stage1_dir>/checkpoint-epoch-N/codebook_mtp. "
            "Default: the highest numeric epoch found."
        ),
    )
    model_group.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, e.g. cuda:0 or cuda:3. CUDA is required.",
    )
    model_group.add_argument(
        "--attn_implementation",
        default="sdpa",
        help=(
            "Attention backend used by both modules: sdpa, eager, or "
            "flash_attention_2 (only if flash-attn is installed)."
        ),
    )

    bench_group = parser.add_argument_group("benchmark")
    bench_group.add_argument(
        "--dtypes",
        default="fp32,fp16,bf16",
        help=(
            "Comma-separated dtypes from fp32,fp16,bf16. "
            "Default: fp32,fp16,bf16."
        ),
    )
    bench_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for the captured frame. Actual voice-clone inference "
            "is batch 1; larger values show scaling. Default: 1."
        ),
    )
    bench_group.add_argument(
        "--eager_frames",
        type=int,
        default=20,
        help=(
            "Timed eager frames per module/dtype. Eager is much slower than "
            "graph replay, so this is smaller than --graph_replays. "
            "Default: 20."
        ),
    )
    bench_group.add_argument(
        "--graph_replays",
        type=int,
        default=200,
        help=(
            "Graph replays per module/dtype. Replay is asynchronous, so the "
            "loop measures throughput of the captured frame. Default: 200."
        ),
    )
    bench_group.add_argument(
        "--warmup_frames",
        type=int,
        default=5,
        help="Untimed eager frames per module/dtype. Default: 5.",
    )
    bench_group.add_argument(
        "--seed",
        type=int,
        default=1234,
        help=(
            "Seed for the synthetic talker-hidden and code-0 embeddings. "
            "The same inputs are used for both modules within a dtype. "
            "Default: 1234."
        ),
    )
    bench_group.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare graph-replay argmax codes with eager codes. Disable only "
            "if you intentionally accept unchecked captures. Default: enabled."
        ),
    )
    bench_group.add_argument(
        "--eager_only",
        action="store_true",
        help=(
            "Skip CUDA Graph capture and report eager metrics only. Useful "
            "for comparing attention backends. The test also falls back to "
            "this mode automatically for flash_attention_2, whose HF wrapper "
            "is not CUDA Graph capture-safe."
        ),
    )
    bench_group.add_argument(
        "--json_out",
        default=None,
        help=(
            "If set, write setup, per-module results, and pairwise "
            "comparisons to this JSON file."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_model_path(model_path: str) -> str:
    """Prefer the local HF snapshot to avoid hub round-trips during loading."""
    if os.path.isdir(os.path.expanduser(model_path)):
        return os.path.abspath(os.path.expanduser(model_path))
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        return model_path


def _is_mtp_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(
        os.path.join(path, "model.safetensors")
    )


def _epoch_of(path: str) -> int:
    match = re.search(r"checkpoint-epoch-(\d+)", path)
    return int(match.group(1)) if match else -1


def resolve_mtp_checkpoint(args: argparse.Namespace) -> str:
    if args.mtp_checkpoint:
        path = os.path.abspath(os.path.expanduser(args.mtp_checkpoint))
        if not _is_mtp_checkpoint(path):
            raise FileNotFoundError(
                f"{path} is not a codebook_mtp checkpoint "
                "(expected config.json and model.safetensors)"
            )
        return path

    root = os.path.abspath(os.path.expanduser(args.mtp_stage1_dir))
    if _is_mtp_checkpoint(root):
        return root

    if args.mtp_epoch is not None:
        path = os.path.join(
            root, f"checkpoint-epoch-{args.mtp_epoch}", "codebook_mtp"
        )
        if not _is_mtp_checkpoint(path):
            raise FileNotFoundError(f"no codebook_mtp checkpoint at {path}")
        return path

    candidates = [
        path
        for path in glob.glob(
            os.path.join(root, "checkpoint-epoch-*", "codebook_mtp")
        )
        if _is_mtp_checkpoint(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint-epoch-*/codebook_mtp under {root}"
        )
    candidates.sort(key=_epoch_of)
    return candidates[-1]


# ---------------------------------------------------------------------------
# Decode cores
# ---------------------------------------------------------------------------


def code_predictor_frame(
    predictor,
    talker_hidden: torch.Tensor,
    code0_embed: torch.Tensor,
    num_code_groups: int,
) -> torch.Tensor:
    """Reproduce one greedy frame of the original code predictor.

    This is the same 15-step loop that ``GenerationMixin.generate`` runs for
    ``do_sample=False`` and ``max_new_tokens=num_code_groups-1``: prefill with
    ``[talker_hidden, code0]``, then feed each argmax code back as a single
    token with a growing KV cache.
    """
    outputs = predictor.forward(
        inputs_embeds=torch.cat((talker_hidden, code0_embed), dim=1),
        use_cache=True,
        generation_steps=0,
    )
    past_key_values = outputs.past_key_values
    code = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    codes = [code]

    for step in range(1, num_code_groups - 1):
        outputs = predictor.forward(
            input_ids=code,
            past_key_values=past_key_values,
            use_cache=True,
            generation_steps=step,
        )
        past_key_values = outputs.past_key_values
        code = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        codes.append(code)

    return torch.cat(codes, dim=1)


def mtp_frame(
    mtp,
    talker_hidden: torch.Tensor,
    code0_embed: torch.Tensor,
    num_code_groups: int,
) -> torch.Tensor:
    """Decode one greedy frame with the MTP core used by ``generate``."""
    if hasattr(mtp, "reset_cache"):
        # Eager single-frame semantics.  For a frame-local MTP this is a
        # no-op; for a time-attention MTP it guarantees frame 0 / empty cache.
        mtp.reset_cache()
    codes, _ = mtp._decode_frame(
        talker_hidden,
        code0_embed,
        target_codes=None,
        do_sample=False,
        num_steps=num_code_groups - 1,
    )
    return torch.stack(codes, dim=1)


# ---------------------------------------------------------------------------
# Timing / capture
# ---------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def time_eager(fn, frames: int, warmup: int) -> float:
    """Return mean eager module-core latency in milliseconds per frame."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync()
        started = time.perf_counter()
        for _ in range(frames):
            fn()
        _sync()
        elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / max(frames, 1)


def capture_graph(fn, warmup: int = 3):
    """Capture one call of ``fn`` and return ``(graph, output)``.

    PyTorch requires warmup on a side stream so that lazy CUDA initialization
    and allocator setup do not happen during capture.
    """
    _sync()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    _sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            output = fn()
    _sync()
    return graph, output


def time_graph(graph, replays: int, warmup: int = 10) -> float:
    """Return mean CUDA Graph replay latency in milliseconds per frame."""
    for _ in range(warmup):
        graph.replay()
    _sync()
    started = time.perf_counter()
    for _ in range(replays):
        graph.replay()
    _sync()
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / max(replays, 1)


def make_inputs(
    talker_hidden_size: int,
    embed_dim: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    talker_hidden = torch.randn(
        batch_size, 1, talker_hidden_size,
        device=device, dtype=dtype, generator=generator,
    )
    code0_embed = torch.randn(
        batch_size, 1, embed_dim,
        device=device, dtype=dtype, generator=generator,
    )
    return talker_hidden, code0_embed


def benchmark_module(
    name: str,
    fn,
    args: argparse.Namespace,
    dtype_name: str,
    reset_state=None,
) -> dict:
    eager_ms = time_eager(fn, args.eager_frames, args.warmup_frames)

    graph_ms = None
    verified = None
    graph_skipped = bool(args.eager_only)
    if not args.eager_only:
        eager_codes = fn().detach().clone()
        graph, graph_codes = capture_graph(fn)
        graph.replay()
        _sync()

        if args.verify:
            verified = bool(torch.equal(eager_codes, graph_codes))

        graph_ms = time_graph(graph, args.graph_replays)

        del graph, graph_codes, eager_codes
    if reset_state is not None:
        # Drop references to graph-private KV-cache tensors before the next
        # dtype cast and before emptying the allocator cache.
        reset_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "module": name,
        "dtype": dtype_name,
        "eager_ms_per_frame": eager_ms,
        "graph_ms_per_frame": graph_ms,
        "graph_speedup": (
            eager_ms / max(graph_ms, 1e-12)
            if graph_ms is not None
            else None
        ),
        "verified": verified,
        "graph_skipped": graph_skipped,
        "batch_size": args.batch_size,
        "eager_frames": args.eager_frames,
        "graph_replays": args.graph_replays,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if abs(value) >= 1000:
        text = f"{value:,.1f}"
    elif abs(value) >= 100:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text + unit


def print_per_module_table(results: dict, dtypes: list[str]) -> None:
    header = (
        f"{'dtype':<8}{'module':<18}{'eager ms/frame':>17}"
        f"{'graph ms/frame':>17}{'graph speedup':>15}{'verified':>10}"
    )
    print("\nPer-module results")
    print(header)
    print("-" * len(header))
    for dtype_name in dtypes:
        for module in ("code_predictor", "mtp"):
            result = results.get(dtype_name, {}).get(module)
            if result is None:
                continue
            print(
                f"{dtype_name:<8}{module:<18}"
                f"{_fmt(result['eager_ms_per_frame']):>17}"
                f"{_fmt(result['graph_ms_per_frame']):>17}"
                f"{_fmt(result['graph_speedup'], 'x'):>15}"
                f"{_fmt(result['verified']):>10}"
            )


def print_pairwise_table(results: dict, dtypes: list[str]) -> None:
    header = (
        f"{'dtype':<8}{'code pred graph':>18}{'MTP graph':>14}"
        f"{'MTP speedup':>14}{'verified':>10}"
        f"{'  |  code pred eager':>20}{'MTP eager':>12}"
        f"{'MTP speedup':>14}"
    )
    print("\nCUDA Graph: code predictor vs MTP (primary result)")
    print(header)
    print("-" * len(header))
    for dtype_name in dtypes:
        pair = results.get(dtype_name)
        if not pair or "code_predictor" not in pair or "mtp" not in pair:
            continue
        base = pair["code_predictor"]
        mtp = pair["mtp"]
        verified = None
        if base["verified"] is not None and mtp["verified"] is not None:
            verified = base["verified"] and mtp["verified"]
        graph_ratio = None
        if (
            base["graph_ms_per_frame"] is not None
            and mtp["graph_ms_per_frame"] is not None
        ):
            graph_ratio = base["graph_ms_per_frame"] / max(
                mtp["graph_ms_per_frame"], 1e-12
            )
        eager_ratio = base["eager_ms_per_frame"] / max(
            mtp["eager_ms_per_frame"], 1e-12
        )
        print(
            f"{dtype_name:<8}"
            f"{_fmt(base['graph_ms_per_frame']):>18}"
            f"{_fmt(mtp['graph_ms_per_frame']):>14}"
            f"{_fmt(graph_ratio, 'x'):>14}"
            f"{_fmt(verified):>10}"
            f"{_fmt(base['eager_ms_per_frame']):>20}"
            f"{_fmt(mtp['eager_ms_per_frame']):>12}"
            f"{_fmt(eager_ratio, 'x'):>14}"
        )


def print_summary(
    args: argparse.Namespace,
    checkpoint: str,
    model_path: str,
    dims: dict,
    results: dict,
    dtypes: list[str],
) -> None:
    print("\n" + "=" * 96)
    print("Setup")
    print("=" * 96)
    print(f"  model:              {args.model_path}")
    print(f"  resolved model:     {model_path}")
    print(f"  MTP checkpoint:     {checkpoint}")
    print(f"  device:             {args.device}")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"  GPU:                {torch.cuda.get_device_name(args.device)}")
    print(f"  torch:              {torch.__version__}")
    print(f"  dtypes:             {', '.join(dtypes)}")
    print(f"  batch_size:         {args.batch_size}")
    print(
        f"  code dimensions:    talker_hidden={dims['talker_hidden']}, "
        f"embed={dims['embed']}, code_groups={dims['num_code_groups']}"
    )
    print(
        f"  layer passes/frame: {dims['code_predictor_layer_passes']} "
        f"(code predictor) -> {dims['mtp_layer_passes']} (MTP)"
    )
    print(
        f"  MTP config:         shared_layers={dims['mtp_shared_layers']}, "
        f"use_time_attention={dims['mtp_use_time_attention']}"
    )
    print(f"  eager frames:       {args.eager_frames}")
    print(f"  graph replays:      {args.graph_replays}")
    print(f"  attention backend:  {args.attn_implementation}")
    print(
        "  execution mode:     "
        + ("eager-only (CUDA Graph skipped)" if args.eager_only else "eager + CUDA Graph")
    )

    print_per_module_table(results, dtypes)
    print_pairwise_table(results, dtypes)

    print(
        "\nNotes\n"
        "  * graph speedup = eager / graph; it measures launch/Python overhead "
        "removed by CUDA Graphs.\n"
        "  * MTP speedup = code predictor / MTP within the same dtype and mode.\n"
        "  * Graph captures one frame of 15 residual codebooks, greedy, with "
        "fixed static inputs.\n"
        "  * For the frame-local MTP checkpoint this matches normal per-frame "
        "inference.  A time-attention MTP\n"
        "    is captured at frame 0 with an empty cache.\n"
        "  * verified=no means the replayed argmax codes differ from eager; "
        "treat that row as invalid.\n"
        "  * In eager-only mode the graph columns are n/a; compare the eager "
        "columns instead."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA Graphs require a CUDA device; no CUDA device is available."
        )
    if not args.device.startswith("cuda"):
        raise ValueError(
            "CUDA Graphs require --device cuda:N; CPU execution is not supported."
        )
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.eager_frames < 1 or args.graph_replays < 1:
        raise ValueError("--eager_frames and --graph_replays must be >= 1")

    if args.attn_implementation.startswith("flash_attention") and not args.eager_only:
        print(
            "Note: HF's flash_attention_2 wrapper is not CUDA Graph "
            "capture-safe in transformers 4.57 (it performs a GPU->CPU sync "
            "in _is_packed_sequence). Switching to eager-only mode. Pass "
            "--eager_only explicitly to silence this notice."
        )
        args.eager_only = True

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    requested = [item.strip().lower() for item in args.dtypes.split(",")]
    dtypes = []
    for name in requested:
        if not name:
            continue
        if name not in DTYPES:
            raise ValueError(
                f"unknown dtype {name!r}; choose from {', '.join(DTYPES)}"
            )
        if name == "bf16" and not torch.cuda.is_bf16_supported():
            print(
                "Skipping bf16: this GPU does not support bfloat16 "
                f"({torch.cuda.get_device_name(args.device)})."
            )
            continue
        if name not in dtypes:
            dtypes.append(name)
    if not dtypes:
        raise ValueError("no usable dtype requested")

    checkpoint = resolve_mtp_checkpoint(args)
    model_path = resolve_model_path(args.model_path)
    print(f"Using MTP checkpoint: {checkpoint}")
    if model_path != args.model_path:
        print(f"Resolved local snapshot: {model_path}")
    print(f"Loading {args.model_path} in fp32 for dtype casting ...")

    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )
    tts.model.eval()
    talker = tts.model.talker
    talker_hidden = int(talker.config.hidden_size)
    embed_dim = int(talker.get_input_embeddings().weight.shape[1])
    num_code_groups = int(talker.config.num_code_groups)

    base_predictor = talker.code_predictor
    print(f"Baseline predictor: {type(base_predictor).__name__}")
    print(f"Installing MTP from {checkpoint} ...")
    mtp = install_codebook_mtp(
        tts.model,
        checkpoint=checkpoint,
        init_from_code_predictor=False,
    )
    mtp.eval()
    base_predictor.eval()

    # The two decoder modules are self-contained now; free the talker and the
    # speech tokenizer to keep graph-capture pools small.
    del tts, talker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    residual_codebooks = num_code_groups - 1
    dims = {
        "talker_hidden": talker_hidden,
        "embed": embed_dim,
        "num_code_groups": num_code_groups,
        "code_predictor_layer_passes": (
            int(base_predictor.config.num_hidden_layers) * residual_codebooks
        ),
        "mtp_layer_passes": (
            int(getattr(mtp.config, "shared_layers", 0)) + residual_codebooks
        ),
        "mtp_shared_layers": int(getattr(mtp.config, "shared_layers", -1)),
        "mtp_use_time_attention": bool(
            getattr(mtp.config, "use_time_attention", True)
        ),
    }

    results: dict = {}
    for dtype_name in dtypes:
        dtype = DTYPES[dtype_name]
        print(f"\n=== dtype {dtype_name} ===")

        base_predictor.to(dtype=dtype)
        mtp.to(dtype=dtype)
        base_predictor.eval()
        mtp.eval()
        if hasattr(mtp, "reset_cache"):
            mtp.reset_cache()

        hidden, code0 = make_inputs(
            talker_hidden,
            embed_dim,
            args.batch_size,
            dtype,
            device,
            args.seed,
        )

        def run_base():
            return code_predictor_frame(
                base_predictor, hidden, code0, num_code_groups
            )

        def run_mtp():
            return mtp_frame(mtp, hidden, code0, num_code_groups)

        # Warm up both before the first timed eager pass and capture.
        with torch.no_grad():
            for _ in range(args.warmup_frames):
                run_base()
                run_mtp()
        _sync()

        print(f"  benchmarking code predictor ({dtype_name}) ...")
        results.setdefault(dtype_name, {})["code_predictor"] = benchmark_module(
            "code_predictor", run_base, args, dtype_name
        )
        print(f"  benchmarking MTP ({dtype_name}) ...")
        results[dtype_name]["mtp"] = benchmark_module(
            "mtp",
            run_mtp,
            args,
            dtype_name,
            reset_state=(
                mtp.reset_cache if hasattr(mtp, "reset_cache") else None
            ),
        )

        printed = results[dtype_name]
        for key, label in (
            ("code_predictor", "code predictor"),
            ("mtp", "MTP"),
        ):
            row = printed[key]
            if row["graph_ms_per_frame"] is None:
                graph_text = "graph=n/a (eager-only)"
            else:
                graph_text = (
                    f"graph={row['graph_ms_per_frame']:.3f} ms/frame, "
                    f"graph speedup={row['graph_speedup']:.2f}x"
                )
            print(
                f"  {label:<15} eager={row['eager_ms_per_frame']:.3f} ms/frame, "
                f"{graph_text}, verified={_fmt(row['verified'])}"
            )

        del hidden, code0, run_base, run_mtp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(args, checkpoint, model_path, dims, results, dtypes)

    if args.json_out:
        payload = {
            "setup": {
                "model_path": args.model_path,
                "model_path_resolved": model_path,
                "mtp_checkpoint": checkpoint,
                "device": args.device,
                "gpu": (
                    torch.cuda.get_device_name(args.device)
                    if args.device.startswith("cuda")
                    else None
                ),
                "torch": torch.__version__,
                "dtypes": dtypes,
                "batch_size": args.batch_size,
                "eager_frames": args.eager_frames,
                "graph_replays": args.graph_replays,
                "warmup_frames": args.warmup_frames,
                "seed": args.seed,
                "attn_implementation": args.attn_implementation,
                "verify": args.verify,
                "eager_only": args.eager_only,
                "dimensions": dims,
            },
            "results": results,
        }
        directory = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
