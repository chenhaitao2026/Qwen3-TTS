#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the original code predictor against the codebook-MTP decoder.

OVERVIEW
========
This is a standalone benchmark.  It only imports the public
``install_codebook_mtp`` helper and monkey-patches timing wrappers on the
loaded objects; no file in ``qwen_tts/`` or ``finetuning/`` is modified.

The script loads the 0.6B Base voice-clone model once, then runs two passes:

1. Baseline: original ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``.
2. MTP: the same talker with ``finetuning/output_notime_dist_1023/mtp_stage1``
   installed in place via ``install_codebook_mtp``.

Because the talker weights, reference prompt, target text and sampling
parameters are identical, the difference is attributable to the residual
codebook decoder.

WHAT IS MEASURED
================
A. End-to-end voice clone
   ``Qwen3TTSModel.generate_voice_clone`` with the same ``VoiceClonePromptItem``.
   Reported per model (mean over ``--repeat`` timed runs):
     * full latency (wrapper)          include code generation + vocoder
     * code generation (model.generate)
     * vocoder + I/O                   full - model.generate
     * RTF                             full latency / generated audio seconds
     * talker prefill                   one prompt-prefill forward
     * decode stage                    sum of all per-frame talker forwards
     * talker decode per frame
     * code predictor / MTP per frame  measured inside the real generation loop
     * code predictor / MTP total
     * talker non-predictor per frame  talker decode - predictor time
     * predictor share of decode

   The two models may stop at slightly different frame counts because the
   sampled residual codes differ.  The script prints ``generated frames`` and
   ``audio seconds`` so the user can see whether the end-to-end totals are
   comparable.  The per-frame rows are the fair latency comparison.

B. Standalone module latency
   Both modules receive identical synthetic inputs:
     ``inputs_embeds = cat([talker_hidden, code0_embed], dim=1)``
   with shape ``[batch=1, 2, hidden]``, ``max_new_tokens = num_code_groups-1``
   and greedy decoding.  The loop runs ``--module_frames`` frames.  Reported:
     * per frame / total / frames per second
     * parameter count (the MTP is larger: 19 layers instead of 5)
   For a frame-local MTP (``use_time_attention=False``), the script can also
   time ``forward_sequence`` over all frames in one teacher-forced pass.  This
   is a compute reference only; it is not on the autoregressive talker path.

REQUIREMENTS
============
* The Qwen3-TTS package must be importable (run from the repo root).
* ``Qwen/Qwen3-TTS-12Hz-0.6B-Base`` weights, tokenizer and speech tokenizer.
  The script prefers the local Hugging Face snapshot to avoid hub round-trips.
* A reference wav for voice cloning.  The default points at LJSpeech
  ``LJ001-0001.wav``; override with ``--ref_audio``.  ``--ref_text`` is
  required for ICL mode (the default).
* CUDA is recommended.  CPU works but is much slower.

QUICK START
===========
Run from the repository root:

    python bench_mtp_vs_code_predictor.py --device cuda:0

This uses the highest MTP epoch under
``finetuning/output_notime_dist_1023/mtp_stage1`` and performs:
1 warmup + 3 timed voice-clone generations per model, then 20 warmup + 200
timed frames for the standalone module benchmark.

COMMON COMMANDS
===============
Specify an epoch:

    python bench_mtp_vs_code_predictor.py --mtp_epoch 2

Specify any MTP checkpoint directory that contains ``config.json`` and
``model.safetensors``:

    python bench_mtp_vs_code_predictor.py \
        --mtp_checkpoint finetuning/output_notime_dist_1023/mtp_stage1/checkpoint-epoch-5/codebook_mtp

Use a different reference audio / target text:

    python bench_mtp_vs_code_predictor.py \
        --ref_audio /path/to/ref.wav \
        --ref_text "Reference transcript" \
        --text "Text to synthesize." \
        --language Auto

Faster smoke test (about one generation per model, short text, few frames):

    python bench_mtp_vs_code_predictor.py \
        --text "Hello world." \
        --max_new_tokens 32 \
        --warmup 0 --repeat 1 \
        --module_frames 10 --module_warmup 2 \
        --no-sequence_bench

Only one half of the benchmark:

    python bench_mtp_vs_code_predictor.py --skip_generate   # module only
    python bench_mtp_vs_code_predictor.py --skip_module     # end-to-end only

Save machine-readable results and the first waveform of each model:

    python bench_mtp_vs_code_predictor.py \
        --json_out bench.json \
        --save_wav_dir bench_wavs

Use another GPU or dtype:

    python bench_mtp_vs_code_predictor.py --device cuda:3 --dtype bf16

MTP CHECKPOINT SELECTION
========================
``--mtp_stage1_dir`` points at the stage-1 output directory.  The script
accepts either:
  * the ``codebook_mtp`` checkpoint directory itself, or
  * a directory containing ``checkpoint-epoch-*/codebook_mtp``.
Default selection is the highest numeric epoch.  ``--mtp_epoch N`` and
``--mtp_checkpoint PATH`` override it.  The chosen path is printed at startup
and written to JSON.

INTERPRETING THE OUTPUT
=======================
* ``speedup`` is ``baseline / MTP`` for latency rows and
  ``MTP / baseline`` for throughput rows (frames/s).
* ``change`` is always ``(MTP / baseline - 1) * 100``.  Negative means the MTP
  is faster/lower; positive means it is slower/higher.
* Contextual rows (audio seconds, generated frames, parameter count,
  predictor share) intentionally have no speedup column.
* Do not compare end-to-end totals from runs with very different frame counts;
  use the per-frame rows instead.

NOTES / TROUBLESHOOTING
=======================
* ``flash-attn is not installed`` is harmless; the default is SDPA for both
  models.  Pass ``--attn_implementation flash_attention_2`` only if installed.
* At batch=1, HF's ``flash_attention_2`` path can be *slower* than SDPA for
  this autoregressive workload.  Its per-layer ``_is_packed_sequence`` check
  builds a GPU scalar and the caller converts it to a Python bool, adding a
  GPU->CPU sync on every attention call.  The talker has 28 layers and the
  original code predictor has 5 layers x 15 codebooks = 75 attention calls per
  frame, so the eager decode loop accumulates thousands of syncs.  SDPA has no
  such check and may already dispatch to a FlashAttention-style kernel.
  FlashAttention's advantage appears at long prefill / larger batches, or with
  a KV-cache-specific decode kernel.
* The MTP decoder (``MTPAttention``) calls ``F.scaled_dot_product_attention``
  directly and ignores ``config._attn_implementation``.  Therefore
  ``--attn_implementation`` only changes the talker and the original code
  predictor; it does not change the MTP residual decoder.
* If a Hugging Face repo ID is not cached, the script falls back to the ID and
  transformers downloads it.  Set ``HF_HUB_OFFLINE=1`` only if the snapshot is
  already complete.
* The MTP checkpoint used here (``output_notime_dist_1023``) is frame-local
  (``use_time_attention=False``).  With a time-attention checkpoint the
  predictor keeps a KV cache across frames; the standalone benchmark then
  includes that growing cache, which matches real inference.
* The standalone MTP module has more parameters than the original 5-layer
  predictor but performs far fewer per-frame layer passes (19 vs 75), so it is
  still faster.  Parameter count alone is not the latency predictor.
* All timings are wall-clock and include Python/HF generation overhead.  Keep
  other GPU jobs off the selected device for stable results.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import sys
import time
from contextlib import contextmanager

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
DEFAULT_REF_AUDIO = "/Downloads/Datasets/LJSpeech-1.1/wavs/LJ001-0001.wav"
DEFAULT_REF_TEXT = (
    "Printing, in the only sense with which we are at present concerned, "
    "differs from most if not from all the arts and crafts represented in "
    "the Exhibition"
)
DEFAULT_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EPILOG = """\
examples:
  # Default: highest MTP epoch, 1 warmup + 3 timed runs, 200 module frames
  python bench_mtp_vs_code_predictor.py --device cuda:0

  # Pick a specific stage-1 epoch (or an explicit checkpoint directory)
  python bench_mtp_vs_code_predictor.py --mtp_epoch 2
  python bench_mtp_vs_code_predictor.py \\
      --mtp_checkpoint finetuning/output_notime_dist_1023/mtp_stage1/checkpoint-epoch-5/codebook_mtp

  # Different reference audio / target text
  python bench_mtp_vs_code_predictor.py \\
      --ref_audio /path/to/ref.wav \\
      --ref_text "Reference transcript" \\
      --text "Text to synthesize."

  # Quick smoke test
  python bench_mtp_vs_code_predictor.py \\
      --text "Hello world." --max_new_tokens 32 \\
      --warmup 0 --repeat 1 --module_frames 10 --module_warmup 2 \\
      --no-sequence_bench

  # Only one half of the benchmark
  python bench_mtp_vs_code_predictor.py --skip_generate
  python bench_mtp_vs_code_predictor.py --skip_module

  # Save JSON and the first waveform of each model
  python bench_mtp_vs_code_predictor.py \\
      --json_out bench.json --save_wav_dir bench_wavs

what is timed:
  * End-to-end: generate_voice_clone with the same reference prompt, text,
    seed and sampling settings for the original code predictor and the MTP.
    Reported: full latency, model.generate, vocoder/I-O, RTF, talker prefill,
    talker decode per frame, code predictor/MTP per frame, and predictor share.
  * Standalone: identical synthetic inputs, batch=1, greedy, --module_frames.
    Reported: per-frame, total, frames/s, parameter count.  For a frame-local
    MTP, forward_sequence is also timed as a compute reference only.

notes:
  * The script changes nothing outside itself; the MTP is installed through
    install_codebook_mtp().
  * The two models can stop at different frame counts.  Compare per-frame and
    per-second rows, not just the end-to-end total.
  * speedup = baseline/MTP for latency and MTP/baseline for throughput.
    change = (MTP/baseline - 1) * 100.
  * The flash-attn warning is harmless; the default attention backend is SDPA.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original Qwen3-TTS 0.6B code predictor with a "
            "codebook-MTP decoder on end-to-end voice cloning and standalone "
            "module latency."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model_path",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help=(
            "Base voice-clone model (HF repo ID or local directory). The MTP "
            "is installed into this model in place. Default: "
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base."
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
        default=None,
        help="cuda:0, cuda:1, ... or cpu. Default: cuda:0 if available.",
    )
    model_group.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Model dtype. bf16 is the normal speed-benchmark setting.",
    )
    model_group.add_argument(
        "--attn_implementation",
        default="sdpa",
        help=(
            "Attention backend used for both models: sdpa, eager, or "
            "flash_attention_2 (only if flash-attn is installed)."
        ),
    )

    data_group = parser.add_argument_group("voice clone input")
    data_group.add_argument(
        "--ref_audio",
        default=DEFAULT_REF_AUDIO,
        help=(
            "Reference wav used to build the voice-clone prompt. "
            f"Default: {DEFAULT_REF_AUDIO}."
        ),
    )
    data_group.add_argument(
        "--ref_text",
        default=DEFAULT_REF_TEXT,
        help=(
            "Transcript of --ref_audio. Required in ICL mode "
            "(default, i.e. --x_vector_only_mode not set)."
        ),
    )
    data_group.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Target text to synthesize.",
    )
    data_group.add_argument(
        "--language",
        default="Auto",
        help="Target language passed to generate_voice_clone. Default: Auto.",
    )
    data_group.add_argument(
        "--x_vector_only_mode",
        action="store_true",
        help=(
            "Use only the speaker embedding; no reference codes or ref_text. "
            "Useful when the reference transcript is unavailable."
        ),
    )
    data_group.add_argument(
        "--non_streaming_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Passed to generate_voice_clone. Default: true (non-streaming "
            "text input)."
        ),
    )

    gen_group = parser.add_argument_group("generation")
    gen_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help=(
            "Upper bound on generated codec frames. The 12 Hz codec uses "
            "12.5 frames per audio second, so 2048 is roughly 164 s. "
            "Default: 2048."
        ),
    )
    gen_group.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample talker codebook 0. Default: true.",
    )
    gen_group.add_argument(
        "--top_k", type=int, default=50, help="Talker top-k. Default: 50."
    )
    gen_group.add_argument(
        "--top_p", type=float, default=1.0, help="Talker top-p. Default: 1.0."
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Talker temperature. Default: 0.9.",
    )
    gen_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Talker repetition penalty. Default: 1.05.",
    )
    gen_group.add_argument(
        "--subtalker_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample residual codebooks in the code predictor/MTP. "
            "Default: true."
        ),
    )
    gen_group.add_argument(
        "--subtalker_top_k",
        type=int,
        default=50,
        help="Residual-codebook top-k. Default: 50.",
    )
    gen_group.add_argument(
        "--subtalker_top_p",
        type=float,
        default=1.0,
        help="Residual-codebook top-p. Default: 1.0.",
    )
    gen_group.add_argument(
        "--subtalker_temperature",
        type=float,
        default=0.9,
        help="Residual-codebook temperature. Default: 0.9.",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=1234,
        help=(
            "Base RNG seed. Timed run i uses seed+i for both models, so their "
            "sampling streams are comparable. Default: 1234."
        ),
    )

    bench_group = parser.add_argument_group("benchmark")
    bench_group.add_argument(
        "--warmup",
        type=int,
        default=1,
        help=(
            "Untimed voice-clone generations per model before measuring. "
            "Use at least 1 to exclude CUDA lazy-init and allocator warmup. "
            "Default: 1."
        ),
    )
    bench_group.add_argument(
        "--repeat",
        type=int,
        default=3,
        help=(
            "Timed voice-clone generations per model; results are averaged. "
            "Default: 3."
        ),
    )
    bench_group.add_argument(
        "--module_frames",
        type=int,
        default=200,
        help=(
            "Number of sequential frames in the standalone predictor/MTP "
            "benchmark. Default: 200 (16 s of audio at 12.5 Hz)."
        ),
    )
    bench_group.add_argument(
        "--module_warmup",
        type=int,
        default=20,
        help=(
            "Untimed frames for the standalone predictor/MTP benchmark. "
            "Default: 20."
        ),
    )
    bench_group.add_argument(
        "--skip_generate",
        action="store_true",
        help=(
            "Skip voice-clone generation; run only the standalone module "
            "benchmark."
        ),
    )
    bench_group.add_argument(
        "--skip_module",
        action="store_true",
        help="Skip the standalone module benchmark; run only generation.",
    )
    bench_group.add_argument(
        "--sequence_bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For a frame-local MTP, also time forward_sequence over all "
            "--module_frames in one teacher-forced pass. This is a compute "
            "reference, not the autoregressive talker path. Default: enabled."
        ),
    )
    bench_group.add_argument(
        "--save_wav_dir",
        default=None,
        help=(
            "If set, write the first baseline and MTP waveform to this "
            "directory for listening checks."
        ),
    )
    bench_group.add_argument(
        "--json_out",
        default=None,
        help=(
            "If set, write setup, mean end-to-end metrics, standalone module "
            "metrics and forward_sequence metrics to this JSON file."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MTP checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_model_path(model_path: str) -> str:
    """Prefer the local HF snapshot, avoiding hub round-trips while timing.

    ``AutoProcessor.from_pretrained(..., fix_mistral_regex=True)`` performs a
    model_info() call for repo IDs, which is slow on a flaky mirror.  A local
    snapshot path skips that.  If the model is not cached, the original ID is
    returned and transformers downloads it normally.
    """
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
# Timing helpers
# ---------------------------------------------------------------------------


class CallTimer:
    """Time a bound method on an object without editing the source class."""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self._records = []
        self._patched = []

    def wrap(self, obj, name: str) -> "CallTimer":
        original = getattr(obj, name)

        if self.cuda:
            @functools.wraps(original)
            def timed(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    end.record()
                    self._records.append((start, end))
        else:
            @functools.wraps(original)
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self._records.append((start, time.perf_counter()))

        setattr(obj, name, timed)
        self._patched.append((obj, name, original))
        return self

    def restore(self) -> None:
        for obj, name, original in reversed(self._patched):
            setattr(obj, name, original)
        self._patched.clear()

    def reset(self) -> None:
        self._records.clear()

    def elapsed_ms(self) -> list[float]:
        if not self._records:
            return []
        if self.cuda:
            torch.cuda.synchronize()
            return [
                start.elapsed_time(end) for start, end in self._records
            ]
        return [(end - start) * 1000.0 for start, end in self._records]


@contextmanager
def timed_bound_method(obj, name: str):
    """Wall-clock time one call of a bound method, including CUDA work."""
    original = getattr(obj, name)
    box = {"seconds": None}

    @functools.wraps(original)
    def timed(*args, **kwargs):
        _sync()
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _sync()
            box["seconds"] = time.perf_counter() - start

    setattr(obj, name, timed)
    try:
        yield box
    finally:
        setattr(obj, name, original)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# End-to-end generation benchmark
# ---------------------------------------------------------------------------


def generation_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=args.subtalker_do_sample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        non_streaming_mode=args.non_streaming_mode,
    )


def _run_generation(
    tts: Qwen3TTSModel,
    prompt,
    args: argparse.Namespace,
    seed: int,
    instrument: bool,
    cuda: bool,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    talker_timer = None
    predictor_timer = None
    if instrument:
        predictor = tts.model.talker.code_predictor
        talker_timer = CallTimer(cuda).wrap(tts.model.talker, "forward")
        predictor_timer = CallTimer(cuda).wrap(predictor, "generate")

    _sync()
    wall_start = time.perf_counter()
    try:
        with timed_bound_method(tts.model, "generate") as generate_box:
            wavs, sample_rate = tts.generate_voice_clone(
                text=args.text,
                language=args.language,
                voice_clone_prompt=prompt,
                **generation_kwargs(args),
            )
    finally:
        if instrument:
            predictor_timer.restore()
            talker_timer.restore()
    _sync()
    total_seconds = time.perf_counter() - wall_start

    talker_ms = talker_timer.elapsed_ms() if instrument else []
    predictor_ms = predictor_timer.elapsed_ms() if instrument else []

    # The first talker call is the prompt prefill; every later call is one
    # audio frame.  The predictor is called once per audio frame.
    talker_prefill_ms = talker_ms[0] if len(talker_ms) > 1 else None
    talker_decode_ms = talker_ms[1:] if len(talker_ms) > 1 else []
    frames = len(predictor_ms)

    other_per_frame = None
    if frames and len(talker_decode_ms) == len(predictor_ms):
        other_per_frame = _mean(
            [
                talker - predictor
                for talker, predictor in zip(talker_decode_ms, predictor_ms)
            ]
        )

    audio_seconds = sum(len(wav) for wav in wavs) / float(sample_rate)
    model_generate_seconds = generate_box["seconds"]
    return {
        "wavs": wavs,
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "frames": frames,
        "total_seconds": total_seconds,
        "model_generate_seconds": model_generate_seconds,
        "vocoder_other_seconds": (
            total_seconds - model_generate_seconds
            if model_generate_seconds is not None
            else None
        ),
        "rtf": total_seconds / max(audio_seconds, 1e-8),
        "prefill_ms": talker_prefill_ms,
        "talker_decode_total_ms": sum(talker_decode_ms),
        "talker_decode_per_frame_ms": _mean(talker_decode_ms),
        "predictor_total_ms": sum(predictor_ms),
        "predictor_per_frame_ms": _mean(predictor_ms),
        "talker_other_per_frame_ms": other_per_frame,
        "predictor_share": (
            sum(predictor_ms) / sum(talker_decode_ms)
            if predictor_ms and talker_decode_ms
            else None
        ),
    }


def benchmark_generation(
    tts: Qwen3TTSModel,
    prompt,
    args: argparse.Namespace,
    label: str,
    cuda: bool,
) -> dict:
    for warmup in range(args.warmup):
        _run_generation(
            tts, prompt, args, args.seed + 10_000 + warmup,
            instrument=False, cuda=cuda,
        )
        print(f"  [{label}] warmup {warmup + 1}/{args.warmup} done")

    runs = []
    for index in range(args.repeat):
        run = _run_generation(
            tts, prompt, args, args.seed + index, instrument=True, cuda=cuda
        )
        runs.append(run)
        print(
            f"  [{label}] run {index + 1}/{args.repeat}: "
            f"{run['total_seconds']:.2f}s, "
            f"frames={run['frames']}, "
            f"audio={run['audio_seconds']:.2f}s, "
            f"predictor={run['predictor_per_frame_ms']:.3f}ms/frame"
        )

    numeric_keys = [
        "audio_seconds",
        "frames",
        "total_seconds",
        "model_generate_seconds",
        "vocoder_other_seconds",
        "rtf",
        "prefill_ms",
        "talker_decode_total_ms",
        "talker_decode_per_frame_ms",
        "predictor_total_ms",
        "predictor_per_frame_ms",
        "talker_other_per_frame_ms",
        "predictor_share",
    ]
    return {
        "label": label,
        "runs": runs,
        "mean": {key: _mean([run[key] for run in runs]) for key in numeric_keys},
    }


# ---------------------------------------------------------------------------
# Standalone module benchmark
# ---------------------------------------------------------------------------


def _make_module_inputs(tts: Qwen3TTSModel, batch_size: int):
    talker = tts.model.talker
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype
    hidden_size = int(talker.config.hidden_size)
    vocab_size = int(talker.config.vocab_size)

    generator = torch.Generator(device=device).manual_seed(1234)
    talker_hidden = torch.randn(
        batch_size, 1, hidden_size, device=device, dtype=dtype,
        generator=generator,
    )
    code0 = torch.randint(
        0, vocab_size, (batch_size, 1), device=device, generator=generator
    )
    code0_embed = talker.get_input_embeddings()(code0)
    return talker_hidden, code0_embed


def benchmark_module(
    predictor,
    talker_hidden: torch.Tensor,
    code0_embed: torch.Tensor,
    num_code_groups: int,
    frames: int,
    warmup: int,
    cuda: bool,
) -> dict:
    inputs_embeds = torch.cat((talker_hidden, code0_embed), dim=1)
    kwargs = dict(
        inputs_embeds=inputs_embeds,
        max_new_tokens=num_code_groups - 1,
        do_sample=False,
    )

    if hasattr(predictor, "reset_cache"):
        predictor.reset_cache()

    with torch.no_grad():
        for _ in range(warmup):
            predictor.generate(**kwargs)

    if hasattr(predictor, "reset_cache"):
        predictor.reset_cache()

    _sync()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(frames):
            predictor.generate(**kwargs)
    _sync()
    elapsed = time.perf_counter() - started

    return {
        "frames": frames,
        "total_ms": elapsed * 1000.0,
        "per_frame_ms": elapsed * 1000.0 / max(frames, 1),
        "frames_per_second": frames / max(elapsed, 1e-8),
        "parameters_m": sum(p.numel() for p in predictor.parameters()) / 1e6,
    }


def benchmark_mtp_sequence(
    mtp,
    talker_hidden: torch.Tensor,
    code0_embed: torch.Tensor,
    num_code_groups: int,
    frames: int,
    warmup: int,
    repeats: int = 5,
) -> dict:
    """Time the frame-parallel teacher-forced pass of the MTP.

    Only meaningful for a frame-local MTP (use_time_attention=False): frames
    do not depend on each other, so flattening time into one batch is exact.

    ``talker_hidden`` is ``[1, frames, H]`` and ``code0_embed`` is
    ``[1, frames, E]``; the embeddings of codebook 0 must come from the
    talker's embedding table, which the MTP does not own.
    """
    device = talker_hidden.device
    vocab_size = int(mtp.config.vocab_size)
    generator = torch.Generator(device=device).manual_seed(2024)
    codec_ids = torch.randint(
        0, vocab_size, (1, frames, num_code_groups), device=device,
        generator=generator,
    )
    codec_ids[..., 0] = 0

    audio_mask = torch.ones(1, frames, dtype=torch.bool, device=device)
    kwargs = dict(
        talker_hidden=talker_hidden,
        codec_ids=codec_ids,
        audio_mask=audio_mask,
        code0_embeds=code0_embed,
        attention_mask=audio_mask,
    )

    with torch.no_grad():
        for _ in range(warmup):
            mtp.forward_sequence(**kwargs)

    times = []
    for _ in range(repeats):
        _sync()
        started = time.perf_counter()
        with torch.no_grad():
            mtp.forward_sequence(**kwargs)
        _sync()
        times.append(time.perf_counter() - started)
    times.sort()
    median = times[len(times) // 2]
    return {
        "frames": frames,
        "median_ms": median * 1000.0,
        "per_frame_ms": median * 1000.0 / max(frames, 1),
        "frames_per_second": frames / max(median, 1e-8),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_value(value, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "%":
        return f"{value * 100:.1f}%"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.3f}"


def print_table(title: str, rows, baseline: str, mtp: str) -> None:
    """Print one comparison table.

    ``lower_is_better`` is True for latency, False for throughput, and None
    for contextual rows that should not get a speedup column.
    """
    print(f"\n{title}")
    header = (
        f"{'metric':<38}{baseline:>14}{mtp:>14}"
        f"{'speedup':>10}{'change':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, base_value, mtp_value, unit, lower_is_better in rows:
        speedup = ""
        change = ""
        if (
            lower_is_better is not None
            and base_value is not None
            and mtp_value is not None
            and base_value != 0
        ):
            speedup_value = (
                base_value / mtp_value
                if lower_is_better
                else mtp_value / base_value
            )
            speedup = f"{speedup_value:.2f}x"
            change = f"{(mtp_value / base_value - 1.0) * 100:+.1f}%"
        print(
            f"{name:<38}"
            f"{_format_value(base_value, unit):>14}"
            f"{_format_value(mtp_value, unit):>14}"
            f"{speedup:>10}{change:>10}"
        )


def print_summary(
    args: argparse.Namespace,
    baseline_gen: dict | None,
    mtp_gen: dict | None,
    baseline_module: dict | None,
    mtp_module: dict | None,
    mtp_sequence: dict | None,
    mtp_info: dict,
) -> None:
    print("\n" + "=" * 92)
    print("Setup")
    print("=" * 92)
    print(f"  model:            {args.model_path}")
    print(f"  MTP checkpoint:   {mtp_info['checkpoint']}")
    print(f"  shared_layers:    {mtp_info['shared_layers']}")
    print(f"  use_time_attention: {mtp_info['use_time_attention']}")
    print(f"  num_code_groups:  {mtp_info['num_code_groups']}")
    print(
        f"  layer passes/frame: {mtp_info['baseline_layer_passes_per_frame']} "
        f"(code predictor) -> {mtp_info['mtp_layer_passes_per_frame']} (MTP)"
    )
    print(f"  device/dtype:     {args.device} / {args.dtype}")
    if torch.cuda.is_available() and args.device != "cpu":
        print(f"  GPU:              {torch.cuda.get_device_name(args.device)}")
    print(f"  torch:            {torch.__version__}")

    if baseline_gen is not None and mtp_gen is not None:
        base = baseline_gen["mean"]
        mtp_mean = mtp_gen["mean"]
        print_table(
            "End-to-end voice-clone generation (mean over runs)",
            [
                ("audio seconds", base["audio_seconds"], mtp_mean["audio_seconds"], "s", None),
                ("generated frames", base["frames"], mtp_mean["frames"], "", None),
                ("full latency (wrapper)", base["total_seconds"], mtp_mean["total_seconds"], "s", True),
                ("  code generation (model.generate)", base["model_generate_seconds"], mtp_mean["model_generate_seconds"], "s", True),
                ("  vocoder + I/O", base["vocoder_other_seconds"], mtp_mean["vocoder_other_seconds"], "s", True),
                ("RTF (full latency / audio)", base["rtf"], mtp_mean["rtf"], "", True),
                ("talker prefill", base["prefill_ms"], mtp_mean["prefill_ms"], "ms", True),
                ("decode stage (sum talker steps)", base["talker_decode_total_ms"], mtp_mean["talker_decode_total_ms"], "ms", True),
                ("talker decode per frame", base["talker_decode_per_frame_ms"], mtp_mean["talker_decode_per_frame_ms"], "ms", True),
                ("code predictor / MTP per frame", base["predictor_per_frame_ms"], mtp_mean["predictor_per_frame_ms"], "ms", True),
                ("code predictor / MTP total", base["predictor_total_ms"], mtp_mean["predictor_total_ms"], "ms", True),
                ("talker non-predictor per frame", base["talker_other_per_frame_ms"], mtp_mean["talker_other_per_frame_ms"], "ms", True),
                ("predictor share of decode", base["predictor_share"], mtp_mean["predictor_share"], "%", None),
            ],
            "baseline",
            "MTP",
        )
        print(
            "  Note: end-to-end totals include the speech tokenizer and can "
            "differ slightly because the two models may stop at different "
            "frame counts.  Per-frame rows are the fair comparison."
        )

    if baseline_module is not None and mtp_module is not None:
        print_table(
            f"Standalone module benchmark ({baseline_module['frames']} frames, "
            "batch=1, greedy)",
            [
                ("per frame", baseline_module["per_frame_ms"], mtp_module["per_frame_ms"], "ms", True),
                ("total", baseline_module["total_ms"], mtp_module["total_ms"], "ms", True),
                ("frames per second", baseline_module["frames_per_second"], mtp_module["frames_per_second"], "", False),
                ("module parameters", baseline_module["parameters_m"], mtp_module["parameters_m"], "M", None),
            ],
            "code pred",
            "MTP",
        )

    if mtp_sequence is not None:
        print(
            f"\nMTP frame-parallel forward_sequence "
            f"({mtp_sequence['frames']} frames in one teacher-forced pass): "
            f"{mtp_sequence['median_ms']:.2f} ms total, "
            f"{mtp_sequence['per_frame_ms']:.4f} ms/frame, "
            f"{mtp_sequence['frames_per_second']:.1f} frames/s"
        )
        print(
            "  (Not on the talker's autoregressive path; the talker does not "
            "know future frames or codes. Shown as a compute reference.)"
        )


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not cuda:
        raise RuntimeError("CUDA device requested but CUDA is not available")

    if not os.path.isfile(args.ref_audio):
        raise FileNotFoundError(
            f"reference audio not found: {args.ref_audio}. "
            "Pass --ref_audio /path/to/ref.wav."
        )

    checkpoint = resolve_mtp_checkpoint(args)
    print(f"Using MTP checkpoint: {checkpoint}")

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_path = resolve_model_path(args.model_path)
    if model_path != args.model_path:
        print(f"Resolved local snapshot: {model_path}")
    print(f"Loading {args.model_path} on {args.device} ({args.dtype}) ...")
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )
    tts.model.eval()
    if hasattr(tts.model, "speech_tokenizer") and hasattr(
        tts.model.speech_tokenizer, "eval"
    ):
        tts.model.speech_tokenizer.eval()

    talker = tts.model.talker
    baseline_predictor = talker.code_predictor
    residual_codebooks = int(talker.config.num_code_groups) - 1
    baseline_layer_passes = (
        int(baseline_predictor.config.num_hidden_layers) * residual_codebooks
    )
    print(f"Baseline predictor: {type(baseline_predictor).__name__}")

    # Build the voice-clone prompt once; both models use the identical prompt.
    print("Building voice-clone prompt (outside the timed region) ...")
    prompt = tts.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only_mode,
    )

    baseline_gen = None
    baseline_module = None
    mtp_gen = None
    mtp_module = None
    mtp_sequence = None

    if not args.skip_generate:
        print("\nBenchmarking original code predictor (end-to-end) ...")
        baseline_gen = benchmark_generation(
            tts, prompt, args, "baseline", cuda
        )

    if not args.skip_module:
        print("\nBenchmarking original code predictor (standalone) ...")
        talker_hidden, code0_embed = _make_module_inputs(tts, batch_size=1)
        baseline_module = benchmark_module(
            baseline_predictor,
            talker_hidden,
            code0_embed,
            int(talker.config.num_code_groups),
            frames=args.module_frames,
            warmup=args.module_warmup,
            cuda=cuda,
        )
        print(
            f"  {baseline_module['per_frame_ms']:.3f} ms/frame, "
            f"{baseline_module['frames_per_second']:.1f} frames/s, "
            f"{baseline_module['parameters_m']:.2f}M params"
        )

    # Install the MTP in place.  Talker weights are untouched.
    print(f"\nInstalling MTP from {checkpoint} ...")
    mtp = install_codebook_mtp(
        tts.model,
        checkpoint=checkpoint,
        init_from_code_predictor=False,
    )
    mtp.eval()
    mtp.reset_cache()

    mtp_info = {
        "checkpoint": checkpoint,
        "shared_layers": int(getattr(mtp.config, "shared_layers", -1)),
        "use_time_attention": bool(
            getattr(mtp.config, "use_time_attention", True)
        ),
        "num_code_groups": int(getattr(mtp.config, "num_code_groups", -1)),
        "parameters_m": sum(p.numel() for p in mtp.parameters()) / 1e6,
        "baseline_layer_passes_per_frame": baseline_layer_passes,
        "mtp_layer_passes_per_frame": (
            int(getattr(mtp.config, "shared_layers", 0)) + residual_codebooks
        ),
    }
    print(
        f"MTP: shared_layers={mtp_info['shared_layers']}, "
        f"use_time_attention={mtp_info['use_time_attention']}, "
        f"params={mtp_info['parameters_m']:.2f}M"
    )
    print(
        f"Layer passes per frame: {mtp_info['baseline_layer_passes_per_frame']} "
        f"(code predictor) -> {mtp_info['mtp_layer_passes_per_frame']} (MTP)"
    )

    if not args.skip_generate:
        print("\nBenchmarking MTP (end-to-end) ...")
        mtp_gen = benchmark_generation(tts, prompt, args, "MTP", cuda)

    if not args.skip_module:
        print("\nBenchmarking MTP (standalone) ...")
        talker_hidden, code0_embed = _make_module_inputs(tts, batch_size=1)
        mtp_module = benchmark_module(
            mtp,
            talker_hidden,
            code0_embed,
            int(talker.config.num_code_groups),
            frames=args.module_frames,
            warmup=args.module_warmup,
            cuda=cuda,
        )
        print(
            f"  {mtp_module['per_frame_ms']:.3f} ms/frame, "
            f"{mtp_module['frames_per_second']:.1f} frames/s, "
            f"{mtp_module['parameters_m']:.2f}M params"
        )

        if args.sequence_bench and not mtp_info["use_time_attention"]:
            print("\nBenchmarking MTP frame-parallel forward_sequence ...")
            seq_hidden = torch.randn(
                1, args.module_frames, talker_hidden.shape[-1],
                device=talker_hidden.device, dtype=talker_hidden.dtype,
            )
            seq_codec = torch.randint(
                0, int(mtp.config.vocab_size),
                (1, args.module_frames, int(talker.config.num_code_groups)),
                device=talker_hidden.device,
            )
            seq_codec[..., 0] = 0
            seq_code0 = talker.get_input_embeddings()(seq_codec[..., 0])
            mtp_sequence = benchmark_mtp_sequence(
                mtp,
                talker_hidden=seq_hidden,
                code0_embed=seq_code0,
                num_code_groups=int(talker.config.num_code_groups),
                frames=args.module_frames,
                warmup=2,
            )
            del seq_hidden, seq_codec, seq_code0

    if args.save_wav_dir:
        import soundfile as sf

        os.makedirs(args.save_wav_dir, exist_ok=True)
        for tag, result in (("baseline", baseline_gen), ("mtp", mtp_gen)):
            if result is None:
                continue
            for index, wav in enumerate(result["runs"][0]["wavs"]):
                path = os.path.join(args.save_wav_dir, f"{tag}-{index}.wav")
                sf.write(path, wav, result["runs"][0]["sample_rate"])
                print(f"Wrote {path}")

    print_summary(
        args,
        baseline_gen,
        mtp_gen,
        baseline_module,
        mtp_module,
        mtp_sequence,
        mtp_info,
    )

    if args.json_out:
        payload = {
            "setup": {
                "model_path": args.model_path,
                "model_path_resolved": model_path,
                "device": args.device,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "seed": args.seed,
                "text": args.text,
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "x_vector_only_mode": args.x_vector_only_mode,
                "generation": generation_kwargs(args),
                "mtp": mtp_info,
            },
            "baseline": {
                "generation": baseline_gen["mean"] if baseline_gen else None,
                "module": baseline_module,
            },
            "mtp": {
                "generation": mtp_gen["mean"] if mtp_gen else None,
                "module": mtp_module,
                "forward_sequence": mtp_sequence,
            },
        }
        # Waveforms are not JSON serializable.
        save_json(args.json_out, payload)


if __name__ == "__main__":
    main()
