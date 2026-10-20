# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Generate speech with a codebook-MTP Qwen3-TTS checkpoint.

Examples
--------
Use a full model saved by `train_codebook_mtp.py --train_talker`:

    python infer_codebook_mtp.py \
        --model_path output/mtp_stage2/checkpoint-epoch-0 \
        --mode custom_voice --speaker vivian \
        --text "Hello from the codebook MTP decoder." \
        --output_wav out.wav

Use a standalone stage-1 MTP checkpoint on top of the base model:

    python infer_codebook_mtp.py \
        --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --mtp_checkpoint output/mtp_stage1/checkpoint-epoch-2/codebook_mtp \
        --mode voice_clone --ref_audio ref.wav --ref_text "reference text" \
        --text "Hello from the codebook MTP decoder." \
        --output_wav out.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from qwen_tts.core.models.codebook_mtp import install_codebook_mtp  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference with a Drama-3-style codebook-MTP decoder."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Base model or a full checkpoint that already contains an MTP.",
    )
    parser.add_argument(
        "--mtp_checkpoint",
        type=str,
        default=None,
        help="Optional standalone MTP checkpoint from stage 1.",
    )
    parser.add_argument(
        "--mode",
        choices=("custom_voice", "voice_clone", "voice_design"),
        default="custom_voice",
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--language", type=str, default="Auto")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--ref_text", type=str, default=None)
    parser.add_argument(
        "--x_vector_only_mode",
        action="store_true",
        help="Voice clone with speaker embedding only (no ICL reference codes).",
    )
    parser.add_argument("--output_wav", type=str, default="output.wav")
    parser.add_argument(
        "--non_streaming_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="sdpa, eager, or flash_attention_2 if flash-attn is installed.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--subtalker_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _generation_kwargs(args: argparse.Namespace):
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )

    if args.mtp_checkpoint is not None:
        install_codebook_mtp(
            qwen3tts.model,
            checkpoint=args.mtp_checkpoint,
            init_from_code_predictor=False,
        )
        print(f"installed MTP from {args.mtp_checkpoint}")
    else:
        predictor_type = type(qwen3tts.model.talker.code_predictor).__name__
        print(f"using code predictor: {predictor_type}")

    qwen3tts.model.eval()
    if hasattr(qwen3tts.model.talker.code_predictor, "reset_cache"):
        qwen3tts.model.talker.code_predictor.reset_cache()
    generate_kwargs = _generation_kwargs(args)

    started = time.time()
    if args.mode == "custom_voice":
        if not args.speaker:
            raise ValueError("--speaker is required for --mode custom_voice")
        wavs, sample_rate = qwen3tts.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            **generate_kwargs,
        )
    elif args.mode == "voice_design":
        if not args.instruct:
            raise ValueError("--instruct is required for --mode voice_design")
        wavs, sample_rate = qwen3tts.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            **generate_kwargs,
        )
    else:
        if not args.ref_audio:
            raise ValueError("--ref_audio is required for --mode voice_clone")
        wavs, sample_rate = qwen3tts.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only_mode,
            **generate_kwargs,
        )
    elapsed = time.time() - started

    os.makedirs(os.path.dirname(os.path.abspath(args.output_wav)), exist_ok=True)
    if len(wavs) == 1:
        sf.write(args.output_wav, wavs[0], sample_rate)
        outputs = [args.output_wav]
    else:
        root, extension = os.path.splitext(args.output_wav)
        outputs = []
        for index, wav in enumerate(wavs):
            path = f"{root}-{index}{extension or '.wav'}"
            sf.write(path, wav, sample_rate)
            outputs.append(path)

    audio_seconds = sum(len(wav) for wav in wavs) / float(sample_rate)
    print(f"wrote {', '.join(outputs)}")
    print(
        f"audio={audio_seconds:.2f}s elapsed={elapsed:.2f}s "
        f"RTF={elapsed / max(audio_seconds, 1e-8):.3f}"
    )


if __name__ == "__main__":
    main()
