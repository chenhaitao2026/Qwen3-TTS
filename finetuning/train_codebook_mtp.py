# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Train the Drama-3-style codebook-MTP decoder for Qwen3-TTS.

The talker still predicts codebook 0.  This script trains the module that
predicts codebooks 1..15 from the talker hidden state and the real audio codes.

Recommended recipe
------------------
Stage 1 (default): freeze the talker and train only the MTP on real WAV codes.

    python train_codebook_mtp.py \
        --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --train_jsonl data/train_with_codes.jsonl \
        --output_model_path output/mtp_stage1 \
        --lr 2e-4 --num_epochs 3

Stage 2 (optional): unfreeze the talker and jointly fine-tune with a small
talker learning rate.  Keep `--mtp_loss_weight` around 0.5; do not make the MTP
loss dominate the codebook-0 loss.

    python train_codebook_mtp.py \
        --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --init_mtp_checkpoint output/mtp_stage1/checkpoint-epoch-2/codebook_mtp \
        --train_jsonl data/train_with_codes.jsonl \
        --output_model_path output/mtp_stage2 \
        --train_talker --talker_lr 5e-6 --lr 2e-5 \
        --lr_scheduler cosine --warmup_ratio 0.03 \
        --mtp_loss_weight 0.5 --num_epochs 1

`--lr_scheduler constant` (the default) is the historical flat-LR behaviour;
`cosine` or `linear` add a linear warmup of `--warmup_ratio` total steps and
decay to `--min_lr_ratio` of the initial LR.  The schedule advances once per
optimizer step and follows `--max_steps`.

The JSONL must be the output of `finetuning/prepare_data.py`, i.e. it must
contain real `audio_codes` produced from the target WAV.  See
`finetuning/README_codebook_mtp.md` for the full discussion.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset import TTSDataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from qwen_tts.core.models.codebook_mtp import (  # noqa: E402
    install_codebook_mtp,
    save_codebook_mtp,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a codebook-MTP decoder on real Qwen3-TTS audio codes."
    )
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Base Qwen3-TTS checkpoint (or a directory containing one).",
    )
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="output/codebook_mtp",
    )
    parser.add_argument(
        "--init_mtp_checkpoint",
        type=str,
        default=None,
        help="Optional MTP checkpoint from a previous stage.",
    )

    # Data / optimization
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4, help="MTP learning rate.")
    parser.add_argument(
        "--talker_lr",
        type=float,
        default=5e-6,
        help="Talker learning rate for the optional joint stage.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after N optimizer steps per epoch (0 = no limit).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=("constant", "cosine", "linear"),
        default="constant",
        help=(
            "Learning-rate schedule, stepped once per optimizer step.  "
            "'constant' keeps the historical behaviour (no scheduler)."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Fraction of total optimizer steps used for linear warmup.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Final LR as a fraction of the initial LR when decaying.",
    )

    # MTP architecture
    parser.add_argument(
        "--shared_layers",
        type=int,
        default=4,
        help="Number of shared full-width layers (Drama 3 uses 4).",
    )
    parser.add_argument(
        "--no_time_attention",
        action="store_true",
        help="Frame-local MTP (no KV cache, same information as the old predictor).",
    )
    parser.add_argument(
        "--time_sliding_window",
        type=int,
        default=0,
        help="Causal history in audio frames (0 = full history).",
    )

    # Training objective
    parser.add_argument(
        "--train_talker",
        action="store_true",
        help="Jointly fine-tune talker + MTP (stage 2).  Saves a full model.",
    )
    parser.add_argument(
        "--mtp_loss_weight",
        type=float,
        default=0.5,
        help="Weight of MTP loss in the joint stage (ignored in stage 1).",
    )
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.0,
        help="KL weight against the frozen original code_predictor.",
    )
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument(
        "--scheduled_sampling",
        type=float,
        default=0.0,
        help="Probability of feeding back the MTP's own previous code.",
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help="Also save a loadable full model for stage 1.  Implied by --train_talker.",
    )

    # Runtime
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="sdpa, eager, or flash_attention_2 if flash-attn is installed.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def _speaker_embedding(raw_model, batch, device, dtype):
    encoder = getattr(raw_model, "speaker_encoder", None)
    if encoder is None:
        return None

    ref_mels = batch["ref_mels"].to(device=device, dtype=dtype)
    lengths = batch["ref_mel_lens"]
    if torch.equal(lengths, lengths[:1].expand_as(lengths)):
        return encoder(ref_mels[:, : int(lengths[0].item())]).detach()

    embeddings = []
    for index, length in enumerate(lengths.tolist()):
        embeddings.append(encoder(ref_mels[index : index + 1, : int(length)]))
    return torch.cat(embeddings, dim=0).detach()


def _talker_inputs(raw_model, batch, speaker_embedding):
    talker = raw_model.talker
    input_ids = batch["input_ids"]
    text_embedding = talker.text_projection(
        talker.model.text_embedding(input_ids[:, :, 0])
    ) * batch["text_embedding_mask"]
    codec_embedding = talker.model.codec_embedding(
        input_ids[:, :, 1]
    ) * batch["codec_embedding_mask"]
    if speaker_embedding is not None:
        codec_embedding = codec_embedding.clone()
        codec_embedding[:, 6, :] = speaker_embedding

    inputs_embeds = text_embedding + codec_embedding
    codec_mask = batch["codec_mask"].unsqueeze(-1)
    for index in range(1, talker.config.num_code_groups):
        residual = talker.code_predictor.get_input_embeddings()[index - 1](
            batch["codec_ids"][:, :, index]
        )
        inputs_embeds = inputs_embeds + residual * codec_mask
    return inputs_embeds


def _teacher_logits(teacher, talker, hidden, codec_ids, audio_mask):
    """Run the frozen old code predictor, frame by frame, in one batch.

    The old predictor is frame-local, so all valid audio frames can be stacked
    into the batch dimension.  Returns `[B, S, Q, V]` with zeros outside the
    audio frames.
    """
    if teacher is None:
        return None

    batch_size, seq_len, _ = hidden.shape
    positions = audio_mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    batch_index = positions[:, 0]
    time_index = positions[:, 1]

    frames = [hidden[batch_index, time_index].unsqueeze(1)]
    frames.append(
        talker.get_input_embeddings()(
            codec_ids[batch_index, time_index, 0]
        ).unsqueeze(1)
    )
    for index in range(1, talker.config.num_code_groups):
        frames.append(
            teacher.get_input_embeddings()[index - 1](
                codec_ids[batch_index, time_index, index]
            ).unsqueeze(1)
        )
    teacher_inputs = torch.cat(frames, dim=1)

    with torch.no_grad():
        logits = teacher.forward_finetune(inputs_embeds=teacher_inputs).logits
    output = hidden.new_zeros(
        (batch_size, seq_len, logits.shape[1], logits.shape[2])
    )
    output[batch_index, time_index] = logits.to(output.dtype)
    return output


def _forward_loss(raw_model, mtp, teacher, batch, args):
    device = raw_model.device
    dtype = next(raw_model.parameters()).dtype
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    speaker_embedding = _speaker_embedding(raw_model, batch, device, dtype)
    inputs_embeds = _talker_inputs(raw_model, batch, speaker_embedding)

    if args.train_talker:
        outputs = raw_model.talker(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["codec_0_labels"],
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[0][-1]
        talker_loss = outputs.loss
    else:
        with torch.no_grad():
            outputs = raw_model.talker(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[0][-1].detach()
        talker_loss = None

    # `hidden[:, t]` is the state that predicts the codec token at `t + 1`.
    mtp_hidden = hidden[:, :-1]
    target_codes = batch["codec_ids"][:, 1:]
    target_mask = batch["codec_mask"][:, 1:]
    mtp_attention_mask = batch["attention_mask"][:, :-1]
    code0_embeds = raw_model.talker.get_input_embeddings()(
        target_codes[..., 0]
    )

    distill_teacher = None
    if args.distill_weight > 0:
        distill_teacher = _teacher_logits(
            teacher,
            raw_model.talker,
            mtp_hidden,
            target_codes,
            target_mask,
        )

    mtp_output = mtp.forward_sequence(
        mtp_hidden,
        target_codes,
        audio_mask=target_mask,
        code0_embeds=code0_embeds,
        attention_mask=mtp_attention_mask,
        teacher_logits=distill_teacher,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        scheduled_sampling_ratio=args.scheduled_sampling,
    )
    mtp_loss = mtp_output.loss
    if args.distill_weight > 0 and mtp_output.distill_loss is not None:
        mtp_loss = mtp_loss + args.distill_weight * mtp_output.distill_loss

    if args.train_talker:
        loss = talker_loss + args.mtp_loss_weight * mtp_loss
    else:
        loss = mtp_loss

    return loss, talker_loss, mtp_output


def _save_full_model(raw_model, local_model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ignore = shutil.ignore_patterns(
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.index.json",
    )
    shutil.copytree(local_model_path, output_dir, dirs_exist_ok=True, ignore=ignore)

    # Start from the original config and only update the predictor fields.
    # `raw_model.config.to_dict()` can add serialization-only keys to nested
    # sub-configs; the base JSON is the format the loader expects.
    with open(os.path.join(local_model_path, "config.json"), "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    predictor_config = raw_model.config.talker_config.code_predictor_config
    predictor_dict = config_dict.setdefault("talker_config", {}).setdefault(
        "code_predictor_config", {}
    )
    for key in (
        "mtp_type",
        "mtp_shared_layers",
        "mtp_use_time_attention",
        "mtp_time_sliding_window",
        "shared_layers",
        "use_time_attention",
        "time_sliding_window",
        "codebook_loss_weights",
    ):
        if hasattr(predictor_config, key):
            predictor_dict[key] = getattr(predictor_config, key)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, ensure_ascii=False)

    state_dict = {
        key: value.detach().to("cpu").contiguous()
        for key, value in raw_model.state_dict().items()
    }
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def _steps_per_epoch(args, num_batches: int, num_processes: int) -> int:
    """Optimizer steps per epoch, accounting for sharding, --max_steps, accumulation."""
    if num_processes > 1:
        num_batches = math.ceil(num_batches / num_processes)
    if args.max_steps:
        num_batches = min(num_batches, args.max_steps)
    return max(1, math.ceil(num_batches / args.gradient_accumulation_steps))


def _build_lr_scheduler(optimizer, args, steps_per_epoch: int):
    """Linear warmup followed by cosine/linear decay, advanced per optimizer step.

    The scheduler is deliberately *not* handed to `accelerator.prepare`:
    accelerate's AcceleratedScheduler advances `num_processes` times per
    synchronized step when batches are not split, which would tie the schedule
    to the world size instead of the optimizer step count.  We step it
    ourselves under `accelerator.sync_gradients`.
    """
    if args.lr_scheduler == "constant":
        return None, 0, 0

    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    decay_steps = max(1, total_steps - warmup_steps)
    min_ratio = args.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            # LambdaLR starts at step 0, so start from base_lr / warmup_steps
            # rather than exactly zero.
            return float(step + 1) / float(warmup_steps)
        # Anchor progress so that the very last optimizer step lands exactly on
        # min_lr_ratio; the clamp keeps extra steps (if the estimate is off by
        # one due to sharding) at the floor instead of overshooting.
        progress = (step - warmup_steps) / max(1, decay_steps - 1)
        progress = min(1.0, max(0.0, progress))
        if args.lr_scheduler == "linear":
            return 1.0 - (1.0 - min_ratio) * progress
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    return scheduler, warmup_steps, total_steps


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup_ratio must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min_lr_ratio must be in [0, 1].")

    local_model_path = (
        args.init_model_path
        if os.path.isdir(args.init_model_path)
        else snapshot_download(args.init_model_path)
    )
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    qwen3tts = Qwen3TTSModel.from_pretrained(
        local_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    config = AutoConfig.from_pretrained(local_model_path)

    # Keep a frozen copy of the old predictor before replacing it.  This is
    # only used for optional soft-target distillation.
    teacher = None
    if args.distill_weight > 0:
        teacher = copy.deepcopy(qwen3tts.model.talker.code_predictor)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

    mtp = install_codebook_mtp(
        qwen3tts.model,
        checkpoint=args.init_mtp_checkpoint,
        shared_layers=args.shared_layers,
        use_time_attention=not args.no_time_attention,
        time_sliding_window=args.time_sliding_window or None,
        init_from_code_predictor=args.init_mtp_checkpoint is None,
    )

    if args.train_talker:
        for parameter in qwen3tts.model.parameters():
            parameter.requires_grad = True
        if getattr(qwen3tts.model, "speaker_encoder", None) is not None:
            for parameter in qwen3tts.model.speaker_encoder.parameters():
                parameter.requires_grad = False
        if args.gradient_checkpointing:
            qwen3tts.model.talker.model.gradient_checkpointing = True
            qwen3tts.model.talker.gradient_checkpointing = True
    else:
        for parameter in qwen3tts.model.parameters():
            parameter.requires_grad = False
        for parameter in mtp.parameters():
            parameter.requires_grad = True
        # The residual embedding tables are also part of the talker's input
        # embedding (the talker sums all codebook embeddings).  Keep the
        # pretrained tables fixed while the talker is frozen so that its input
        # distribution does not drift; stage 2 unfreezes them jointly.
        for parameter in mtp.get_input_embeddings().parameters():
            parameter.requires_grad = False

    with open(args.train_jsonl, "r", encoding="utf-8") as handle:
        train_data = [json.loads(line) for line in handle if line.strip()]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    mtp_parameters = [p for p in mtp.parameters() if p.requires_grad]
    if args.train_talker:
        talker_parameters = [
            p
            for name, p in qwen3tts.model.named_parameters()
            if p.requires_grad and not name.startswith("talker.code_predictor.")
        ]
        parameter_groups = [
            {"params": mtp_parameters, "lr": args.lr},
            {"params": talker_parameters, "lr": args.talker_lr},
        ]
    else:
        parameter_groups = [{"params": mtp_parameters, "lr": args.lr}]
    optimizer = AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.dtype == "bf16" else "no",
    )
    steps_per_epoch = _steps_per_epoch(args, len(dataloader), accelerator.num_processes)
    scheduler, warmup_steps, total_steps = _build_lr_scheduler(
        optimizer, args, steps_per_epoch
    )
    model, optimizer, dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, dataloader
    )
    raw_model = accelerator.unwrap_model(model)
    if teacher is not None:
        teacher.to(device=accelerator.device, dtype=next(raw_model.parameters()).dtype)

    accelerator.print(
        f"MTP params: {sum(p.numel() for p in mtp_parameters):,} | "
        f"train_talker={args.train_talker} | "
        f"time_attention={not args.no_time_attention} | "
        f"distill_weight={args.distill_weight}"
    )
    if scheduler is None:
        accelerator.print(f"lr schedule: constant | lr={args.lr:g}")
    else:
        accelerator.print(
            f"lr schedule: {args.lr_scheduler} | steps/epoch={steps_per_epoch} | "
            f"total_steps={total_steps} | warmup_steps={warmup_steps} | "
            f"min_lr_ratio={args.min_lr_ratio}"
        )

    save_full = args.save_full_model or args.train_talker
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        running_talker = 0.0
        running_mtp = 0.0
        running_codebooks = None
        logged_steps = 0

        for step, batch in enumerate(dataloader):
            if args.max_steps and step >= args.max_steps:
                break
            with accelerator.accumulate(model):
                loss, talker_loss, mtp_output = _forward_loss(
                    raw_model, mtp, teacher, batch, args
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in raw_model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float().item()
            running_mtp += mtp_output.loss.detach().float().item()
            if talker_loss is not None:
                running_talker += talker_loss.detach().float().item()
            per_codebook = mtp_output.per_codebook_loss.detach().float().cpu()
            running_codebooks = (
                per_codebook
                if running_codebooks is None
                else running_codebooks + per_codebook
            )
            logged_steps += 1

            if step % args.log_every == 0:
                message = (
                    f"epoch {epoch} | step {step} | loss {loss.item():.4f} | "
                    f"mtp {mtp_output.loss.item():.4f}"
                )
                if talker_loss is not None:
                    message += f" | talker {talker_loss.item():.4f}"
                if scheduler is not None:
                    message += f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                accelerator.print(message)

        if accelerator.is_main_process and logged_steps:
            average_codebooks = running_codebooks / logged_steps
            accelerator.print(
                f"epoch {epoch} averages | loss {running_loss / logged_steps:.4f} | "
                f"mtp {running_mtp / logged_steps:.4f} | "
                f"talker {running_talker / logged_steps if args.train_talker else 0.0:.4f}"
            )
            accelerator.print(
                "per-codebook NLL | "
                + " ".join(
                    f"{index + 1}:{value:.4f}"
                    for index, value in enumerate(average_codebooks.tolist())
                )
            )

            checkpoint = os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            )
            save_codebook_mtp(
                raw_model.talker.code_predictor,
                os.path.join(checkpoint, "codebook_mtp"),
            )
            if save_full:
                _save_full_model(raw_model, local_model_path, checkpoint)
            accelerator.print(f"saved {checkpoint}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
