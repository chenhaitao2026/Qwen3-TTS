# coding=utf-8
# Train a codebook-MTP decoder on the same JSONL as finetuning/sft_12hz.py.
#
# The talker still predicts codebook 0. This script trains the decoder that
# predicts codebooks 1..N-1 from that hidden state. By default each frame is
# independent: no prefill and no attention over earlier frames, so inference
# only replaces code_predictor. Pass --attend_time for the Drama 3 cache.
# By default the talker stays frozen and every residual
# codebook is still summed into its input, so a pretrained talker does not have
# to be retrained. Set --vq_backbone_codebooks 1 --fusion_scale --train_talker
# to match Drama 3's partial fusion; that changes the talker input and the
# frozen checkpoint is no longer valid.
#
#   python train_codebook_mtp.py \
#     --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
#     --train_jsonl train_with_codes.jsonl \
#     --output_model_path output/codebook_mtp

import argparse
import json
import os

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.core.models.codebook_mtp import (
    Qwen3TTSCodebookMTP,
    Qwen3TTSCodebookMTPConfig,
    save_codebook_mtp,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output/codebook_mtp")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.5)
    parser.add_argument("--shared_layers", type=int, default=2)
    parser.add_argument("--attend_time", action="store_true", help="cache and attend over earlier frames")
    parser.add_argument("--sliding_window", type=int, default=0, help="used only with --attend_time; 0 keeps full causal attention")
    parser.add_argument("--hidden_size", type=int, default=0, help="0 uses the code_predictor width")
    parser.add_argument(
        "--vq_backbone_codebooks",
        type=int,
        default=-1,
        help="-1 feeds every residual codebook back into the talker",
    )
    parser.add_argument("--fusion_scale", action="store_true")
    parser.add_argument("--train_talker", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def build_decoder(talker_config, args):
    overrides = {
        "shared_layers": args.shared_layers,
        "attend_time": args.attend_time,
        "sliding_window": args.sliding_window or None,
        "fusion_scale": args.fusion_scale,
    }
    if args.hidden_size:
        overrides["hidden_size"] = args.hidden_size
    if args.vq_backbone_codebooks >= 0:
        overrides["vq_backbone_codebooks"] = args.vq_backbone_codebooks
    return Qwen3TTSCodebookMTP(Qwen3TTSCodebookMTPConfig.from_talker_config(talker_config, **overrides))


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(args.init_model_path)
    decoder = build_decoder(config.talker_config, args)
    if (
        not args.train_talker
        and decoder.config.vq_backbone_codebooks != config.talker_config.num_code_groups - 1
    ):
        accelerator.print(
            "warning: partial codebook fusion changes the talker input; "
            "pass --train_talker or the frozen talker will see a different embedding"
        )
    decoder.gradient_checkpointing = args.gradient_checkpointing
    qwen3tts.model.talker.codebook_mtp = decoder

    if not args.train_talker:
        for name, param in qwen3tts.model.named_parameters():
            param.requires_grad = name.startswith("talker.codebook_mtp.")

    with open(args.train_jsonl) as handle:
        train_data = [json.loads(line) for line in handle]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    optimizer = AdamW((p for p in qwen3tts.model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    model, optimizer, loader = accelerator.prepare(qwen3tts.model, optimizer, loader)

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                speaker_embedding = model.speaker_encoder(
                    batch["ref_mels"].to(model.device).to(model.dtype)
                ).detach()
                text_embed = model.talker.model.text_embedding(batch["input_ids"][:, :, 0]) * batch["text_embedding_mask"]
                codec_embed = model.talker.model.codec_embedding(batch["input_ids"][:, :, 1]) * batch["codec_embedding_mask"]
                codec_embed[:, 6, :] = speaker_embedding
                # fuse_sequence adds residual-code embeddings on audio rows and
                # leaves the speaker slot and codec-pad rows unchanged.
                inputs = text_embed + model.talker.codebook_mtp.fuse_sequence(
                    codec_embed,
                    model.talker.code_predictor.get_input_embeddings(),
                    batch["codec_ids"],
                    batch["codec_mask"],
                )

                talker_inputs = inputs[:, :-1]
                talker_mask = batch["attention_mask"][:, :-1]
                labels = batch["codec_0_labels"][:, 1:]
                if args.train_talker:
                    outputs = model.talker(
                        inputs_embeds=talker_inputs,
                        attention_mask=talker_mask,
                        labels=labels,
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[0][-1]
                    talker_loss = outputs.loss
                else:
                    with torch.no_grad():
                        outputs = model.talker(
                            inputs_embeds=talker_inputs,
                            attention_mask=talker_mask,
                            labels=labels,
                            output_hidden_states=True,
                        )
                        hidden = outputs.hidden_states[0][-1]
                    hidden = hidden.detach()
                    talker_loss = None

                # hidden[:, t] is the state that predicts the token at t+1.
                positions = talker_mask.long().cumsum(-1) - 1
                positions = positions.clamp(min=0)
                mtp_out = model.talker.codebook_mtp(
                    hidden,
                    batch["codec_ids"][:, 1:],
                    batch["codec_mask"][:, 1:],
                    input_codes=batch["codec_ids"][:, :-1],
                    input_audio_mask=batch["codec_mask"][:, :-1],
                    position_ids=positions,
                    padding_mask=talker_mask,
                )
                loss = mtp_out.loss if talker_loss is None else talker_loss + args.mtp_loss_weight * mtp_out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            path = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            save_codebook_mtp(accelerator.unwrap_model(model).talker.codebook_mtp, path)
            accelerator.print(f"saved {path}")


if __name__ == "__main__":
    main()
