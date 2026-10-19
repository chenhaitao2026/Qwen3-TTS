# coding=utf-8
# Inference with a trained codebook-MTP decoder.
#
# The talker samples codebook 0 as before. talker.forward then runs the MTP
# decoder once per frame instead of code_predictor.generate()'s 15-step loop.
#
#   python infer_codebook_mtp.py \
#     --model_path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#     --mtp_path output/codebook_mtp/checkpoint-epoch-2 \
#     --text "She said she would be here by noon." \
#     --speaker speaker_test \
#     --output output.wav

import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.codebook_mtp import attach_codebook_mtp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mtp_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="speaker_test")
    parser.add_argument("--language", type=str, default="Auto")
    parser.add_argument("--instruct", type=str, default="")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_audio", type=str, default="")
    parser.add_argument("--ref_text", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
    )
    attach_codebook_mtp(tts.model, args.mtp_path)
    if args.ref_audio:
        wavs, sr = tts.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text or None,
        )
    else:
        wavs, sr = tts.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct or None,
        )
    sf.write(args.output, wavs[0], sr)
    print(f"wrote {args.output} ({len(wavs[0])} samples @ {sr} Hz)")


if __name__ == "__main__":
    main()
