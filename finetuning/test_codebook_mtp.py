# coding=utf-8
"""Shape and cache-equivalence checks for the codebook-MTP decoder. No checkpoint required."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.core.models.codebook_mtp import Qwen3TTSCodebookMTP, Qwen3TTSCodebookMTPConfig


def _config(**overrides):
    values = dict(
        vocab_size=32,
        num_code_groups=4,
        talker_hidden_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        shared_layers=2,
        sliding_window=None,
        vq_backbone_codebooks=1,
        fusion_scale=True,
    )
    values.update(overrides)
    return Qwen3TTSCodebookMTPConfig(**values)


def _batch(model, length=5, batch=2, drop_first=False):
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(batch, length, model.config.talker_hidden_size, generator=g)
    codes = torch.randint(0, model.config.vocab_size, (batch, length, model.config.num_code_groups), generator=g)
    audio = torch.ones(batch, length, dtype=torch.bool)
    if drop_first:
        audio[:, 0] = False
    # Input frame at t is the target frame at t-1.
    input_codes = torch.zeros_like(codes)
    input_codes[:, 1:] = codes[:, :-1]
    input_audio = torch.zeros_like(audio)
    input_audio[:, 1:] = audio[:, :-1]
    positions = torch.arange(length).view(1, -1).expand(batch, -1)
    return hidden, codes, audio, input_codes, input_audio, positions


def check_forward_and_fusion():
    model = Qwen3TTSCodebookMTP(_config())
    hidden, codes, audio, input_codes, input_audio, positions = _batch(model, drop_first=True)
    out = model(
        hidden,
        codes,
        audio,
        input_codes=input_codes,
        input_audio_mask=input_audio,
        position_ids=positions,
    )
    assert out.logits.shape == (2, 5, 3, 32), out.logits.shape
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.shared_layers[0].merge.weight.grad is not None
    timed = Qwen3TTSCodebookMTP(_config(attend_time=True))
    timed_out = timed(
        hidden,
        codes,
        audio,
        input_codes=input_codes,
        input_audio_mask=input_audio,
        position_ids=positions,
    )
    timed_out.loss.backward()
    assert timed.bypass_embeddings[0].weight.grad is not None

    code0 = torch.randn(2, 8)
    residual = torch.nn.ModuleList(torch.nn.Embedding(32, 8) for _ in range(3))
    fused = model.fuse_frame(code0, residual, codes[:, 0])
    assert fused.shape == (2, 1, 8)
    seq = model.fuse_sequence(torch.randn(2, 5, 8), residual, codes, audio)
    assert seq.shape == (2, 5, 8)
    # Non-audio rows are not scaled or mixed with residual codes.
    raw = torch.randn(2, 5, 8)
    kept = model.fuse_sequence(raw, residual, codes, audio)
    assert torch.equal(kept[:, 0], raw[:, 0])


def check_cache_matches_full(sliding_window, drop_first=False):
    torch.manual_seed(1)
    model = Qwen3TTSCodebookMTP(_config(sliding_window=sliding_window, attend_time=True)).eval()
    hidden, codes, audio, input_codes, input_audio, positions = _batch(
        model, length=6, batch=2, drop_first=drop_first
    )
    full = model(
        hidden,
        codes,
        audio,
        input_codes=input_codes,
        input_audio_mask=input_audio,
        position_ids=positions,
    ).logits

    model.reset_cache()
    for step in range(hidden.shape[1]):
        decoded = model.decode_step(
            hidden[:, step],
            codes[:, step, 0],
            positions[:, step],
            target_codes=codes[:, step],
            audio_mask=audio[:, step],
        )
        if not torch.allclose(decoded.logits, full[:, step], atol=1e-5, rtol=1e-5):
            delta = (decoded.logits - full[:, step]).abs().max().item()
            raise AssertionError(f"window={sliding_window} step={step} max abs {delta}")


def check_prefill_then_step():
    torch.manual_seed(2)
    model = Qwen3TTSCodebookMTP(_config(sliding_window=4, attend_time=True)).eval()
    hidden, codes, audio, input_codes, input_audio, positions = _batch(model, length=6, batch=1)
    full = model(
        hidden,
        codes,
        audio,
        input_codes=input_codes,
        input_audio_mask=input_audio,
        position_ids=positions,
    ).logits
    model.prefill(
        hidden[:, :4],
        codes[:, :4],
        audio[:, :4],
        input_codes=input_codes[:, :4],
        input_audio_mask=input_audio[:, :4],
        position_ids=positions[:, :4],
    )
    for step in range(4, 6):
        decoded = model.decode_step(
            hidden[:, step],
            codes[:, step, 0],
            positions[:, step],
            target_codes=codes[:, step],
            audio_mask=audio[:, step],
        )
        if not torch.allclose(decoded.logits, full[:, step], atol=1e-5, rtol=1e-5):
            delta = (decoded.logits - full[:, step]).abs().max().item()
            raise AssertionError(f"prefill step={step} max abs {delta}")


def check_frame_local_ignores_history():
    torch.manual_seed(3)
    model = Qwen3TTSCodebookMTP(_config()).eval()
    assert model.config.attend_time is False
    hidden, codes, audio, input_codes, input_audio, positions = _batch(model, length=4, batch=1)
    first = model(
        hidden, codes, audio, input_codes=input_codes, input_audio_mask=input_audio, position_ids=positions
    ).logits
    hidden2 = hidden.clone()
    hidden2[:, 0] = torch.randn_like(hidden2[:, 0])
    codes2 = codes.clone()
    codes2[:, 0] = (codes2[:, 0] + 1) % model.config.vocab_size
    second = model(
        hidden2, codes2, audio, input_codes=input_codes, input_audio_mask=input_audio, position_ids=positions + 10
    ).logits
    if not torch.allclose(first[:, 1:], second[:, 1:], atol=1e-5, rtol=1e-5):
        raise AssertionError("frame-local logits changed when another frame changed")
    stepped = model.decode_step(hidden[:, 2], codes[:, 2, 0], positions[:, 2] + 7, target_codes=codes[:, 2], audio_mask=audio[:, 2])
    if not torch.allclose(stepped.logits, first[:, 2], atol=1e-5, rtol=1e-5):
        raise AssertionError("decode_step diverged from the frame-local forward")
    try:
        model.prefill(hidden, codes, audio)
    except RuntimeError:
        pass
    else:
        raise AssertionError("prefill should be rejected when attend_time is false")


if __name__ == "__main__":
    check_forward_and_fusion()
    check_frame_local_ignores_history()
    check_cache_matches_full(None)
    check_cache_matches_full(None, drop_first=True)
    check_cache_matches_full(3)
    check_prefill_then_step()
    print("codebook-mtp checks passed")
