# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Codebook-MTP decoder for Qwen3-TTS, following Fish Audio Drama 3
# (arXiv:2610.18010, Section 2.4) and fish-speech's FishOmniAudioDecoder.
#
# Qwen3-TTS predicts codebook 0 with the talker, then runs code_predictor once
# per residual codebook. On the 12 Hz model that is 15 sequential forwards, and
# each forward re-reads the same weights. Batch-1 decode is memory-bandwidth
# bound, so the cost is bytes read and kernel launches, not the tiny intra-frame
# attention.
#
# Drama 3 stacks one layer per codebook and attends causally along time. Each
# layer runs once per frame, so its weights are read once. Residual codebooks
# stay sequential — layer k is sampled before layer k+1 — but a step is one
# layer, not a full network.
#
# Blocks here are Qwen3-style (RMSNorm, RoPE, GQA, SwiGLU), not Drama 3's gated
# attention, so they match the rest of this repo.
#
# `attend_time` selects the context:
#   False (default): one frame only. No prefill, no KV cache, no previous-frame
#     codes. This is a drop-in for `code_predictor`: the talker hidden and code 0
#     are the only inputs.
#   True: Drama 3. Layers attend causally over time, prefill writes the prompt
#     into the cache, and `sliding_window` bounds that cache.

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class Qwen3TTSCodebookMTPConfig(PretrainedConfig):
    model_type = "qwen3_tts_codebook_mtp"

    def __init__(
        self,
        vocab_size=2048,
        num_code_groups=16,
        talker_hidden_size=1024,
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        shared_layers=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attend_time=False,
        sliding_window=None,
        # Residual codes fed back into the talker, counting from code 1.
        # num_code_groups - 1 keeps Qwen3-TTS's sum of every codebook, so a frozen
        # talker still sees its pretrained input.
        # 1 matches Drama 3 (only code 1 goes back; codes 2..N-1 enter the first
        # shared layer) and requires joint talker training.
        vq_backbone_codebooks=None,
        fusion_scale=False,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
        self.talker_hidden_size = talker_hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.shared_layers = shared_layers
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attend_time = attend_time
        self.sliding_window = sliding_window
        if vq_backbone_codebooks is None:
            vq_backbone_codebooks = num_code_groups - 1
        self.vq_backbone_codebooks = vq_backbone_codebooks
        self.fusion_scale = fusion_scale
        self.initializer_range = initializer_range
        if self.num_code_groups < 2:
            raise ValueError("num_code_groups must be >= 2")
        if self.shared_layers < 1:
            raise ValueError("shared_layers must be >= 1")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if not 0 <= self.vq_backbone_codebooks <= self.num_code_groups - 1:
            raise ValueError("vq_backbone_codebooks out of range")

    @classmethod
    def from_talker_config(cls, talker_config, **overrides):
        predictor = talker_config.code_predictor_config
        head_dim = getattr(predictor, "head_dim", None) or (
            predictor.hidden_size // predictor.num_attention_heads
        )
        values = dict(
            vocab_size=predictor.vocab_size,
            num_code_groups=talker_config.num_code_groups,
            talker_hidden_size=talker_config.hidden_size,
            hidden_size=predictor.hidden_size,
            intermediate_size=predictor.intermediate_size,
            num_attention_heads=predictor.num_attention_heads,
            num_key_value_heads=predictor.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=predictor.rms_norm_eps,
            rope_theta=predictor.rope_theta,
            attention_bias=predictor.attention_bias,
        )
        values.update(overrides)
        return cls(**values)

    @property
    def num_residual_codebooks(self):
        return self.num_code_groups - 1

    @property
    def num_layers(self):
        return self.shared_layers + self.num_residual_codebooks


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, position_ids, theta):
    """Half-split RoPE. q/k are [B, heads, S, D], position_ids is [B, S]."""
    half = q.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    cos, sin = emb.cos(), emb.sin()
    q_out = (q.float() * cos) + (_rotate_half(q.float()) * sin)
    k_out = (k.float() * cos) + (_rotate_half(k.float()) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, heads, length, dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, heads, n_rep, length, dim)
    return hidden_states.reshape(batch, heads * n_rep, length, dim)


def causal_mask(query_positions, key_positions, key_padding, sliding_window):
    """Bool [B, 1, Sq, Sk], True where attention is allowed."""
    allowed = key_positions[:, None, :] <= query_positions[:, :, None]
    allowed = allowed & key_padding[:, None, :]
    if sliding_window is not None:
        in_window = (query_positions[:, :, None] - key_positions[:, None, :]) < sliding_window
        allowed = allowed & in_window
    return allowed[:, None, :, :]


def sdpa(q, k, v, allowed, scale):
    if allowed is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    additive = torch.zeros(allowed.shape, device=q.device, dtype=torch.float32)
    additive = additive.masked_fill(~allowed, torch.finfo(torch.float32).min)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive, scale=scale)


class MTPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden, position_ids, past_kv):
        batch, length, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, position_ids, self.rope_theta)
        present = (k, v)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        return present, k, v, q


class MTPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MTPAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MTPMLP(config)
        self.sliding_window = config.sliding_window
        self.attend_time = config.attend_time

    def forward(self, hidden, position_ids, past_kv, past_positions, key_padding=None, past_padding=None):
        residual = hidden
        normed = self.input_layernorm(hidden)
        if not self.attend_time:
            # Position 0 and no cache: the frame index and earlier frames are not inputs.
            position_ids = torch.zeros_like(position_ids)
            past_kv = None
        present, k, v, q = self.self_attn(normed, position_ids, past_kv)
        if not self.attend_time:
            length = hidden.shape[1]
            allowed = torch.eye(length, dtype=torch.bool, device=hidden.device).view(1, 1, length, length)
        else:
            if key_padding is None:
                key_padding = torch.ones(position_ids.shape, dtype=torch.bool, device=position_ids.device)
            # Same mask for a full sequence and for cached steps. SDPA's is_causal
            # path is a different kernel and does not match this decode.
            if past_positions is None:
                key_positions = position_ids
                full_padding = key_padding
            else:
                key_positions = torch.cat((past_positions, position_ids), dim=1)
                if past_padding is None:
                    past_padding = torch.ones(past_positions.shape, dtype=torch.bool, device=position_ids.device)
                full_padding = torch.cat((past_padding, key_padding), dim=1)
            allowed = causal_mask(position_ids, key_positions, full_padding, self.sliding_window)
        attn = sdpa(
            q,
            repeat_kv(k, self.self_attn.num_kv_groups),
            repeat_kv(v, self.self_attn.num_kv_groups),
            allowed,
            self.self_attn.scaling,
        )
        attn = attn.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + self.self_attn.o_proj(attn)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, present


class MTPLayer(nn.Module):
    """u = W[RMSNorm(e); RMSNorm(h)], then one pre-norm block and a final RMSNorm."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.code_embedding = nn.Embedding(config.vocab_size, dim)
        self.embed_norm = RMSNorm(dim, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(dim, eps=config.rms_norm_eps)
        self.merge = nn.Linear(2 * dim, dim, bias=False)
        self.block = MTPBlock(config)
        self.final_norm = RMSNorm(dim, eps=config.rms_norm_eps)

    def forward(self, hidden, codes, keep, extra, position_ids, past_kv, past_positions, key_padding=None, past_padding=None):
        embedded = self.code_embedding(codes.clamp(0, self.code_embedding.num_embeddings - 1))
        embedded = embedded * keep.to(embedded.dtype).unsqueeze(-1)
        if extra is not None:
            embedded = embedded + extra
        mixed = self.merge(torch.cat((self.embed_norm(embedded), self.hidden_norm(hidden)), dim=-1))
        out, present = self.block(
            mixed.to(hidden.dtype), position_ids, past_kv, past_positions, key_padding, past_padding
        )
        return self.final_norm(out).to(hidden.dtype), present


@dataclass
class MTPForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


@dataclass
class MTPDecodeOutput:
    codes: torch.Tensor
    logits: torch.Tensor


class Qwen3TTSCodebookMTP(nn.Module):
    """Predict residual codebooks 1..N-1 of the frame whose code 0 the talker just scored.

    `forward` teacher-forces every code and is parallel over time. `decode_step`
    runs each layer once and samples the residual head before the next layer.
    """

    def __init__(self, config: Qwen3TTSCodebookMTPConfig):
        super().__init__()
        self.config = config
        if config.talker_hidden_size == config.hidden_size:
            self.hidden_proj = nn.Identity()
        else:
            self.hidden_proj = nn.Linear(config.talker_hidden_size, config.hidden_size, bias=True)
        self.shared_layers = nn.ModuleList(MTPLayer(config) for _ in range(config.shared_layers))
        self.residual_layers = nn.ModuleList(MTPLayer(config) for _ in range(config.num_residual_codebooks))
        self.codebook_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_residual_codebooks)
        )
        n_bypass = config.num_code_groups - 1 - config.vq_backbone_codebooks
        self.bypass_embeddings = (
            nn.ModuleList(nn.Embedding(config.vocab_size, config.hidden_size) for _ in range(n_bypass))
            if n_bypass
            else None
        )
        self.gradient_checkpointing = False
        # Not registered as buffers: inference tensors must not enter state_dict.
        n_layers = config.num_layers
        self._state = {
            "kv": [None] * n_layers,
            "positions": [None] * n_layers,
            "padding": [None] * n_layers,
            "prev_codes": None,
            "prev_audio": None,
        }
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def reset_cache(self):
        n = self.config.num_layers
        self._state["kv"] = [None] * n
        self._state["positions"] = [None] * n
        self._state["padding"] = [None] * n
        self._state["prev_codes"] = None
        self._state["prev_audio"] = None

    def _bypass_sum(self, input_codes, input_audio_mask):
        if self.bypass_embeddings is None or input_codes is None:
            return None
        first = 1 + self.config.vq_backbone_codebooks
        total = None
        for offset, table in enumerate(self.bypass_embeddings):
            ids = input_codes[..., first + offset].clamp(0, table.num_embeddings - 1)
            embedded = table(ids)
            total = embedded if total is None else total + embedded
        return total * input_audio_mask.to(total.dtype).unsqueeze(-1)

    def _cache_get(self, index):
        return self._state["kv"][index], self._state["positions"][index], self._state["padding"][index]

    def _cache_put(self, index, present, position_ids, key_padding):
        past_kv, past_pos, past_pad = self._cache_get(index)
        k, v = present
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
            position_ids = torch.cat((past_pos, position_ids), dim=1)
            key_padding = torch.cat((past_pad, key_padding), dim=1)
        window = self.config.sliding_window
        if window is not None and k.shape[2] > window:
            k = k[:, :, -window:]
            v = v[:, :, -window:]
            position_ids = position_ids[:, -window:]
            key_padding = key_padding[:, -window:]
        self._state["kv"][index] = (k, v)
        self._state["positions"][index] = position_ids
        self._state["padding"][index] = key_padding

    def _run_layer(self, index, layer, hidden, codes, keep, extra, position_ids, use_cache, key_padding):
        use_cache = use_cache and self.config.attend_time
        past_kv, past_pos, past_pad = self._cache_get(index) if use_cache else (None, None, None)
        if self.gradient_checkpointing and self.training:
            extra_arg = extra if extra is not None else hidden.new_zeros(0)

            def _fn(hidden, codes, keep, position_ids, key_padding, extra_arg):
                extra_in = None if extra_arg.numel() == 0 else extra_arg
                out, _present = layer(hidden, codes, keep, extra_in, position_ids, None, None, key_padding, None)
                return out

            return torch.utils.checkpoint.checkpoint(
                _fn, hidden, codes, keep, position_ids, key_padding, extra_arg, use_reentrant=False
            ), None
        hidden, present = layer(
            hidden, codes, keep, extra, position_ids, past_kv, past_pos, key_padding, past_pad
        )
        if use_cache:
            self._cache_put(index, present, position_ids, key_padding)
        return hidden, present

    def _layers(
        self,
        hidden,
        code0,
        residual_codes,
        audio_mask,
        bypass,
        position_ids,
        use_cache,
        sample_fn,
        key_padding,
    ):
        """`residual_codes` is [B, S, num_residual] teacher codes, or None when sampling.

        Residual layer k injects code k-1 of the frame being completed. Layer 0
        therefore injects code 0, not the first residual code.
        """
        logits = []
        sampled = [code0]
        layer_index = 0
        for index, layer in enumerate(self.shared_layers):
            extra = bypass if index == 0 else None
            hidden, _present = self._run_layer(
                layer_index, layer, hidden, code0, audio_mask, extra, position_ids, use_cache, key_padding
            )
            layer_index += 1
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            if sample_fn is None:
                prev = code0 if index == 0 else residual_codes[..., index - 1]
            else:
                prev = sampled[-1]
            hidden, _present = self._run_layer(
                layer_index, layer, hidden, prev, audio_mask, None, position_ids, use_cache, key_padding
            )
            step_logits = head(hidden)
            logits.append(step_logits)
            if sample_fn is not None:
                sampled.append(sample_fn(step_logits))
            layer_index += 1
        return torch.stack(logits, dim=-2), sampled

    def _loss(self, logits, target_codes, audio_mask):
        if not audio_mask.any():
            return logits.sum() * 0
        terms = [
            F.cross_entropy(logits[:, :, index][audio_mask], target_codes[:, :, index + 1][audio_mask])
            for index in range(self.config.num_residual_codebooks)
        ]
        return torch.stack(terms).mean()

    def forward(
        self,
        talker_hidden,
        target_codes,
        audio_mask,
        input_codes=None,
        input_audio_mask=None,
        position_ids=None,
        padding_mask=None,
        use_cache=False,
    ):
        """Teacher-forced pass, parallel over time.

        `talker_hidden` [B, S, H] predicts `target_codes` [B, S, Q]. `audio_mask`
        [B, S] selects supervised frames. `input_codes` are the frames the backbone
        consumed at those positions; their bypassed codebooks enter the first layer.
        `padding_mask` is 1/True on real tokens.
        """
        if use_cache and self.training:
            raise RuntimeError("use_cache is for inference prefill only")
        if use_cache and not self.config.attend_time:
            use_cache = False
        batch, length, _ = talker_hidden.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=talker_hidden.device).view(1, -1).expand(batch, -1)
        if not self.config.attend_time:
            input_codes = None
        if input_audio_mask is None:
            input_audio_mask = torch.zeros(batch, length, dtype=torch.bool, device=talker_hidden.device)
        if padding_mask is None:
            padding_mask = torch.ones(batch, length, dtype=torch.bool, device=talker_hidden.device)
        else:
            padding_mask = padding_mask.bool()
        audio_mask = audio_mask.bool()
        hidden = self.hidden_proj(talker_hidden)
        bypass = self._bypass_sum(input_codes, input_audio_mask)
        logits, _sampled = self._layers(
            hidden,
            target_codes[..., 0],
            target_codes[..., 1:],
            audio_mask,
            bypass,
            position_ids,
            use_cache,
            sample_fn=None,
            key_padding=padding_mask,
        )
        if use_cache and length:
            # The next decode step's backbone input is this last target frame.
            self._state["prev_codes"] = target_codes[:, -1].detach()
            self._state["prev_audio"] = audio_mask[:, -1].detach()
        return MTPForwardOutput(logits=logits, loss=self._loss(logits, target_codes, audio_mask))

    @torch.no_grad()
    def prefill(
        self,
        talker_hidden,
        target_codes,
        audio_mask,
        input_codes=None,
        input_audio_mask=None,
        position_ids=None,
        padding_mask=None,
    ):
        if not self.config.attend_time:
            raise RuntimeError("prefill is only used when attend_time=True")
        was_training = self.training
        self.eval()
        self.reset_cache()
        self.forward(
            talker_hidden,
            target_codes,
            audio_mask,
            input_codes=input_codes,
            input_audio_mask=input_audio_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            use_cache=True,
        )
        if was_training:
            self.train()

    def decode_step(
        self,
        talker_hidden,
        code0,
        position_ids,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        target_codes=None,
        audio_mask=None,
    ):
        """One frame. `talker_hidden` is the state that scored `code0`."""
        if talker_hidden.dim() == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        if code0.dim() > 1:
            code0 = code0.reshape(code0.shape[0], -1)[:, 0]
        if position_ids.dim() == 1:
            position_ids = position_ids.view(-1, 1)
        batch = talker_hidden.shape[0]
        if audio_mask is None:
            audio_mask = (code0 >= 0) & (code0 < self.config.vocab_size)
        audio_mask = audio_mask.reshape(batch, 1).bool()
        if self.config.attend_time and self._state["prev_codes"] is not None:
            input_codes = self._state["prev_codes"].unsqueeze(1)
            input_audio = self._state["prev_audio"].view(batch, 1)
        else:
            input_codes = None
            input_audio = torch.zeros(batch, 1, dtype=torch.bool, device=talker_hidden.device)
        hidden = self.hidden_proj(talker_hidden)
        bypass = self._bypass_sum(input_codes, input_audio)
        if target_codes is None:
            sample_fn = lambda step_logits: sample_codebook(
                step_logits[:, 0], temperature, top_k, top_p, do_sample
            ).view(batch, 1)
            teacher = None
        else:
            sample_fn = None
            teacher = target_codes.unsqueeze(1) if target_codes.dim() == 2 else target_codes
        logits, sampled = self._layers(
            hidden,
            code0.view(batch, 1),
            None if teacher is None else teacher[..., 1:],
            audio_mask,
            bypass,
            position_ids,
            use_cache=self.config.attend_time,
            sample_fn=sample_fn,
            key_padding=torch.ones(batch, 1, dtype=torch.bool, device=talker_hidden.device),
        )
        codes = teacher[:, 0] if teacher is not None else torch.cat(sampled, dim=-1)
        if self.config.attend_time:
            self._state["prev_codes"] = codes.detach()
            self._state["prev_audio"] = audio_mask[:, 0].detach()
        return MTPDecodeOutput(codes=codes, logits=logits[:, 0])

    def fuse_frame(self, code0_embed, residual_embeddings, codes):
        """Talker input for one generated frame. `codes` is [B, num_code_groups]."""
        if code0_embed.dim() == 2:
            code0_embed = code0_embed.unsqueeze(1)
        parts = [code0_embed]
        for index in range(self.config.vq_backbone_codebooks):
            parts.append(residual_embeddings[index](codes[:, index + 1]).unsqueeze(1))
        fused = torch.stack(parts, dim=0).sum(0)
        if self.config.fusion_scale and len(parts) > 1:
            fused = fused / math.sqrt(len(parts))
        return fused

    def fuse_sequence(self, code0_embed, residual_embeddings, codec_ids, audio_mask):
        """Add residual-code embeddings onto a code-0 sequence. Non-audio rows are unchanged."""
        if audio_mask.dim() == 3:
            audio_mask = audio_mask.squeeze(-1)
        audio = audio_mask.bool()
        residual = torch.zeros_like(code0_embed)
        for index in range(self.config.vq_backbone_codebooks):
            residual = residual + residual_embeddings[index](codec_ids[:, :, index + 1])
        residual = residual * audio.unsqueeze(-1).to(dtype=residual.dtype)
        if self.config.fusion_scale and self.config.vq_backbone_codebooks > 0:
            mixed = (code0_embed + residual) / math.sqrt(1 + self.config.vq_backbone_codebooks)
            return torch.where(audio.unsqueeze(-1), mixed, code0_embed)
        return code0_embed + residual


def sample_codebook(logits, temperature, top_k, top_p, do_sample):
    """One id per row from [B, vocab] logits."""
    if (not do_sample) or temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    scores = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None and 0 < top_k < scores.shape[-1]:
        cutoff = scores.topk(top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, torch.finfo(scores.dtype).min)
    if top_p is not None and top_p < 1:
        ordered, indices = scores.sort(dim=-1, descending=True)
        remove = ordered.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[..., 0] = False
        scores = scores.scatter(-1, indices, ordered.masked_fill(remove, torch.finfo(scores.dtype).min))
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(-1)


def attach_codebook_mtp(model, path, device=None, dtype=None):
    """Load a trained decoder onto `model.talker` so `generate()` uses it."""
    from safetensors.torch import load_file

    config = Qwen3TTSCodebookMTPConfig.from_pretrained(path)
    decoder = Qwen3TTSCodebookMTP(config)
    decoder.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
    if device is None or dtype is None:
        try:
            ref = next(model.talker.parameters())
            device = ref.device if device is None else device
            dtype = ref.dtype if dtype is None else dtype
        except StopIteration:
            pass
    if device is not None or dtype is not None:
        decoder.to(device=device, dtype=dtype)
    decoder.eval()
    decoder.reset_cache()
    model.talker.codebook_mtp = decoder
    return decoder


def save_codebook_mtp(decoder, path):
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    decoder.config.save_pretrained(path)
    save_file({k: v.detach().cpu() for k, v in decoder.state_dict().items()}, os.path.join(path, "model.safetensors"))
