# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Codebook-MTP decoder for Qwen3-TTS.
#
# This module is a drop-in replacement for
# `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`.  It keeps the
# same public surface used by the talker:
#
#   * get_input_embeddings() / get_output_embeddings()
#   * forward_finetune(inputs_embeds=..., labels=...) -> (logits, loss)
#   * generate(inputs_embeds=..., max_new_tokens=15, ...) -> .sequences
#
# The difference is how the residual codebooks are produced.  The original
# code predictor is a 5-layer depth transformer that is run once per codebook
# (15 sequential passes, i.e. 75 layer passes per frame).  Following the
# codebook-MTP decoder of Fish Audio Drama 3 (arXiv:2610.18010, Sec. 2.4), each
# residual codebook gets its own full-width layer, the layers are stacked along
# the codebook axis, and every layer attends causally along time.  A frame thus
# costs `shared_layers + (num_code_groups - 1)` single-position layer passes
# (19 for the 16-codebook Qwen3-TTS 12 Hz model) and each layer is read once.
#
# Two operating modes are supported:
#
#   use_time_attention=False
#       Frame-local.  No state is carried between frames, so the module is an
#       exact drop-in even if the host model is not modified.  This has the
#       same information access as the original code predictor.
#
#   use_time_attention=True (default)
#       Drama 3 layout.  Every layer keeps its own KV cache over previous
#       frames.  The talker calls `reset_cache()` at the beginning of an
#       utterance (see `Qwen3TTSTalkerForConditionalGeneration.forward`).
#       A sliding window can bound the cache.
#
# The module is intentionally written without depending on the HF attention
# backends: inference is a batch of single-position steps, where the custom
# SDPA path is both simple and predictable.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput

from .configuration_qwen3_tts import (Qwen3TTSTalkerCodePredictorConfig,
                                      Qwen3TTSTalkerConfig)
from .modeling_qwen3_tts import (Qwen3TTSRMSNorm, Qwen3TTSRotaryEmbedding,
                                 Qwen3TTSTalkerCodePredictorOutputWithPast,
                                 Qwen3TTSTalkerTextMLP,
                                 Qwen3TTSTalkerTextPreTrainedModel,
                                 apply_rotary_pos_emb)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Qwen3TTSCodebookMTPConfig(Qwen3TTSTalkerCodePredictorConfig):
    """Configuration of the codebook-MTP decoder.

    It intentionally derives from `Qwen3TTSTalkerCodePredictorConfig` so that
    `Qwen3TTSConfig`/`Qwen3TTSTalkerConfig` can keep serialising a
    `code_predictor_config` without a new sub-config type.  The MTP-specific
    fields are also written as `mtp_*` attributes on the host config by
    `install_codebook_mtp`, which is what makes a full `save_pretrained` /
    `from_pretrained` round-trip construct this class automatically.
    """

    model_type = "qwen3_tts_codebook_mtp"

    def __init__(
        self,
        shared_layers: int = 4,
        use_time_attention: bool = True,
        time_sliding_window: Optional[int] = None,
        talker_hidden_size: Optional[int] = None,
        codebook_loss_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shared_layers = int(shared_layers)
        self.use_time_attention = bool(use_time_attention)
        self.time_sliding_window = None if time_sliding_window in (0, None) else int(time_sliding_window)
        self.talker_hidden_size = talker_hidden_size
        self.codebook_loss_weights = list(codebook_loss_weights) if codebook_loss_weights is not None else None
        if self.shared_layers < 0:
            raise ValueError("shared_layers must be >= 0")
        if self.use_time_attention and self.time_sliding_window is not None and self.time_sliding_window < 1:
            raise ValueError("time_sliding_window must be a positive integer or None")

    @classmethod
    def from_code_predictor_config(cls, config, talker_config=None, **overrides):
        values = config.to_dict()
        # These are derived from the base class / runtime and should not be
        # passed as keyword arguments twice.
        for key in ("model_type", "transformers_version", "architectures"):
            values.pop(key, None)
        if talker_config is not None:
            values["talker_hidden_size"] = getattr(talker_config, "hidden_size", values.get("talker_hidden_size"))
            values["num_code_groups"] = getattr(talker_config, "num_code_groups", values.get("num_code_groups"))
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class MTPAttention(nn.Module):
    """GQA + QK-norm + RoPE, with an optional per-layer time cache.

    Parameter names mirror `Qwen3TTSAttention` so that the five pretrained
    code-predictor blocks can be copied into the MTP blocks directly.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, cache=None):
        batch, seq_len, _ = hidden_states.shape

        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        present = (key, value)
        if cache is not None:
            past_key, past_value = cache
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
            present = (key, value)

        key = _repeat_kv(key, self.num_kv_groups)
        value = _repeat_kv(value, self.num_kv_groups)

        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=seq_len > 1,
                scale=self.scaling,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                scale=self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output), present


class MTPBlock(nn.Module):
    """A pre-norm transformer block with the same parameter names as a
    `Qwen3TTSDecoderLayer`."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MTPAttention(config)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3TTSTalkerTextMLP(config)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, position_embeddings, attention_mask, cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class MTPLayer(nn.Module):
    """Merge one code embedding with the hidden state from the layer below.

    u = W [ RMSNorm(e) ; RMSNorm(h) ]
    h' = RMSNorm(Block(u))
    """

    def __init__(self, config, embed_dim: int):
        super().__init__()
        self.embed_norm = Qwen3TTSRMSNorm(embed_dim, eps=config.rms_norm_eps)
        self.hidden_norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.merge = nn.Linear(embed_dim + config.hidden_size, config.hidden_size, bias=False)
        self.block = MTPBlock(config)
        self.final_norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, code_embed, hidden_states, position_embeddings, attention_mask=None, cache=None):
        merged = self.merge(
            torch.cat(
                (self.embed_norm(code_embed), self.hidden_norm(hidden_states)),
                dim=-1,
            )
        )
        hidden_states, present = self.block(
            merged, position_embeddings, attention_mask, cache
        )
        return self.final_norm(hidden_states), present


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class MTPSequenceOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    per_codebook_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None


@dataclass
class MTPGenerationOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_codebook(logits, temperature=0.9, top_k=50, top_p=1.0, do_sample=True):
    """Draw one id per row from `[B, vocab]` logits."""
    if (not do_sample) or temperature is None or float(temperature) <= 0:
        return logits.argmax(dim=-1)

    scores = logits.float() / max(float(temperature), 1e-5)
    if top_k is not None and 0 < int(top_k) < scores.shape[-1]:
        cutoff = scores.topk(int(top_k), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p is not None and float(top_p) < 1.0:
        ordered, indices = scores.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, ordered)
    probs = scores.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------


class Qwen3TTSCodebookMTPModel(nn.Module):
    def __init__(self, config, embedding_dim: int):
        super().__init__()
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        self.num_residual_codebooks = config.num_code_groups - 1
        shared_layers = int(getattr(config, "shared_layers", 4))
        if shared_layers < 0:
            raise ValueError("shared_layers must be >= 0")

        # Same shape and names as the original code predictor, so the tables
        # can be copied directly and the talker can keep using them.
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(self.num_residual_codebooks)]
        )
        self.shared_layers = nn.ModuleList(
            [MTPLayer(config, embedding_dim) for _ in range(shared_layers)]
        )
        self.codebook_layers = nn.ModuleList(
            [MTPLayer(config, embedding_dim) for _ in range(self.num_residual_codebooks)]
        )
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.codec_embedding

    def set_input_embeddings(self, value):
        self.codec_embedding = value


class Qwen3TTSCodebookMTPForConditionalGeneration(Qwen3TTSTalkerTextPreTrainedModel):
    """Drop-in replacement for the talker's `code_predictor`."""

    config_class = Qwen3TTSCodebookMTPConfig
    base_model_prefix = "talker.code_predictor"
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, talker_config: Optional[Qwen3TTSTalkerConfig] = None):
        super().__init__(config)
        talker_hidden_size = (
            getattr(talker_config, "hidden_size", None)
            if talker_config is not None
            else getattr(config, "talker_hidden_size", None)
        )
        if talker_hidden_size is None:
            raise ValueError(
                "talker_hidden_size is required: pass `talker_config` or set "
                "`config.talker_hidden_size`."
            )
        self.talker_hidden_size = int(talker_hidden_size)
        self.embed_dim = self.talker_hidden_size
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        self.num_residual_codebooks = config.num_code_groups - 1

        self.model = Qwen3TTSCodebookMTPModel(config, self.embed_dim)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(self.num_residual_codebooks)
            ]
        )

        if config.hidden_size != self.talker_hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                self.talker_hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()

        weights = getattr(config, "codebook_loss_weights", None)
        if weights is None:
            weights = [1.0] * self.num_residual_codebooks
        if len(weights) != self.num_residual_codebooks:
            raise ValueError(
                f"codebook_loss_weights must have {self.num_residual_codebooks} entries"
            )
        self.register_buffer(
            "codebook_loss_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=False,
        )

        self._num_layers = len(self.model.shared_layers) + len(self.model.codebook_layers)
        self._kv_cache = [None] * self._num_layers
        self._frame_index = 0
        self._cache_batch_size = None

        self.post_init()

    # ------------------------------------------------------------------
    # Compatibility with the original code predictor
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # ------------------------------------------------------------------
    # Cache handling
    # ------------------------------------------------------------------

    def reset_cache(self):
        self._kv_cache = [None] * self._num_layers
        self._frame_index = 0
        self._cache_batch_size = None

    def _maybe_reset_cache(self, batch_size: int):
        if self._cache_batch_size != batch_size:
            self.reset_cache()
            self._cache_batch_size = batch_size

    def _trim_cache(self, cache, sliding_window):
        if cache is None or sliding_window is None:
            return cache
        key, value = cache
        if key.shape[2] > sliding_window:
            key = key[:, :, -int(sliding_window):]
            value = value[:, :, -int(sliding_window):]
        return (key, value)

    def _past_cache(self, cache):
        """Trim the history *before* the current key is appended.

        The current frame is always visible.  With `time_sliding_window=1`
        this means no history at all, not one stale key plus the current one.
        """
        window = getattr(self.config, "time_sliding_window", None)
        if cache is None or window is None:
            return cache
        max_past = int(window) - 1
        if max_past <= 0:
            return None
        return self._trim_cache(cache, max_past)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    @staticmethod
    def _additive_causal_mask(
        audio_mask,
        attention_mask,
        dtype,
        frame_local=False,
        sliding_window=None,
    ):
        """Return a `[B, 1, S, S]` additive mask, or None when it is causal.

        `frame_local=True` reproduces `use_time_attention=False`: every query
        sees only the key at its own position.  `sliding_window` bounds the
        causal history in audio-frame units.
        """
        if audio_mask is None and attention_mask is None:
            if not frame_local and sliding_window is None:
                return None
            raise ValueError(
                "audio_mask or attention_mask is required for frame-local or "
                "sliding-window attention"
            )
        batch, seq_len = (
            audio_mask.shape if audio_mask is not None else attention_mask.shape
        )
        device = (
            audio_mask.device if audio_mask is not None else attention_mask.device
        )

        key_valid = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        if attention_mask is not None:
            key_valid = key_valid & attention_mask.bool()
        if audio_mask is not None:
            key_valid = key_valid & audio_mask.bool()

        if not frame_local and sliding_window is None and key_valid.all():
            # The MTP only sees audio frames at inference, so a contiguous
            # causal mask is exactly right when every position is an audio
            # frame and every key is valid.
            return None

        idx = torch.arange(seq_len, device=device)
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None, :, :]
        if frame_local:
            allowed = eye & key_valid[:, None, None, :]
        else:
            causal = idx[None, :] <= idx[:, None]
            allowed = causal[None, None, :, :] & key_valid[:, None, None, :]
            if sliding_window is not None:
                # The window is measured in audio frames, not in the padded
                # text+audio sequence positions.
                audio_pos = (
                    audio_mask.long().cumsum(dim=-1) - 1
                    if audio_mask is not None
                    else idx.expand(batch, -1)
                )
                allowed = allowed & (
                    audio_pos[:, None, :] > audio_pos[:, :, None] - int(sliding_window)
                )[:, None, :, :]

        # A query with no valid key (padding and text positions in the full
        # sequence) attends to itself so that its unused output is finite and
        # backward stays free of NaNs.
        has_key = allowed.any(dim=-1, keepdim=True)
        allowed = torch.where(has_key, allowed, eye)

        additive = torch.zeros(allowed.shape, device=device, dtype=torch.float32)
        additive = additive.masked_fill(
            ~allowed, torch.finfo(torch.float32).min
        )
        return additive.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Core layer stacks
    # ------------------------------------------------------------------

    def _project_hidden(self, talker_hidden):
        return self.small_to_mtp_projection(talker_hidden)

    def _position_embeddings(self, hidden_states, position_ids):
        return self.model.rotary_emb(hidden_states, position_ids)

    def _shared_forward(self, hidden_states, code0_embed, position_embeddings, attention_mask, caches=None):
        for index, layer in enumerate(self.model.shared_layers):
            cache = None if caches is None else self._past_cache(caches[index])
            hidden_states, present = layer(
                code0_embed, hidden_states, position_embeddings, attention_mask, cache
            )
            if caches is not None:
                caches[index] = self._trim_cache(
                    present, getattr(self.config, "time_sliding_window", None)
                )
        return hidden_states

    def _codebook_forward(
        self,
        hidden_states,
        code_embeds,
        position_embeddings,
        attention_mask,
        caches=None,
        cache_offset=0,
        codebook_weights=None,
        labels=None,
    ):
        """Run the per-codebook layers.

        `code_embeds[k-1]` is the embedding injected into codebook layer `k`.
        Returns logits `[B, S, Q, V]` if `code_embeds` has `Q` entries.
        """
        logits = []
        for offset, (layer, head) in enumerate(zip(self.model.codebook_layers, self.lm_head)):
            cache = None if caches is None else caches[cache_offset + offset]
            hidden_states, present = layer(
                code_embeds[offset], hidden_states, position_embeddings, attention_mask, cache
            )
            if caches is not None:
                caches[cache_offset + offset] = present
            logits.append(head(hidden_states))
        return torch.stack(logits, dim=2)  # [B, S, Q, V]

    def _loss(self, logits, target_codes, audio_mask, codebook_weights=None):
        """Mean cross entropy over codebooks, each codebook normalized on its
        own frames (Drama 3, Eq. 9)."""
        if audio_mask is None:
            audio_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        audio_mask = audio_mask.bool()
        if not audio_mask.any():
            zero = logits.sum() * 0.0
            return zero, torch.zeros(self.num_residual_codebooks, device=logits.device)

        terms = []
        for index in range(self.num_residual_codebooks):
            target = target_codes[..., index + 1][audio_mask]
            terms.append(
                F.cross_entropy(logits[:, :, index][audio_mask], target, reduction="mean")
            )
        terms = torch.stack(terms)
        if codebook_weights is None:
            codebook_weights = self.codebook_loss_weights
        weights = torch.as_tensor(
            codebook_weights, device=terms.device, dtype=terms.dtype
        )
        loss = (terms * weights).sum() / weights.sum().clamp_min(1e-8)
        return loss, terms

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def forward_finetune(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ) -> Qwen3TTSTalkerCodePredictorOutputWithPast:
        """Teacher-forced, frame-local pass.

        `inputs_embeds` is `[B, num_code_groups, embed_dim]`:
            position 0       talker hidden state
            position 1       embedding of codebook 0 of the frame
            position k       embedding of codebook k-1 (k = 2..Q)
        `labels` is `[B, Q]` with codes 1..Q.  This is exactly the tensor the
        existing `forward_sub_talker_finetune` builds, so this method is
        binary compatible with the original code predictor.
        """
        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required for forward_finetune")
        if inputs_embeds.shape[1] < self.num_code_groups:
            raise ValueError(
                f"expected at least {self.num_code_groups} positions, got {inputs_embeds.shape[1]}"
            )

        batch = inputs_embeds.shape[0]
        device = inputs_embeds.device

        hidden_states = self._project_hidden(inputs_embeds[:, :1])  # [B, 1, H]
        code0_embed = inputs_embeds[:, 1:2]                          # [B, 1, E]
        position_ids = torch.zeros(batch, 1, dtype=torch.long, device=device)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)

        hidden_states = self._shared_forward(
            hidden_states, code0_embed, position_embeddings, attention_mask=None
        )

        logits = []
        for index in range(self.num_residual_codebooks):
            # Position index+1 holds the embedding of codebook index (0-based).
            code_embed = inputs_embeds[:, index + 1:index + 2]
            hidden_states, _ = self.model.codebook_layers[index](
                code_embed, hidden_states, position_embeddings, attention_mask=None
            )
            logits.append(self.lm_head[index](hidden_states))
        logits = torch.cat(logits, dim=1)  # [B, Q, V]

        loss = None
        if labels is not None:
            if labels.shape[1] != self.num_residual_codebooks:
                raise ValueError(
                    f"labels must have {self.num_residual_codebooks} columns, got {labels.shape[1]}"
                )
            terms = []
            for index in range(self.num_residual_codebooks):
                terms.append(
                    F.cross_entropy(
                        logits[:, index],
                        labels[:, index],
                        ignore_index=-100,
                        reduction="mean",
                    )
                )
            terms = torch.stack(terms)
            weights = self.codebook_loss_weights.to(terms.dtype)
            loss = (terms * weights).sum() / weights.sum().clamp_min(1e-8)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(loss=loss, logits=logits)

    def forward_sequence(
        self,
        talker_hidden,
        codec_ids,
        audio_mask=None,
        code0_embeds=None,
        attention_mask=None,
        position_ids=None,
        return_logits=False,
        teacher_logits=None,
        distill_weight=0.0,
        distill_temperature=1.0,
        codebook_weights=None,
        scheduled_sampling_ratio=0.0,
        generator=None,
    ) -> MTPSequenceOutput:
        """Teacher-forced pass over a whole sequence, parallel over time.

        This is the Drama 3 training layout: all positions and all codebooks
        are computed in `shared_layers + Q` passes instead of the original
        `Q * depth_layers` per-frame passes.

        Args:
            talker_hidden: `[B, S, talker_hidden]`; position `t` is the state
                that predicts codebook 0 of the frame at `t+1`.
            codec_ids: `[B, S, num_code_groups]`; the target frame at each
                position (the caller usually passes `codec_ids[:, 1:]`).
            audio_mask: `[B, S]`, True on positions that are audio frames.
                It is used both for the loss and for the time-attention keys.
            code0_embeds: `[B, S, embed_dim]`; embedding of codebook 0 of the
                target frame.  The talker owns this table, so the caller must
                pass it (the code predictor does not have a code-0 table).
            attention_mask: `[B, S]`, True on non-padding tokens.
            position_ids: `[B, S]`; defaults to the audio-frame index
                (`cumsum(audio_mask) - 1`), matching inference.  In
                frame-local mode the default is all zeros.
            return_logits: return the full `[B, S, Q, V]` tensor.
            teacher_logits: optional `[B, S, Q, V]` soft targets from the old
                code predictor, used for distillation.
            distill_weight: deprecated compatibility argument.  The returned
                `distill_loss` is unweighted; the caller applies the weight.
            scheduled_sampling_ratio: probability of feeding back the model's
                own argmax code (for codebook >= 2) instead of the ground
                truth during training.  This is a DAgger-style regulariser,
                not a replacement for real-code supervision.
        """
        if code0_embeds is None:
            raise ValueError(
                "`code0_embeds` is required. Compute it with the talker's "
                "`codec_embedding(codec_ids[..., 0])`."
            )
        batch, seq_len, _ = talker_hidden.shape
        device = talker_hidden.device

        if audio_mask is None:
            audio_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        else:
            audio_mask = audio_mask.bool()
        if attention_mask is None:
            attention_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.bool()

        use_time_attention = bool(getattr(self.config, "use_time_attention", True))
        if position_ids is None:
            if use_time_attention:
                position_ids = (audio_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
            else:
                position_ids = torch.zeros_like(audio_mask, dtype=torch.long)
        else:
            position_ids = position_ids.long()

        hidden_states = self._project_hidden(talker_hidden)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)
        attn_mask = self._additive_causal_mask(
            audio_mask,
            attention_mask,
            hidden_states.dtype,
            frame_local=not use_time_attention,
            sliding_window=(
                getattr(self.config, "time_sliding_window", None)
                if use_time_attention
                else None
            ),
        )

        hidden_states = self._shared_forward(
            hidden_states,
            code0_embeds,
            position_embeddings,
            attention_mask=attn_mask,
        )

        logits_list = []
        per_codebook = []
        distill_terms = []
        prev_codes = None

        for index in range(self.num_residual_codebooks):
            if index == 0:
                code_embed = code0_embeds
            else:
                if prev_codes is None:
                    prev_codes = codec_ids[..., index]
                code_embed = self.model.codec_embedding[index - 1](prev_codes)

            hidden_states, _ = self.model.codebook_layers[index](
                code_embed, hidden_states, position_embeddings, attention_mask=attn_mask
            )
            logits = self.lm_head[index](hidden_states)  # [B, S, V]

            target = codec_ids[..., index + 1]
            if audio_mask.any():
                per_codebook.append(
                    F.cross_entropy(
                        logits[audio_mask], target[audio_mask], reduction="mean"
                    )
                )
            else:
                per_codebook.append(logits.sum() * 0.0)

            if teacher_logits is not None:
                t_logits = teacher_logits[:, :, index, :][audio_mask]
                s_logits = logits[audio_mask]
                temperature = max(float(distill_temperature), 1e-5)
                distill_terms.append(
                    F.kl_div(
                        F.log_softmax(s_logits / temperature, dim=-1),
                        F.softmax(t_logits / temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * (temperature ** 2)
                )

            if return_logits:
                logits_list.append(logits)

            # Feed the next layer.  By default this is teacher forcing from
            # the ground-truth code.  Scheduled sampling mixes in the model's
            # own argmax for a random subset of frames.
            next_codes = codec_ids[..., index + 1]
            if scheduled_sampling_ratio > 0 and index + 1 < self.num_residual_codebooks:
                with torch.no_grad():
                    sampled = logits.argmax(dim=-1)
                    if scheduled_sampling_ratio >= 1.0:
                        use_sample = torch.ones_like(next_codes, dtype=torch.bool)
                    else:
                        use_sample = (
                            torch.rand(
                                next_codes.shape,
                                device=device,
                                generator=generator,
                            )
                            < scheduled_sampling_ratio
                        )
                    next_codes = torch.where(use_sample, sampled, next_codes)
            prev_codes = next_codes

        per_codebook = torch.stack(per_codebook)
        weights = torch.as_tensor(
            codebook_weights
            if codebook_weights is not None
            else self.codebook_loss_weights,
            device=per_codebook.device,
            dtype=per_codebook.dtype,
        )
        loss = (per_codebook * weights).sum() / weights.sum().clamp_min(1e-8)

        distill_loss = None
        if distill_terms:
            distill_loss = torch.stack(distill_terms).mean()

        return MTPSequenceOutput(
            loss=loss,
            per_codebook_loss=per_codebook,
            logits=torch.stack(logits_list, dim=2) if return_logits else None,
            distill_loss=distill_loss,
        )

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def _decode_frame(
        self,
        talker_hidden,
        code0_embed,
        target_codes=None,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        num_steps=None,
        return_logits=False,
    ):
        """Decode the residual codebooks of exactly one frame.

        `target_codes`, when given for teacher forcing, contains the residual
        codes in order (`codec_ids[..., 1:]`, shape `[B, Q]`).  A full frame
        `[B, Q + 1]` including codebook 0 is also accepted for convenience.
        """
        if talker_hidden.dim() == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        if code0_embed.dim() == 2:
            code0_embed = code0_embed.unsqueeze(1)
        if talker_hidden.shape[1] != 1:
            raise ValueError("decode_frame expects exactly one position")

        batch = talker_hidden.shape[0]
        device = talker_hidden.device
        steps = self.num_residual_codebooks if num_steps is None else int(num_steps)
        steps = max(0, min(steps, self.num_residual_codebooks))

        if target_codes is not None and target_codes.shape[-1] == self.num_code_groups:
            # Be forgiving: accept either the full 16-code frame or the 15
            # residual labels.  The first column is codebook 0, which is an
            # input to the first MTP layer, not a target.
            target_codes = target_codes[..., 1:]

        use_cache = bool(getattr(self.config, "use_time_attention", True))
        if use_cache:
            self._maybe_reset_cache(batch)
            frame_index = self._frame_index
            position_ids = torch.full((batch, 1), frame_index, dtype=torch.long, device=device)
        else:
            position_ids = torch.zeros(batch, 1, dtype=torch.long, device=device)

        hidden_states = self._project_hidden(talker_hidden)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)

        caches = self._kv_cache if use_cache else None
        hidden_states = self._shared_forward(
            hidden_states, code0_embed, position_embeddings, attention_mask=None, caches=caches
        )

        codes = []
        logits_list = []
        for index in range(steps):
            if index == 0:
                code_embed = code0_embed
            else:
                code_embed = self.model.codec_embedding[index - 1](codes[-1].view(batch, 1))
            layer_offset = len(self.model.shared_layers) + index
            cache = (
                None
                if caches is None
                else self._past_cache(caches[layer_offset])
            )
            hidden_states, present = self.model.codebook_layers[index](
                code_embed, hidden_states, position_embeddings, attention_mask=None, cache=cache
            )
            if caches is not None:
                caches[layer_offset] = self._trim_cache(
                    present, getattr(self.config, "time_sliding_window", None)
                )
            logits = self.lm_head[index](hidden_states[:, 0])  # [B, V]
            logits_list.append(logits)

            if target_codes is not None:
                targets = target_codes
                if targets.dim() == 3:
                    targets = targets[:, 0]
                code = targets[:, index].to(device=device, dtype=torch.long)
            else:
                code = sample_codebook(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            codes.append(code)

        if use_cache:
            self._frame_index += 1

        return codes, logits_list

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds=None,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        **kwargs,
    ):
        """Generate all residual codebooks of one frame.

        The talker calls this with `inputs_embeds = cat([past_hidden, code0])`
        and `max_new_tokens = num_code_groups - 1`.  It returns an object whose
        `.sequences` is `[B, num_code_groups - 1]`, exactly like the original
        code predictor.
        """
        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required")
        if inputs_embeds.shape[1] < 2:
            raise ValueError("`inputs_embeds` must contain [talker_hidden, code0]")

        steps = self.num_residual_codebooks if max_new_tokens is None else int(max_new_tokens)
        steps = max(0, min(steps, self.num_residual_codebooks))

        codes, logits = self._decode_frame(
            inputs_embeds[:, 0],
            inputs_embeds[:, 1],
            target_codes=None,
            do_sample=False if do_sample is None else bool(do_sample),
            temperature=1.0 if temperature is None else temperature,
            top_k=50 if top_k is None else top_k,
            top_p=1.0 if top_p is None else top_p,
            num_steps=steps,
            return_logits=kwargs.get("return_logits", False),
        )
        sequences = (
            torch.stack(codes, dim=1)
            if codes
            else inputs_embeds.new_zeros((inputs_embeds.shape[0], 0), dtype=torch.long)
        )
        return MTPGenerationOutput(
            sequences=sequences,
            logits=torch.stack(logits, dim=1) if (logits and kwargs.get("return_logits")) else None,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ):
        """Compatibility shim.

        The talker uses `generate()` during decoding and
        `forward_finetune()` during training.  `forward()` is kept for callers
        that feed a teacher-forced 16-token frame; it returns all codebook
        logits.  A single-frame `inputs_embeds` without labels returns the
        first residual codebook's logits, which is the only codebook that can
        be computed without already knowing the lower codes.
        """
        if inputs_embeds is not None and (
            labels is not None or inputs_embeds.shape[1] > 2
        ):
            return self.forward_finetune(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        if inputs_embeds is None or inputs_embeds.shape[1] < 2:
            raise ValueError(
                "forward() expects teacher-forced 16-token embeddings or "
                "[talker_hidden, code0]. Use generate() for decoding."
            )
        codes, logits = self._decode_frame(
            inputs_embeds[:, 0],
            inputs_embeds[:, 1],
            target_codes=None,
            do_sample=False,
            num_steps=1,
            return_logits=True,
        )
        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            logits=logits[0].unsqueeze(1) if logits else None,
            generation_steps=(generation_steps or 0) + 1,
        )

    # ------------------------------------------------------------------
    # Weight transfer from the original code predictor
    # ------------------------------------------------------------------

    @torch.no_grad()
    def load_from_code_predictor(self, old_code_predictor):
        """Warm-start from a `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`.

        Embeddings, output heads, the small-to-MTP projection and every
        transformer block are copied.  Old blocks are reused round-robin for
        the new shared/per-codebook layers.  The merge projection is
        initialised so that the first forward is close to
        `0.5 * proj(code_embed) + 0.5 * hidden`.
        """
        # Embeddings and heads.
        old_embeddings = old_code_predictor.get_input_embeddings()
        new_embeddings = self.get_input_embeddings()
        for old, new in zip(old_embeddings, new_embeddings):
            new.load_state_dict(old.state_dict())

        old_heads = old_code_predictor.get_output_embeddings()
        for old, new in zip(old_heads, self.lm_head):
            new.load_state_dict(old.state_dict())

        # Projection.
        old_projection = getattr(old_code_predictor, "small_to_mtp_projection", None)
        if old_projection is not None:
            if isinstance(old_projection, nn.Identity) and isinstance(
                self.small_to_mtp_projection, nn.Identity
            ):
                pass
            elif isinstance(old_projection, nn.Linear) and isinstance(
                self.small_to_mtp_projection, nn.Linear
            ):
                self.small_to_mtp_projection.load_state_dict(old_projection.state_dict())

        old_layers = list(old_code_predictor.model.layers)
        old_norm_state = old_code_predictor.model.norm.state_dict()

        def copy_block(dst_layer, src_layer):
            dst_layer.block.load_state_dict(src_layer.state_dict(), strict=False)
            dst_layer.final_norm.load_state_dict(old_norm_state)

        for index, layer in enumerate(self.model.shared_layers):
            copy_block(layer, old_layers[index % len(old_layers)])
        num_shared = len(self.model.shared_layers)
        for index, layer in enumerate(self.model.codebook_layers):
            copy_block(layer, old_layers[(num_shared + index) % len(old_layers)])

        # Initialise merge = 0.5 * [projection | identity].
        hidden_size = self.config.hidden_size
        embed_dim = self.embed_dim
        eye = torch.eye(hidden_size, device=self.codebook_loss_weights.device)
        if isinstance(self.small_to_mtp_projection, nn.Linear):
            projection_weight = self.small_to_mtp_projection.weight
        else:
            projection_weight = eye
        for layer in list(self.model.shared_layers) + list(self.model.codebook_layers):
            weight = layer.merge.weight
            weight.zero_()
            weight[:, :embed_dim] = 0.5 * projection_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            weight[:, embed_dim:] = 0.5 * eye.to(
                device=weight.device, dtype=weight.dtype
            )

        return self


# ---------------------------------------------------------------------------
# Install / save / load helpers
# ---------------------------------------------------------------------------


def _get_talker(model):
    """Return the talker from any of the public wrapper/levels."""
    if hasattr(model, "talker"):
        return model.talker
    if hasattr(model, "model") and hasattr(model.model, "talker"):
        return model.model.talker
    if hasattr(model, "code_predictor") and hasattr(model, "model"):
        # Already a Qwen3TTSTalkerForConditionalGeneration.
        return model
    raise TypeError(
        "expected a Qwen3TTSModel, Qwen3TTSForConditionalGeneration, or "
        "Qwen3TTSTalkerForConditionalGeneration"
    )


def install_codebook_mtp(
    model,
    checkpoint: Optional[str] = None,
    shared_layers: int = 4,
    use_time_attention: bool = True,
    time_sliding_window: Optional[int] = None,
    init_from_code_predictor: bool = True,
    codebook_loss_weights=None,
    device=None,
    dtype=None,
):
    """Replace `model.talker.code_predictor` with a codebook-MTP decoder.

    Args:
        model: `Qwen3TTSModel` or its inner causal-LM model.
        checkpoint: optional directory written by `save_codebook_mtp`.
        init_from_code_predictor: copy embeddings/heads/blocks from the
            existing code predictor before training.  Ignored when
            `checkpoint` is given.
    """
    talker = _get_talker(model)
    talker_config = talker.config
    old_code_predictor = getattr(talker, "code_predictor", None)

    if checkpoint is not None:
        mtp = load_codebook_mtp(
            checkpoint,
            talker_config=talker_config,
            device=device,
            dtype=dtype,
        )
    else:
        config = Qwen3TTSCodebookMTPConfig.from_code_predictor_config(
            talker_config.code_predictor_config,
            talker_config,
            shared_layers=shared_layers,
            use_time_attention=use_time_attention,
            time_sliding_window=time_sliding_window,
            codebook_loss_weights=codebook_loss_weights,
        )
        mtp = Qwen3TTSCodebookMTPForConditionalGeneration(config, talker_config)
        if init_from_code_predictor and old_code_predictor is not None:
            mtp.load_from_code_predictor(old_code_predictor)

    # Make full-model save/load construct the MTP automatically.  Store both
    # the `mtp_*` names (easy to discover in config.json) and the plain names
    # consumed by the MTP modules when `from_pretrained` rebuilds the talker.
    predictor_config = talker_config.code_predictor_config
    predictor_config.mtp_type = "codebook_mtp"
    predictor_config.mtp_shared_layers = int(
        getattr(mtp.config, "shared_layers", shared_layers)
    )
    predictor_config.mtp_use_time_attention = bool(
        getattr(mtp.config, "use_time_attention", use_time_attention)
    )
    predictor_config.mtp_time_sliding_window = getattr(
        mtp.config, "time_sliding_window", time_sliding_window
    )
    predictor_config.shared_layers = predictor_config.mtp_shared_layers
    predictor_config.use_time_attention = predictor_config.mtp_use_time_attention
    predictor_config.time_sliding_window = predictor_config.mtp_time_sliding_window
    if getattr(mtp.config, "codebook_loss_weights", None) is not None:
        predictor_config.codebook_loss_weights = list(
            mtp.config.codebook_loss_weights
        )

    if device is None or dtype is None:
        try:
            reference = next(talker.parameters())
            device = reference.device if device is None else device
            dtype = reference.dtype if dtype is None else dtype
        except StopIteration:
            pass
    if device is not None or dtype is not None:
        mtp.to(device=device, dtype=dtype)

    mtp.reset_cache()
    talker.code_predictor = mtp
    return mtp


def save_codebook_mtp(mtp, path: str):
    """Save only the MTP decoder (small, easy to move between checkpoints)."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    mtp.config.save_pretrained(path)
    state = {key: value.detach().cpu() for key, value in mtp.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"))


def load_codebook_mtp(
    path: str,
    talker_config: Optional[Qwen3TTSTalkerConfig] = None,
    device=None,
    dtype=None,
):
    """Load a checkpoint written by `save_codebook_mtp`."""
    from safetensors.torch import load_file

    config = Qwen3TTSCodebookMTPConfig.from_pretrained(path)
    if talker_config is None:
        from types import SimpleNamespace

        talker_config = SimpleNamespace(
            hidden_size=config.talker_hidden_size,
            num_code_groups=config.num_code_groups,
        )
    mtp = Qwen3TTSCodebookMTPForConditionalGeneration(config, talker_config)
    state = load_file(os.path.join(path, "model.safetensors"))
    mtp.load_state_dict(state)
    if device is not None or dtype is not None:
        mtp.to(device=device, dtype=dtype)
    mtp.eval()
    mtp.reset_cache()
    return mtp
