# Codebook-MTP decoder for Qwen3-TTS

`qwen_tts/core/models/codebook_mtp.py` implements a Drama-3-style codebook
multi-token-prediction (MTP) decoder and makes it a drop-in replacement for
`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`.

The talker still generates codebook 0.  The MTP generates the 15 residual
codebooks (`1..15`) for the same frame.

## What changes

| | Original code predictor | Codebook-MTP |
|---|---|---|
| Per-frame layer passes | 5 layers x 15 codebooks = **75** | 4 shared + 15 codebook layers = **19** |
| Weight reads per frame | 75 layer reads | 19 layer reads |
| Time context | none (frame-local) | causal KV cache per layer |
| Public API | `forward_finetune`, `generate`, embeddings, heads | identical |

The new module keeps the names and shapes of the existing
`codec_embedding`, `lm_head`, and `small_to_mtp_projection`, so the pretrained
tables can be copied.  The talker code only gains two small hooks:

* `mtp_type="codebook_mtp"` in `code_predictor_config` selects the new class
  when a full model is rebuilt.
* a `reset_cache()` call at talker prefill drops the previous utterance's KV
  cache.

Nothing else in the external Qwen3-TTS API changes.

## Install / replace

```python
from qwen_tts.core.models.codebook_mtp import install_codebook_mtp

mtp = install_codebook_mtp(
    model,                       # Qwen3TTSModel or its inner model
    shared_layers=4,
    use_time_attention=True,
    init_from_code_predictor=True,   # warm-start from the 5-layer predictor
)
```

For a standalone stage-1 checkpoint:

```python
install_codebook_mtp(model, checkpoint="output/mtp_stage1/checkpoint-epoch-2/codebook_mtp")
```

`save_codebook_mtp(mtp, path)` and `load_codebook_mtp(path)` handle the small
standalone checkpoint.  A full model saved after `--train_talker` contains the
MTP automatically and does not need `--mtp_checkpoint` at inference.

## Training data: real WAV codes, not generated WAV codes

Use the codes produced from the **real target WAV** by
`finetuning/prepare_data.py`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

The MTP is a conditional decoder.  Its job is to model
`p(code_1..15 | talker_hidden, code_0, previous residual codes)`.  The real
tokenized WAV is the only unbiased sample from that target distribution.

Codes decoded from **generated** WAVs are on-policy samples from the current
model.  They are useful only as an optional regularizer:

* scheduled sampling / DAgger: replace a small fraction of previous
  codebooks with the model's own predictions during training;
* a small auxiliary loss on generated rollouts.

Do **not** train on generated codes alone.  Errors in the generated audio and
in the tokenizer become the target, the model learns to imitate its own
mistakes, and codebook diversity usually collapses.  Keep real codes as the
majority target.  `train_codebook_mtp.py` therefore uses real codes and exposes
`--scheduled_sampling` for the safe, within-frame version of this idea.

## Recommended two-stage recipe

### Stage 1: train the MTP with the talker frozen

This is the important stage.  The talker is already a strong codebook-0
predictor; only the residual decoder is new.

```bash
cd finetuning
python train_codebook_mtp.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl train_with_codes.jsonl \
  --output_model_path output/mtp_stage1 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lr 2e-4 \
  --num_epochs 3 \
  --shared_layers 4
```

The script:

1. freezes every talker parameter;
2. keeps the copied residual embedding tables frozen (they are also part of
   the talker's input embedding, so changing them would move the frozen
   talker's input distribution);
3. computes talker hidden states with `torch.no_grad()`;
4. aligns `hidden[:, :-1]` with `codec_ids[:, 1:]` and `codec_mask[:, 1:]`;
5. trains the MTP with per-codebook cross entropy (Drama 3 Eq. 9);
6. saves `checkpoint-epoch-N/codebook_mtp`.

Learning rates around `1e-4..3e-4` work for the warm-started MTP.  Use a
smaller rate if you initialize from a strong MTP checkpoint.

### Learning-rate schedule

By default the learning rate is constant (no scheduler, the historical
behaviour).  `--lr_scheduler cosine` or `linear` adds a linear warmup over
`--warmup_ratio` of the total optimizer steps and then decays to
`--min_lr_ratio` of the initial rate:

```bash
python train_codebook_mtp.py \
  ... \
  --lr_scheduler cosine --warmup_ratio 0.03 --min_lr_ratio 0.1
```

The schedule is advanced once per optimizer step (not per micro-batch), so it
matches the real number of updates, and it follows `--max_steps`.  Each
parameter group keeps its own base rate (`--lr` for the MTP, `--talker_lr` for
the talker) and is scaled by the same factor.

* **Stage 1** does not need a schedule; a short warmup is the useful part if
  the first steps look unstable.  For very small datasets the flat rate is
  simpler and differences are small.
* **Stage 2** is where a schedule matters: a flat `--talker_lr 5e-6` for a full
  epoch keeps nudging a 1.7B pretrained talker late in training.  Use
  `--lr_scheduler cosine --warmup_ratio 0.03` and keep `--talker_lr` at or
  below `5e-6`.

Without a scheduler you can emulate a step decay by stopping after N epochs and
restarting from `checkpoint-epoch-N-1/codebook_mtp` with `--init_mtp_checkpoint`
and a smaller `--lr`.

### Optional distillation from the old code predictor

The original 5-layer predictor is a useful teacher because it was trained on
the same data.  Distill its **distribution**, not its sampled codes:

```bash
python train_codebook_mtp.py \
  ... \
  --distill_weight 0.3 \
  --distill_temperature 1.0
```

The script keeps a frozen copy of the old predictor, runs it frame by frame on
the same talker hidden state and ground-truth previous codes, and adds a
per-codebook KL term.  Keep real-code cross entropy as the primary loss.  A
teacher argmax is just another hard label and can propagate teacher errors;
soft logits preserve the teacher's uncertainty across residual codebooks.

### Stage 2 (optional): joint SFT with the talker

```bash
python train_codebook_mtp.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --init_mtp_checkpoint output/mtp_stage1/checkpoint-epoch-2/codebook_mtp \
  --train_jsonl train_with_codes.jsonl \
  --output_model_path output/mtp_stage2 \
  --train_talker \
  --talker_lr 5e-6 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --warmup_ratio 0.03 \
  --mtp_loss_weight 0.5 \
  --num_epochs 1
```

Joint loss:

```text
L = L_talker(code_0) + lambda * L_mtp(code_1..15)
```

`lambda = 0.5` is a good starting point.  Do not use a huge MTP weight
(`10`, `50`, ...).  The residual codebooks are a refinement of codebook 0;
letting them dominate makes the talker drift, hurts codebook-0 accuracy, and
usually creates an exposure-bias loop: worse code 0 -> worse residuals -> worse
next hidden state.  If you must run a single-stage SFT, use `lambda` around
`0.3..1.0`, not a large value, and prefer stage 1 first.

Stage 2 saves a full model (`checkpoint-epoch-N/`), so inference can load it
directly.  The full checkpoint is saved as a CustomVoice model, matching
`finetuning/sft_12hz.py`:

* the speaker embeddings extracted from each sample's `ref_audio` are written
  into `talker.model.codec_embedding` (and the separate `speaker_encoder` is
  not saved);
* `talker_config.spk_id` / `talker_config.spk_is_dialect` are added to
  `config.json`, and `tts_model_type` becomes `custom_voice`;
* the speaker keys come from the dataset's `speaker` field.  Samples without
  that field use `--speaker_name` (default: `speaker_test`).

Use one of those saved speaker names at inference time.  A full checkpoint is
intended for `generate_custom_voice`; use the base model plus a stage-1 MTP
checkpoint if you need voice cloning.

## Inference

Full model from stage 2:

```bash
python infer_codebook_mtp.py \
  --model_path output/mtp_stage2/checkpoint-epoch-0 \
  --mode custom_voice \
  --speaker speaker_test \
  --text "Hello from the codebook MTP decoder." \
  --output_wav out.wav
```

Stage-1 MTP on top of the base model:

```bash
python infer_codebook_mtp.py \
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --mtp_checkpoint output/mtp_stage1/checkpoint-epoch-2/codebook_mtp \
  --mode voice_clone \
  --ref_audio ref.wav \
  --ref_text "Reference transcript" \
  --text "Hello from the codebook MTP decoder." \
  --output_wav out.wav
```

The script prints elapsed time and RTF.  Measure on your own hardware and
batch size; the 75 -> 19 layer-pass reduction is the MTP-only reduction, while
end-to-end latency is dominated by the talker for short utterances.

## Frame-local fallback

`use_time_attention=False` (`--no_time_attention` in training) gives a
frame-local decoder with the same information access as the original code
predictor and no KV cache.  It is the safest choice if you cannot retrain or
if you need bit-for-bit frame independence.  Drama 3's quality benefit comes
from the causal time attention, so the default is `True`.

### Frame-local training flow (`--no_time_attention`)

`use_time_attention=False` only removes the MTP's own cross-frame attention.
The talker hidden still contains the talker's causal history, so the MTP sees
the same information as the original frame-local code predictor.

Training is otherwise the same two-stage recipe:

```bash
python train_codebook_mtp.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl train_with_codes.jsonl \
  --output_model_path output/mtp_stage1_local \
  --no_time_attention \
  --lr 2e-4 --num_epochs 3
```

Inside `forward_sequence`:

1. `position_ids` are all zero, so RoPE is the identity;
2. the attention mask is the identity (`frame_local=True`), so a frame cannot
   see any other frame;
3. the 4 shared + 15 codebook layers still run once per frame, but with no KV
   cache;
4. codebooks inside a frame are still predicted sequentially, because layer
   `k` consumes the code predicted by layer `k-1`;
5. the loss is unchanged: per-codebook CE on `codec_mask[:, 1:]`.

At inference `_decode_frame` uses `position_ids=0` and no cache, so
`reset_cache()` is effectively a no-op and frames can be decoded in any order.

Because frames are independent, a faster variant is to flatten all valid
frames into the batch dimension before the MTP pass:

```python
hidden_flat = mtp_hidden[target_mask]        # [N, H]
codes_flat = target_codes[target_mask]       # [N, 16]
code0_flat = code0_embeds[target_mask]       # [N, E]
# call forward_sequence with B=N, S=1 and all-ones masks
```

This avoids the `S x S` identity mask.  The current script keeps `[B, S]` for
simplicity; both forms are mathematically equivalent in frame-local mode.

For stage 2, the checkpoint config wins: if the stage-1 checkpoint was saved
with `use_time_attention=False`, stage 2 stays frame-local even without
passing `--no_time_attention` again.

## Important training alignment

For a talker hidden state at position `t`:

```python
hidden = talker_outputs.hidden_states[0][-1][:, :-1]
target_codes = batch["codec_ids"][:, 1:]
target_mask = batch["codec_mask"][:, 1:]
code0_embeds = talker.get_input_embeddings()(target_codes[..., 0])
```

`hidden[:, t]` predicts the frame at `t + 1`.  Do not pass unshifted codes,
and do not include the EOS position in the MTP loss.

## Limitations

* The old `finetuning/sft_12hz.py` still calls `forward_finetune` on isolated
  16-token frames.  That path remains compatible but does not train the
  causal time attention.  Use `train_codebook_mtp.py` for the full decoder.
* Stage 1 caches nothing across batches; it recomputes frozen talker hidden
  states each epoch.  For very large datasets, precompute hidden states once
  with the same alignment and train the MTP on the cached tensors.
* The MTP has 19 layers instead of 5, so its parameter count is larger.  The
  per-frame weight-read count is still much lower, and the talker dominates
  total parameters.
