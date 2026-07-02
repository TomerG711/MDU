# Null-anchor uncond source & ref_model

How the fork chooses **which model** produces uncond logits `logits_u` for null-anchor KL, and when a second **frozen ref_model** is loaded.

See also: [METRICS.md](../eval_outputs/METRICS.md) (τ, RougeL).

---

## Quick reference

| `--null_anchor_source` | `match_mode=random` | denoise modes (token_id, position, …) | traj rollout |
|------------------------|---------------------|---------------------------------------|--------------|
| **auto** (default) | frozen SFT `ref_model` | trainable CFG `model` (Q masked) | frozen SFT `ref_model` |
| **frozen_sft** | frozen SFT (null-prompt uncond) | frozen SFT (null-prompt uncond) | frozen SFT |
| **pre_sft_cond** | frozen pre-SFT instruct (conditional) | frozen pre-SFT instruct (conditional) | frozen pre-SFT instruct (conditional) |
| **trainable_cfg** | trainable CFG | trainable CFG | trainable CFG |
| **ema** | EMA copy (Q-masked uncond forward) | EMA copy | EMA copy |

**EMA:** separate `ema_model` initialized from SFT; after each optimizer step  
`θ_ema ← m·θ_ema + (1−m)·θ_student` (`--null_anchor_ema_decay`, default `0.999`).  
Works with any `match_mode`. Does not load `ref_model`.

**NPO** always uses frozen `ref_model` regardless of `null_anchor_source`.

---

## Code path (forget loss)

`MDUTrainer.compute_loss` picks the forget function:

```
match_mode=random + loss_type=null_anchor + traj_rollout=false
  → _random_sft_null_anchor_loss          (τ sweep / run_main.sh)

match_mode=random + loss_type=null_anchor + traj_rollout=true
  → _traj_rollout_na_loss

else
  → _denoise_novel_ga_loss                (token_id, position, gap, …)
```

Within each null-anchor path, anchor logits come from `_null_anchor_uncond_logits`:

- **`anchor=frozen_sft`** → `_ref_forward_logits` → frozen SFT `ref_model` with null-prompt inputs (optional GPU split via `--ref_device`)
- **`anchor=pre_sft_cond`** → `_ref_forward_logits` → frozen **pre-TOFU instruct** `ref_model` with **same conditional inputs** as student (`[Q | masked A]`)
- **`anchor=trainable_cfg`** → `model(...)` with question positions masked, `torch.no_grad()`
- **`anchor=ema`** → `ema_model(...)` with question positions masked, `torch.no_grad()`; weights updated each step

Resolution logic: `resolve_null_anchor_uncond()` in `src/unlearn_run_utils.py`.

---

## When is ref_model loaded?

`needs_ref_model(data_args)` is true when:

- `loss_type=npo`, or
- `loss_type=null_anchor` **and** resolved anchor is `frozen_sft` or `pre_sft_cond`

If `trainable_cfg` or `auto` with denoise-only null-anchor, **only one model** is loaded (single-GPU friendly).

### `pre_sft_cond` ref checkpoint (`--ref_model_name_or_path`)

| Backbone | Default when empty |
|----------|-------------------|
| LLaDA | `GSAI-ML/LLaDA-8B-Instruct` |
| Dream | `Dream-org/Dream-v0-Instruct-7B` |

Student (`--model_name_or_path`) remains the TOFU-SFT checkpoint. Ref loads via `get_model(model_args, model_name_or_path=ref_path)` kwarg (avoids `BASE_MODELS_DIR` resolution on HF ids).

`null_prompt_mode` is **ignored** when `pre_sft_cond` is active.

---

## Gradients (why ref GPU split is safe)

1. Uncond forward is always under `torch.no_grad()`.
2. `_null_anchor_kl` uses `logits_u.detach()` and `log_pt.detach()`.
3. Only `logits_c` from the trainable `model` receives gradients.

`ref_device=cuda:1` only moves inference; logits are moved back to the trainable device before KL.

---

## Logs

**Once per run** (search logs for `[mdu-setup]`):

```
[mdu-setup] forget=_random_sft_null_anchor_loss match_mode=random loss_type=null_anchor
            anchor=pre_sft_cond anchor_input=conditional null_anchor_source=pre_sft_cond
            τ=0.25 ref_model=loaded@cuda:1 ref_path=GSAI-ML/LLaDA-8B-Instruct
```

**Per forget batch** (null-anchor paths only):

```
[null-anchor] ... τ=0.5 ... anchor=pre_sft_cond anchor_input=conditional kl̄=...
```

Resolved `anchor=` is what actually ran (not the raw `null_anchor_source` flag when `auto`).

**Checkpoint:** `train_config.json` → `mdu_setup` block mirrors the same fields.

---

## Null prompt mode (`--null_prompt_mode`)

Controls **only the uncond forward** input for null-anchor KL. The conditional forward is unchanged (`[visible Q | masked answer]`, full bidirectional attention).

| Mode | `input_ids` on Q | `attention_mask` on Q (uncond only) | Interpretation |
|------|------------------|-------------------------------------|----------------|
| **`mask`** (default) | `[MASK]` | full attn (`None` in training — see below) | MDU paper / CFG-style discrete null |
| **`empty`** | unchanged (real Q tokens) | **0** (synthesized; Q not attended to) | “No prompt signal” via attention |
| **`pad`** | `pad_token_id` | full attn | Neutral filler vs `[MASK]` |

**Collator note:** `MDUCollator.after` pops `attention_mask` before `compute_loss`, so training batches have `attention_mask=None`. For `empty` mode the builder **must synthesize** a mask with zeros on Q positions; otherwise the mode is a silent no-op.

Q positions are detected via `labels != -100` (same as legacy `mask` mode).

Orthogonal to `--null_anchor_source` for uncond anchors (frozen / EMA / trainable picks **which weights** run anchor; `null_prompt_mode` picks **what input** they see). **Not used** when `null_anchor_source=pre_sft_cond` (conditional anchor always uses full Q+A).

**Logs:** `[mdu-setup]` includes `null_prompt_mode=empty` when not default. Checkpoint names gain `_nullprompt_{mode}` when mode ≠ `mask`.

---

## Examples

```bash
# Paper-style τ sweep (default): random + frozen SFT ref on GPU 1
CUDA_DEVICES=0,1 NULL_ANCHOR_SOURCE=auto bash run_main.sh tofu_llada 0.5

# Random masking but CFG anchor (no second model)
CUDA_DEVICES=0 NULL_ANCHOR_SOURCE=trainable_cfg bash run_main.sh tofu_llada 0.5

# Empty-prompt uncond (Q tokens present, attention masked on Q)
CUDA_DEVICES=0 NULL_ANCHOR_SOURCE=trainable_cfg NULL_PROMPT_MODE=empty \
  bash run_main.sh tofu_llada 0.25

# Pre-SFT conditional anchor (frozen instruct, same Q+A as student)
CUDA_DEVICES=0,1 NULL_ANCHOR_SOURCE=pre_sft_cond bash run_main.sh tofu_llada 0.25

# Override pre-SFT ref path
REF_MODEL_NAME_OR_PATH=./checkpoints/my-base-instruct \
  NULL_ANCHOR_SOURCE=pre_sft_cond CUDA_DEVICES=0,1 bash run_main.sh tofu_llada 0.25

# Token-id trajectory with frozen anchor (experiment)
NULL_ANCHOR_SOURCE=frozen_sft CUDA_DEVICES=0,1 \
  accelerate launch ... --match_mode token_id --null_anchor_source frozen_sft
```
