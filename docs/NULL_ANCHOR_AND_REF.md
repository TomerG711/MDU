# Null-anchor uncond source & ref_model

How the fork chooses **which model** produces uncond logits `logits_u` for null-anchor KL, and when a second **frozen ref_model** is loaded.

See also: [METRICS.md](../eval_outputs/METRICS.md) (τ, RougeL).

---

## Quick reference

| `--null_anchor_source` | `match_mode=random` | denoise modes (token_id, position, …) | traj rollout |
|------------------------|---------------------|---------------------------------------|--------------|
| **auto** (default) | frozen SFT `ref_model` | trainable CFG `model` (Q masked) | frozen SFT `ref_model` |
| **frozen_sft** | frozen SFT | frozen SFT | frozen SFT |
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

Within each null-anchor path, uncond logits come from `_null_anchor_uncond_logits`:

- **`uncond=frozen_sft`** → `_ref_forward_logits` → frozen `ref_model` (optional GPU split via `--ref_device`)
- **`uncond=trainable_cfg`** → `model(...)` with question positions masked, `torch.no_grad()`
- **`uncond=ema`** → `ema_model(...)` with question positions masked, `torch.no_grad()`; weights updated each step

Resolution logic: `null_anchor_uses_frozen_ref()` in `src/unlearn_run_utils.py`.

---

## When is ref_model loaded?

`needs_ref_model(data_args)` is true when:

- `loss_type=npo`, or
- `loss_type=null_anchor` **and** resolved uncond is `frozen_sft`

If `trainable_cfg` or `auto` with denoise-only null-anchor, **only one model** is loaded (single-GPU friendly).

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
            uncond=frozen_sft null_anchor_source=auto τ=0.5 ref_model=loaded@cuda:1
```

**Per forget batch** (null-anchor paths only):

```
[null-anchor] ... τ=0.5 ... uncond=frozen_sft kl̄=...
```

Resolved `uncond=` is what actually ran (not the raw `null_anchor_source` flag when `auto`).

**Checkpoint:** `train_config.json` → `mdu_setup` block mirrors the same fields.

---

## Examples

```bash
# Paper-style τ sweep (default): random + frozen SFT ref on GPU 1
CUDA_DEVICES=0,1 NULL_ANCHOR_SOURCE=auto bash run_main.sh tofu_llada 0.5

# Random masking but CFG anchor (no second model)
CUDA_DEVICES=0 NULL_ANCHOR_SOURCE=trainable_cfg bash run_main.sh tofu_llada 0.5

# Token-id trajectory with frozen anchor (experiment)
NULL_ANCHOR_SOURCE=frozen_sft CUDA_DEVICES=0,1 \
  accelerate launch ... --match_mode token_id --null_anchor_source frozen_sft
```
