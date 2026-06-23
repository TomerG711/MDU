# Change review: local fork vs upstream `main`

Audit of all uncommitted work relative to `origin/main` (TomerG711/MDU).  
**No code was modified during this review** — findings only.

---

## Summary

| Verdict | Count | Meaning |
|---------|-------|---------|
| Justified | Most infra/eval/sweep changes | Replication tooling, HF data, eval provenance |
| Bug / regression | 1 critical | `unlearn_mdu_llada.py` null-anchor in `_denoise_novel_ga_loss` |
| Uncertain / verify | 2 | `apply_disable_data_parallel` timing; eval RougeL = recall vs paper F1 |
| Intentional fork divergence | Several | Checkpoint layout, W&B, 2-GPU ref split |

---

## File-by-file

### `src/unlearn_run_utils.py` (new)

**Purpose:** Shared training utilities — HF TOFU loading, checkpoint naming, W&B, ref GPU placement, `train_config.json` provenance.

| Change | Verdict |
|--------|---------|
| `load_forget_retain_rows` + HF `locuslab/TOFU` | **Justified** — matches eval pipeline; avoids local JSONL dependency |
| `resolve_run_directory` / `default_checkpoint_name` | **Justified** — named checkpoints under `./checkpoints/` |
| `place_ref_model` / `resolve_ref_device` | **Justified** — OOM mitigation on 2×GPU |
| `apply_disable_data_parallel` | **Partially justified, fragile** — sets `training_args._n_gpu = 1` before Trainer init, but HF Trainer may reset `_n_gpu` in `_setup_devices()`; logs from runs did not always show “DataParallel disabled” |
| `save_final_checkpoint` + `copy_model_python_files` | **Justified** — replaces hardcoded `./checkpoints/llada-tofu-sft/checkpoint-final` copy path |
| `prepare_wandb_run_name` | **Justified** — fixes stale `./checkpoints/_mdu_run` W&B display name |

---

### `scripts/tofu_data.py` (new)

**Purpose:** Shared TOFU loader for train + eval (HF or local JSONL).

| Change | Verdict |
|--------|---------|
| `load_tofu_hf` / `load_tofu_split` | **Justified** — single source of truth for eval splits |

---

### `src/unlearn_mdu_llada.py` (modified)

| Change | Verdict |
|--------|---------|
| Import `unlearn_run_utils` | **Justified** |
| `_ref_forward_logits` + `ref_device` | **Justified** for multi-GPU ref forwards |
| **`_denoise_novel_ga_loss` null_anchor: `model(...)` → `_ref_forward_logits(...)`** | **FIXED** — restored via `--null_anchor_source auto` (default) |
| `_random_sft_null_anchor_loss` → `_ref_forward_logits` | **Justified** (same as upstream `ref_model`, adds GPU routing) |
| NPO / traj_rollout → `_ref_forward_logits` | **Justified** |
| HF data args, checkpoint args, W&B | **Justified** |
| Remove inline `load_tofu` | **Justified** |

**Critical detail (upstream TomerG711/leegeoru ~L867–870):**

```python
# UPSTREAM (position/token_id path) — CFG: same trainable model, Q masked
outputs_u = model(input_ids=noised_u, ...)
```

```python
# LOCAL (incorrect for trajectory modes)
logits_u = self._ref_forward_logits(noised_u, ...)
```

Upstream uses **trainable `model`** for uncond in `_denoise_novel_ga_loss`; **frozen `ref_model`** only in `_random_sft_null_anchor_loss`. Local change was collateral from ref-GPU refactor.

**Note:** `unlearn_mdu_dream.py` still has upstream behavior (`model` in denoise path) — **inconsistency between backbones**.

---

### `src/unlearn_mdu_dream.py` (modified)

Same infra as llada except **`_denoise_novel_ga_loss` null_anchor still uses `model(...)`** — matches upstream.

| Change | Verdict |
|--------|---------|
| Ref device / `_ref_forward_logits` / run utils | **Justified** |
| Denoise null-anchor unchanged | **Correct vs upstream** |

---

### `scripts/eval_tofu_llada.py` (modified)

| Change | Verdict |
|--------|---------|
| `--experiment` / `--run_id` / `eval_outputs/` layout | **Justified** — provenance, manifest, summary |
| HF `tofu_split` loading | **Justified** |
| `split_meta.json`, git fingerprint, script sha256 | **Justified** for replication audit |
| RougeL still `.recall` | **Documented gap** — MDU paper Table 2 “rL” is LCS **F1**; TOFU paper uses recall; our table mixes ours (recall) vs paper (F1) |
| Early abort / `finalize_split_output` | **Justified** |

`eval_tofu_dream.py`: smaller parallel changes — **justified**.

---

### `scripts/run_mdu_tau_sweep.sh` (new)

| Change | Verdict |
|--------|---------|
| Sequential τ sweep, pipelined eval | **Justified** |
| Eval wait fix (`$!` + `_eval_wait_any`, not `wait -n` in subshell) | **Justified** — see `test_eval_wait_bug.sh` |
| `SKIP_TAU_*`, resume flags | **Justified** |
| Default `PER_DEVICE_BATCH=2`, `GRAD_ACCUM=8` | **Justified** (OOM); differs from paper 4×4 — note in replication |
| `match_mode random`, no `novel_percentile` in CLI | **OK for null_anchor** — random path uses `_random_sft_null_anchor_loss`, not trajectory weights |

---

### `scripts/test_eval_wait_bug.sh` (new)

**Justified** — documents and tests sweep eval wait regression.

---

### `run_main.sh` (modified)

| Change | Verdict |
|--------|---------|
| Checkpoint-based output vs `--output_dir` arg | **Justified** |
| `REF_DEVICE`, `DISABLE_DP`, W&B | **Justified** |
| Default `LLADA_BASE_SFT=./checkpoints/...` | **Justified** for this fork’s layout |
| Removed `--novel_percentile 100` from CLI | **Neutral for null_anchor+random**; would matter for GA/NPO random or trajectory modes |

---

### `requirements.txt` (modified)

Added peft, tyro, wandb, lm-eval, etc. + dllm clone note.

**Justified** — documents actual runtime deps for this fork.

---

### `eval_outputs/` (reports only in PR)

| Artifact | Included | Verdict |
|----------|----------|---------|
| `RESULTS.md`, `METRICS.md`, `README.md` | Yes | **Justified** |
| `summary.json`, `manifest.json`, `results.json`, `split_meta.json` | Yes | **Justified** |
| `details.jsonl`, `run.log` | No (gitignore) | Too large / raw logs |

---

## Post-fix self-review (2026-06-22)

### Fixed: `_denoise_novel_ga_loss` null-anchor regression

| Path | Upstream | After fix (`null_anchor_source=auto`) |
|------|----------|----------------------------------------|
| `_denoise_novel_ga_loss` (position/token_id/…) | trainable `model`, Q-masked CFG | **same** via `_null_anchor_uncond_logits` |
| `_random_sft_null_anchor_loss` | frozen `ref_model` | **same** (with optional `_ref_forward_logits` GPU split) |
| `_traj_rollout_na_loss` | frozen `ref_model` | **same** via helper |
| NPO (all paths) | frozen `ref_model` | **unchanged** |

### New: `--null_anchor_source`

| Value | Uncond logits | Loads ref? |
|-------|---------------|------------|
| `auto` (default) | random/traj → frozen SFT; denoise → trainable CFG | only when random/traj/NPO |
| `frozen_sft` | always frozen ref | yes (null_anchor) |
| `trainable_cfg` | always trainable CFG | no (null_anchor); single-GPU OK |

### Script snapshots

- Training: `unlearn_mdu_llada.py` / `unlearn_mdu_dream.py` copied to checkpoint dir at start + final save (`train_config.json` → `training_script_snapshot`).
- Eval: `eval_tofu_llada.py` copied to `eval_outputs/<experiment>/<run_id>/` on first split.

### Upstream parity check

Compared `origin/main` denoise null-anchor block: restored `model(input_ids=noised_u, …)` behavior under `auto`.  
Random/traj still use `ref_model` as upstream. NPO still requires ref. No change to retain path or GA losses.

---

## Impact on completed τ runs

τ sweep used `match_mode=random` + `null_anchor` + default `null_anchor_source=auto` → `_random_sft_null_anchor_loss` → **frozen `ref_model`**.  
The `_denoise_novel_ga_loss` bug **did not affect** those runs (unchanged behavior for random+auto).

Runs may still differ from paper due to: RougeL recall vs F1, batch 2×8 vs 4×4, eval MC setup, SFT checkpoint path.

---

## Recommended follow-ups (not done in this PR)

1. ~~Revert `_denoise_novel_ga_loss` null-anchor~~ **Done** — use `--null_anchor_source auto` (default).
2. Confirm `apply_disable_data_parallel` runs after Trainer device setup or verify no silent DataParallel.
3. Optionally switch eval to `rougeL.fmeasure` for paper-aligned rL.
4. Align `METRICS.md` random/`novel_percentile` description with actual `compute_loss` branch.
5. Add `scripts/verify_ref_device_equivalence.py` for OOM-safe ref GPU roundtrip test.
