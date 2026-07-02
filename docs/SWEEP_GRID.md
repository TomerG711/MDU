# MDU grid sweep: τ × match_mode × null_anchor_source

Orchestrated by `scripts/run_mdu_tau_sweep.sh`.

## Default: τ sweep only (one config per invocation)

The script loops over **τ only**. Set `MATCH_MODE` and `NULL_ANCHOR_SOURCE` once per run (~5 checkpoints ≈ 100G).

```bash
# Default: random + frozen (legacy names; skips existing ckpts)
bash scripts/run_mdu_tau_sweep.sh

# Next config — delete or archive prior ckpts first if disk-limited
MATCH_MODE=token_id NULL_ANCHOR_SOURCE=frozen_sft bash scripts/run_mdu_tau_sweep.sh
MATCH_MODE=token_id NULL_ANCHOR_SOURCE=trainable_cfg bash scripts/run_mdu_tau_sweep.sh
MATCH_MODE=position NULL_ANCHOR_SOURCE=frozen_sft bash scripts/run_mdu_tau_sweep.sh
# ... etc. (6 configs × 5 τ = 30 total, run separately)
```

| Env | Default |
|-----|---------|
| `MATCH_MODE` | `random` |
| `NULL_ANCHOR_SOURCE` | `frozen_sft` |
| `NULL_PROMPT_MODE` | `mask` |
| `REF_MODEL_NAME_OR_PATH` | *(empty; see per-source defaults)* |
| `LLADA_PRE_SFT_REF` | `GSAI-ML/LLaDA-8B-Instruct` |
| `TAUS` | `0 0.25 0.5 0.75 1` |

## Full grid (30 runs = 6 separate invocations)

From `run_main.sh` + `configs/mdu_tofu.yaml`:

| Setting | Upstream value | Our sweep |
|---------|----------------|-----------|
| `loss_type` | *(not passed; code default `ga` — likely release oversight)* | **`null_anchor` explicit** |
| `match_mode` | *(not passed; code default **`token_id`**)* | grid includes `token_id`, `position`, `random` |
| `novel_percentile` | **100** | **100** for `token_id`/`position` |
| `denoise_steps` | 128 | 128 |
| `null_anchor_eta` | 0 | 0 |
| `null_anchor_kl_dir` | forward | forward |
| `alpha` (retain) | 1.0 | 1.0 |
| batch | 4×4 = 16 | 2×8 = 16 (OOM) |
| anchor | `frozen_unconditional: true` in yaml | `frozen_sft` in grid; `trainable_cfg` for ablation |

**Important:** Upstream does **not** use `match_mode=random` in `run_main.sh`. Paper TOFU likely used **`token_id`** + **`novel_percentile=100`** + frozen unconditional anchor. Our earlier τ sweep used `random` + `frozen_sft` (different token-selection strategy).

### Per `match_mode` behavior

| `match_mode` | Forget function | Token selection | `novel_percentile` |
|--------------|-----------------|-----------------|-------------------|
| `random` | `_random_sft_null_anchor_loss` | Diffusion random mask (not percentile) | N/A |
| `token_id` | `_denoise_novel_ga_loss` | Late unmask → token-id set | **100** = all late tokens |
| `position` | `_denoise_novel_ga_loss` | Late unmask positions directly | **100** |

### Per `null_anchor_source`

| Source | Anchor | GPUs |
|--------|--------|------|
| `frozen_sft` | Frozen SFT `ref_model`, null-prompt uncond | `CUDA_DEVICES_FROZEN=0,1` |
| `pre_sft_cond` | Frozen pre-TOFU instruct (`GSAI-ML/LLaDA-8B-Instruct` default), **conditional** same Q+A | `CUDA_DEVICES_FROZEN=0,1` |
| `trainable_cfg` | Trainable `model`, Q masked | `CUDA_DEVICES_TRAINABLE=0` |

See [NULL_ANCHOR_AND_REF.md](NULL_ANCHOR_AND_REF.md).

## Naming

**Legacy** (existing random + frozen_sft runs — not re-trained if checkpoint exists):

- Checkpoint: `mdu_llada_forget10_nullanchor_tau0p5`
- Eval: `eval_outputs/mdu_tau0p5/2026-06-22_mdu_tau0p5_v1`

**New grid** (example):

- Checkpoint: `mdu_llada_forget10_token_id_cfg_tau0p5`
- Eval: `eval_outputs/mdu_token_id_cfg/2026-06-22_tau0p5_v1`

**Non-default null prompt** (example `empty`):

- Checkpoint: `mdu_llada_forget10_position_cfg_nullprompt_empty_tau0p25`
- Eval: `eval_outputs/mdu_position_cfg_nullprompt_empty/2026-06-22_tau0p25_v1`

**Pre-SFT conditional anchor** (example):

- Checkpoint: `mdu_llada_forget10_position_presftcond_tau0p25`
- Eval: `eval_outputs/mdu_position_presftcond/2026-07-02_tau0p25_v1`

## Pre-SFT conditional anchor (`pre_sft_cond`)

Frozen pre-TOFU instruct ref with **conditional** inputs (same `[Q|masked A]` as student). Default ref: `GSAI-ML/LLaDA-8B-Instruct`. Works with any `match_mode`.

```bash
MATCH_MODE=position NULL_ANCHOR_SOURCE=pre_sft_cond \
  GRADIENT_CHECKPOINTING=1 TAUS="0.25 0.5" bash scripts/run_mdu_tau_sweep.sh
```

Compare vs `mdu_position_cfg` (best forget rL **0.016** @ τ=0.25).

## Empty-prompt ablation

Compare against mask baseline (`position + trainable_cfg @ τ=0.25`: forget rL 0.016, retain rL 0.846 in `eval_outputs/RESULTS.md`).

**Note:** `empty` gave a small forget win on `random` (~0.01–0.05 rL); **position+empty not run** — still worth completing:

```bash
MATCH_MODE=position NULL_ANCHOR_SOURCE=trainable_cfg NULL_PROMPT_MODE=empty \
  GRADIENT_CHECKPOINTING=1 TAUS="0.25 0.5" bash scripts/run_mdu_tau_sweep.sh
```

Quick single-τ probe:

```bash
MATCH_MODE=position NULL_ANCHOR_SOURCE=trainable_cfg NULL_PROMPT_MODE=empty \
  GRADIENT_CHECKPOINTING=1 TAUS="0.25" bash scripts/run_mdu_tau_sweep.sh
```

If forget/retain are flat or worse on both axes, `empty` is not worth pursuing; try `pad` only if warranted.

## Examples

```bash
DRY_RUN=1 bash scripts/run_mdu_tau_sweep.sh
MATCH_MODE=token_id NULL_ANCHOR_SOURCE=frozen_sft TAUS="0.5" bash scripts/run_mdu_tau_sweep.sh
```

## Upstream paper TOFU settings (leegeoru/MDU)
