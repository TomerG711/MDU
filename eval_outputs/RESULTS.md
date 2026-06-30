# TOFU eval results (LLaDA, forget10)

Consolidated comparison vs [MDU paper Table 2](https://arxiv.org/abs/2605.18253).  
Per-run provenance: `summary.json` / `manifest.json` under each `<experiment>/<run_id>/`.

**Metrics caveat:** **rL** in tables below is RougeL **recall** (`eval_tofu_llada.py`, TOFU-style). The MDU paper Table 2 **rL** column is RougeL **F1** — numbers are not directly comparable without re-eval or conversion. See [METRICS.md](METRICS.md).

**Eval config (all our runs):** `max_new_tokens=128`, `steps=256`, `mask_samples=128`, `seed=42`, `truth_ratio=false`, HF `locuslab/TOFU`.

**Direction:** lower is better on **Forget**; higher is better on **Retain**, **RA**, **WF**.

**Anchor naming:** *frozen anchor* = `null_anchor_source=frozen_sft` (frozen SFT `ref_model`); *trainable anchor* = `null_anchor_source=trainable_cfg` (trainable model, Q masked). For `token_id` / `position` denoise paths, **`auto` resolves to the same trainable CFG anchor** as `trainable_cfg` (only naming, logging, and checkpoint paths differ).

---

## Sweep status (2026-06-29)

| Sweep | `match_mode` | Anchor | τ values | Eval splits | Status |
|-------|--------------|--------|----------|-------------|--------|
| SFT baseline | — | — | — | 4/4 | complete |
| `mdu_tau*` | `random` | frozen | 0 … 1 | 20/20 | complete |
| `mdu_random_cfg` | `random` | trainable | 0 … 1 | 20/20 | complete |
| `mdu_random_ema` | `random` | **ema** (decay=0.999) | 0.25, 0.5 | 8/8 | complete |
| `mdu_random_ema0p99` | `random` | **ema** (decay=0.99) | 0.25, 0.5 | 8/8 | complete |
| `mdu_position_frozen` | `position` | frozen | 0 … 1 | 20/20 | complete |
| `mdu_position_cfg` | `position` | trainable | 0 … 1 | 20/20 | complete |
| `mdu_token_id_frozen` | `token_id` | frozen | 0 … 1 | 20/20 | complete |
| `mdu_token_id_cfg` | `token_id` | trainable | 0 … 1 | 20/20 | complete |

**Full grid:** 6 configs × 5 τ = 30 runs complete, plus EMA smoke (2 decays × 2 τ). All eval splits validated: `status=completed`, expected line counts (400 / 400 / 117 / 100).

---

## Random + EMA anchor — smoke tests complete

**Purpose:** Validate `null_anchor_source=ema` (lagged student copy as null anchor). Sweeps completed 2026-06-29.

**Training (both):** `match_mode=random`, `null_anchor_source=ema`, 2-GPU layout (`CUDA_DEVICES=0,1`, `ref_device=auto`), 9 ep, lr=1e-5, batch 2×8. τ = **0.25, 0.5** only.

### decay = 0.999 (`mdu_random_ema`)

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.25** | 0.232 | 0.016 | 0.798 | 0.462 | 0.507 | 0.172 | 0.789 | 0.275 |
| **τ=0.50** | 0.567 | 0.465 | 0.749 | 0.583 | 0.565 | 0.157 | 0.794 | 0.238 |

| τ | Eval | W&B |
|---|------|-----|
| 0.25 | [`mdu_random_ema/2026-06-29_tau0p25_v1`](./mdu_random_ema/2026-06-29_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/r31nzt3p) |
| 0.5 | [`mdu_random_ema/2026-06-29_tau0p5_v1`](./mdu_random_ema/2026-06-29_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/xfij8d5o) |

**vs random+frozen:** Nearly identical (e.g. τ=0.25 forget rL 0.232 vs 0.233). Decay=0.999 moves θ_ema only ~36% toward the student over 450 steps → anchor ≈ frozen θ₀.

### decay = 0.99 (`mdu_random_ema0p99`)

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.25** | 0.213 | 0.014 | 0.795 | 0.459 | 0.537 | 0.172 | 0.804 | 0.276 |
| **τ=0.50** | 0.554 | 0.451 | 0.746 | 0.580 | 0.561 | 0.158 | 0.790 | 0.240 |

| τ | Eval | W&B |
|---|------|-----|
| 0.25 | [`mdu_random_ema0p99/2026-06-29_tau0p25_v1`](./mdu_random_ema0p99/2026-06-29_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/can69ujp) |
| 0.5 | [`mdu_random_ema0p99/2026-06-29_tau0p5_v1`](./mdu_random_ema0p99/2026-06-29_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/132fga2g) |

**vs decay=0.999:** Small but consistent forget improvement (τ=0.25 rL 0.213 vs 0.232; τ=0.5 rL 0.554 vs 0.567). Retain essentially unchanged.

**vs random+frozen:** Slightly better forget at both τ; still nowhere near trainable (τ=0.25 rL 0.213 vs 0.183; τ=0.5 rL 0.554 vs 0.486).

**vs random+trainable:** EMA₀.₉₉ does **not** close the gap. The τ≥0.5 forget collapse (~rL 0.55, p ~0.45) is shared by frozen, both EMA decays, and only partially improved by trainable.

**Takeaway:** On `random` match_mode, anchor type (frozen / slow EMA / fast EMA) is a **second-order knob**. Decay must be ≪0.999 to matter at all; even 0.99 only nudges metrics. Trajectory `match_mode` + trainable anchor dominate forget quality. EMA is unlikely to rescue `random`; if pursued further, try **0.995–0.998** (middle ground) on **`position+ema`**, not another `random` sweep.

**Logs:** `sweep_logs/mdu_tau_sweep_random_ema_{2026-06-29,ema0p99_2026-06-29}.log`

Checkpoints: `checkpoints/mdu_llada_forget10_random_ema_tau{0p25,0p5}/`, `checkpoints/mdu_llada_forget10_random_ema0p99_tau{0p25,0p5}/`.

---

## Paper reference

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Paper (Base SFT)** | 0.884 | 0.380 | 0.870 | 0.330 | 0.611 | 0.041 | 0.835 | 0.143 |
| **Paper (MDU τ=0.00)** | 0.069 | 0.000 | 0.868 | 0.381 | 0.629 | 0.093 | 0.848 | 0.193 |
| **Paper (MDU τ=0.25)** | 0.135 | 0.000 | 0.857 | 0.392 | 0.616 | 0.116 | 0.842 | 0.204 |
| **Paper (MDU τ=0.50)** | 0.098 | 0.001 | 0.853 | 0.447 | 0.645 | 0.133 | 0.842 | 0.205 |
| **Paper (MDU τ=0.75)** | 0.078 | 0.040 | 0.684 | 0.535 | 0.612 | 0.155 | 0.827 | 0.233 |
| **Paper (MDU τ=1.00)** | 0.034 | 0.074 | 0.511 | 0.485 | 0.568 | 0.110 | 0.777 | 0.187 |

---

## SFT baseline

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ours** | 0.854 | 0.511 | 0.857 | 0.509 | 0.603 | 0.139 | 0.773 | 0.247 |

| Eval |
|------|
| [`sft_baseline/2026-06-22_sft_v1`](./sft_baseline/2026-06-22_sft_v1/) |

---

## Random + frozen anchor (`mdu_tau*`)

**Training:** `match_mode=random`, frozen anchor (`null_anchor_source=frozen_sft`, auto), `novel_percentile` unused, 9 ep, lr=1e-5, batch 2×8, `ref_device=auto`, GPUs 0+1. Sweep completed 2026-06-22.

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.087 | 0.001 | 0.871 | 0.536 | 0.526 | 0.155 | 0.792 | 0.257 |
| **τ=0.25** | 0.233 | 0.016 | 0.797 | 0.461 | 0.504 | 0.172 | 0.794 | 0.276 |
| **τ=0.50** | 0.566 | 0.465 | 0.753 | 0.582 | 0.553 | 0.156 | 0.786 | 0.238 |
| **τ=0.75** | 0.564 | 0.480 | 0.720 | 0.573 | 0.569 | 0.135 | 0.799 | 0.220 |
| **τ=1.00** | 0.561 | 0.470 | 0.702 | 0.560 | 0.593 | 0.123 | 0.790 | 0.210 |

| τ | Eval |
|---|------|
| 0 | [`mdu_tau0/2026-06-22_mdu_tau0_v1`](./mdu_tau0/2026-06-22_mdu_tau0_v1/) |
| 0.25 | [`mdu_tau0p25/2026-06-22_mdu_tau0p25_v1`](./mdu_tau0p25/2026-06-22_mdu_tau0p25_v1/) |
| 0.5 | [`mdu_tau0p5/2026-06-22_mdu_tau0p5_v1`](./mdu_tau0p5/2026-06-22_mdu_tau0p5_v1/) |
| 0.75 | [`mdu_tau0p75/2026-06-22_mdu_tau0p75_v1`](./mdu_tau0p75/2026-06-22_mdu_tau0p75_v1/) |
| 1 | [`mdu_tau1/2026-06-22_mdu_tau1_v1`](./mdu_tau1/2026-06-22_mdu_tau1_v1/) |

τ=0 and τ=0.25 forget are close to paper; **τ≥0.5** forget stays high (rL ~0.56) vs paper (~0.03–0.10).

---

## Random + trainable anchor (`mdu_random_cfg`)

**Training:** `match_mode=random`, trainable anchor, single GPU, no `ref_model`, `GRADIENT_CHECKPOINTING` off. Sweep completed 2026-06-24.

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.087 | 0.001 | 0.871 | 0.536 | 0.526 | 0.155 | 0.792 | 0.257 |
| **τ=0.25** | 0.183 | 0.010 | 0.792 | 0.440 | 0.522 | 0.169 | 0.792 | 0.273 |
| **τ=0.50** | 0.486 | 0.384 | 0.721 | 0.565 | 0.538 | 0.159 | 0.765 | 0.242 |
| **τ=0.75** | 0.507 | 0.417 | 0.690 | 0.566 | 0.555 | 0.134 | 0.779 | 0.216 |
| **τ=1.00** | 0.482 | 0.386 | 0.668 | 0.541 | 0.528 | 0.119 | 0.769 | 0.202 |

**vs frozen anchor (forget rL):** τ=0 identical; τ≥0.25 trainable is 0.05–0.08 lower than frozen.

| τ | Eval | W&B |
|---|------|-----|
| 0 | [`2026-06-24_tau0_v1`](./mdu_random_cfg/2026-06-24_tau0_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/cs4ar6lx) |
| 0.25 | [`2026-06-24_tau0p25_v1`](./mdu_random_cfg/2026-06-24_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/v0zu0j9s) |
| 0.5 | [`2026-06-24_tau0p5_v1`](./mdu_random_cfg/2026-06-24_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/41btc0gx) |
| 0.75 | [`2026-06-24_tau0p75_v1`](./mdu_random_cfg/2026-06-24_tau0p75_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/6iwrq8wq) |
| 1 | [`2026-06-24_tau1_v1`](./mdu_random_cfg/2026-06-24_tau1_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/tfimfpj9) |

---

## Position + frozen anchor (`mdu_position_frozen`)

**Training:** `match_mode=position`, frozen anchor, `novel_percentile=100`, `denoise_steps=128`, `GRADIENT_CHECKPOINTING=1`, `ref_device=auto`, GPUs 0+1, 9 ep, lr=1e-5, batch 2×8. Sweep completed 2026-06-26 (`sweep_logs/mdu_tau_sweep_position_frozen_2026-06-25.log`). First attempt without GC OOM'd at step 12; relaunched with `GRADIENT_CHECKPOINTING=1`.

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.060 | 0.002 | 0.871 | 0.502 | 0.490 | 0.116 | 0.806 | 0.223 |
| **τ=0.25** | 0.022 | 0.000 | 0.852 | 0.522 | 0.512 | 0.123 | 0.791 | 0.207 |
| **τ=0.50** | 0.108 | 0.025 | 0.801 | 0.525 | 0.481 | 0.120 | 0.777 | 0.204 |
| **τ=0.75** | 0.160 | 0.183 | 0.643 | 0.489 | 0.462 | 0.102 | 0.752 | 0.189 |
| **τ=1.00** | 0.172 | 0.225 | 0.544 | 0.475 | 0.451 | 0.095 | 0.750 | 0.178 |

**vs trainable anchor (forget rL):** τ=0 identical (0.060); τ=0.25–0.75 frozen is 0.006–0.05 higher (weaker unlearn); τ=1.0 tied (~0.17). **Retain:** similar at τ≤0.25; frozen drops harder at τ≥0.5 (e.g. τ=1: 0.544 vs 0.631 trainable).

| τ | Eval | W&B |
|---|------|-----|
| 0 | [`2026-06-25_tau0_v1`](./mdu_position_frozen/2026-06-25_tau0_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/pk23qkx1) |
| 0.25 | [`2026-06-25_tau0p25_v1`](./mdu_position_frozen/2026-06-25_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/llnlzp1i) |
| 0.5 | [`2026-06-25_tau0p5_v1`](./mdu_position_frozen/2026-06-25_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/ergpy8xv) |
| 0.75 | [`2026-06-25_tau0p75_v1`](./mdu_position_frozen/2026-06-25_tau0p75_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/7zsrbt71) |
| 1 | [`2026-06-25_tau1_v1`](./mdu_position_frozen/2026-06-25_tau1_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/ucozgdsa) |

---

## Position + trainable anchor (`mdu_position_cfg`)

**Training:** `match_mode=position`, trainable anchor, `novel_percentile=100`, `denoise_steps=128`, `GRADIENT_CHECKPOINTING=1`, single GPU, 9 ep, lr=1e-5, batch 2×8. Sweep completed 2026-06-25 (`sweep_logs/mdu_tau_sweep_position_cfg_2026-06-25.log`).

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.060 | 0.002 | 0.871 | 0.502 | 0.490 | 0.116 | 0.806 | 0.223 |
| **τ=0.25** | 0.016 | 0.000 | 0.846 | 0.519 | 0.487 | 0.122 | 0.776 | 0.206 |
| **τ=0.50** | 0.057 | 0.002 | 0.819 | 0.535 | 0.456 | 0.118 | 0.760 | 0.195 |
| **τ=0.75** | 0.120 | 0.063 | 0.766 | 0.519 | 0.466 | 0.101 | 0.727 | 0.182 |
| **τ=1.00** | 0.171 | 0.127 | 0.631 | 0.490 | 0.470 | 0.087 | 0.730 | 0.168 |

**vs paper (forget rL):** τ=0.0/0.25/0.5 match or beat paper; τ=0.75–1.0 retain drops more than paper at high τ.

**vs random frozen anchor (forget rL):** position+trainable is far stronger at all τ (e.g. τ=0.5: 0.057 vs 0.566 random frozen).

| τ | Eval | W&B |
|---|------|-----|
| 0 | [`2026-06-24_tau0_v1`](./mdu_position_cfg/2026-06-24_tau0_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/1mau0sqx) |
| 0.25 | [`2026-06-24_tau0p25_v1`](./mdu_position_cfg/2026-06-24_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/90ecgqzg) |
| 0.5 | [`2026-06-25_tau0p5_v1`](./mdu_position_cfg/2026-06-25_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/5x3q1n8j) |
| 0.75 | [`2026-06-25_tau0p75_v1`](./mdu_position_cfg/2026-06-25_tau0p75_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/xpfgxzx6) |
| 1 | [`2026-06-25_tau1_v1`](./mdu_position_cfg/2026-06-25_tau1_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/tpeeyviu) |

---

## Token ID + frozen anchor (`mdu_token_id_frozen`)

**Training:** `match_mode=token_id`, frozen anchor, `novel_percentile=100`, `denoise_steps=128`, `GRADIENT_CHECKPOINTING=1`, `ref_device=auto`, GPUs 0+1, 9 ep, lr=1e-5, batch 2×8. Sweep completed 2026-06-27 (`sweep_logs/mdu_tau_sweep_token_id_frozen_2026-06-26.log`). Paper-likely upstream config (code default `match_mode=token_id`; upstream `run_main.sh` passes `novel_percentile=100`).

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.061 | 0.001 | 0.876 | 0.509 | 0.514 | 0.124 | 0.817 | 0.214 |
| **τ=0.25** | 0.048 | 0.000 | 0.872 | 0.524 | 0.519 | 0.131 | 0.794 | 0.221 |
| **τ=0.50** | 0.343 | 0.080 | 0.787 | 0.473 | 0.506 | 0.121 | 0.790 | 0.210 |
| **τ=0.75** | 0.365 | 0.303 | 0.740 | 0.515 | 0.527 | 0.119 | 0.772 | 0.211 |
| **τ=1.00** | 0.423 | 0.319 | 0.747 | 0.513 | 0.545 | 0.113 | 0.758 | 0.211 |

**vs paper (forget rL):** τ=0.0/0.25 match or beat paper; **τ≥0.5 forget degrades sharply** (0.34–0.42 vs paper ~0.03–0.10) while retain stays relatively high (~0.74–0.79).

**vs position+frozen anchor (forget rL):** τ=0 tied (~0.06); τ=0.25–1.0 token_id is much weaker (e.g. τ=0.5: 0.343 vs 0.108 position frozen, 0.057 position trainable).

| τ | Eval | W&B |
|---|------|-----|
| 0 | [`2026-06-26_tau0_v1`](./mdu_token_id_frozen/2026-06-26_tau0_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/t6nylcbc) |
| 0.25 | [`2026-06-26_tau0p25_v1`](./mdu_token_id_frozen/2026-06-26_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/ua27z2n2) |
| 0.5 | [`2026-06-26_tau0p5_v1`](./mdu_token_id_frozen/2026-06-26_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/37f917nv) |
| 0.75 | [`2026-06-26_tau0p75_v1`](./mdu_token_id_frozen/2026-06-26_tau0p75_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/d1f404vm) |
| 1 | [`2026-06-26_tau1_v1`](./mdu_token_id_frozen/2026-06-26_tau1_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/qbmour9c) |

---

## Token ID + trainable anchor (`mdu_token_id_cfg`)

**Training:** `match_mode=token_id`, trainable anchor, `novel_percentile=100`, `denoise_steps=128`, `GRADIENT_CHECKPOINTING=1`, single GPU, 9 ep, lr=1e-5, batch 2×8. Sweep completed 2026-06-28 (`sweep_logs/mdu_tau_sweep_token_id_cfg_2026-06-27.log`). First attempt failed on disk-full during τ=0 save; relaunched after checkpoint cleanup.

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **τ=0.00** | 0.061 | 0.001 | 0.876 | 0.509 | 0.514 | 0.124 | 0.817 | 0.214 |
| **τ=0.25** | 0.038 | 0.000 | 0.871 | 0.522 | 0.519 | 0.129 | 0.761 | 0.216 |
| **τ=0.50** | 0.092 | 0.001 | 0.863 | 0.554 | 0.517 | 0.128 | 0.792 | 0.205 |
| **τ=0.75** | 0.245 | 0.135 | 0.766 | 0.541 | 0.490 | 0.117 | 0.741 | 0.196 |
| **τ=1.00** | 0.316 | 0.189 | 0.776 | 0.524 | 0.529 | 0.108 | 0.742 | 0.193 |

**vs token_id+frozen anchor (forget rL):** τ=0 tied (0.061); trainable much stronger at τ≥0.5 (e.g. τ=0.5: 0.092 vs 0.343 frozen).

**vs position+trainable anchor (forget rL):** position+trainable still best overall (τ=0.25: 0.016 vs 0.038); token_id+trainable beats token_id+frozen at all τ>0.

| τ | Eval | W&B |
|---|------|-----|
| 0 | [`2026-06-27_tau0_v1`](./mdu_token_id_cfg/2026-06-27_tau0_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/6jcfkw3i) |
| 0.25 | [`2026-06-27_tau0p25_v1`](./mdu_token_id_cfg/2026-06-27_tau0p25_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/42lekus3) |
| 0.5 | [`2026-06-27_tau0p5_v1`](./mdu_token_id_cfg/2026-06-27_tau0p5_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/fg6unv8z) |
| 0.75 | [`2026-06-27_tau0p75_v1`](./mdu_token_id_cfg/2026-06-27_tau0p75_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/v0kttwty) |
| 1 | [`2026-06-27_tau1_v1`](./mdu_token_id_cfg/2026-06-27_tau1_v1/) | [run](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/7gaoe6gz) |

Checkpoints (weights on disk): `checkpoints/mdu_llada_forget10_token_id_cfg_tau{0,0p25,0p5,0p75,1}/`.

---

## Cross-sweep forget RougeL (τ comparison)

| τ | Paper | Random frozen | EMA .999 | EMA .99 | Random trainable | Position frozen | Position trainable | Token ID frozen | Token ID trainable |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 0.069 | 0.087 | — | — | 0.087 | **0.060** | **0.060** | 0.061 | 0.061 |
| 0.25 | 0.135 | 0.233 | 0.232 | **0.213** | 0.183 | 0.022 | **0.016** | 0.048 | 0.038 |
| 0.50 | 0.098 | 0.566 | 0.567 | 0.554 | 0.486 | 0.108 | **0.057** | 0.343 | 0.092 |
| 0.75 | 0.078 | 0.564 | — | — | 0.507 | 0.160 | **0.120** | 0.365 | 0.245 |
| 1.00 | 0.034 | 0.561 | — | — | 0.482 | 0.172 | 0.171 | 0.423 | 0.316 |

---

## Cross-sweep retain RougeL (τ comparison)

| τ | Paper | Random frozen | EMA .999 | EMA .99 | Random trainable | Position frozen | Position trainable | Token ID frozen | Token ID trainable |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 0.868 | **0.871** | — | — | **0.871** | 0.871 | 0.871 | 0.876 | 0.876 |
| 0.25 | 0.857 | 0.797 | 0.798 | 0.795 | 0.792 | 0.852 | 0.846 | 0.872 | 0.871 |
| 0.50 | 0.853 | 0.753 | 0.749 | 0.746 | 0.721 | 0.801 | 0.819 | 0.787 | **0.863** |
| 0.75 | 0.684 | 0.720 | — | — | 0.690 | 0.643 | **0.766** | 0.740 | 0.766 |
| 1.00 | 0.511 | 0.702 | — | — | 0.668 | 0.544 | 0.631 | 0.747 | **0.776** |

---

## Cross-sweep forget probability p (τ comparison)

Eq. (14) answer probability on `forget10`; **lower is better**.

| τ | Paper | Random frozen | EMA .999 | EMA .99 | Random trainable | Position frozen | Position trainable | Token ID frozen | Token ID trainable |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | **0.000** | 0.001 | — | — | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 |
| 0.25 | **0.000** | 0.016 | 0.016 | 0.014 | 0.010 | **0.000** | **0.000** | **0.000** | **0.000** |
| 0.50 | **0.001** | 0.465 | 0.465 | 0.451 | 0.384 | 0.025 | **0.002** | 0.080 | **0.001** |
| 0.75 | **0.040** | 0.480 | — | — | 0.417 | 0.183 | 0.063 | 0.303 | 0.135 |
| 1.00 | **0.074** | 0.470 | — | — | 0.386 | 0.225 | 0.127 | 0.319 | 0.189 |

Random modes collapse to SFT-like forget **p** (~0.38–0.48) at τ≥0.5; position+trainable stays low through τ=0.5 (**0.002**). At τ≥0.75, paper still leads on forget **p**; our best is position+trainable (0.063–0.127).

---

## Cross-sweep retain probability p (τ comparison)

Eq. (14) answer probability on `retain_perturbed`; **higher is better**.

| τ | Paper | Random frozen | EMA .999 | EMA .99 | Random trainable | Position frozen | Position trainable | Token ID frozen | Token ID trainable |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 0.381 | **0.536** | — | — | **0.536** | 0.502 | 0.502 | 0.509 | 0.509 |
| 0.25 | 0.392 | 0.461 | 0.462 | 0.459 | 0.440 | 0.522 | 0.519 | **0.524** | 0.522 |
| 0.50 | 0.447 | 0.582 | 0.583 | 0.580 | 0.565 | 0.525 | 0.535 | 0.473 | **0.554** |
| 0.75 | **0.535** | 0.573 | — | — | 0.566 | 0.489 | 0.519 | 0.515 | 0.541 |
| 1.00 | 0.485 | 0.560 | — | — | 0.541 | 0.475 | 0.490 | 0.513 | **0.524** |

At τ≥0.5, token_id+trainable retains the highest **p** among our configs while still unlearning; random modes inflate retain **p** but fail forget. Paper’s retain **p** rises at τ=0.75 (0.535) without the random-mode forget collapse.

---

## Notes

- **Paper Base SFT** = LLaDA-8B-Instruct after 1000-epoch TOFU SFT (paper). **Our SFT** is a separate checkpoint (`2026-06-22_sft_v1`); RougeL is similar but Eq. (14) probability runs higher.
- **Position sweep interruptions (resolved):** τ=0.25 unlearn and τ=0.5 eval were each killed once mid-run (no traceback; likely external SIGKILL). Resumed with `nohup`; final metrics use completed eval run roots above. Orphan eval stub `mdu_position_cfg/2026-06-24_tau0p5_v1/` (no splits) is not used.
- **τ=0.5 eval run_id:** completed eval is `2026-06-25_tau0p5_v1` (resume date), not `2026-06-24_tau0p5_v1`.
- **Position+frozen OOM (resolved):** without `GRADIENT_CHECKPOINTING`, both position+frozen and position+trainable OOM at optimizer step 12 (`mdu #75`, NOVEL=134/153). GC run completed all 5 τ.
- **Token ID vs position:** upstream default `match_mode=token_id`, but position+trainable dominates on forget at most τ. Token_id+trainable recovers much of the τ≥0.5 gap vs token_id+frozen, but still trails position+trainable.
- **Token_id+trainable disk failure (resolved):** first run failed on disk during τ=0 checkpoint save; relaunched 2026-06-27 after deleting completed-sweep checkpoints.
- **EMA anchor on `random`:** decay=0.999 ≈ frozen; decay=0.99 gives a small forget bump (τ=0.25 rL 0.213) but retains the τ≥0.5 collapse. Anchor type is secondary to `match_mode`; trainable+trajectory (`position`/`token_id`) drives real unlearning.
- **Best forget overall:** position+trainable at τ=0.25 (rL=0.016). **Best paper-likely config (token_id):** token_id+trainable at τ=0.25–0.5 (rL=0.038–0.092).
