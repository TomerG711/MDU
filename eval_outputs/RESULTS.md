# TOFU eval results (LLaDA, forget10)

Consolidated comparison vs [MDU paper Table 2](https://arxiv.org/abs/2605.18253).  
Per-run provenance: `summary.json` / `manifest.json` under each `<experiment>/<run_id>/`.

**Eval config (all our runs):** `max_new_tokens=128`, `steps=256`, `mask_samples=128`, `seed=42`, `truth_ratio=false`, HF `locuslab/TOFU`.

**Direction:** lower is better on **Forget**; higher is better on **Retain**, **RA**, **WF**.

## Main table

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Paper (Base SFT)** | 0.884 | 0.380 | 0.870 | 0.330 | 0.611 | 0.041 | 0.835 | 0.143 |
| **Ours (SFT baseline)** | 0.854 | 0.511 | 0.857 | 0.509 | 0.603 | 0.139 | 0.773 | 0.247 |
| **Paper (MDU τ=0.00)** | 0.069 | 0.000 | 0.868 | 0.381 | 0.629 | 0.093 | 0.848 | 0.193 |
| **Ours (MDU frozen τ=0.00)** | 0.087 | 0.001 | 0.871 | 0.536 | 0.526 | 0.155 | 0.792 | 0.257 |
| **Paper (MDU τ=0.25)** | 0.135 | 0.000 | 0.857 | 0.392 | 0.616 | 0.116 | 0.842 | 0.204 |
| **Ours (MDU frozen τ=0.25)** | 0.233 | 0.016 | 0.797 | 0.461 | 0.504 | 0.172 | 0.794 | 0.276 |
| **Paper (MDU τ=0.50)** | 0.098 | 0.001 | 0.853 | 0.447 | 0.645 | 0.133 | 0.842 | 0.205 |
| **Ours (MDU frozen τ=0.50)** | 0.566 | 0.465 | 0.753 | 0.582 | 0.553 | 0.156 | 0.786 | 0.238 |
| **Paper (MDU τ=0.75)** | 0.078 | 0.040 | 0.684 | 0.535 | 0.612 | 0.155 | 0.827 | 0.233 |
| **Ours (MDU frozen τ=0.75)** | 0.564 | 0.480 | 0.720 | 0.573 | 0.569 | 0.135 | 0.799 | 0.220 |
| **Paper (MDU τ=1.00)** | 0.034 | 0.074 | 0.511 | 0.485 | 0.568 | 0.110 | 0.777 | 0.187 |
| **Ours (MDU frozen τ=1.00)** | 0.561 | 0.470 | 0.702 | 0.560 | 0.593 | 0.123 | 0.790 | 0.210 |

## Our run roots

| Row | Checkpoint | Eval run |
|-----|------------|----------|
| SFT baseline | [`checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU`](../checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU) | [`sft_baseline/2026-06-22_sft_v1`](./sft_baseline/2026-06-22_sft_v1/) |
| MDU frozen τ=0 | *(deleted)* | [`mdu_tau0/2026-06-22_mdu_tau0_v1`](./mdu_tau0/2026-06-22_mdu_tau0_v1/) |
| MDU frozen τ=0.25 | *(deleted)* | [`mdu_tau0p25/2026-06-22_mdu_tau0p25_v1`](./mdu_tau0p25/2026-06-22_mdu_tau0p25_v1/) |
| MDU frozen τ=0.5 | *(deleted)* | [`mdu_tau0p5/2026-06-22_mdu_tau0p5_v1`](./mdu_tau0p5/2026-06-22_mdu_tau0p5_v1/) |
| MDU frozen τ=0.75 | *(deleted)* | [`mdu_tau0p75/2026-06-22_mdu_tau0p75_v1`](./mdu_tau0p75/2026-06-22_mdu_tau0p75_v1/) |
| MDU frozen τ=1 | *(deleted)* | [`mdu_tau1/2026-06-22_mdu_tau1_v1`](./mdu_tau1/2026-06-22_mdu_tau1_v1/) |

## Notes

- **Paper Base SFT** = LLaDA-8B-Instruct after 1000-epoch TOFU SFT (paper Appendix B.1). **Our SFT** is a separate checkpoint (`2026-06-22_sft_v1`); RougeL is similar but Eq. (14) probability runs higher.
- **MDU training (frozen, random):** `null_anchor`, `match_mode=random`, `null_anchor_source=frozen_sft` (auto), 9 epochs, lr=1e-5, batch 2×8, `ref_device=auto`. τ sweep completed 2026-06-22. Checkpoints deleted after eval; metrics preserved under `mdu_tau*/`.
- **τ=0** and **τ=0.25** forget are close to paper (rL 0.087 / 0.233 vs 0.069 / 0.135). **τ≥0.5** forget is much weaker than paper (rL ~0.56–0.57 vs paper ~0.03–0.10) while retain also drops more than paper reports.

---

## Random + trainable CFG anchor (`mdu_random_cfg`, 2026-06-24)

**Training:** `match_mode=random`, `null_anchor_source=trainable_cfg`, single GPU (`CUDA_VISIBLE_DEVICES=0`), no `ref_model`. Otherwise same as frozen sweep (9 ep, lr=1e-5, batch 2×8).

**Sweep status:** completed 2026-06-24 15:37 — all 5 τ trained, eval'd (4 splits each), W&B logged.

### Metrics

| | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CFG τ=0.00** | 0.087 | 0.001 | 0.871 | 0.536 | 0.526 | 0.155 | 0.792 | 0.257 |
| **CFG τ=0.25** | 0.183 | 0.010 | 0.792 | 0.440 | 0.522 | 0.169 | 0.792 | 0.273 |
| **CFG τ=0.50** | 0.486 | 0.384 | 0.721 | 0.565 | 0.538 | 0.159 | 0.765 | 0.242 |
| **CFG τ=0.75** | 0.507 | 0.417 | 0.690 | 0.566 | 0.555 | 0.134 | 0.779 | 0.216 |
| **CFG τ=1.00** | 0.482 | 0.386 | 0.668 | 0.541 | 0.528 | 0.119 | 0.769 | 0.202 |

### Frozen vs trainable (random masking, forget rL only)

| τ | Frozen SFT ref | Trainable CFG | Δ (CFG − frozen) |
|---|:---:|:---:|:---:|
| 0.00 | 0.087 | 0.087 | 0.000 |
| 0.25 | 0.233 | 0.183 | −0.050 |
| 0.50 | 0.566 | 0.486 | −0.080 |
| 0.75 | 0.564 | 0.507 | −0.057 |
| 1.00 | 0.561 | 0.482 | −0.079 |

τ=0 eval is identical (target is uniform; anchor source irrelevant). τ≥0.25 shows meaningful separation.

### Run roots

| τ | Checkpoint | Eval | W&B |
|---|------------|------|-----|
| 0 | [`mdu_llada_forget10_random_cfg_tau0`](../checkpoints/mdu_llada_forget10_random_cfg_tau0) | [`mdu_random_cfg/2026-06-24_tau0_v1`](./mdu_random_cfg/2026-06-24_tau0_v1/) | [cs4ar6lx](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/cs4ar6lx) |
| 0.25 | [`..._tau0p25`](../checkpoints/mdu_llada_forget10_random_cfg_tau0p25) | [`..._tau0p25_v1`](./mdu_random_cfg/2026-06-24_tau0p25_v1/) | [v0zu0j9s](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/v0zu0j9s) |
| 0.5 | [`..._tau0p5`](../checkpoints/mdu_llada_forget10_random_cfg_tau0p5) | [`..._tau0p5_v1`](./mdu_random_cfg/2026-06-24_tau0p5_v1/) | [41btc0gx](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/41btc0gx) |
| 0.75 | [`..._tau0p75`](../checkpoints/mdu_llada_forget10_random_cfg_tau0p75) | [`..._tau0p75_v1`](./mdu_random_cfg/2026-06-24_tau0p75_v1/) | [6iwrq8wq](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/6iwrq8wq) |
| 1 | [`..._tau1`](../checkpoints/mdu_llada_forget10_random_cfg_tau1) | [`..._tau1_v1`](./mdu_random_cfg/2026-06-24_tau1_v1/) | [tfimfpj9](https://wandb.ai/model-validation/unlearning-dllms-MDU/runs/tfimfpj9) |

### Validation (2026-06-24)

| Check | Result |
|-------|--------|
| Sweep log | `sweep_logs/mdu_tau_sweep_random_cfg_2026-06-24.log` — all τ COMPLETE |
| Checkpoints (5/5) | weights + `train_config.json` status=completed |
| W&B (5/5) | `wandb_run.json` in each checkpoint |
| Eval splits (20/20) | forget10, retain_perturbed, world_facts, real_authors × 5 τ — all `completed`, line counts OK |
| Manifests (5/5) | `manifest.json` + `summary.json` per run |
