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
| **Ours (MDU τ=0.00)** | 0.087 | 0.001 | 0.871 | 0.536 | 0.526 | 0.155 | 0.792 | 0.257 |
| **Paper (MDU τ=0.25)** | 0.135 | 0.000 | 0.857 | 0.392 | 0.616 | 0.116 | 0.842 | 0.204 |
| **Ours (MDU τ=0.25)** | 0.233 | 0.016 | 0.797 | 0.461 | 0.504 | 0.172 | 0.794 | 0.276 |
| **Paper (MDU τ=0.50)** | 0.098 | 0.001 | 0.853 | 0.447 | 0.645 | 0.133 | 0.842 | 0.205 |
| **Ours (MDU τ=0.50)** | 0.566 | 0.465 | 0.753 | 0.582 | 0.553 | 0.156 | 0.786 | 0.238 |
| **Paper (MDU τ=0.75)** | 0.078 | 0.040 | 0.684 | 0.535 | 0.612 | 0.155 | 0.827 | 0.233 |
| **Ours (MDU τ=0.75)** | 0.564 | 0.480 | 0.720 | 0.573 | 0.569 | 0.135 | 0.799 | 0.220 |
| **Paper (MDU τ=1.00)** | 0.034 | 0.074 | 0.511 | 0.485 | 0.568 | 0.110 | 0.777 | 0.187 |
| **Ours (MDU τ=1.00)** | 0.561 | 0.470 | 0.702 | 0.560 | 0.593 | 0.123 | 0.790 | 0.210 |

## Our run roots

| Row | Checkpoint | Eval run |
|-----|------------|----------|
| SFT baseline | [`checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU`](../checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU) | [`sft_baseline/2026-06-22_sft_v1`](./sft_baseline/2026-06-22_sft_v1/) |
| MDU τ=0 | [`checkpoints/mdu_llada_forget10_nullanchor_tau0`](../checkpoints/mdu_llada_forget10_nullanchor_tau0) | [`mdu_tau0/2026-06-22_mdu_tau0_v1`](./mdu_tau0/2026-06-22_mdu_tau0_v1/) |
| MDU τ=0.25 | [`checkpoints/mdu_llada_forget10_nullanchor_tau0p25`](../checkpoints/mdu_llada_forget10_nullanchor_tau0p25) | [`mdu_tau0p25/2026-06-22_mdu_tau0p25_v1`](./mdu_tau0p25/2026-06-22_mdu_tau0p25_v1/) |
| MDU τ=0.5 | [`checkpoints/mdu_llada_forget10_nullanchor_tau0p5`](../checkpoints/mdu_llada_forget10_nullanchor_tau0p5) | [`mdu_tau0p5/2026-06-22_mdu_tau0p5_v1`](./mdu_tau0p5/2026-06-22_mdu_tau0p5_v1/) |
| MDU τ=0.75 | [`checkpoints/mdu_llada_forget10_nullanchor_tau0p75`](../checkpoints/mdu_llada_forget10_nullanchor_tau0p75) | [`mdu_tau0p75/2026-06-22_mdu_tau0p75_v1`](./mdu_tau0p75/2026-06-22_mdu_tau0p75_v1/) |
| MDU τ=1 | [`checkpoints/mdu_llada_forget10_nullanchor_tau1`](../checkpoints/mdu_llada_forget10_nullanchor_tau1) | [`mdu_tau1/2026-06-22_mdu_tau1_v1`](./mdu_tau1/2026-06-22_mdu_tau1_v1/) |

## Notes

- **Paper Base SFT** = LLaDA-8B-Instruct after 1000-epoch TOFU SFT (paper Appendix B.1). **Our SFT** is a separate checkpoint (`2026-06-22_sft_v1`); RougeL is similar but Eq. (14) probability runs higher.
- **MDU training (ours):** `null_anchor`, `random`, 9 epochs, lr=1e-5, batch 2×8, `ref_device=auto`. τ sweep completed 2026-06-22 (τ=0/0.5 manual or pre-sweep; τ=0.25/0.75/1 via `run_mdu_tau_sweep.sh`).
- **τ=0** and **τ=0.25** forget are close to paper (rL 0.087 / 0.233 vs 0.069 / 0.135). **τ≥0.5** forget is much weaker than paper (rL ~0.56–0.57 vs paper ~0.03–0.10) while retain also drops more than paper reports.
