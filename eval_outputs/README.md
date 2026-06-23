# TOFU evaluation outputs

Structured runs from `scripts/eval_tofu_llada.py` with `--experiment` and `--run_id`.

## Layout

```
eval_outputs/
  <experiment>/              # e.g. sft_baseline, mdu_tau0p5
    <run_id>/                # e.g. 2026-06-22_sft_v1 (or auto timestamp)
      README.txt
      manifest.json          # provenance + all splits (rebuilt after each split)
      summary.json           # aggregate scores (paper-style table input)
      splits/
        forget10/
        retain_perturbed/
        real_authors/
        world_facts/
```

## Running a full TOFU eval (4 splits)

```bash
source .venv/bin/activate
cd /path/to/MDU

EXPERIMENT=sft_baseline          # or mdu_tau0p5, ga_baseline, ...
RUN_ID=2026-06-22_my_run        # reuse across all 4 splits; omit for auto timestamp
MODEL=./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU

for SPLIT in forget10 retain_perturbed real_authors world_facts; do
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_tofu_llada.py \
    --model "$MODEL" \
    --experiment "$EXPERIMENT" \
    --run_id "$RUN_ID" \
    --tofu_split "$SPLIT"
done
```

Run two splits in parallel on 2 GPUs with the **same** `--run_id`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_tofu_llada.py ... --tofu_split forget10 &
CUDA_VISIBLE_DEVICES=1 python scripts/eval_tofu_llada.py ... --tofu_split retain_perturbed &
wait
```

## Paper reference (LLaDA, TOFU forget10, Table 2)

| Method | Forget rL | Forget p | Retain rL | Retain p | RA rL | RA p | WF rL | WF p |
|--------|-----------|----------|-----------|----------|-------|------|-------|------|
| Base SFT | 0.884 | 0.380 | 0.870 | 0.330 | 0.611 | 0.041 | 0.835 | 0.143 |
| GA | 0.348 | 0.020 | 0.361 | 0.018 | 0.591 | 0.068 | 0.865 | 0.156 |
| GD | 0.533 | 0.061 | 0.676 | 0.169 | 0.645 | 0.053 | 0.845 | 0.164 |
| NPO | 0.372 | 0.009 | 0.726 | 0.138 | 0.606 | 0.033 | 0.844 | 0.145 |
| SimNPO | 0.485 | 0.036 | 0.804 | 0.273 | 0.640 | 0.055 | 0.836 | 0.164 |
| WGA | 0.122 | 0.010 | 0.696 | 0.304 | 0.577 | 0.089 | 0.815 | 0.157 |
| DPO | 0.479 | 0.168 | 0.796 | 0.401 | 0.699 | 0.046 | 0.834 | 0.125 |
| MDU τ=0.00 | 0.069 | 0.000 | 0.868 | 0.381 | 0.629 | 0.093 | 0.848 | 0.193 |
| MDU τ=0.25 | 0.135 | 0.000 | 0.857 | 0.392 | 0.616 | 0.116 | 0.842 | 0.204 |
| MDU τ=0.50 | 0.098 | 0.001 | 0.853 | 0.447 | 0.645 | 0.133 | 0.842 | 0.205 |
| MDU τ=0.75 | 0.078 | 0.040 | 0.684 | 0.535 | 0.612 | 0.155 | 0.827 | 0.233 |
| MDU τ=1.00 | 0.034 | 0.074 | 0.511 | 0.485 | 0.568 | 0.110 | 0.777 | 0.187 |

Source: MDU paper Table 2 (LLaDA-8B-Instruct). Forget ↓ better; Retain / RA / WF ↑ better.

## Completed runs

See **[RESULTS.md](./RESULTS.md)** for the consolidated paper vs ours comparison table.

| Experiment | Run ID | Notes |
|------------|--------|-------|
| `sft_baseline` | `2026-06-22_sft_v1` | Our TOFU SFT checkpoint; 4/4 splits OK |
| `mdu_tau0` | `2026-06-22_mdu_tau0_v1` | MDU null-anchor τ=0; 4/4 splits OK |
| `mdu_tau0p5` | `2026-06-22_mdu_tau0p5_v1` | MDU null-anchor τ=0.5; 4/4 splits OK |

### Adding a new run

1. Pick `--experiment` and `--run_id`.
2. Run all four splits (same `run_id`).
3. Add a row to the **Completed runs** table above.
4. Update **[RESULTS.md](./RESULTS.md)** with scores from `<experiment>/<run_id>/summary.json`.
