# Masked Diffusion Unlearning (MDU)

Reference implementation for the paper
**"Machine Unlearning for Masked Diffusion Language Models"**
([arXiv preprint](https://arxiv.org/abs/TBD)).

This release contains the MDU training objective and the evaluation
protocols required to reproduce the main results on TOFU forget10 and
RWKU. Baseline implementations (GA, GD, NPO, SimNPO, WGA, DPO) and
backbone training code (Dream / LLaDA SFT) are not included here.

---

## 1. Repository layout

```
MDU/
├── README.md
├── LICENSE
├── requirements.txt
├── run_main.sh                  # MDU runner (TOFU / RWKU, both backbones)
├── configs/
│   ├── mdu_tofu.yaml             # MDU hyperparameters used on TOFU
│   └── mdu_rwku.yaml             # MDU hyperparameters used on RWKU
├── src/
│   ├── unlearn_mdu_llada.py      # MDU loss for LLaDA-8B-Instruct
│   └── unlearn_mdu_dream.py      # MDU loss for Dream-7B-Instruct
└── scripts/
    ├── eval_tofu_llada.py        # TOFU eval for LLaDA
    ├── eval_tofu_dream.py        # TOFU eval for Dream
    ├── eval_rwku_dream.py        # RWKU eval for Dream (8 metrics)
    ├── convert_rwku_to_tofu.py
    └── convert_rwku_dream_to_tofu.py
```

---

## 2. Environment

```bash
conda create -n mdu python=3.10 -y && conda activate mdu
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.7.0+cu128, transformers 4.57.0,
NVIDIA H200 (141 GB HBM3e).

---

## 3. Data and backbones

All experiments use only public assets:

- **TOFU** (forget10 split). The repository expects
  `forget10.json`, `retain_perturbed.json`, `real_authors.json`,
  `world_facts.json` from the original TOFU release.
- **RWKU**. Use `scripts/convert_rwku_dream_to_tofu.py` to build
  per-entity `forget.jsonl` and `dpo.jsonl` from the RWKU pair JSON.
- **Backbones**:
  - `GSAI-ML/LLaDA-8B-Instruct` (HuggingFace)
  - `Dream-org/Dream-v0-Instruct-7B` (HuggingFace)

For TOFU we additionally fine-tune each backbone on the full TOFU corpus
to instil the target knowledge (LLaDA: 1000 epoch, Dream: 300 epoch).
The SFT code itself is not part of this release; the resulting Base SFT
checkpoint is consumed by `run_main.sh` via the `LLADA_BASE_SFT` /
`DREAM_BASE_SFT` paths.

---

## 4. MDU objective

Given a forget pair $(x, y)$ from $\mathcal{D}_f$ and a partially masked
response $y_t \sim q(\cdot \mid y, t)$ at noise level
$t \sim \mathcal{U}[0, 1]$, MDU minimizes a forward KL from the trainable
prompt-conditional prediction $p^{c}_\theta(\cdot \mid x, y_t)$ to a
$\tau$-sharpened frozen unconditional anchor at every masked position:

$$
\mathcal{L}_{\mathrm{MDU}}(\theta)
= \mathbb{E}\!\left[
\frac{1}{|\mathcal{M}_t|}
\sum_{i \in \mathcal{M}_t}
\mathrm{KL}\!\left(
p^{c}_\theta(\cdot \mid x, y_t)
\,\Big\|\,
\frac{1}{Z_i}\, p^{u}_{\theta_0}(\cdot \mid m, y_t)^{\tau}
\right)
\right],
$$

where $\mathcal{M}_t$ is the set of masked positions, $m$ is a null
prompt of identical length to $x$, and
$Z_i = \sum_{v} p^{u}_{\theta_0}(v \mid m, y_t)^{\tau}$ is the
normalization constant.

- $\tau \in [0, 1]$ controls the sharpness of the unconditional anchor.
- $\theta_0$ is the model at the start of unlearning; the unconditional
  branch is detached and never updated.
- $\tau = 0$ reduces to a uniform anchor; $\tau = 1$ recovers an
  ESD-style anchor against the base model's null-prompt prediction.

In practice we add a weighted reconstruction loss
$\lambda \, \mathcal{L}_{\mathrm{sft}}(\theta; \mathcal{D}_r)$ on a
retain set whenever one is available (TOFU). On RWKU we use $\lambda=0$
since no entity-level retain set is provided.

The full optimization step is implemented in
`src/unlearn_mdu_{llada,dream}.py` (look for `null_anchor_tau`,
`null_anchor_eta`, `null_anchor_kl_dir`).

---

## 5. Reproducing the main results

### 5.1 TOFU forget10 (LLaDA-8B-Instruct, Dream-7B-Instruct)

```bash
# Edit the paths at the top of run_main.sh first.
# Pass τ ∈ {0, 0.25, 0.5, 0.75, 1}.

LR=1e-5 EPO=9 bash run_main.sh tofu_llada 0.5 ./outputs/llada_tofu_tau0p5
LR=1e-5 EPO=5 bash run_main.sh tofu_dream 0.5 ./outputs/dream_tofu_tau0p5
```

Then evaluate:
```bash
python scripts/eval_tofu_llada.py --model ./outputs/llada_tofu_tau0p5/checkpoint-final
python scripts/eval_tofu_dream.py --model ./outputs/dream_tofu_tau0p5/checkpoint-final
```

### 5.2 RWKU (Dream-7B-Instruct)

```bash
# 1) Convert RWKU pair JSON to per-entity forget.jsonl / dpo.jsonl
python scripts/convert_rwku_dream_to_tofu.py

# 2) Unlearn one entity (e.g. τ=0.5).
SUBJECT=1_Stephen_King
LR=1e-5 EPO=3 bash run_main.sh rwku_dream 0.5 \
    ./outputs/dream_rwku_${SUBJECT}_tau0p5 ${SUBJECT}

# 3) Evaluate (8 metrics: F-L1/L2/L3, N-L1/L2, MMLU, TruthfulQA, TriviaQA).
TARGET="Stephen King"
python scripts/eval_rwku_dream.py \
    --model ./outputs/dream_rwku_${SUBJECT}_tau0p5/checkpoint-final \
    --target_subject "${TARGET}" \
    --output_dir ./outputs/dream_rwku_${SUBJECT}_tau0p5/eval
```

For the full sweep (10 entities), wrap `run_main.sh rwku_dream` in a
loop over the canonical subject names listed in the paper.

---

## 6. Citation

```bibtex
@article{lee2026mdu,
  title  = {Machine Unlearning for Masked Diffusion Language Models},
  author = {Lee, Georu and Jeong, Seungwon and Kim, Hoki and Park, Jinseong and Lee, Woojin},
  journal= {arXiv preprint},
  year   = {2026}
}
```

## 7. License

This repository is released under the MIT License (see `LICENSE`).
External assets retain their original licenses: TOFU (MIT), RWKU
(Apache-2.0), LLaDA-8B-Instruct (MIT), and Dream-v0-Instruct-7B
(Apache-2.0).
