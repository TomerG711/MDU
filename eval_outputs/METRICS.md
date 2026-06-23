# MDU metrics & training knobs

Short reference for **τ** (null-anchor temperature), **RougeL**, and **`match_mode`** (which answer tokens get the forget loss).

---

## τ (null-anchor temperature)

MDU forget loss uses a **null anchor** KL on masked answer positions. The target distribution is a τ-tempered version of the frozen unconditional model **pᵘ_θ₀** (question masked).

**Equation (paper Eq. 5; code in `_null_anchor_kl`):**

```
L_forget = E[ KL( p^c_θ  ||  p* ) ]

p*(v)  ∝  p^u_θ0(v | m, y_t)^τ
```

- **p^c_θ** — trainable conditional model (full prompt x).
- **p^u_θ0** — frozen reference, question masked (m).
- **p*** — τ-tempered anchor; KL pulls conditional toward p*.

In code: `target_logits = tau * logits_u.detach()`, then forward KL between `log_softmax(logits_c)` and `log_softmax(target_logits)`.

| τ | Anchor p* | Effect in words |
|---|-----------|-----------------|
| **0** | Uniform over vocab | “Forget everything equally” — pushes conditional toward max entropy; outputs often **corrupted / nonsensical**. |
| **0.25–0.75** | Partially sharpened uncond | Interpolation: still forgets, but anchor keeps some structure from θ₀. |
| **1** | Full frozen uncond pᵘ_θ₀ | “Match what θ₀ predicts when the question is masked” — forget can stay **fluent** (wrong facts, refusals) but **hurts retain** more. |

**Important:** τ is **not** “τ=0 forget, τ=1 retain.” Both ends are forget objectives. Retain is a **separate** SFT term (`alpha`). The tradeoff is *how* you forget (garbled vs. fluent substitution) vs. how much utility you keep — see paper Table 2.

Our τ sweep uses `match_mode=random` with `novel_percentile=100` (all answer tokens masked for null-anchor KL).

---

## RougeL (recall, precision, F1)

RougeL scores the **longest common subsequence (LCS)** of word tokens (stemmed in our eval). For ground truth **y** and prediction **ŷ**:

```
recall    = |LCS| / |y|
precision = |LCS| / |ŷ|
F1        = 2 · recall · precision / (recall + precision)
```

### What the papers call “RougeL”

| Source | Definition |
|--------|------------|
| **MDU paper** ([arXiv:2605.18253](https://arxiv.org/abs/2605.18253), Appendix B.1) | **rL = LCS F1** between ŷ and y. Table 2 column “rL” is this F1 score. |
| **TOFU paper** ([arXiv:2401.06121](https://arxiv.org/abs/2401.06121)) | **ROUGE-L recall** (“acts as a surrogate for accuracy”). |
| **This repo** (`eval_tofu_llada.py`) | `rouge_scorer` → **`["rougeL"].recall`** (TOFU-style, **not** MDU F1). |

So MDU Table 2 “rL” is **F1**, while our eval logs **recall** under the key `rougeL`. The two differ whenever prediction length ≠ reference length (see examples below). For strict paper replication, use `.fmeasure` instead of `.recall`.

In `RESULTS.md`, column **rL** = our RougeL **recall**; column **p** = Eq. (7) answer **probability**, not Rouge precision.

**Direction:** on **forget**, lower overlap = better unlearning (whether you track recall or F1).

### Three toy examples

Assume word-level LCS (no stemming).

**1. Recall best** — prediction covers almost all of GT, plus extra words:

| | Text |
|---|------|
| GT | `the cat sat on the mat` |
| Pred | `the cat sat on the mat today` |

LCS length 6 → recall **6/6 = 1.0**, precision **6/7 ≈ 0.86**, F1 **≈ 0.92**.

**2. Precision best** — short prediction, only a small slice of GT:

| | Text |
|---|------|
| GT | `the cat sat on the mat` |
| Pred | `cat mat` |

LCS length 2 → recall **2/6 ≈ 0.33**, precision **2/2 = 1.0**, F1 **0.50**.

**3. All same** — identical strings:

| | Text |
|---|------|
| GT | `hello world` |
| Pred | `hello world` |

LCS length 2 → recall **1.0**, precision **1.0**, F1 **1.0**.

---

## `match_mode` (which tokens get forget loss)

`match_mode` controls **which answer positions** receive gradient in the forget path (`_denoise_novel_ga_loss` / `_random_weights` in `unlearn_mdu_llada.py`). It does **not** change eval metrics.

Common flow for trajectory modes (`token_id`, `position`, …):

1. **Pass 1 (no grad):** run a denoising rollout on the answer; record **unmask step** per position.
2. Take the **latest** `novel_percentile`% of positions by unmask step (“late” tokens).
3. **Pass 2 (grad):** apply GA / null-anchor loss on the selected positions.

### `random`

Skips the trajectory entirely (`_random_weights`).

- Uniformly samples `novel_percentile`% of **content** answer tokens (special tokens excluded).
- No dependence on model confidence or unmask order.
- Used in our τ sweep with `novel_percentile=100` → **every answer token** is a forget target.

### `position` (default when not `token_id`)

Uses **where** in the sequence the model unmasked late.

- After Pass 1, `late_positions` = top `novel_percentile`% by unmask step.
- Forget loss applies to those **indices**, regardless of which token ID was generated there.
- Intuition: punish tokens the model resolves **last** during diffusion (often factual / hard spans).

### `token_id`

Uses **what token** appeared in late unmask slots, then maps back to GT.

- Collect token IDs generated at `late_positions` → set `late_token_ids`.
- For each GT answer position, if `gt_token_id ∈ late_token_ids`, apply loss there (even if that GT position was unmasked earlier).
- Intuition: forget **vocabulary items** the model treats as “late-revealed facts,” not just late slots.
- Retain SFT can mirror this (`match_mode=token_id` on retain path).

### Other modes (brief)

| Mode | Idea |
|------|------|
| `factual_filter`, `tid_factual` | Late positions + attention-entropy cache to keep “factual” tokens. |
| `tid_prob`, `prob_sigmoid` | Intersect late token IDs with low step-0 p(GT). |
| `gap`, `gap_tid` | Precomputed single-mask **gap** cache; bottom-% uncertain tokens. |
| `cat_oracle` | Oracle category labels per token (e.g. category 3 = factual span). |

---

## See also

- Results table: [`RESULTS.md`](./RESULTS.md)
- Eval script: [`scripts/eval_tofu_llada.py`](../scripts/eval_tofu_llada.py)
- Training: [`src/unlearn_mdu_llada.py`](../src/unlearn_mdu_llada.py)
