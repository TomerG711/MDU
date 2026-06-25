"""
LLaDA MDU Unlearning on TOFU

Implementation features:
  1. Position-based matching
  2. Continuous weighting (w = (step/max_step)^beta) or binary mask
  3. Bounded GA / NPO loss options (NPO requires ref_model)
  4. Single-pass approximation mode (skip full denoising trajectory)
  5. Trajectory caching (reuse Pass 1 every N steps)

Algorithm:
  forget batch:
    Pass 1 (no_grad): prompt → N-step denoising → unmask order tracking
    Late positions (top p%) → continuous weight by unmask step
    Pass 2 (grad): weighted GA on late positions
  retain batch:
    normal SFT loss

  total_loss = -forget_weighted_nll + alpha * retain_nll

Run:
    cd ./dllm
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        src/unlearn_mdu_llada.py \
        --model_name_or_path ./checkpoints/llada-tofu-sft/checkpoint-final \
        --tofu_path ./TOFU/forget10.json \
        --retain_path ./TOFU/retain_perturbed.json \
        --output_dir ./outputs/mdu-llada-tofu \
        --novel_percentile 30 \
        --denoise_steps 32 \
        --alpha 1.0 \
        --weight_beta 1.0 \
        --loss_type ga
"""

import os
import math
import json
import string
import sys
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
import accelerate
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets

import dllm
from dllm.core.trainers import MDLMTrainer, MDLMConfig
from dllm.core.samplers.utils import get_num_transfer_tokens

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unlearn_run_utils import (  # noqa: E402
    apply_disable_data_parallel,
    build_train_config,
    configure_wandb,
    load_forget_retain_rows,
    maybe_add_wandb_callback,
    place_ref_model,
    needs_ref_model,
    null_anchor_uses_frozen_ref,
    copy_script_snapshot,
    build_mdu_setup_summary,
    format_mdu_setup_log,
    prepare_wandb_run_name,
    resolve_run_directory,
    rows_to_messages,
    save_final_checkpoint,
    write_train_config,
)

logger = dllm.utils.get_default_logger(__name__)


# ── Collator ──────────────────────────────────────────────────────────────────

class MDUCollator(dllm.utils.CollatorWrapper):
    def before(self, features):
        self._is_forget = torch.tensor(
            [f.pop("is_forget", 1) for f in features], dtype=torch.bool
        )
        self._prompt_lens = [f.pop("prompt_len", 0) for f in features]
        keep = {"input_ids", "labels", "attention_mask"}
        for f in features:
            for key in list(f.keys()):
                if key not in keep:
                    f.pop(key)
        return features

    def after(self, outputs):
        outputs.pop("attention_mask", None)
        outputs["is_forget"] = self._is_forget
        outputs["prompt_lens"] = self._prompt_lens
        return outputs


# ── Trainer ───────────────────────────────────────────────────────────────────

class MDUTrainer(MDLMTrainer):
    """
    MDU + retain GD.

    Key improvements over v1:
      - Position-based matching (no token ID leakage)
      - Continuous weighting: w = (step/max_step)^beta
      - Multiple loss types: ga, bounded_ga
      - Single-pass approximation mode
      - Trajectory caching
    """

    def __init__(self, novel_percentile=30, denoise_steps=32, alpha=1.0,
                 max_new_tokens=128,
                 weight_beta=1.0, loss_type="ga", loss_cap=5.0,
                 npo_beta=0.2, ref_model=None,
                 single_pass=False, cache_interval=0,
                 adaptive_threshold=False, adaptive_k=1.0,
                 conf_threshold=0.0,
                 weight_type="power", sigmoid_k=8.0, sigmoid_center=0.5,
                 match_mode="token_id",
                 factual_cache_path="", factual_late_pct=0.30, factual_entropy_pct=0.67,
                 factual_entropy_reverse=False,
                 prob_filter_pct=0.30,
                 gap_cache_path="", gap_bottom_pct=0.30,
                 retain_mode="sft", retain_late_pct=0.30,
                 log_mask_categories_path="",
                 diagnostic_csv="",
                 diagnostic_interval=10,
                 cat_cache_path="", cat_target_cats="3",
                 null_anchor_eta=0.0, null_anchor_kl_dir="forward",
                 null_anchor_tau=1.0,
                 null_anchor_traj_rollout=False,
                 null_anchor_traj_steps=16,
                 null_anchor_exclude_special=False,
                 null_anchor_source="auto",
                 ref_device=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.novel_percentile = novel_percentile
        self.denoise_steps = denoise_steps
        self.alpha = alpha
        self.max_new_tokens = max_new_tokens
        self.weight_beta = weight_beta
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_k = adaptive_k
        self.conf_threshold = conf_threshold  # >0: select tokens with confidence < tau
        self.weight_type = weight_type
        self.sigmoid_k = sigmoid_k
        self.sigmoid_center = sigmoid_center  # sigmoid((step_pct - center) * k)
        self.loss_type = loss_type
        self.loss_cap = loss_cap
        self.npo_beta = npo_beta
        self.null_anchor_eta = float(null_anchor_eta)
        self.null_anchor_kl_dir = null_anchor_kl_dir
        self.null_anchor_tau = float(null_anchor_tau)
        self.null_anchor_traj_rollout = bool(null_anchor_traj_rollout)
        self.null_anchor_traj_steps = int(null_anchor_traj_steps)
        self.null_anchor_exclude_special = bool(null_anchor_exclude_special)
        self.null_anchor_source = (null_anchor_source or "auto").strip().lower()
        # Will be populated lazily once tokenizer is available (in _maybe_init_special_ids).
        self._special_token_ids = None
        self.ref_model = ref_model
        self._ref_device = ref_device  # None → colocated with trainable model
        self.single_pass = single_pass
        self.match_mode = match_mode  # "token_id" (v1 style) or "position" (v2 style) or "factual_filter"
        self.cache_interval = cache_interval
        # Factual filter: top late_pct% by step → top entropy_pct% (bottom by entropy) as factual
        self.factual_cache_path = factual_cache_path
        self.factual_late_pct = factual_late_pct
        self.factual_entropy_pct = factual_entropy_pct
        self.factual_entropy_reverse = factual_entropy_reverse
        self.prob_filter_pct = prob_filter_pct  # tid_prob: bottom K% by full-mask p(GT) (0~1)
        self.gap_cache_path = gap_cache_path
        self.gap_bottom_pct = gap_bottom_pct
        self.gap_cache = None
        if gap_cache_path and os.path.exists(gap_cache_path):
            import json as _json
            with open(gap_cache_path) as _f:
                self.gap_cache = _json.load(_f)
            logger.info(f"[gap] loaded {len(self.gap_cache)} entries from {gap_cache_path}")
        self.retain_mode = retain_mode  # "sft" (default random diffusion) or "late"
        self.retain_late_pct = retain_late_pct
        self.factual_cache = None
        if factual_cache_path and os.path.exists(factual_cache_path):
            import json as _json
            with open(factual_cache_path) as _f:
                self.factual_cache = _json.load(_f)
            logger.info(f"[factual_filter] loaded {len(self.factual_cache)} entries "
                        f"from {factual_cache_path}")
        self._log_count = 0
        self._global_step = 0
        self._exclude_ids = None
        self._cached_weights = {}  # batch hash → weight tensor
        self.log_mask_categories_path = log_mask_categories_path
        if log_mask_categories_path:
            os.makedirs(os.path.dirname(log_mask_categories_path) or ".", exist_ok=True)
        self.diagnostic_csv = diagnostic_csv
        self.diagnostic_interval = diagnostic_interval
        self._diag_header_written = False
        if diagnostic_csv:
            os.makedirs(os.path.dirname(diagnostic_csv) or ".", exist_ok=True)
        self.cat_cache_path = cat_cache_path
        self.cat_target_cats = set(int(c) for c in str(cat_target_cats).split(",") if c.strip())
        self.cat_cache = None
        if cat_cache_path and os.path.exists(cat_cache_path):
            import json as _json
            with open(cat_cache_path) as _f:
                self.cat_cache = _json.load(_f)
            logger.info(
                f"[cat_oracle] loaded {len(self.cat_cache)} entries from {cat_cache_path}; "
                f"target_cats={sorted(self.cat_target_cats)}"
            )

    def _ref_forward_logits(self, input_ids, attention_mask=None):
        """Forward through frozen ref_model; logits returned on ``input_ids.device``."""
        if self.ref_model is None:
            raise RuntimeError("ref_model is required for this loss")
        if self._ref_device is None:
            outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = self._postprocess_outputs(outputs)
            return outputs.logits
        ref_dev = torch.device(self._ref_device)
        with torch.no_grad():
            ref_ids = input_ids.to(ref_dev)
            ref_attn = attention_mask.to(ref_dev) if attention_mask is not None else None
            outputs = self.ref_model(input_ids=ref_ids, attention_mask=ref_attn)
            outputs = self._postprocess_outputs(outputs)
            return outputs.logits.to(input_ids.device)

    def _null_anchor_uses_frozen_ref(self) -> bool:
        return null_anchor_uses_frozen_ref(
            loss_type=self.loss_type,
            null_anchor_source=self.null_anchor_source,
            match_mode=self.match_mode,
            null_anchor_traj_rollout=self.null_anchor_traj_rollout,
        )

    def _null_anchor_uncond_logits(self, model, noised_u, attention_mask=None):
        """Uncond logits for null-anchor KL: frozen SFT ref or trainable CFG (Q-masked)."""
        with torch.no_grad():
            if self._null_anchor_uses_frozen_ref():
                if self.ref_model is None:
                    raise RuntimeError(
                        "null_anchor_source requires frozen ref_model but ref_model is None"
                    )
                return self._ref_forward_logits(noised_u, attention_mask)
            outputs_u = model(input_ids=noised_u, attention_mask=attention_mask)
            outputs_u = self._postprocess_outputs(outputs_u)
            return outputs_u.logits

    def _null_anchor_uncond_label(self) -> str:
        return "frozen_sft" if self._null_anchor_uses_frozen_ref() else "trainable_cfg"

    def _get_exclude_ids(self, device):
        if self._exclude_ids is not None:
            return self._exclude_ids.to(device)
        tok = self.processing_class
        exclude = set()
        for t in [tok.eos_token, tok.pad_token, tok.mask_token,
                  '<|endoftext|>', '<|eot_id|>']:
            if t:
                ids = tok.encode(t, add_special_tokens=False)
                exclude.update(ids)
        for ch in string.punctuation + " ":
            ids = tok.encode(ch, add_special_tokens=False)
            exclude.update(ids)
        self._exclude_ids = torch.tensor(sorted(exclude), dtype=torch.long)
        return self._exclude_ids.to(device)

    def _single_pass_weights(self, model, input_ids, labels, attention_mask, prompt_lens):
        """Single forward pass approximation: use gt_prob at full masking as novelty signal."""
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)

        weight = torch.zeros(b, l, device=device, dtype=torch.float)

        with torch.no_grad():
            # Mask all answer positions
            noised = torch.where(maskable_mask, mask_id, input_ids)
            logits = model(input_ids=noised, attention_mask=attention_mask).logits
            logits = self._postprocess_outputs(
                type("O", (), {"logits": logits})()
            ).logits
            probs = F.softmax(logits.float(), dim=-1)

            for i in range(b):
                resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
                if len(resp_pos) == 0:
                    continue

                gt_probs = probs[i, resp_pos, :]
                gt_token_ids = input_ids[i, resp_pos]
                gt_prob_vals = gt_probs.gather(1, gt_token_ids.unsqueeze(1)).squeeze(1)

                # Filter exclude tokens
                for j, pos in enumerate(resp_pos.tolist()):
                    tid = input_ids[i, pos].item()
                    if torch.isin(torch.tensor(tid, device=device), exclude_ids).item():
                        gt_prob_vals[j] = 1.0  # high prob → low weight

                # Binary selection (weight_beta deprecated: always uniform 1.0)
                w = torch.ones_like(gt_prob_vals)
                weight[i, resp_pos] = w

        return weight, maskable_mask

    def _random_weights(self, input_ids, labels, attention_mask):
        """Random token selection: sample novel_percentile% of answer tokens uniformly."""
        import random as _random
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)
        weight = torch.zeros(b, l, device=device, dtype=torch.float)

        for i in range(b):
            resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
            # filter exclude tokens
            content_pos = [p.item() for p in resp_pos
                           if not torch.isin(input_ids[i, p], exclude_ids).item()]
            if not content_pos:
                continue
            k = min(len(content_pos), max(1, int(len(content_pos) * self.novel_percentile / 100)))
            selected = _random.sample(content_pos, k)
            for pos in selected:
                weight[i, pos] = 1.0

        return weight, maskable_mask

    def _cat_oracle_weights(self, input_ids, labels, attention_mask, prompt_lens):
        """Cat-oracle: weight=1 on answer positions whose pre-labelled category is in cat_target_cats.

        Cache key = comma-joined prompt_ids (prefix of input_ids[i, :prompt_len]).
        Cache value = [cat per answer token], aligned 1:1 with input_ids[i, prompt_len:].
        """
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)
        exclude_id_set = set(exclude_ids.tolist())
        weight = torch.zeros(b, l, device=device, dtype=torch.float)

        if self.cat_cache is None:
            logger.warning("[cat_oracle] cat_cache not loaded — no GA applied this batch")
            return weight, maskable_mask

        miss = 0
        targets = self.cat_target_cats
        for i in range(b):
            prompt_len = int(prompt_lens[i])
            resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
            if len(resp_pos) == 0:
                continue
            p_ids = input_ids[i, :prompt_len].tolist()
            key = ",".join(str(t) for t in p_ids)
            cats = self.cat_cache.get(key)
            if cats is None:
                miss += 1
                continue
            for pos in resp_pos.tolist():
                rel = pos - prompt_len
                if rel < 0 or rel >= len(cats):
                    continue
                if cats[rel] not in targets:
                    continue
                tid = input_ids[i, pos].item()
                if tid in exclude_id_set:
                    continue
                weight[i, pos] = 1.0
        if miss and self._log_count < 3:
            logger.warning(f"[cat_oracle] {miss}/{b} samples missed cache lookup this batch")
            self._log_count += 1
        return weight, maskable_mask

    def _denoising_trajectory_weights(self, model, input_ids, labels, attention_mask, prompt_lens):
        """Full denoising trajectory: track unmask order and compute position-based weights."""
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)

        weight = torch.zeros(b, l, device=device, dtype=torch.float)

        with torch.no_grad():
            for i in range(b):
                prompt_len = prompt_lens[i]
                resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
                if len(resp_pos) == 0:
                    continue

                seq = input_ids[i].clone()
                seq[resp_pos] = mask_id
                x = seq.unsqueeze(0)

                mask_index = (x == mask_id)[:, prompt_len:]
                num_transfers = get_num_transfer_tokens(
                    mask_index=mask_index,
                    steps=self.denoise_steps,
                    scheduler=self.scheduler,
                    stochastic=False,
                )[0].tolist()

                unmask_step = {}
                unmask_conf = {}  # pos → confidence at unmask time
                step0_p_gt = {}  # pos → full-answer-mask p(GT) (tid_prob mode)
                step0_captured = False

                for step, n_unmask in enumerate(num_transfers):
                    if n_unmask == 0:
                        continue

                    logits = model(input_ids=x).logits
                    logits = self._postprocess_outputs(
                        type("O", (), {"logits": logits})()
                    ).logits
                    probs = F.softmax(logits.float(), dim=-1)
                    conf = probs.max(dim=-1).values[0]
                    pred = probs.argmax(dim=-1)[0]

                    if self.match_mode in ("tid_prob", "prob_sigmoid") and not step0_captured:
                        for pos in resp_pos.tolist():
                            gt_tid = input_ids[i, pos].item()
                            step0_p_gt[pos] = probs[0, pos, gt_tid].item()
                        step0_captured = True

                    masked_pos = (x[0] == mask_id).nonzero(as_tuple=True)[0]
                    masked_pos = masked_pos[masked_pos >= prompt_len]
                    if len(masked_pos) == 0:
                        break

                    conf_masked = conf[masked_pos]
                    n = min(n_unmask, len(masked_pos))
                    _, top_idx = conf_masked.topk(n, largest=True)
                    unmask_pos = masked_pos[top_idx]

                    for pos in unmask_pos.tolist():
                        unmask_step[pos] = step
                        unmask_conf[pos] = conf[pos].item()
                        x[0, pos] = pred[pos]

                if not unmask_step:
                    continue

                # Filter exclude tokens from generated sequence
                content_by_step = []
                for pos, step in unmask_step.items():
                    gen_tid = x[0, pos].item()
                    if torch.isin(torch.tensor(gen_tid, device=device), exclude_ids).item():
                        continue
                    content_by_step.append((pos, step))

                if not content_by_step:
                    continue

                max_step = max(s for _, s in content_by_step)
                if max_step == 0:
                    max_step = 1

                sorted_by_step = sorted(content_by_step, key=lambda kv: kv[1], reverse=True)

                if self.conf_threshold > 0.0:
                    # Confidence threshold: select tokens with confidence < tau at unmasking time
                    late_positions = [pos for pos, _ in sorted_by_step
                                      if unmask_conf.get(pos, 1.0) < self.conf_threshold]
                    if not late_positions:
                        late_positions = [sorted_by_step[0][0]]  # at least one position
                elif self.adaptive_threshold:
                    all_steps_list = [s for _, s in content_by_step]
                    mean_s = sum(all_steps_list) / len(all_steps_list)
                    std_s = (sum((s - mean_s) ** 2 for s in all_steps_list) / len(all_steps_list)) ** 0.5
                    threshold = mean_s + self.adaptive_k * std_s
                    k = max(1, sum(1 for _, s in sorted_by_step if s >= threshold))
                    late_positions = [pos for pos, _ in sorted_by_step[:k]]
                else:
                    # Top novel_percentile% by unmask step (late tokens)
                    k = max(1, int(len(sorted_by_step) * self.novel_percentile / 100))
                    late_positions = [pos for pos, _ in sorted_by_step[:k]]

                late_steps = {pos: step for pos, step in sorted_by_step
                              if pos in set(late_positions)}

                resp_pos_set = set(resp_pos.tolist())
                exclude_id_set = set(exclude_ids.tolist())  # O(1) lookup

                def _calc_weight(step_val):
                    # weight_beta deprecated: always return binary 1.0
                    return 1.0

                import math as _math
                def _sigmoid_w(step_val):
                    step_pct = step_val / max_step
                    return 1.0 / (1.0 + _math.exp(-(step_pct - self.sigmoid_center) * self.sigmoid_k))

                def _power_w(step_val):
                    # weight_beta deprecated: binary selection always
                    return 1.0

                w_fn = _sigmoid_w if self.weight_type == "sigmoid" else _power_w

                if self.match_mode == "tid_factual":
                    # v1 token-id matching + entropy filter
                    # 1) top late_pct% late positions (by step)
                    # 2) late_token_ids = set of model-predicted IDs at those late positions
                    # 3) candidates = GT positions where gt_tid ∈ late_token_ids
                    # 4) entropy filter: bottom factual_entropy_pct% by cache entropy
                    # 5) binary weight = 1.0 on survivors
                    assert self.factual_cache is not None, "factual_cache_path required"
                    p_ids = input_ids[i, :prompt_len].tolist()
                    key = ",".join(str(t) for t in p_ids)
                    entry = self.factual_cache.get(key)
                    # {rel: (cache_tid, entropy)} — keep tid so we can verify alignment
                    tid_ent_by_rel = {r[0]: (r[1], r[2]) for r in entry["rels"]} if entry else {}

                    # top late_pct% by step
                    n_late = max(1, int(len(content_by_step) * self.factual_late_pct))
                    late_pos_steps_tid = sorted_by_step[:n_late]
                    late_token_ids = set(x[0, p].item() for p, _ in late_pos_steps_tid)

                    # GT candidates (token_id matching)
                    candidates = []  # (pos, entropy)
                    for pos in resp_pos.tolist():
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        if gt_tid not in late_token_ids:
                            continue
                        rel = pos - prompt_len
                        te = tid_ent_by_rel.get(rel, None)
                        if te is None:
                            continue
                        cache_tid, ent = te
                        # Defensive: rel-only lookup is unsafe if prompt_len drifts.
                        # Verify the cached tid matches the actual GT token id.
                        if cache_tid != gt_tid:
                            logger.warning(
                                f"[tid_factual] cache tid mismatch at rel={rel}: "
                                f"gt={gt_tid} cache={cache_tid} — skipping position"
                            )
                            continue
                        candidates.append((pos, ent))

                    if not candidates:
                        continue

                    # bottom (or top if reverse) K% by entropy
                    candidates.sort(key=lambda x: x[1], reverse=self.factual_entropy_reverse)
                    n_keep = max(1, int(len(candidates) * self.factual_entropy_pct))
                    for p, _ in candidates[:n_keep]:
                        weight[i, p] = 1.0
                elif self.match_mode == "tid_prob":
                    # v1 tid match × full-answer-mask p(GT) bottom K% intersection.
                    # 1) top late_pct% late positions → late_token_ids
                    # 2) step-0 full-mask p(GT) per resp_pos → bottom prob_filter_pct% → prob_bottom_tids
                    # 3) surviving_tids = late_token_ids ∩ prob_bottom_tids
                    # 4) weight = 1.0 on GT positions where gt_tid ∈ surviving_tids
                    n_late = max(1, int(len(content_by_step) * self.factual_late_pct))
                    late_pos_steps_tp = sorted_by_step[:n_late]
                    late_token_ids = set(x[0, p].item() for p, _ in late_pos_steps_tp)

                    prob_candidates = []  # (pos, gt_tid, logp)
                    for pos in resp_pos.tolist():
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        p_gt = step0_p_gt.get(pos, 1.0)
                        logp = _math.log(max(p_gt, 1e-12))
                        prob_candidates.append((pos, gt_tid, logp))

                    if not prob_candidates:
                        continue

                    prob_candidates.sort(key=lambda t: t[2])  # ascending = bottom first
                    n_prob = max(1, int(len(prob_candidates) * self.prob_filter_pct))
                    prob_bottom_tids = set(t[1] for t in prob_candidates[:n_prob])

                    surviving = late_token_ids & prob_bottom_tids
                    if not surviving:
                        continue

                    # tid → max unmask step (for sigmoid weighting)
                    tid_to_step = {}
                    for p, s in late_pos_steps_tp:
                        tid = x[0, p].item()
                        if tid not in tid_to_step or s > tid_to_step[tid]:
                            tid_to_step[tid] = s

                    for pos in resp_pos.tolist():
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        if gt_tid not in surviving:
                            continue
                        if self.weight_type == "sigmoid":
                            weight[i, pos] = w_fn(tid_to_step[gt_tid])
                        else:
                            weight[i, pos] = 1.0
                elif self.match_mode == "prob_sigmoid":
                    # Inverted: prob_bottom hard filter × sigmoid(step_pct) soft weight on all tokens.
                    # No late hard cut. weight = w_fn(max_step of gt_tid in generation)
                    # iff gt_tid in prob_bottom_K. Then sigmoid Bernoulli (or power weight) applies.
                    all_step_by_id = {}
                    for p, s in content_by_step:
                        tid = x[0, p].item()
                        if tid not in all_step_by_id or s > all_step_by_id[tid]:
                            all_step_by_id[tid] = s

                    prob_candidates = []
                    for pos in resp_pos.tolist():
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        p_gt = step0_p_gt.get(pos, 1.0)
                        prob_candidates.append((pos, gt_tid, _math.log(max(p_gt, 1e-12))))

                    if not prob_candidates:
                        continue

                    prob_candidates.sort(key=lambda t: t[2])
                    n_prob = max(1, int(len(prob_candidates) * self.prob_filter_pct))
                    prob_bottom_tids = set(t[1] for t in prob_candidates[:n_prob])

                    for pos in resp_pos.tolist():
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        if gt_tid not in prob_bottom_tids:
                            continue
                        if gt_tid not in all_step_by_id:
                            continue
                        weight[i, pos] = w_fn(all_step_by_id[gt_tid])
                elif self.match_mode == "factual_filter":
                    # top factual_late_pct% by step → within that, keep bottom factual_entropy_pct%
                    # by entropy (= factual). weight=1.0 uniform on survivors.
                    assert self.factual_cache is not None, "factual_cache_path required"
                    # prompt key = comma-joined unpadded prompt ids
                    p_ids = input_ids[i, :prompt_len].tolist()
                    key = ",".join(str(t) for t in p_ids)
                    entry = self.factual_cache.get(key)
                    if entry is None:
                        # no cache hit → fall back: apply GA to all top late_pct positions
                        n_late = max(1, int(len(content_by_step) * self.factual_late_pct))
                        late_pos = [p for p, _ in sorted_by_step[:n_late]]
                        for p in late_pos:
                            if input_ids[i, p].item() in exclude_id_set:
                                continue
                            weight[i, p] = 1.0
                        continue
                    ent_by_rel = {r[0]: r[2] for r in entry["rels"]}

                    # top late_pct% by step
                    n_late = max(1, int(len(content_by_step) * self.factual_late_pct))
                    late_pos_steps = sorted_by_step[:n_late]

                    # gather (pos, rel, entropy) for late tokens that exist in cache
                    late_with_ent = []
                    for p, s in late_pos_steps:
                        rel = p - prompt_len
                        if rel in ent_by_rel and input_ids[i, p].item() not in exclude_id_set:
                            late_with_ent.append((p, rel, ent_by_rel[rel]))
                    if not late_with_ent:
                        continue

                    # keep bottom factual_entropy_pct% by entropy
                    late_with_ent.sort(key=lambda x: x[2])  # ascending entropy
                    n_keep = max(1, int(len(late_with_ent) * self.factual_entropy_pct))
                    factual_positions = late_with_ent[:n_keep]
                    for p, _, _ in factual_positions:
                        weight[i, p] = 1.0
                elif self.weight_type == "sigmoid":
                    # Sigmoid mode: token_id matching (GT ∩ generated), sigmoid weight on all tokens
                    # token_id → max unmask step among all generated positions with that token
                    all_step_by_id = {}
                    for p, s in content_by_step:
                        tid = x[0, p].item()
                        if tid not in all_step_by_id or s > all_step_by_id[tid]:
                            all_step_by_id[tid] = s
                    for pos in resp_pos:
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        if gt_tid not in all_step_by_id:
                            continue
                        weight[i, pos] = w_fn(all_step_by_id[gt_tid])
                elif self.match_mode == "gap":
                    # bottom-K% by single-mask gap: pick uncertain tokens from a precomputed gap cache
                    assert self.gap_cache is not None, "gap_cache_path required for match_mode=gap"
                    p_ids = input_ids[i, :prompt_len].tolist()
                    key = ",".join(str(t) for t in p_ids)
                    entry = self.gap_cache.get(key)
                    if entry is not None:
                        # Collect (rel_pos, gap) for all response tokens
                        gap_candidates = []
                        for pos in resp_pos.tolist():
                            rel = pos - prompt_len
                            gt_tid = input_ids[i, pos].item()
                            if gt_tid in exclude_id_set:
                                continue
                            info = entry.get(str(rel))
                            if info is not None:
                                gap_candidates.append((pos, info["gap"]))
                        if gap_candidates:
                            # Sort by gap ascending (most uncertain first)
                            gap_candidates.sort(key=lambda t: t[1])
                            n_select = max(1, int(len(gap_candidates) * self.gap_bottom_pct))
                            for pos, _ in gap_candidates[:n_select]:
                                weight[i, pos] = 1.0
                    else:
                        logger.warning(f"[gap] cache miss for sample, applying GA to all response tokens")
                        for pos in resp_pos.tolist():
                            gt_tid = input_ids[i, pos].item()
                            if gt_tid in exclude_id_set:
                                continue
                            weight[i, pos] = 1.0
                elif self.match_mode == "gap_tid":
                    # Intersection: (gap-bottom K%) ∩ (late K% positions where gen==GT)
                    # gap_bottom_pct -> fraction selected from the gap cache
                    # factual_late_pct -> fraction of late-unmasking positions from the denoising trajectory
                    assert self.gap_cache is not None, "gap_cache_path required for match_mode=gap_tid"
                    p_ids = input_ids[i, :prompt_len].tolist()
                    key = ",".join(str(t) for t in p_ids)
                    entry = self.gap_cache.get(key)
                    if entry is None:
                        logger.warning(f"[gap_tid] cache miss — skip sample")
                        continue

                    # 1) gap-bottom K%
                    gap_candidates = []
                    for pos in resp_pos.tolist():
                        rel = pos - prompt_len
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        info = entry.get(str(rel))
                        if info is not None:
                            gap_candidates.append((pos, info["gap"]))
                    if not gap_candidates:
                        continue
                    gap_candidates.sort(key=lambda t: t[1])
                    n_gap = max(1, int(len(gap_candidates) * self.gap_bottom_pct))
                    gap_set = set(p for p, _ in gap_candidates[:n_gap])

                    # 2) late K% (by unmask step) with gen==GT
                    n_late = max(1, int(len(content_by_step) * self.factual_late_pct))
                    tid_set = set()
                    for pos, _ in sorted_by_step[:n_late]:
                        if pos not in resp_pos_set:
                            continue
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        if x[0, pos].item() == gt_tid:
                            tid_set.add(pos)

                    # 3) intersection → weight=1.0
                    for pos in (gap_set & tid_set):
                        weight[i, pos] = 1.0
                elif self.match_mode == "token_id":
                    # v1 style: late token IDs → search GT for matching IDs
                    late_token_ids = set(x[0, p].item() for p in late_positions)
                    for pos in resp_pos:
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid not in late_token_ids:
                            continue
                        if gt_tid in exclude_id_set:
                            continue
                        matching_steps = [late_steps[p] for p in late_positions
                                          if x[0, p].item() == gt_tid]
                        step = max(matching_steps)
                        weight[i, pos] = w_fn(step)
                else:
                    # position mode: use late positions directly
                    for pos in late_positions:
                        if pos not in resp_pos_set:
                            continue
                        gt_tid = input_ids[i, pos].item()
                        if gt_tid in exclude_id_set:
                            continue
                        step = late_steps[pos]
                        weight[i, pos] = w_fn(step)

        return weight, maskable_mask

    def _gap_weights(self, input_ids, labels, attention_mask, prompt_lens):
        """Pre-computed single-mask gap cache → bottom K% tokens get weight=1.0.
        No denoising trajectory needed — pure cache lookup."""
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)
        exclude_id_set = set(exclude_ids.tolist())
        weight = torch.zeros(b, l, device=device, dtype=torch.float)

        for i in range(b):
            prompt_len = prompt_lens[i] if i < len(prompt_lens) else 0
            resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
            if len(resp_pos) == 0:
                continue

            p_ids = input_ids[i, :prompt_len].tolist()
            key = ",".join(str(t) for t in p_ids)
            entry = self.gap_cache.get(key)
            if entry is None:
                logger.warning(f"[gap] cache miss, applying GA to all response tokens")
                for pos in resp_pos.tolist():
                    gt_tid = input_ids[i, pos].item()
                    if gt_tid not in exclude_id_set:
                        weight[i, pos] = 1.0
                continue

            gap_candidates = []
            for pos in resp_pos.tolist():
                rel = pos - prompt_len
                gt_tid = input_ids[i, pos].item()
                if gt_tid in exclude_id_set:
                    continue
                info = entry.get(str(rel))
                if info is not None:
                    gap_candidates.append((pos, info["gap"]))

            if gap_candidates:
                gap_candidates.sort(key=lambda t: t[1])  # ascending = most uncertain first
                n_select = max(1, int(len(gap_candidates) * self.gap_bottom_pct))
                for pos, _ in gap_candidates[:n_select]:
                    weight[i, pos] = 1.0

        return weight, maskable_mask

    def _denoise_novel_ga_loss(self, model, input_ids, labels, attention_mask, prompt_lens):
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id

        # Select weight computation method
        if self.match_mode == "gap":
            weight, maskable_mask = self._gap_weights(input_ids, labels, attention_mask, prompt_lens)
        elif self.match_mode == "random":
            weight, maskable_mask = self._random_weights(input_ids, labels, attention_mask)
        elif self.match_mode == "cat_oracle":
            weight, maskable_mask = self._cat_oracle_weights(
                input_ids, labels, attention_mask, prompt_lens
            )
        elif self.single_pass:
            weight, maskable_mask = self._single_pass_weights(
                model, input_ids, labels, attention_mask, prompt_lens
            )
        else:
            weight, maskable_mask = self._denoising_trajectory_weights(
                model, input_ids, labels, attention_mask, prompt_lens
            )

        # For sigmoid mode: weight = masking probability → sample stochastic mask
        if self.weight_type == "sigmoid":
            # weight[i,pos] = P(mask pos) = sigmoid((step_pct - 0.5) * k)
            rand_mask = torch.rand_like(weight) < weight   # stochastic masking
            novel_mask = rand_mask & maskable_mask
            # loss weight = uniform (1.0) for all masked positions
            loss_weight = novel_mask.float()
        else:
            novel_mask = weight > 0
            loss_weight = weight

        # ── Write-only mask logging hook (no gradient impact) ──
        if self.log_mask_categories_path and self.is_world_process_zero():
            with torch.no_grad():
                try:
                    epoch = float(self.state.epoch) if self.state is not None else -1.0
                    step = int(self.state.global_step) if self.state is not None else self._log_count
                    entries = []
                    for i in range(input_ids.shape[0]):
                        positions = novel_mask[i].nonzero(as_tuple=True)[0].tolist()
                        if not positions:
                            continue
                        entries.append({
                            "epoch": epoch,
                            "step": step,
                            "prompt_len": int(prompt_lens[i]) if i < len(prompt_lens) else 0,
                            "input_ids": input_ids[i].detach().cpu().tolist(),
                            "mask_positions": positions,
                        })
                    if entries:
                        with open(self.log_mask_categories_path, "a") as f:
                            for e in entries:
                                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                except Exception as _e:
                    logger.warning(f"[log_mask_categories] write failed: {_e}")

        n_novel = novel_mask.sum().item()
        n_maskable = maskable_mask.sum().item()
        pct = 100 * n_novel / max(1, n_maskable)
        w_sum = weight.sum().item()
        logger.info(
            f"[mdu #{self._log_count}] "
            f"NOVEL={n_novel}/{n_maskable} ({pct:.1f}%) weight_sum={w_sum:.2f}"
        )
        print(
            f"[mdu #{self._log_count}] "
            f"NOVEL={n_novel}/{n_maskable} ({pct:.1f}%) weight_sum={w_sum:.2f}",
            flush=True,
        )
        self._log_count += 1

        if n_novel == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), None

        # Pass 2: mask novel positions → compute loss
        noised = torch.where(novel_mask, mask_id, input_ids)
        outputs = model(input_ids=noised, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # ── Loss computation based on loss_type ──
        if self.loss_type == "ga":
            token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
            weighted_nll = token_nll * loss_weight
            nll = weighted_nll.sum() / loss_weight.sum().clamp_min(1)

        elif self.loss_type == "bounded_ga":
            token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
            token_nll = torch.clamp(token_nll, max=self.loss_cap)
            weighted_nll = token_nll * loss_weight
            nll = weighted_nll.sum() / loss_weight.sum().clamp_min(1)

        elif self.loss_type == "null_anchor":
            # CFG-flavor: pull p_c → p_∅ on selected positions.
            # Need an extra uncond forward (Q masked).
            q_mask = (~maskable_mask) & attention_mask.bool() if attention_mask is not None else (~maskable_mask)
            noised_u = torch.where(q_mask, mask_id, noised)
            logits_u = self._null_anchor_uncond_logits(model, noised_u, attention_mask)
            kl_per_token = self._null_anchor_kl(logits, logits_u)  # (B, T)
            weighted_kl = kl_per_token * loss_weight
            # Minimize directly (compute_loss adds it, doesn't negate)
            nll = weighted_kl.sum() / loss_weight.sum().clamp_min(1)

        elif self.loss_type == "npo":
            # NPO: -logsigmoid(beta * (current_loss - ref_loss))
            token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
            weighted_nll = token_nll * loss_weight
            current_loss = weighted_nll.sum() / loss_weight.sum().clamp_min(1)

            with torch.no_grad():
                ref_logits = self._ref_forward_logits(noised, attention_mask)
                ref_nll = F.cross_entropy(ref_logits.transpose(1, 2), input_ids, reduction="none")
                ref_weighted = ref_nll * loss_weight
                ref_loss = ref_weighted.sum() / loss_weight.sum().clamp_min(1)

            neg_log_ratio = current_loss - ref_loss
            npo_loss = -F.logsigmoid(self.npo_beta * neg_log_ratio).mean() * 2 / self.npo_beta
            # Return positive so that compute_loss negates it (loss = loss - forget_nll)
            nll = npo_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # ── Diagnostic logging (write-only, no gradient impact) ──
        if (self.diagnostic_csv and self.is_world_process_zero()
                and self._log_count % self.diagnostic_interval == 0):
            with torch.no_grad():
                try:
                    log_probs_diag = F.log_softmax(logits.float(), dim=-1)
                    probs_diag = log_probs_diag.exp()
                    H = -(probs_diag * log_probs_diag).sum(dim=-1)  # (B, T)
                    p_GT = probs_diag.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
                    log_p_GT = torch.log(p_GT.clamp_min(1e-30))
                    # Analytical gradient magnitude on GT logit
                    grad_GA = (1.0 - p_GT)                       # GA's |∂L/∂z_GT|

                    sel = loss_weight > 0
                    nonsel = (~sel) & (novel_mask if novel_mask.shape == sel.shape else sel)

                    def _mean(t, m):
                        return float(t[m].mean().item()) if m.any() else float('nan')

                    epoch = float(self.state.epoch) if self.state is not None else -1.0
                    step = int(self.state.global_step) if self.state is not None else self._log_count

                    row = {
                        "step": step,
                        "epoch": round(epoch, 4),
                        "loss_type": self.loss_type,
                        "loss_value": float(nll.item()),
                        "n_selected": int(sel.sum().item()),
                        "n_nonselected": int(nonsel.sum().item()) if nonsel.any() else 0,
                        "p_GT_sel": _mean(p_GT, sel),
                        "H_sel": _mean(H, sel),
                        "H_nonsel": _mean(H, nonsel) if nonsel.any() else float('nan'),
                        "grad_GA_sel": _mean(grad_GA, sel),
                    }
                    if not self._diag_header_written and not os.path.exists(self.diagnostic_csv):
                        with open(self.diagnostic_csv, "w") as f:
                            f.write(",".join(row.keys()) + "\n")
                        self._diag_header_written = True
                    elif not self._diag_header_written:
                        self._diag_header_written = True
                    with open(self.diagnostic_csv, "a") as f:
                        f.write(",".join(str(v) for v in row.values()) + "\n")
                except Exception as _e:
                    logger.warning(f"[diagnostic] write failed: {_e}")

        return nll, outputs

    def _null_anchor_kl(self, logits_c, logits_u):
        """KL between cond and ESD-style target distribution.

        target_logits = logits_u - eta * (logits_c - logits_u)         (detached)
        eta = 0  → target = logits_u  (current "anchor" form)
        eta = 1  → target = 2*logits_u - logits_c  (ESD canonical, anti-c)

        kl_dir:
          forward : KL(p_c ‖ sg(p_target))    ← default
          reverse : KL(sg(p_target) ‖ p_c)
          js      : 0.5 [KL(p_c‖p_m) + KL(p_target‖p_m)]  with p_m = average
        Returns: (B, T) per-token KL scalar.
        """
        eta = self.null_anchor_eta
        tau = self.null_anchor_tau
        # target logits (with stop-grad on logits_u; logits_c.detach() if eta != 0 to keep target frozen)
        if eta == 0.0:
            target_logits = logits_u.detach()
        else:
            target_logits = (logits_u - eta * (logits_c - logits_u)).detach()
        # τ-temperature unification: τ=1 → NA (current), τ=0 → uniform anchor, τ>1 → sharper
        target_logits = tau * target_logits
        # Optionally mask special tokens (EOS/pad/mask/etc.) in BOTH cond and target
        # so KL is over the non-special vocab subset and remains finite.
        logits_c_eff = logits_c.float()
        target_logits = target_logits.float()
        if self.null_anchor_exclude_special:
            if self._special_token_ids is None and self.processing_class is not None:
                ids = list(getattr(self.processing_class, 'all_special_ids', []) or [])
                self._special_token_ids = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(0, dtype=torch.long)
            ids = self._special_token_ids
            if ids is not None and ids.numel() > 0:
                ids_dev = ids.to(target_logits.device)
                logits_c_eff = logits_c_eff.clone()
                target_logits = target_logits.clone()
                logits_c_eff.index_fill_(-1, ids_dev, float('-inf'))
                target_logits.index_fill_(-1, ids_dev, float('-inf'))
        log_pc = F.log_softmax(logits_c_eff, dim=-1)
        log_pt = F.log_softmax(target_logits, dim=-1).detach()
        # Helper: KL with safe handling of -inf positions where p_c = 0 (avoid 0 * inf = nan)
        def _safe_kl(p, log_p, log_q):
            term = log_p - log_q  # may have inf - inf = nan at masked positions
            term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
            return (p * term).sum(dim=-1)
        if self.null_anchor_kl_dir == "reverse":
            pt = log_pt.exp()
            return _safe_kl(pt, log_pt, log_pc)
        if self.null_anchor_kl_dir == "js":
            pc = log_pc.exp(); pt = log_pt.exp()
            pm = 0.5 * (pc + pt)
            log_pm = pm.clamp_min(1e-12).log()
            kl1 = _safe_kl(pc, log_pc, log_pm)
            kl2 = _safe_kl(pt, log_pt, log_pm)
            return 0.5 * (kl1 + kl2)
        # default: forward KL(p_c ‖ p_target)
        pc = log_pc.exp()
        return _safe_kl(pc, log_pc, log_pt)

    def _traj_rollout_na_loss(self, model, input_ids, labels, attention_mask):
        """Trajectory-rollout NA loss (option C: TRUE inference rollout w/ model preds).

        Faithful to the CFG-style trajectory loss:
            L_traj = E[ KL( p_θ(·|x, ŷ_s) ‖ p_θ(·|m, ŷ_s) ) ]
        where ŷ_s is the partially-generated state along the model's OWN inference
        trajectory (not GT-forced).

        1. Run partial inference rollout using the CURRENT model:
           - Start with all answer positions masked.
           - Each step: model(no_grad) forward → confidence top-k → place model's
             argmax preds at those positions.
        2. Stop at a random step ∈ [1, T-1] → use that ŷ_s as training state.
        3. Standard NA loss on currently-masked answer positions.

        Cost per batch: ~T/2 forwards (rollout, no_grad) + 2 forwards (training, c+u).
        """
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id
        T = max(int(self.null_anchor_traj_steps), 2)

        maskable_mask = labels != -100  # answer positions

        # Trajectory uses the CURRENT trainable model (matches user's math).
        # NOTE: rollout itself is no_grad, so no backprop through trajectory.
        traj_model = model

        # Random stop step per batch (uniform in [1, T-1])
        import random as _rnd
        stop_step = _rnd.randint(1, T - 1)

        with torch.no_grad():
            x = torch.where(maskable_mask, torch.full_like(input_ids, mask_id), input_ids)
            for step in range(stop_step):
                mask_idx = (x == mask_id) & maskable_mask
                if not mask_idx.any():
                    break
                # Number of positions to unmask this step (linear scheduler)
                n_remaining = mask_idx.sum(dim=1).float()  # (B,)
                steps_left = max(T - step, 1)
                n_unmask = (n_remaining / steps_left).ceil().long().clamp(min=1)

                out = traj_model(input_ids=x, attention_mask=attention_mask)
                out = self._postprocess_outputs(out)
                probs = F.softmax(out.logits.float(), dim=-1)
                conf = probs.max(dim=-1).values  # (B, L)
                pred = probs.argmax(dim=-1)       # (B, L)  ← model's predicted tokens

                # Per-sample top-k confident masked positions → place MODEL'S PREDS (option C)
                for i in range(b):
                    pos = mask_idx[i].nonzero(as_tuple=True)[0]
                    if len(pos) == 0:
                        continue
                    k = int(min(n_unmask[i].item(), len(pos)))
                    _, top_idx = conf[i, pos].topk(k, largest=True)
                    chosen = pos[top_idx]
                    x[i, chosen] = pred[i, chosen]  # model's argmax preds (true inference)

        # Training-time forward at the rollout y_t
        noised = x
        masked_mask = (noised == mask_id) & maskable_mask
        if not masked_mask.any():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            # still need an outputs object for callers; do a dummy forward
            outputs_dummy = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_dummy = self._postprocess_outputs(outputs_dummy)
            return zero, outputs_dummy

        # cond forward (grad)
        outputs_c = model(input_ids=noised, attention_mask=attention_mask)
        outputs_c = self._postprocess_outputs(outputs_c)
        logits_c = outputs_c.logits

        # uncond forward (mask Q positions)
        if attention_mask is not None:
            q_mask = (~maskable_mask) & attention_mask.bool()
        else:
            q_mask = ~maskable_mask
        noised_u = torch.where(q_mask, mask_id, noised)
        logits_u = self._null_anchor_uncond_logits(model, noised_u, attention_mask)

        kl_per_token = self._null_anchor_kl(logits_c, logits_u)
        null_loss = (kl_per_token * masked_mask.float()).sum() / masked_mask.sum().clamp_min(1)

        n_masked = int(masked_mask.sum().item())
        n_total = int(maskable_mask.sum().item())
        logger.info(
            f"[na-traj-rollout] stop_step={stop_step}/{T}  masked={n_masked}/{n_total} "
            f"({100.0*n_masked/max(n_total,1):.1f}%) η={self.null_anchor_eta} τ={self.null_anchor_tau} "
            f"dir={self.null_anchor_kl_dir} uncond={self._null_anchor_uncond_label()} kl̄={null_loss.item():.4f}"
        )
        return null_loss, outputs_c

    def _random_sft_null_anchor_loss(self, model, input_ids, labels, attention_mask):
        """SFT-style random masking + KL(p_c ‖ sg(p_∅)) — CFG-flavor null-anchor.

        cond   : forward on [Q | partially-masked answer]              (gradient O)
        uncond : forward on [<mask>×Q | partially-masked answer]       (gradient X)
        loss   : KL( softmax(logits_c) ‖ stop_grad(softmax(logits_u)) ) on masked answer pos.

        We *minimize* this loss (no negation), so caller must add (not subtract).
        """
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id

        maskable_mask = labels != -100  # answer positions only
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (torch.rand((b, l), device=device) < p_mask) & maskable_mask
        noised = torch.where(masked_mask, mask_id, input_ids)

        # cond forward (gradient O)
        outputs_c = model(input_ids=noised, attention_mask=attention_mask)
        outputs_c = self._postprocess_outputs(outputs_c)
        logits_c = outputs_c.logits

        # uncond forward: also mask Q positions
        if attention_mask is not None:
            q_mask = (~maskable_mask) & attention_mask.bool()
        else:
            q_mask = ~maskable_mask
        noised_u = torch.where(q_mask, mask_id, noised)
        logits_u = self._null_anchor_uncond_logits(model, noised_u, attention_mask)

        kl_per_token = self._null_anchor_kl(logits_c, logits_u)
        null_loss = (kl_per_token * masked_mask.float()).sum() / masked_mask.sum().clamp_min(1)

        n_masked = masked_mask.sum().item()
        n_maskable = maskable_mask.sum().item()
        logger.info(
            f"[null-anchor] masked={n_masked}/{n_maskable} "
            f"({100*n_masked/max(1,n_maskable):.1f}%) "
            f"η={self.null_anchor_eta} τ={self.null_anchor_tau} dir={self.null_anchor_kl_dir} "
            f"uncond={self._null_anchor_uncond_label()} kl̄={null_loss.item():.4f}"
        )
        return null_loss, outputs_c

    def _random_ga_loss(self, model, input_ids, labels, attention_mask):
        """SFT-style random diffusion masking + GA loss (no denoising trajectory)."""
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (torch.rand((b, l), device=device) < p_mask) & maskable_mask
        noised = torch.where(masked_mask, self.processing_class.mask_token_id, input_ids)

        outputs = model(input_ids=noised, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        loss_weights = self._compute_loss_weights(
            t=t, inputs={"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask},
            masked_mask=masked_mask,
        )
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        ga_loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)

        n_masked = masked_mask.sum().item()
        n_maskable = maskable_mask.sum().item()
        logger.info(f"[random-ga] masked={n_masked}/{n_maskable} ({100*n_masked/max(1,n_maskable):.1f}%)")
        return ga_loss, outputs

    def _random_npo_loss(self, model, input_ids, labels, attention_mask):
        """SFT-style random diffusion masking + NPO loss (no denoising trajectory)."""
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (torch.rand((b, l), device=device) < p_mask) & maskable_mask
        noised = torch.where(masked_mask, self.processing_class.mask_token_id, input_ids)

        outputs = model(input_ids=noised, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        loss_weights = self._compute_loss_weights(
            t=t, inputs={"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask},
            masked_mask=masked_mask,
        )
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        current_loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)

        with torch.no_grad():
            ref_logits = self._ref_forward_logits(noised, attention_mask)
            ref_nll = F.cross_entropy(ref_logits.transpose(1, 2), input_ids, reduction="none")
            ref_nll = ref_nll * loss_weights * masked_mask.to(ref_nll.dtype)
            ref_loss = ref_nll.sum() / maskable_mask.sum().clamp_min(1)

        neg_log_ratio = current_loss - ref_loss
        npo_loss = -F.logsigmoid(self.npo_beta * neg_log_ratio).mean() * 2 / self.npo_beta

        n_masked = masked_mask.sum().item()
        n_maskable = maskable_mask.sum().item()
        logger.info(f"[random-npo] masked={n_masked}/{n_maskable} ({100*n_masked/max(1,n_maskable):.1f}%)")
        return npo_loss, outputs

    def _retain_sft_loss(self, model, input_ids, labels, attention_mask):
        b, l = input_ids.shape
        maskable_mask = labels != -100
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=input_ids.device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (torch.rand((b, l), device=input_ids.device) < p_mask) & maskable_mask
        noised = torch.where(masked_mask, self.processing_class.mask_token_id, input_ids)

        outputs = model(input_ids=noised, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        loss_weights = self._compute_loss_weights(
            t=t, inputs={"input_ids": input_ids, "labels": labels}, masked_mask=masked_mask
        )
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        return token_nll.sum() / maskable_mask.sum().clamp_min(1)

    def _retain_late_sft_loss(self, model, input_ids, labels, attention_mask, prompt_lens):
        """
        Retain SFT with token-id matching, mirroring forget path (match_mode=token_id).
        1) Run denoising trajectory on retain sample.
        2) Select top retain_late_pct% late-unmask positions (content only).
        3) Build late_token_ids = set of model-predicted tids at those positions.
        4) Mark every GT response position whose gt_tid ∈ late_token_ids.
        5) Mask those positions, compute plain-mean NLL (binary weights, no 1/t).
        """
        b, l = input_ids.shape
        device = input_ids.device
        mask_id = self.processing_class.mask_token_id
        maskable_mask = labels != -100
        exclude_ids = self._get_exclude_ids(device)
        exclude_id_set = set(exclude_ids.tolist())

        novel_mask = torch.zeros_like(maskable_mask, dtype=torch.bool)

        with torch.no_grad():
            for i in range(b):
                prompt_len = prompt_lens[i] if i < len(prompt_lens) else 0
                resp_pos = maskable_mask[i].nonzero(as_tuple=True)[0]
                if len(resp_pos) == 0:
                    continue

                # Build noised: prompt kept, response fully masked
                x = input_ids[i:i+1].clone()
                x[0, resp_pos] = mask_id

                mask_index_resp = (x == mask_id)[:, prompt_len:]
                num_transfers = get_num_transfer_tokens(
                    mask_index=mask_index_resp,
                    steps=self.denoise_steps,
                    scheduler=self.scheduler,
                    stochastic=False,
                )[0].tolist()

                unmask_step = {}
                xc = x.clone()
                for step, n_unmask in enumerate(num_transfers):
                    if n_unmask == 0:
                        continue
                    logits_step = model(input_ids=xc).logits
                    probs = F.softmax(logits_step[0].float(), dim=-1)
                    conf = probs.max(dim=-1).values
                    pred = probs.argmax(dim=-1)
                    masked_pos = (xc[0] == mask_id).nonzero(as_tuple=True)[0]
                    masked_pos = masked_pos[masked_pos >= prompt_len]
                    if len(masked_pos) == 0:
                        break
                    n_pick = min(n_unmask, len(masked_pos))
                    _, top_idx = conf[masked_pos].topk(n_pick, largest=True)
                    for pos in masked_pos[top_idx].tolist():
                        unmask_step[pos] = step
                        xc[0, pos] = pred[pos]

                if not unmask_step:
                    continue

                # Exclude punctuation / special tokens from content
                content = []
                for pos, step in unmask_step.items():
                    gen_tid = xc[0, pos].item()
                    if gen_tid in exclude_id_set:
                        continue
                    content.append((pos, step, gen_tid))
                if not content:
                    continue

                content.sort(key=lambda kv: -kv[1])  # late first
                n_late = max(1, int(len(content) * self.retain_late_pct))
                late_entries = content[:n_late]

                # tid-matching (mirrors forget match_mode=token_id)
                late_token_ids = set(tid for _, _, tid in late_entries)
                for pos in resp_pos.tolist():
                    gt_tid = input_ids[i, pos].item()
                    if gt_tid in exclude_id_set:
                        continue
                    if gt_tid not in late_token_ids:
                        continue
                    novel_mask[i, pos] = True

        n_novel = novel_mask.sum().item()
        n_maskable = maskable_mask.sum().item()
        logger.info(
            f"[retain-late #{self._log_count}] "
            f"RETAIN_LATE={n_novel}/{n_maskable} ({100*n_novel/max(1,n_maskable):.1f}%)"
        )

        if n_novel == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        noised = torch.where(novel_mask, mask_id, input_ids)
        outputs = model(input_ids=noised, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        weighted = token_nll * novel_mask.to(token_nll.dtype)
        return weighted.sum() / novel_mask.to(token_nll.dtype).sum().clamp_min(1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        is_forget = inputs.get(
            "is_forget",
            torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        )
        prompt_lens = inputs.get("prompt_lens", [0] * input_ids.shape[0])

        forget_idx = is_forget.nonzero(as_tuple=True)[0]
        retain_idx = (~is_forget).nonzero(as_tuple=True)[0]

        loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        outputs = None

        if len(forget_idx) > 0:
            f_ids = input_ids[forget_idx]
            f_labels = labels[forget_idx]
            f_attn = attention_mask[forget_idx] if attention_mask is not None else None
            f_prompt_lens = [prompt_lens[i] for i in forget_idx.tolist()]
            if self.match_mode == "random" and self.loss_type == "ga":
                forget_nll, outputs = self._random_ga_loss(
                    model, f_ids, f_labels, f_attn
                )
            elif self.match_mode == "random" and self.loss_type == "npo":
                forget_nll, outputs = self._random_npo_loss(
                    model, f_ids, f_labels, f_attn
                )
            elif self.match_mode == "random" and self.loss_type == "null_anchor":
                if self.null_anchor_traj_rollout:
                    forget_nll, outputs = self._traj_rollout_na_loss(
                        model, f_ids, f_labels, f_attn
                    )
                else:
                    forget_nll, outputs = self._random_sft_null_anchor_loss(
                        model, f_ids, f_labels, f_attn
                    )
            else:
                forget_nll, outputs = self._denoise_novel_ga_loss(
                    model, f_ids, f_labels, f_attn, f_prompt_lens
                )
            if self.loss_type in ("npo", "null_anchor"):
                # These losses are minimized directly (not negated like GA)
                loss = loss + forget_nll
            else:
                loss = loss - forget_nll

        if len(retain_idx) > 0 and self.alpha != 0:
            r_ids = input_ids[retain_idx]
            r_labels = labels[retain_idx]
            r_attn = attention_mask[retain_idx] if attention_mask is not None else None
            if self.retain_mode == "late":
                r_prompt_lens = [prompt_lens[i] for i in retain_idx.tolist()]
                retain_nll = self._retain_late_sft_loss(
                    model, r_ids, r_labels, r_attn, r_prompt_lens
                )
            else:
                retain_nll = self._retain_sft_loss(model, r_ids, r_labels, r_attn)
            loss = loss + self.alpha * retain_nll

        self._global_step += 1
        return (loss, outputs) if return_outputs else loss


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "./checkpoints/llada-tofu-sft/checkpoint-final"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    tofu_split: str = field(
        default="forget10",
        metadata={"help": "HF TOFU config name (e.g. forget10). Empty string → use tofu_path."},
    )
    retain_tofu_split: str = field(
        default="retain_perturbed",
        metadata={"help": "HF retain config. Empty string → use retain_path."},
    )
    hf_dataset: str = field(default="locuslab/TOFU", metadata={"help": "HF dataset repo."})
    hf_split: str = field(default="train", metadata={"help": "HF split name."})
    tofu_path: str = field(
        default="./data/tofu/forget10.json",
        metadata={"help": "Local forget JSONL when tofu_split is empty."},
    )
    retain_path: str = field(
        default="./data/tofu/retain_perturbed.json",
        metadata={"help": "Local retain JSONL when retain_tofu_split is empty."},
    )
    mask_prompt_loss: bool = field(default=True)
    novel_percentile: int = field(
        default=30,
        metadata={"help": "Treat the last N% of unmasking steps as novel positions."}
    )
    denoise_steps: int = field(
        default=32,
        metadata={"help": "Number of denoising steps; more is more accurate but slower."}
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Legacy argument for older runs; unused in GT-mask Pass 1."}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Retain loss coefficient (lambda in the paper)."}
    )
    weight_beta: float = field(
        default=0.0,
        metadata={"help": "[DEPRECATED] Ignored; always uses uniform binary weights."}
    )
    loss_type: str = field(
        default="ga",
        metadata={"help": "Loss type: ga | bounded_ga | npo | null_anchor"}
    )
    loss_cap: float = field(
        default=5.0,
        metadata={"help": "NLL upper bound for bounded_ga."}
    )
    npo_beta: float = field(
        default=0.2,
        metadata={"help": "NPO beta (used when loss_type=npo)."}
    )
    null_anchor_eta: float = field(
        default=0.0,
        metadata={"help": "null_anchor: ESD-style negative CFG strength. "
                          "target = logits_∅ - eta · (logits_c - logits_∅). "
                          "eta=0 → simple anchor (current). eta=1 → ESD canonical (anti-c)."},
    )
    null_anchor_kl_dir: str = field(
        default="forward",
        metadata={"help": "null_anchor: KL direction. forward=KL(p_c‖p_∅), reverse=KL(p_∅‖p_c), js=symmetric."},
    )
    null_anchor_tau: float = field(
        default=1.0,
        metadata={"help": "null_anchor: τ-temperature on target logits. "
                          "target = softmax(τ · logits_∅). "
                          "τ=1 → NA (current). τ=0 → uniform anchor. "
                          "τ in (0,1) → smooth interpolation. τ>1 → sharper anchor."},
    )
    null_anchor_traj_rollout: bool = field(
        default=False,
        metadata={"help": "null_anchor: if True, replace random masking with a partial inference "
                          "trajectory rollout (option B: low_conf unmask order, GT tokens placed). "
                          "Cost: ~T/2 extra forwards per batch."},
    )
    null_anchor_traj_steps: int = field(
        default=16,
        metadata={"help": "null_anchor: T for training-time trajectory rollout (default 16)."},
    )
    null_anchor_exclude_special: bool = field(
        default=False,
        metadata={"help": "null_anchor: if True, mask all_special_ids (EOS/pad/mask/etc.) "
                          "in the target distribution to -inf so cond is not pulled toward "
                          "EOS-heavy uncond predictions."},
    )
    null_anchor_source: str = field(
        default="auto",
        metadata={
            "help": "null_anchor uncond logits source. "
            "auto=upstream (random/traj→frozen SFT ref; denoise modes→trainable CFG). "
            "frozen_sft=always frozen ref_model. trainable_cfg=always trainable model (Q-masked CFG). "
            "When trainable_cfg or auto without random/traj, ref_model is not loaded (single-GPU friendly)."
        },
    )
    single_pass: bool = field(
        default=False,
        metadata={"help": "True: estimate novelty with a single forward pass instead of denoising."}
    )
    cache_interval: int = field(
        default=0,
        metadata={"help": "0=recompute the trajectory every step; N>0=refresh every N steps."}
    )
    adaptive_threshold: bool = field(
        default=False,
        metadata={"help": "True: per-sample adaptive threshold (mean+k*std). False: fixed percentile."}
    )
    adaptive_k: float = field(
        default=1.0,
        metadata={"help": "k value when adaptive_threshold is on (threshold = mean + k * std)."}
    )
    conf_threshold: float = field(
        default=0.0,
        metadata={"help": ">0: select tokens with confidence < tau at unmasking time (adaptive). 0 disables."}
    )
    weight_type: str = field(
        default="power",
        metadata={"help": "Weight function: power=(step/max)^beta | sigmoid=sigmoid((step_pct-0.5)*k)."}
    )
    sigmoid_k: float = field(
        default=8.0,
        metadata={"help": "Sigmoid weight sharpness; used when weight_type=sigmoid."}
    )
    sigmoid_center: float = field(
        default=0.5,
        metadata={"help": "Sigmoid center in [0,1]; larger values concentrate on later tokens (mean ~ 1-center)."}
    )
    match_mode: str = field(
        default="token_id",
        metadata={"help": "token_id | position | factual_filter | tid_factual | tid_prob | gap | gap_tid | cat_oracle"}
    )
    cat_cache_path: str = field(
        default="",
        metadata={"help": "cat-oracle cache JSON path (used by match_mode=cat_oracle); "
                          "{prompt_key: [cat per answer token]}."}
    )
    cat_target_cats: str = field(
        default="3",
        metadata={"help": "cat-oracle: comma-separated category ids treated as unlearn targets. "
                          "Default '3' = factual span tokens. Example '3,4'."}
    )
    factual_cache_path: str = field(
        default="",
        metadata={"help": "attention entropy cache JSON path (used by match_mode=factual_filter)."}
    )
    factual_late_pct: float = field(
        default=0.30,
        metadata={"help": "factual_filter: top-N% by unmask step are placed in the late bin (0..1)."}
    )
    factual_entropy_pct: float = field(
        default=0.67,
        metadata={"help": "factual_filter: within the late bin, the bottom-N% entropy tokens count as factual (0..1)."}
    )
    factual_entropy_reverse: bool = field(
        default=False,
        metadata={"help": "If True, select top-N% by entropy (reverse baseline)."}
    )
    prob_filter_pct: float = field(
        default=0.30,
        metadata={"help": "tid_prob: intersect the bottom-N% p(GT) under full-answer mask with the late tid set (0..1)."}
    )
    gap_cache_path: str = field(
        default="",
        metadata={"help": "single-mask gap cache JSON path (used by match_mode=gap)."}
    )
    gap_bottom_pct: float = field(
        default=0.30,
        metadata={"help": "gap: apply GA on the bottom-N% single-mask gap tokens (0..1)."}
    )
    retain_mode: str = field(
        default="sft",
        metadata={"help": "sft=standard random diffusion masking | late=pick the top retain_late_pct% positions from the denoising trajectory."}
    )
    retain_late_pct: float = field(
        default=0.30,
        metadata={"help": "Top late-unmasking percentile to select when retain_mode=late (0..1)."}
    )
    log_mask_categories_path: str = field(
        default="",
        metadata={"help": "Write-only JSONL path. If set, per-forget-batch mask selection logged "
                          "as {epoch, step, prompt_len, input_ids, mask_positions} for post-hoc "
                          "category analysis. Empty = disabled (no overhead)."}
    )
    diagnostic_csv: str = field(
        default="",
        metadata={"help": "Write-only CSV path for GA diagnostic stats: "
                          "p_GT, H(p), gradient magnitudes at selected/non-selected positions. "
                          "Empty = disabled."}
    )
    diagnostic_interval: int = field(
        default=10,
        metadata={"help": "How often to write diagnostic_csv rows (counted in forget-loss calls)."}
    )
    ref_device: str = field(
        default="auto",
        metadata={
            "help": "Device for frozen ref_model (null_anchor/npo). "
            "auto=cuda:1 when 2+ GPUs visible; same=colocate with trainable model; "
            "or an explicit device e.g. cuda:1."
        },
    )


@dataclass
class TrainingArguments(MDLMConfig):
    output_dir: str = "./checkpoints/_mdu_run"
    checkpoints_root: str = field(
        default="./checkpoints",
        metadata={"help": "Parent directory for named unlearning checkpoints."},
    )
    checkpoint_name: str = field(
        default="",
        metadata={"help": "Checkpoint folder name under checkpoints_root (auto if empty)."},
    )
    group_by_length: bool = True
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    save_strategy: str = "no"
    logging_steps: int = 10
    report_to: str = "none"
    wandb_project: str = field(
        default="unlearning-dllms-MDU",
        metadata={"help": "W&B project when report_to includes wandb."},
    )
    run_name: str = ""
    eval_strategy: str = "no"
    disable_data_parallel: str = field(
        default="auto",
        metadata={
            "help": "Disable nn.DataParallel on multi-GPU single-process runs. "
            "auto=yes when ref_model is on a separate GPU; yes/no to force."
        },
    )


# ── Data ──────────────────────────────────────────────────────────────────────

def sft_map_fn(example, tokenizer, is_forget):
    messages = example["messages"]
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_len": prompt_len,
        "is_forget": int(is_forget),
    }


# ── Train ─────────────────────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    output_dir, checkpoint_name = resolve_run_directory(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    training_args.checkpoint_name = checkpoint_name
    prepare_wandb_run_name(training_args)
    configure_wandb(training_args)
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    ref_model = None
    ref_device_placed = None
    load_ref = needs_ref_model(data_args)
    mdu_setup = build_mdu_setup_summary(
        data_args, ref_loaded=load_ref, ref_device_placed=None,
    )
    if load_ref:
        ref_model = dllm.utils.get_model(model_args=model_args)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        ref_device_placed = place_ref_model(ref_model, data_args.ref_device)
        mdu_setup["ref_device_placed"] = ref_device_placed
    else:
        ref_device_placed = None

    apply_disable_data_parallel(
        training_args,
        training_args.disable_data_parallel,
        ref_device_placed,
    )

    logger.info(format_mdu_setup_log(mdu_setup))
    if training_args.gradient_checkpointing:
        logger.info(
            "[mdu-setup] gradient_checkpointing=True "
            f"(kwargs={training_args.gradient_checkpointing_kwargs})"
        )

    forget_rows, retain_rows, data_source = load_forget_retain_rows(data_args)

    with accelerate.PartialState().local_main_process_first():
        forget_ds = Dataset.from_list(rows_to_messages(forget_rows))
        forget_ds = forget_ds.map(
            partial(sft_map_fn, tokenizer=tokenizer, is_forget=True),
            num_proc=1, desc="Tokenizing forget"
        )
        retain_ds = None
        if data_args.alpha != 0:
            retain_ds = Dataset.from_list(rows_to_messages(retain_rows))
            retain_ds = retain_ds.map(
                partial(sft_map_fn, tokenizer=tokenizer, is_forget=False),
                num_proc=1, desc="Tokenizing retain"
            )
            combined = concatenate_datasets([forget_ds, retain_ds]).shuffle(seed=42)
        else:
            combined = forget_ds.shuffle(seed=42)
        dataset = DatasetDict({"train": combined})
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()

    _TRAIN_SCRIPT_PATH = os.path.abspath(__file__)
    training_script_snapshot = None
    if accelerate.PartialState().is_main_process:
        training_script_snapshot = copy_script_snapshot(_TRAIN_SCRIPT_PATH, output_dir)

    if accelerate.PartialState().is_main_process:
        cfg = build_train_config(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            data_source=data_source,
            checkpoint_name=checkpoint_name,
        )
        cfg["mdu_setup"] = mdu_setup
        if training_script_snapshot is not None:
            cfg["training_script_snapshot"] = training_script_snapshot
        write_train_config(os.path.join(output_dir, "train_config.json"), cfg)
    logger.info(
        f"MDU dataset: forget={len(forget_ds)}, retain={len(retain_ds) if data_args.alpha != 0 else 0}"
    )

    trainer = MDUTrainer(
        novel_percentile=data_args.novel_percentile,
        denoise_steps=data_args.denoise_steps,
        alpha=data_args.alpha,
        max_new_tokens=data_args.max_new_tokens,
        weight_beta=data_args.weight_beta,
        loss_type=data_args.loss_type,
        loss_cap=data_args.loss_cap,
        npo_beta=data_args.npo_beta,
        null_anchor_eta=data_args.null_anchor_eta,
        null_anchor_kl_dir=data_args.null_anchor_kl_dir,
        null_anchor_tau=data_args.null_anchor_tau,
        null_anchor_traj_rollout=data_args.null_anchor_traj_rollout,
        null_anchor_traj_steps=data_args.null_anchor_traj_steps,
        null_anchor_exclude_special=data_args.null_anchor_exclude_special,
        null_anchor_source=data_args.null_anchor_source,
        ref_model=ref_model,
        ref_device=ref_device_placed,
        single_pass=data_args.single_pass,
        cache_interval=data_args.cache_interval,
        adaptive_threshold=data_args.adaptive_threshold,
        adaptive_k=data_args.adaptive_k,
        conf_threshold=data_args.conf_threshold,
        weight_type=data_args.weight_type,
        sigmoid_k=data_args.sigmoid_k,
        sigmoid_center=data_args.sigmoid_center,
        match_mode=data_args.match_mode,
        factual_cache_path=data_args.factual_cache_path,
        factual_late_pct=data_args.factual_late_pct,
        factual_entropy_pct=data_args.factual_entropy_pct,
        factual_entropy_reverse=data_args.factual_entropy_reverse,
        prob_filter_pct=data_args.prob_filter_pct,
        gap_cache_path=data_args.gap_cache_path,
        gap_bottom_pct=data_args.gap_bottom_pct,
        retain_mode=data_args.retain_mode,
        retain_late_pct=data_args.retain_late_pct,
        log_mask_categories_path=data_args.log_mask_categories_path,
        diagnostic_csv=data_args.diagnostic_csv,
        diagnostic_interval=data_args.diagnostic_interval,
        cat_cache_path=data_args.cat_cache_path,
        cat_target_cats=data_args.cat_target_cats,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=MDUCollator(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=-100,
            )
        ),
    )
    maybe_add_wandb_callback(
        trainer,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        data_source=data_source,
        checkpoint_name=checkpoint_name,
    )
    trainer.train()
    ckpt_dir = save_final_checkpoint(
        trainer,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        data_source=data_source,
        checkpoint_name=checkpoint_name,
        training_script_path=_TRAIN_SCRIPT_PATH,
    )
    logger.info(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    train()
