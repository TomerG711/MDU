#!/usr/bin/env python3
"""Compare base vs SFT vs unlearned student on fixed masked training forwards.

Replays the random-mask null-anchor setup used in ``_random_sft_null_anchor_loss``
with ``null_anchor_source=pre_sft_cond``: same ``noised`` input for student and
base ref; reports per-masked-position KL(student‖base) and argmax token agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
_DLLM = os.path.join(_REPO_ROOT, "dllm")
for _p in (_SRC, _SCRIPTS, _DLLM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dllm.utils  # noqa: E402
from dllm.core.schedulers import LinearAlphaScheduler  # noqa: E402
from tofu_data import load_tofu_hf  # noqa: E402
from unlearn_mdu_llada import sft_map_fn  # noqa: E402
from unlearn_run_utils import DEFAULT_PRE_SFT_REF_LLAMA, rows_to_messages  # noqa: E402


def null_anchor_kl(
    logits_c: torch.Tensor,
    logits_u: torch.Tensor,
    *,
    tau: float = 0.25,
    eta: float = 0.0,
) -> torch.Tensor:
    """Forward KL(p_c ‖ sg(p_target)) per token; matches MDUTrainer._null_anchor_kl."""
    if eta == 0.0:
        target_logits = logits_u.detach()
    else:
        target_logits = (logits_u - eta * (logits_c - logits_u)).detach()
    target_logits = tau * target_logits.float()
    logits_c_eff = logits_c.float()
    log_pc = F.log_softmax(logits_c_eff, dim=-1)
    log_pt = F.log_softmax(target_logits, dim=-1).detach()
    pc = log_pc.exp()
    term = log_pc - log_pt
    term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
    return (pc * term).sum(dim=-1)


def apply_random_mask(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask_id: int,
    scheduler: LinearAlphaScheduler,
    time_epsilon: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (noised, masked_mask, maskable_mask) for batch size 1."""
    b, seq_len = input_ids.shape
    device = input_ids.device
    maskable_mask = labels != -100

    t = time_epsilon + (1 - time_epsilon) * torch.rand(
        b, device=device, generator=generator
    )
    p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, seq_len)
    rand_u = torch.rand((b, seq_len), device=device, generator=generator)
    masked_mask = (rand_u < p_mask) & maskable_mask
    noised = torch.where(masked_mask, mask_id, input_ids)
    return noised, masked_mask, maskable_mask


@torch.no_grad()
def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[0]


def load_model(path: str, device: torch.device):
    model = dllm.utils.get_model(model_name_or_path=path, dtype=torch.bfloat16)
    model.eval()
    model.to(device)
    return model


def token_str(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def summarize_positions(
    tokenizer,
    input_ids: torch.Tensor,
    masked_mask: torch.Tensor,
    logits_by_name: dict[str, torch.Tensor],
    kl_tensors: dict[str, torch.Tensor],
    *,
    max_positions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = masked_mask[0].nonzero(as_tuple=True)[0].tolist()
    for pos in positions[:max_positions]:
        gt_id = int(input_ids[0, pos].item())
        row: dict[str, Any] = {
            "pos": int(pos),
            "gt_token": token_str(tokenizer, gt_id),
            "gt_id": gt_id,
        }
        for name, logits in logits_by_name.items():
            pred_id = int(logits[0, pos].argmax(-1).item())
            row[f"{name}_pred_id"] = pred_id
            row[f"{name}_pred"] = token_str(tokenizer, pred_id)
            row[f"{name}_matches_gt"] = pred_id == gt_id
        for name, kl in kl_tensors.items():
            row[f"{name}_kl_to_base"] = float(kl[0, pos].item())
        rows.append(row)
    return rows


def process_example(
    example_idx: int,
    batch: dict[str, torch.Tensor],
    *,
    tokenizer,
    models: dict[str, Any],
    scheduler: LinearAlphaScheduler,
    time_epsilon: float,
    mask_id: int,
    tau: float,
    seed: int,
    device: torch.device,
    max_positions: int,
) -> dict[str, Any]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + example_idx)

    noised, masked_mask, maskable_mask = apply_random_mask(
        input_ids,
        labels,
        mask_id=mask_id,
        scheduler=scheduler,
        time_epsilon=time_epsilon,
        generator=gen,
    )

    logits_by_name: dict[str, torch.Tensor] = {}
    for name, model in models.items():
        logits_by_name[name] = forward_logits(model, noised, attention_mask)

    base_logits = logits_by_name["base"]
    kl_tensors: dict[str, torch.Tensor] = {}
    for name in ("sft", "unlearned"):
        kl_tensors[name] = null_anchor_kl(logits_by_name[name], base_logits, tau=tau)

    masked = masked_mask[0]
    n_masked = int(masked.sum().item())
    n_maskable = int(maskable_mask[0].sum().item())

    summary: dict[str, Any] = {
        "example_idx": example_idx,
        "seq_len": int(input_ids.shape[1]),
        "prompt_len": int(batch["prompt_len"]),
        "n_maskable": n_maskable,
        "n_masked": n_masked,
        "mask_frac": n_masked / max(n_maskable, 1),
        "seed": seed + example_idx,
    }

    for name in ("sft", "unlearned"):
        kl = kl_tensors[name][0]
        if n_masked:
            mean_kl = float(kl[masked].mean().item())
            argmax_match_base = float(
                (logits_by_name[name][0, masked].argmax(-1)
                 == base_logits[0, masked].argmax(-1)).float().mean().item()
            )
            argmax_match_gt = float(
                (logits_by_name[name][0, masked].argmax(-1)
                 == input_ids[0, masked]).float().mean().item()
            )
        else:
            mean_kl = float("nan")
            argmax_match_base = float("nan")
            argmax_match_gt = float("nan")
        summary[f"{name}_mean_kl_to_base"] = mean_kl
        summary[f"{name}_argmax_match_base_frac"] = argmax_match_base
        summary[f"{name}_argmax_match_gt_frac"] = argmax_match_gt

    summary["positions"] = summarize_positions(
        tokenizer,
        input_ids,
        masked_mask,
        logits_by_name,
        kl_tensors,
        max_positions=max_positions,
    )

    # Short decoded answer span (answer tokens only, unmasked view)
    ans_start = int(batch["prompt_len"])
    ans_ids = input_ids[0, ans_start:].tolist()
    summary["answer_preview"] = tokenizer.decode(ans_ids, skip_special_tokens=False)[:240]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-path", default="./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU")
    parser.add_argument("--base-path", default=DEFAULT_PRE_SFT_REF_LLAMA)
    parser.add_argument(
        "--unlearned-path",
        default="./checkpoints/mdu_llada_forget10_random_presftcond_tau0p25",
    )
    parser.add_argument("--split", default="forget10")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument(
        "--output",
        default="./diag_outputs/pre_sft_masked_forward_tau0p25.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} tau={args.tau} seed={args.seed} n={args.num_examples}")

    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=args.sft_path)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    rows = load_tofu_hf(args.split)
    rows = rows_to_messages(rows[: args.num_examples])
    print(f"loaded {len(rows)} examples from TOFU split={args.split}")

    examples: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        mapped = sft_map_fn(row, tokenizer, is_forget=True)
        mapped["prompt_len"] = mapped["prompt_len"]
        examples.append(mapped)

    print("loading models (sequential to limit peak VRAM)...")
    model_paths = {
        "base": args.base_path,
        "sft": args.sft_path,
        "unlearned": args.unlearned_path,
    }
    models: dict[str, Any] = {}
    for name, path in model_paths.items():
        print(f"  loading {name}: {path}")
        models[name] = load_model(path, device)

    scheduler = LinearAlphaScheduler()
    results: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        batch = {
            "input_ids": torch.tensor([ex["input_ids"]], dtype=torch.long),
            "labels": torch.tensor([ex["labels"]], dtype=torch.long),
            "prompt_len": ex["prompt_len"],
        }
        print(f"\n=== example {i} (prompt_len={ex['prompt_len']}, seq_len={len(ex['input_ids'])}) ===")
        summary = process_example(
            i,
            batch,
            tokenizer=tokenizer,
            models=models,
            scheduler=scheduler,
            time_epsilon=args.time_epsilon,
            mask_id=mask_id,
            tau=args.tau,
            seed=args.seed,
            device=device,
            max_positions=args.max_positions,
        )
        results.append(summary)
        print(
            f"masked {summary['n_masked']}/{summary['n_maskable']} "
            f"({100*summary['mask_frac']:.1f}%)"
        )
        print(
            f"  sft       KL→base={summary['sft_mean_kl_to_base']:.4f}  "
            f"argmax∩base={summary['sft_argmax_match_base_frac']:.2f}  "
            f"argmax∩gt={summary['sft_argmax_match_gt_frac']:.2f}"
        )
        print(
            f"  unlearned KL→base={summary['unlearned_mean_kl_to_base']:.4f}  "
            f"argmax∩base={summary['unlearned_argmax_match_base_frac']:.2f}  "
            f"argmax∩gt={summary['unlearned_argmax_match_gt_frac']:.2f}"
        )
        for pos_row in summary["positions"][:4]:
            print(
                f"    pos={pos_row['pos']:4d} gt={pos_row['gt_token']!r} "
                f"base={pos_row['base_pred']!r} "
                f"sft={pos_row['sft_pred']!r} "
                f"ul={pos_row['unlearned_pred']!r} "
                f"kl_ul={pos_row.get('unlearned_kl_to_base', float('nan')):.3f}"
            )

    # Aggregate
    def _mean(key: str) -> float:
        vals = [r[key] for r in results if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    aggregate = {
        "tau": args.tau,
        "seed": args.seed,
        "num_examples": len(results),
        "sft_mean_kl_to_base": _mean("sft_mean_kl_to_base"),
        "unlearned_mean_kl_to_base": _mean("unlearned_mean_kl_to_base"),
        "sft_argmax_match_base_frac": _mean("sft_argmax_match_base_frac"),
        "unlearned_argmax_match_base_frac": _mean("unlearned_argmax_match_base_frac"),
        "sft_argmax_match_gt_frac": _mean("sft_argmax_match_gt_frac"),
        "unlearned_argmax_match_gt_frac": _mean("unlearned_argmax_match_gt_frac"),
    }

    print("\n=== aggregate (masked positions only) ===")
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    out = {
        "config": vars(args),
        "aggregate": aggregate,
        "examples": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
