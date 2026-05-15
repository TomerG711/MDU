"""
TOFU evaluation for LLaDA model
Metrics: RougeL, Probability, Truth Ratio

Usage:
    python eval_tofu_llada.py --model <model_path> \
        --forget_file /path/to/forget01.json \
        --perturbed_file /path/to/forget01_perturbed.json \
        --output_dir /path/to/output
"""
import json
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer

import sys
sys.path.insert(0, "./dllm")
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
parser.add_argument("--forget_file", default="./data/tofu/forget01.json")
parser.add_argument("--perturbed_file", default="./data/tofu/forget01_perturbed.json")
parser.add_argument("--output_dir", default=None)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--mask_samples", type=int, default=128, help="Eq.(14) Monte Carlo samples")
parser.add_argument("--mc_batch_size", type=int, default=128, help="MC samples per forward pass")
parser.add_argument("--truth_ratio", action="store_true", help="Compute truth ratio (requires perturbed file)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--early_abort_n", type=int, default=0, help="Check abort after first N samples (0=disabled)")
parser.add_argument("--early_abort_threshold", type=float, default=0.05, help="Abort threshold for avg probability")
parser.add_argument("--early_abort_mode", choices=["below", "above"], default="below",
                    help="below: abort if prob < threshold (retain collapse), above: abort if prob > threshold (forget not working)")
parser.add_argument("--suppress_special", action="store_true",
                    help="Ban EOS/pad/special tokens during generation (forces non-empty output).")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModel.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
).to("cuda").eval()

MASK_TOKEN_ID = tokenizer.mask_token_id or model.config.mask_token_id
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = MASK_TOKEN_ID

sampler = MDLMSampler(model=model, tokenizer=tokenizer)
suppress_ids = list(tokenizer.all_special_ids) if args.suppress_special else None
if args.suppress_special:
    print(f"[suppress_special] banning special token IDs: {suppress_ids}")
sampler_config = MDLMSamplerConfig(
    max_new_tokens=args.max_new_tokens,
    steps=args.steps,
    temperature=0.0,
    remasking="low_confidence",
    return_dict=True,
    suppress_tokens=suppress_ids,
)


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def compute_eq14_loss(input_ids, answer_mask, n_samples=128, batch_size=32):
    """
    Eq.(14)-style conditional estimator (LLaDA Appendix A.2), batched MC.
    input_ids: [1, seq_len]
    answer_mask: [1, seq_len], 1 where response tokens are
    """
    b, seq_len = input_ids.shape
    ans_mask_bool = answer_mask.bool().squeeze(0)
    answer_indices = ans_mask_bool.nonzero(as_tuple=True)[0]
    L = answer_indices.shape[0]
    if L == 0:
        return 0.0

    # Pre-generate all masking patterns
    all_noised = input_ids.expand(n_samples, -1).clone()  # [n_samples, seq_len]
    mask_flags = torch.zeros(n_samples, seq_len, device=input_ids.device, dtype=torch.bool)
    ls = []

    for s in range(n_samples):
        l = torch.randint(1, L + 1, (1,)).item()
        ls.append(l)
        perm = torch.randperm(L, device=input_ids.device)[:l]
        positions = answer_indices[perm]
        all_noised[s, positions] = MASK_TOKEN_ID
        mask_flags[s, positions] = True

    target_ids = input_ids.expand(n_samples, -1)
    total_score = 0.0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        with torch.no_grad():
            logits = model(all_noised[start:end]).logits

        token_nll = F.cross_entropy(
            logits.transpose(1, 2), target_ids[start:end], reduction="none"
        )  # [bs, seq_len]

        for j in range(end - start):
            masked_ce_sum = token_nll[j][mask_flags[start + j]].sum().item()
            total_score += masked_ce_sum / ls[start + j]

    return total_score / n_samples


def build_input(question, answer):
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    full_ids = prompt_ids + answer_ids
    prompt_len = len(prompt_ids)
    answer_mask = [0] * prompt_len + [1] * len(answer_ids)
    return (
        torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to("cuda"),
        torch.tensor(answer_mask, dtype=torch.long).unsqueeze(0).to("cuda"),
    )


def generate_answer(question):
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
    output = sampler.sample([prompt_tensor], config=sampler_config)
    pred_ids = output.sequences[0][len(prompt_ids):]
    pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
    return pred


# Load data
forget_data = load_jsonl(args.forget_file)
perturbed_data = load_jsonl(args.perturbed_file) if args.truth_ratio else []
if args.num_samples:
    forget_data = forget_data[:args.num_samples]
    if perturbed_data:
        perturbed_data = perturbed_data[:args.num_samples]

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rougeL_scores = []
prob_scores = []
truth_ratios = []
details = []

print(f"\nModel: {args.model}")
print(f"Evaluating {len(forget_data)} samples...")
print("=" * 60)

for i, fgt in enumerate(forget_data):
    question = fgt["question"]
    gt_answer = fgt["answer"]
    pert = perturbed_data[i] if i < len(perturbed_data) else {}
    perturbed_answers = pert.get("perturbed_answer", [])

    # 1. RougeL
    pred = generate_answer(question)
    rouge = scorer.score(gt_answer, pred)["rougeL"].recall
    rougeL_scores.append(rouge)

    # 2. Eq.(14) reconstruction loss
    full_ids, answer_mask = build_input(question, gt_answer)
    gt_loss = compute_eq14_loss(full_ids, answer_mask, n_samples=args.mask_samples, batch_size=args.mc_batch_size)
    prob = np.exp(-gt_loss)
    prob_scores.append(prob)

    # 3. Truth Ratio (optional)
    truth_ratio = 1.0
    perturb_margin = 0.0
    if args.truth_ratio and perturbed_answers:
        perturb_losses = []
        for pa in perturbed_answers[:3]:
            p_ids, p_mask = build_input(question, pa)
            pl = compute_eq14_loss(p_ids, p_mask, n_samples=args.mask_samples, batch_size=args.mc_batch_size)
            perturb_losses.append(pl)
        mean_perturb_loss = np.mean(perturb_losses)
        truth_ratio = np.exp(gt_loss - mean_perturb_loss)
        perturb_margin = mean_perturb_loss - gt_loss
    truth_ratios.append(truth_ratio)

    detail = {
        "question": question,
        "gt_answer": gt_answer,
        "pred": pred,
        "rougeL": rouge,
        "gt_loss_eq14": gt_loss,
        "probability": prob,
    }
    if args.truth_ratio:
        detail["truth_ratio"] = truth_ratio
        detail["perturb_margin_eq14"] = perturb_margin
    details.append(detail)

    # Append each sample to the jsonl immediately
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "details.jsonl"), "a") as fout:
            fout.write(json.dumps(detail, ensure_ascii=False) + "\n")

    msg = f"  [{i+1}/{len(forget_data)}] RougeL={rouge:.3f}(avg={np.mean(rougeL_scores):.3f})  Prob={prob:.3f}(avg={np.mean(prob_scores):.3f})"
    if args.truth_ratio:
        msg += f"  TR={truth_ratio:.3f}(avg={np.mean(truth_ratios):.3f})"
    print(msg, flush=True)

    if args.early_abort_n > 0 and (i + 1) == args.early_abort_n:
        avg_prob = np.mean(prob_scores)
        should_abort = (
            (args.early_abort_mode == "below" and avg_prob < args.early_abort_threshold) or
            (args.early_abort_mode == "above" and avg_prob > args.early_abort_threshold)
        )
        if should_abort:
            op = "<" if args.early_abort_mode == "below" else ">"
            reason = f"avg_prob={avg_prob:.4f} {op} {args.early_abort_threshold} after {args.early_abort_n} samples (mode={args.early_abort_mode})"
            print(f"\n*** EARLY ABORT: {reason} ***")
            if args.output_dir:
                abort_result = {
                    "model": args.model,
                    "aborted": True,
                    "abort_reason": reason,
                    "scores": {
                        "rougeL": float(np.mean(rougeL_scores)),
                        "probability": float(avg_prob),
                    },
                    "details": details,
                }
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(abort_result, f, indent=2, ensure_ascii=False)
            sys.exit(1)

print("=" * 60)
print(f"  RougeL:      {np.mean(rougeL_scores):.3f}")
print(f"  Probability: {np.mean(prob_scores):.3f}")
if args.truth_ratio:
    print(f"  Truth Ratio: {np.mean(truth_ratios):.3f}")

if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    scores = {  
        "rougeL": float(np.mean(rougeL_scores)),
        "probability": float(np.mean(prob_scores)),
    }
    if args.truth_ratio:
        scores["truth_ratio"] = float(np.mean(truth_ratios))
    result = {
        "model": args.model,
        "scores": scores,
        "details": details,
    }
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
