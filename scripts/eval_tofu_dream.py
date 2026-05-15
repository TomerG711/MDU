"""
TOFU evaluation for Dream model
Metrics: RougeL, Probability, Truth Ratio

Usage:
    python eval_tofu_dream.py --model <model_path> \
        --forget_file /path/to/forget01.json \
        --perturbed_file /path/to/forget01_perturbed.json \
        --output_dir /path/to/output

Probability: exp(-ELBO_loss_per_token)  (Dream approximation)
Truth Ratio: exp(gt_loss_per_token - perturb_loss_per_token)
RougeL: generated output vs ground truth
"""
import json
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
parser.add_argument("--forget_file", default="./data/tofu/forget01.json")
parser.add_argument("--perturbed_file", default="./data/tofu/forget01_perturbed.json")
parser.add_argument("--output_dir", default=None)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--num_samples", type=int, default=None, help="None = all")
parser.add_argument("--mask_samples", type=int, default=128, help="Eq.(14) Monte Carlo samples")
parser.add_argument("--mc_batch_size", type=int, default=128, help="MC samples per forward pass")
parser.add_argument("--truth_ratio", action="store_true", help="Compute truth ratio (requires perturbed file)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

MASK_TOKEN_ID = tokenizer.mask_token_id or tokenizer.unk_token_id


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def compute_eq14_loss(input_ids, attention_mask, answer_mask, n_samples=128, batch_size=32):
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
            # right_shift_logits (Dream)
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

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
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).input_ids[0]
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    full_ids = torch.cat([prompt_ids, torch.tensor(answer_ids)])
    prompt_len = len(prompt_ids)
    answer_mask = torch.zeros(len(full_ids), dtype=torch.long)
    answer_mask[prompt_len:] = 1
    return full_ids, answer_mask


def generate_answer(question):
    messages = [{"role": "user", "content": question}]
    enc = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = enc.input_ids.to("cuda")
    attention_mask = enc.attention_mask.to("cuda")
    out = model.diffusion_generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens, steps=args.steps,
        temperature=0.0, alg="entropy", return_dict_in_generate=True,
    )
    pred = tokenizer.decode(out.sequences[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
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
    full_ids = full_ids.unsqueeze(0).to("cuda")
    attn_mask = torch.ones_like(full_ids, dtype=torch.bool).to("cuda")
    answer_mask = answer_mask.unsqueeze(0).to("cuda")

    gt_loss = compute_eq14_loss(full_ids, attn_mask, answer_mask, n_samples=args.mask_samples, batch_size=args.mc_batch_size)
    prob = np.exp(-gt_loss)
    prob_scores.append(prob)

    # 3. Truth Ratio (optional)
    truth_ratio = 1.0
    perturb_margin = 0.0
    if args.truth_ratio and perturbed_answers:
        perturb_losses = []
        for pa in perturbed_answers[:3]:
            p_ids, p_mask = build_input(question, pa)
            p_ids = p_ids.unsqueeze(0).to("cuda")
            p_attn = torch.ones_like(p_ids, dtype=torch.bool).to("cuda")
            p_mask = p_mask.unsqueeze(0).to("cuda")
            pl = compute_eq14_loss(p_ids, p_attn, p_mask, n_samples=args.mask_samples, batch_size=args.mc_batch_size)
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

    msg = f"  [{i+1}/{len(forget_data)}] RougeL={np.mean(rougeL_scores):.3f}  Prob={np.mean(prob_scores):.3f}"
    if args.truth_ratio:
        msg += f"  TruthRatio={np.mean(truth_ratios):.3f}"
    print(msg, flush=True)

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
