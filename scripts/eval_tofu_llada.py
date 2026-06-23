"""
TOFU evaluation for LLaDA model
Metrics: RougeL, Probability, Truth Ratio

Usage:
    python eval_tofu_llada.py --model <model_path> \
        --forget_file /path/to/forget01.json \
        --perturbed_file /path/to/forget01_perturbed.json \
        --output_dir /path/to/output

    # Load split from Hugging Face (parquet on hub; same fields as local JSONL)
    python eval_tofu_llada.py --model <model_path> \
        --tofu_split forget10 --truth_ratio --output_dir /path/to/output

    # Structured run layout (recommended for multi-split / multi-method experiments)
    python eval_tofu_llada.py --model <model_path> \
        --experiment sft_baseline --run_id 2026-06-22_baseline \
        --tofu_split forget10
"""
import json
import argparse
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tofu_data import load_tofu_split

sys.path.insert(0, "./dllm")
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

_SCRIPT_PATH = os.path.abspath(__file__)
_TOFU_EVAL_SPLITS = ("forget10", "retain_perturbed", "real_authors", "world_facts")


# ---------------------------------------------------------------------------
# Run tracking (provenance / layout only — does not affect eval logic)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()


def _git_info():
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True, cwd=os.path.dirname(_SCRIPT_PATH),
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip() != ""
        return {"root": root, "commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _script_fingerprint():
    with open(_SCRIPT_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"path": _SCRIPT_PATH, "sha256": digest}


def _environment_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _infer_split_name(args):
    if args.tofu_split:
        return args.tofu_split
    base = os.path.basename(args.forget_file)
    return os.path.splitext(base)[0]


def _args_to_dict(args):
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def resolve_output_layout(args):
    """Map --experiment/--run_id to args.output_dir and run metadata paths."""
    if not args.experiment:
        return args

    if args.output_dir:
        print(
            f"[run] --experiment set; ignoring explicit --output_dir ({args.output_dir})",
            file=sys.stderr,
        )

    split_name = _infer_split_name(args)
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = os.path.join(args.eval_outputs_root, args.experiment, run_id)
    split_dir = os.path.join(run_root, "splits", split_name)

    args.run_id = run_id
    args.output_dir = split_dir
    args._run_root = run_root
    args._split_name = split_name
    return args


def prepare_split_output(args):
    """Create split directory; structured layout extras when --experiment is set."""
    if not args.output_dir:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    details_path = os.path.join(args.output_dir, "details.jsonl")

    if args.experiment:
        if os.path.exists(details_path) and not args.force:
            print(
                f"[run] {details_path} already exists. Pass --force to overwrite, "
                f"or use a new --run_id.",
                file=sys.stderr,
            )
            sys.exit(1)
        open(details_path, "w").close()

        log_path = os.path.join(args.output_dir, "run.log")
        args._log_file = open(log_path, "w", encoding="utf-8")
        args._stdout_orig = sys.stdout
        sys.stdout = _Tee(sys.stdout, args._log_file)

        started_at = datetime.now(timezone.utc).isoformat()
        split_meta = {
            "experiment": args.experiment,
            "run_id": getattr(args, "run_id", None),
            "split": getattr(args, "_split_name", _infer_split_name(args)),
            "status": "running",
            "started_at": started_at,
            "argv": sys.argv,
            "args": _args_to_dict(args),
            "model": args.model,
            "eval_script": _script_fingerprint(),
            "git": _git_info(),
            "environment": _environment_info(),
        }
        with open(os.path.join(args.output_dir, "split_meta.json"), "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2, ensure_ascii=False)

        if getattr(args, "_run_root", None):
            os.makedirs(args._run_root, exist_ok=True)
            run_readme = os.path.join(args._run_root, "README.txt")
            if not os.path.exists(run_readme):
                with open(run_readme, "w", encoding="utf-8") as f:
                    f.write(
                        f"experiment: {args.experiment}\n"
                        f"run_id: {args.run_id}\n"
                        f"paper splits: {', '.join(_TOFU_EVAL_SPLITS)}\n"
                        f"Use the same --experiment and --run_id for all four splits.\n"
                    )
            print(f"[run] output: {args.output_dir}")
            print(f"[run] run_root: {args._run_root}")


def _rebuild_summary(run_root):
    summary = {}
    splits_root = os.path.join(run_root, "splits")
    if not os.path.isdir(splits_root):
        return
    for split_name in sorted(os.listdir(splits_root)):
        results_path = os.path.join(splits_root, split_name, "results.json")
        if not os.path.isfile(results_path):
            continue
        with open(results_path, encoding="utf-8") as f:
            payload = json.load(f)
        entry = {
            "rougeL": payload.get("scores", {}).get("rougeL"),
            "probability": payload.get("scores", {}).get("probability"),
            "aborted": payload.get("aborted", False),
        }
        if "truth_ratio" in payload.get("scores", {}):
            entry["truth_ratio"] = payload["scores"]["truth_ratio"]
        summary[split_name] = entry
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _rebuild_run_manifest(run_root):
    manifest = {
        "experiment": None,
        "run_id": os.path.basename(run_root),
        "run_root": run_root,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "eval_script": _script_fingerprint(),
        "git": _git_info(),
        "environment": _environment_info(),
        "splits": {},
        "paper_splits_expected": list(_TOFU_EVAL_SPLITS),
    }
    splits_root = os.path.join(run_root, "splits")
    if not os.path.isdir(splits_root):
        return
    for split_name in sorted(os.listdir(splits_root)):
        split_dir = os.path.join(splits_root, split_name)
        meta_path = os.path.join(split_dir, "split_meta.json")
        split_entry = {"split_dir": split_dir}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            manifest["experiment"] = manifest["experiment"] or meta.get("experiment")
            split_entry.update({
                "status": meta.get("status"),
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "args": meta.get("args"),
                "model": meta.get("model"),
                "scores": meta.get("scores"),
                "abort_reason": meta.get("abort_reason"),
            })
        results_path = os.path.join(split_dir, "results.json")
        if os.path.isfile(results_path):
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
            split_entry.setdefault("scores", results.get("scores"))
            split_entry["aborted"] = results.get("aborted", False)
        manifest["splits"][split_name] = split_entry
    with open(os.path.join(run_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def finalize_split_output(args, *, status="completed", scores=None, abort_reason=None):
    if not args.experiment or not args.output_dir:
        return

    finished_at = datetime.now(timezone.utc).isoformat()
    meta_path = os.path.join(args.output_dir, "split_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = status
        meta["finished_at"] = finished_at
        if scores is not None:
            meta["scores"] = scores
        if abort_reason:
            meta["abort_reason"] = abort_reason
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if getattr(args, "_run_root", None):
        _rebuild_run_manifest(args._run_root)
        _rebuild_summary(args._run_root)
        print(f"[run] manifest: {os.path.join(args._run_root, 'manifest.json')}")
        print(f"[run] summary:  {os.path.join(args._run_root, 'summary.json')}")

    if getattr(args, "_log_file", None):
        sys.stdout = args._stdout_orig
        args._log_file.close()


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
parser.add_argument(
    "--tofu_split",
    default=None,
    help="TOFU HF config name (e.g. forget10, retain_perturbed, real_authors). "
         "Loads from Hugging Face instead of local JSONL.",
)
parser.add_argument("--hf_dataset", default="locuslab/TOFU", help="HF dataset repo")
parser.add_argument("--hf_split", default="train", help="HF dataset split name")
parser.add_argument("--forget_file", default="./data/tofu/forget01.json")
parser.add_argument("--perturbed_file", default="./data/tofu/forget01_perturbed.json")
parser.add_argument("--output_dir", default=None)
parser.add_argument(
    "--experiment",
    default=None,
    help="Experiment name for structured outputs under eval_outputs/<experiment>/<run_id>/splits/<split>/",
)
parser.add_argument(
    "--run_id",
    default=None,
    help="Run identifier (default: UTC timestamp). Reuse the same id across all four TOFU splits.",
)
parser.add_argument(
    "--eval_outputs_root",
    default="./eval_outputs",
    help="Root directory for structured experiment outputs.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite existing split output (details.jsonl) instead of failing.",
)
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
args = resolve_output_layout(args)
prepare_split_output(args)

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
forget_data, perturbed_data = load_tofu_split(
    tofu_split=args.tofu_split,
    forget_file=args.forget_file,
    perturbed_file=args.perturbed_file,
    hf_dataset=args.hf_dataset,
    hf_split=args.hf_split,
    load_perturbed=args.truth_ratio,
)
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
            abort_scores = {
                "rougeL": float(np.mean(rougeL_scores)),
                "probability": float(avg_prob),
            }
            if args.output_dir:
                abort_result = {
                    "model": args.model,
                    "aborted": True,
                    "abort_reason": reason,
                    "scores": abort_scores,
                    "details": details,
                }
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(abort_result, f, indent=2, ensure_ascii=False)
            finalize_split_output(
                args,
                status="aborted",
                scores=abort_scores,
                abort_reason=reason,
            )
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
    finalize_split_output(args, status="completed", scores=scores)
