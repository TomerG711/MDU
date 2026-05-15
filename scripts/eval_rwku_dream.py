"""
Evaluate Dream model on RWKU:
- forget_level1 / level2 / level3 (RougeL recall, generation)
- neighbor_level1 / level2 (RougeL recall, generation)
- utility_general (MMLU, multiple-choice)
- utility_truthfulness (TruthfulQA MC1, log-prob)
- utility_factuality (TriviaQA, generation EM)
"""
import torch
import argparse
import numpy as np
import dllm  # noqa: F401  — registers Dream model + tokenizer classes for AutoTokenizer/AutoModel
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer

SAMPLES_PER_SUBJECT = None
BATCH_SIZE = 32


def generate_batch(model, tokenizer, questions):
    all_messages = [[{"role": "user", "content": q}] for q in questions]
    encoded = [
        tokenizer.apply_chat_template(m, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        for m in all_messages
    ]
    # pad to same length
    max_len = max(e.input_ids.shape[1] for e in encoded)
    input_ids = torch.zeros(len(questions), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(questions), max_len, dtype=torch.long)
    for i, e in enumerate(encoded):
        l = e.input_ids.shape[1]
        input_ids[i, max_len - l:] = e.input_ids[0]
        attention_mask[i, max_len - l:] = e.attention_mask[0]

    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        steps=64,
        temperature=0.0,
        alg="entropy",
        return_dict_in_generate=True,
    )
    results = []
    for i in range(len(questions)):
        pred = tokenizer.decode(output.sequences[i][max_len:], skip_special_tokens=True).strip()
        results.append(pred)
    return results


def eval_dataset(model, tokenizer, dataset, subjects, label, save_fn=None):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_scores = []
    all_details = []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for subject in subjects:
        subset = [x for x in dataset if x["subject"] == subject]
        if SAMPLES_PER_SUBJECT is not None:
            subset = subset[:SAMPLES_PER_SUBJECT]
        if not subset:
            continue

        subject_scores = []
        for i in range(0, len(subset), BATCH_SIZE):
            batch = subset[i:i + BATCH_SIZE]
            preds = generate_batch(model, tokenizer, [x["query"] for x in batch])
            for item, pred in zip(batch, preds):
                pred_clean = pred.split("<|im_end|>")[0].strip()
                score = scorer.score(item["answer"], pred_clean)["rougeL"].recall
                subject_scores.append(score)
                all_scores.append(score)
                all_details.append({
                    "subject": subject,
                    "query": item["query"],
                    "answer": item["answer"],
                    "pred": pred_clean,
                    "rougeL": score,
                })

        print(f"  [{subject}] RougeL: {sum(subject_scores)/len(subject_scores):.3f}  (n={len(subject_scores)})")
        if save_fn:
            save_fn(all_details)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"  >> {label} Overall RougeL: {avg:.3f}  (total={len(all_scores)})")
    return avg, all_details


MMLU_CHOICES = ["A", "B", "C", "D"]


def eval_mmlu(model, tokenizer, dataset, batch_size=1, max_samples=100):
    print(f"\n{'='*60}")
    print(f"  Utility (MMLU)")
    print(f"{'='*60}")

    dataset = dataset.select(range(min(max_samples, len(dataset))))
    prompts = []
    answers = []
    for sample in dataset:
        dev_set = sample["examples"]
        subject = sample["task"]
        # few-shot prompt
        few_shot = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        for ex in dev_set:
            few_shot += "Question: " + ex["question"]
            for j, c in enumerate(ex["choices"]):
                few_shot += f"\n{MMLU_CHOICES[j]}. {c}"
            few_shot += f"\nAnswer: {MMLU_CHOICES[ex['answer']]}\n\n"
        q = "Question: " + sample["question"]
        for j, c in enumerate(sample["choices"]):
            q += f"\n{MMLU_CHOICES[j]}. {c}"
        q += "\nAnswer:"
        messages = [{"role": "user", "content": few_shot + q}]
        prompts.append(messages)
        answers.append(sample["answer"])

    correct = 0
    total = len(prompts)
    for i in range(0, total, batch_size):
        batch_msgs = prompts[i:i + batch_size]
        batch_ans = answers[i:i + batch_size]
        encoded = [
            tokenizer.apply_chat_template(m, return_tensors="pt", return_dict=True, add_generation_prompt=True)
            for m in batch_msgs
        ]
        max_len = max(e.input_ids.shape[1] for e in encoded)
        b = len(batch_msgs)
        input_ids = torch.zeros(b, max_len, dtype=torch.long)
        attention_mask = torch.zeros(b, max_len, dtype=torch.long)
        for k, e in enumerate(encoded):
            l = e.input_ids.shape[1]
            input_ids[k, max_len - l:] = e.input_ids[0]
            attention_mask[k, max_len - l:] = e.attention_mask[0]
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        output = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=5, steps=64, temperature=0.0,
            alg="entropy", return_dict_in_generate=True,
        )
        for k in range(b):
            pred = tokenizer.decode(output.sequences[k][max_len:], skip_special_tokens=True).strip()
            pred_choice = pred[0].upper() if pred else ""
            if pred_choice == MMLU_CHOICES[batch_ans[k]]:
                correct += 1

    acc = correct / total if total > 0 else 0
    print(f"  >> MMLU Accuracy: {acc:.3f}  (total={total})")
    return acc


def eval_truthfulqa(model, tokenizer, dataset, subjects):
    """TruthfulQA MC1: pick choice with lowest per-token NLL."""
    import torch.nn.functional as F
    print(f"\n{'='*60}")
    print(f"  Utility (TruthfulQA MC1)")
    print(f"{'='*60}")
    items = [r for r in dataset if r.get("subject") in subjects]
    correct = 0
    total = 0
    for subj in subjects:
        sub = [r for r in items if r["subject"] == subj]
        if not sub:
            continue
        sc = 0
        for r in sub:
            q = r["question"]
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gt_idx = labels.index(1) if 1 in labels else 0
            best_score = float("inf")
            best_idx = 0
            for i, c in enumerate(choices):
                msgs = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": c},
                ]
                ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False)[0].to("cuda")
                with torch.no_grad():
                    out = model(input_ids=ids.unsqueeze(0))
                    logits = out.logits[0]
                    target = ids[1:]
                    nll = F.cross_entropy(logits[:-1], target, reduction="sum").item() / max(target.size(0), 1)
                if nll < best_score:
                    best_score = nll
                    best_idx = i
            if best_idx == gt_idx:
                sc += 1
        n = len(sub)
        print(f"  [{subj}] MC1 acc: {sc/n:.3f}  (n={n})")
        correct += sc
        total += n
    acc = correct / total if total else 0
    print(f"  >> TruthfulQA MC1: {acc:.3f}  (total={total})")
    return acc


def eval_triviaqa(model, tokenizer, dataset, subjects, batch_size=BATCH_SIZE):
    """TriviaQA: generate answer, exact-match against any reference."""
    print(f"\n{'='*60}")
    print(f"  Utility (TriviaQA)")
    print(f"{'='*60}")
    items = [r for r in dataset if r.get("subject") in subjects]
    correct = 0
    total = 0
    for subj in subjects:
        sub = [r for r in items if r["subject"] == subj]
        if not sub:
            continue
        sc = 0
        for i in range(0, len(sub), batch_size):
            batch = sub[i:i + batch_size]
            preds = generate_batch(model, tokenizer, [r["question"] for r in batch])
            for r, p in zip(batch, preds):
                p_clean = p.split("<|im_end|>")[0].strip().lower()
                if any(a.lower() in p_clean for a in r["answers"]):
                    sc += 1
        n = len(sub)
        print(f"  [{subj}] TriviaQA EM: {sc/n:.3f}  (n={n})")
        correct += sc
        total += n
    acc = correct / total if total else 0
    print(f"  >> TriviaQA EM: {acc:.3f}  (total={total})")
    return acc


def main():
    import json, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--num_subjects", type=int, default=10)
    parser.add_argument("--target_subject", type=str, default=None,
                        help="if set, evaluate only this single subject (per-subject unlearning eval)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    print("Loading datasets...")
    fl1 = load_dataset("jinzhuoran/RWKU", "forget_level1")["test"]
    fl2 = load_dataset("jinzhuoran/RWKU", "forget_level2")["test"]
    fl3 = load_dataset("jinzhuoran/RWKU", "forget_level3")["test"]
    nl1 = load_dataset("jinzhuoran/RWKU", "neighbor_level1")["test"]
    nl2 = load_dataset("jinzhuoran/RWKU", "neighbor_level2")["test"]
    mmlu = load_dataset("jinzhuoran/RWKU", "utility_general")["test"]
    truth = load_dataset("jinzhuoran/RWKU", "utility_truthfulness")["test"]
    trivia = load_dataset("jinzhuoran/RWKU", "utility_factuality")["test"]

    if args.target_subject:
        subjects = [args.target_subject]
        print(f"\nTarget subject: {args.target_subject}")
    else:
        subjects = list(dict.fromkeys(fl2["subject"]))[:args.num_subjects]
        print(f"\nSubjects ({args.num_subjects}): {subjects}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    details = {}

    def save_results(results, details):
        if args.output_dir:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump({"subjects": subjects, "scores": results, "details": details}, f, indent=2, ensure_ascii=False)

    def make_save_fn(key):
        def fn(d):
            details[key] = d
            save_results(results, details)
        return fn

    results = {}
    results["forget_level1"], details["forget_level1"] = eval_dataset(model, tokenizer, fl1, subjects, "Forget Level1 (fill-in-the-blank)", save_fn=make_save_fn("forget_level1"))
    save_results(results, details)
    results["forget_level2"], details["forget_level2"] = eval_dataset(model, tokenizer, fl2, subjects, "Forget Level2 (Q&A)", save_fn=make_save_fn("forget_level2"))
    save_results(results, details)
    results["forget_level3"], details["forget_level3"] = eval_dataset(model, tokenizer, fl3, subjects, "Forget Level3 (adversarial)", save_fn=make_save_fn("forget_level3"))
    save_results(results, details)
    results["neighbor_level1"], details["neighbor_level1"] = eval_dataset(model, tokenizer, nl1, subjects, "Neighbor Level1 (fill-in-the-blank)", save_fn=make_save_fn("neighbor_level1"))
    save_results(results, details)
    results["neighbor_level2"], details["neighbor_level2"] = eval_dataset(model, tokenizer, nl2, subjects, "Neighbor Level2 (Q&A)", save_fn=make_save_fn("neighbor_level2"))
    save_results(results, details)
    results["mmlu"] = eval_mmlu(model, tokenizer, mmlu, batch_size=BATCH_SIZE, max_samples=100)
    save_results(results, details)
    results["truthfulqa_mc1"] = eval_truthfulqa(model, tokenizer, truth, subjects)
    save_results(results, details)
    results["triviaqa_em"] = eval_triviaqa(model, tokenizer, trivia, subjects, batch_size=BATCH_SIZE)
    save_results(results, details)
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:25s}: {v:.3f}")

    if args.output_dir:
        print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
