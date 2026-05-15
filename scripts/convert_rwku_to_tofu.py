"""
Convert RWKU Target_LLaDA pair files into TOFU-style JSONL for use with
unlearn_mdu_llada.py trainer.

Output: single merged JSONL with {"question": prompt, "answer": response[0]}

Run:
    python ./scripts/eval/convert_rwku_to_tofu.py
"""
import json, os, glob

SRC_ROOT = "./RWKU/LLaMA-Factory/data/RWKU/Target_LLaDA"
OUT_DIR = "./TOFU"
os.makedirs(OUT_DIR, exist_ok=True)

# Merged 10-subject forget set
out_merged = os.path.join(OUT_DIR, "rwku_forget_10subj.jsonl")

# Per-subject files too (for single-subject unlearning if wanted)
SUBJECTS = sorted(os.listdir(SRC_ROOT))
print(f"Found {len(SUBJECTS)} subjects")

merged = []
for subj in SUBJECTS:
    src = os.path.join(SRC_ROOT, subj, "pair_LLaDA-8B-Instruct-full.json")
    if not os.path.exists(src):
        print(f"SKIP (no pair file): {subj}")
        continue
    data = json.load(open(src))
    items = []
    for item in data:
        responses = item.get("response", [])
        if not responses:
            continue
        # Use first response as canonical answer
        items.append({
            "question": item["prompt"],
            "answer": responses[0] if isinstance(responses, list) else responses,
            "subject": item.get("subject"),
            "id": item.get("id"),
        })
    # Per-subject file
    per_subj = os.path.join(OUT_DIR, f"rwku_forget_{subj}.jsonl")
    with open(per_subj, "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"  {subj}: {len(items)} pairs → {per_subj}")
    merged.extend(items)

with open(out_merged, "w") as f:
    for x in merged:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"\nMerged: {len(merged)} pairs → {out_merged}")

# Also create a simple retain_placeholder (we use noRetain anyway, but trainer needs path)
placeholder = os.path.join(OUT_DIR, "rwku_retain_placeholder.jsonl")
if not os.path.exists(placeholder):
    # Empty-safe: just use forget as placeholder (won't be trained on since alpha=0)
    with open(placeholder, "w") as f:
        for x in merged[:10]:  # tiny sample
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"Placeholder retain: {placeholder}")
