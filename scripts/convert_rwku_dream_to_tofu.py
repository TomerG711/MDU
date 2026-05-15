"""Convert RWKU pair JSON to TOFU-style jsonl for Dream unlearning.

Usage:
    python ./scripts/unlearn/convert_rwku_dream_to_tofu.py

Input  (per subject):
    ./RWKU/LLaMA-Factory/data/RWKU/Target_Dream_v2_full10/<subject>/
        pair_Dream-v0-Instruct-7B-full.json    # list of {id, subject, intro, prompt, response=[neg, pos]}

Output (per subject, jsonl one record per line):
    ./RWKU/data/dream_subset/<subject>/
        forget.jsonl    # {"question": <intro>, "answer": <positive>}            (ascent: GA/NPO/WGA/SimNPO/NA)
        dpo.jsonl       # {"question": <intro>, "answer": <positive>,
                        #  "perturbed_answer": [<negative>]}                     (DPO: chosen=neg via perturbed[0])
"""
import json
import os
from pathlib import Path

SRC_ROOT = Path("./RWKU/LLaMA-Factory/data/RWKU/Target_Dream_v2_full10")
DST_ROOT = Path("./RWKU/data/dream_subset")
PAIR_FILE = "pair_Dream-v0-Instruct-7B-full.json"


def convert_subject(subject_dir: Path):
    src = subject_dir / PAIR_FILE
    if not src.exists():
        raise FileNotFoundError(f"missing pair file: {src}")
    pairs = json.loads(src.read_text(encoding="utf-8"))

    out_dir = DST_ROOT / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    forget_path = out_dir / "forget.jsonl"
    dpo_path = out_dir / "dpo.jsonl"

    n = 0
    with forget_path.open("w", encoding="utf-8") as f_forget, dpo_path.open("w", encoding="utf-8") as f_dpo:
        for item in pairs:
            intro = item["intro"]
            response = item["response"]
            if len(response) < 2:
                continue
            negative, positive = response[0], response[1]
            if not (intro and negative and positive):
                continue
            f_forget.write(json.dumps({"question": intro, "answer": positive}, ensure_ascii=False) + "\n")
            f_dpo.write(json.dumps({"question": intro, "answer": positive, "perturbed_answer": [negative]}, ensure_ascii=False) + "\n")
            n += 1

    return n


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()],
                      key=lambda p: int(p.name.split("_", 1)[0]))
    print(f"Found {len(subjects)} subjects under {SRC_ROOT}")
    total = 0
    for sd in subjects:
        n = convert_subject(sd)
        total += n
        print(f"  [{sd.name}]  {n} entries -> {DST_ROOT / sd.name}/{{forget,dpo}}.jsonl")
    print(f"Total: {total} entries across {len(subjects)} subjects")


if __name__ == "__main__":
    main()
