"""Load TOFU splits from local JSONL files or Hugging Face."""

from __future__ import annotations

import json
from typing import Optional

DEFAULT_HF_DATASET = "locuslab/TOFU"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_tofu_hf(
    split: str,
    dataset_name: str = DEFAULT_HF_DATASET,
    hf_split: str = "train",
) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split, split=hf_split)
    return [dict(row) for row in ds]


def resolve_tofu_paths(
    tofu_split: Optional[str],
    forget_file: Optional[str],
    perturbed_file: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Map a TOFU config name (e.g. forget10) to local JSONL paths."""
    if tofu_split is None:
        return forget_file, perturbed_file

    base = tofu_split.removesuffix("_perturbed")
    forget_path = forget_file or f"./data/tofu/{base}.json"
    perturbed_path = perturbed_file or f"./data/tofu/{base}_perturbed.json"
    return forget_path, perturbed_path


def load_tofu_split(
    *,
    tofu_split: Optional[str] = None,
    forget_file: Optional[str] = None,
    perturbed_file: Optional[str] = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_split: str = "train",
    load_perturbed: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Load a TOFU forget split and optional perturbed split.

    If ``tofu_split`` is set (e.g. ``forget10``), rows are fetched from
    Hugging Face via ``load_dataset(hf_dataset, tofu_split)``. The HF hub
    stores these configs as parquet/arrow, not raw JSONL, but each row has
    the same ``question`` / ``answer`` fields as the local files.

    Otherwise, local JSONL paths are used (defaults: forget01).
    """
    if tofu_split is not None:
        forget_data = load_tofu_hf(tofu_split, dataset_name=hf_dataset, hf_split=hf_split)
        perturbed_data = []
        if load_perturbed:
            perturbed_name = (
                tofu_split if tofu_split.endswith("_perturbed") else f"{tofu_split}_perturbed"
            )
            perturbed_data = load_tofu_hf(
                perturbed_name, dataset_name=hf_dataset, hf_split=hf_split
            )
        return forget_data, perturbed_data

    forget_path, perturbed_path = resolve_tofu_paths(
        tofu_split, forget_file, perturbed_file
    )
    forget_data = load_jsonl(forget_path)
    perturbed_data = load_jsonl(perturbed_path) if load_perturbed else []
    return forget_data, perturbed_data
