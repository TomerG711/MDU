#!/usr/bin/env python3
"""Unit tests for pre_sft_cond conditional anchor resolution and input building."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from unlearn_run_utils import (  # noqa: E402
    DEFAULT_PRE_SFT_REF_DREAM,
    DEFAULT_PRE_SFT_REF_LLAMA,
    build_null_anchor_anchor_inputs,
    resolve_null_anchor_uncond,
    resolve_ref_model_path,
)


def _model_args(student: str, ref: str = "") -> SimpleNamespace:
    return SimpleNamespace(model_name_or_path=student, ref_model_name_or_path=ref)


def test_resolve_pre_sft_cond_all_match_modes():
    for mode in ("random", "position", "token_id"):
        got = resolve_null_anchor_uncond(
            loss_type="null_anchor",
            null_anchor_source="pre_sft_cond",
            match_mode=mode,
            null_anchor_traj_rollout=False,
        )
        assert got == "pre_sft_cond", f"match_mode={mode} got {got}"


def test_resolve_pre_sft_cond_aliases():
    for alias in ("pre_sft", "base", "base_instruct", "presftcond"):
        got = resolve_null_anchor_uncond(
            loss_type="null_anchor",
            null_anchor_source=alias,
            match_mode="position",
            null_anchor_traj_rollout=False,
        )
        assert got == "pre_sft_cond", f"alias={alias} got {got}"


def test_auto_never_resolves_pre_sft_cond():
    got = resolve_null_anchor_uncond(
        loss_type="null_anchor",
        null_anchor_source="auto",
        match_mode="position",
        null_anchor_traj_rollout=False,
    )
    assert got == "trainable_cfg"


def test_resolve_ref_model_path_defaults():
    llada = _model_args("./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU")
    assert (
        resolve_ref_model_path(llada, "pre_sft_cond")
        == DEFAULT_PRE_SFT_REF_LLAMA
    )
    dream = _model_args("./checkpoints/dream-tofu-sft/checkpoint-final")
    assert (
        resolve_ref_model_path(dream, "pre_sft_cond")
        == DEFAULT_PRE_SFT_REF_DREAM
    )
    assert (
        resolve_ref_model_path(llada, "frozen_sft")
        == llada.model_name_or_path
    )


def test_resolve_ref_model_path_explicit_override():
    llada = _model_args(
        "./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU",
        ref="/custom/base-instruct",
    )
    assert resolve_ref_model_path(llada, "pre_sft_cond") == "/custom/base-instruct"
    assert resolve_ref_model_path(llada, "frozen_sft") == "/custom/base-instruct"


def test_build_anchor_inputs_pre_sft_cond():
    noised = torch.tensor([[1, 2, 3, 4]])
    maskable = torch.tensor([[False, False, True, True]])
    attn = torch.tensor([[1, 1, 1, 1]])
    for npm in ("mask", "empty", "pad"):
        noised_u, attn_u = build_null_anchor_anchor_inputs(
            noised,
            maskable,
            attn,
            mask_token_id=99,
            pad_token_id=0,
            null_prompt_mode=npm,
            anchor_resolved="pre_sft_cond",
        )
        assert torch.equal(noised_u, noised)
        assert torch.equal(attn_u, attn)


def test_build_anchor_inputs_uncond_delegates():
    noised = torch.tensor([[10, 20, 30, 40]])
    maskable = torch.tensor([[False, False, True, True]])
    noised_u, attn_u = build_null_anchor_anchor_inputs(
        noised,
        maskable,
        None,
        mask_token_id=99,
        pad_token_id=0,
        null_prompt_mode="mask",
        anchor_resolved="frozen_sft",
    )
    assert noised_u[0, 0].item() == 99
    assert noised_u[0, 1].item() == 99
    assert noised_u[0, 2].item() == 30
    assert attn_u is None


def main() -> int:
    tests = [
        test_resolve_pre_sft_cond_all_match_modes,
        test_resolve_pre_sft_cond_aliases,
        test_auto_never_resolves_pre_sft_cond,
        test_resolve_ref_model_path_defaults,
        test_resolve_ref_model_path_explicit_override,
        test_build_anchor_inputs_pre_sft_cond,
        test_build_anchor_inputs_uncond_delegates,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
