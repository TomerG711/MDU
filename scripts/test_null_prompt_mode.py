#!/usr/bin/env python3
"""Smoke tests for build_null_anchor_uncond_inputs (mask / empty / pad)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearn_run_utils import build_null_anchor_uncond_inputs  # noqa: E402


def _fixture():
    # [Q Q | A A | pad] — labels -100 on Q+pad, answer on A
    noised = torch.tensor([[10, 11, 20, 21, 0]], dtype=torch.long)
    maskable = torch.tensor([[False, False, True, True, False]])
    return noised, maskable


def test_mask_replaces_q_ids_attn_none():
    noised, maskable = _fixture()
    mask_id = 99
    noised_u, attn_u = build_null_anchor_uncond_inputs(
        noised,
        maskable,
        attention_mask=None,
        mask_token_id=mask_id,
        pad_token_id=0,
        null_prompt_mode="mask",
    )
    assert torch.equal(noised_u, torch.tensor([[99, 99, 20, 21, 99]]))
    assert attn_u is None


def test_empty_keeps_ids_synthesizes_attn_when_none():
    noised, maskable = _fixture()
    noised_u, attn_u = build_null_anchor_uncond_inputs(
        noised,
        maskable,
        attention_mask=None,
        mask_token_id=99,
        pad_token_id=0,
        null_prompt_mode="empty",
    )
    assert torch.equal(noised_u, noised)
    assert attn_u is not None
    assert torch.equal(attn_u, torch.tensor([[0, 0, 1, 1, 0]]))


def test_empty_respects_existing_attn_and_zeros_q():
    noised, maskable = _fixture()
    attn_in = torch.tensor([[1, 1, 1, 1, 0]])
    noised_u, attn_u = build_null_anchor_uncond_inputs(
        noised,
        maskable,
        attention_mask=attn_in,
        mask_token_id=99,
        pad_token_id=0,
        null_prompt_mode="empty",
    )
    assert torch.equal(noised_u, noised)
    assert torch.equal(attn_u, torch.tensor([[0, 0, 1, 1, 0]]))


def test_pad_replaces_q_with_pad_id():
    noised, maskable = _fixture()
    pad_id = 42
    noised_u, attn_u = build_null_anchor_uncond_inputs(
        noised,
        maskable,
        attention_mask=None,
        mask_token_id=99,
        pad_token_id=pad_id,
        null_prompt_mode="pad",
    )
    assert torch.equal(noised_u, torch.tensor([[42, 42, 20, 21, 42]]))
    assert attn_u is None


def test_pad_requires_pad_token_id():
    noised, maskable = _fixture()
    try:
        build_null_anchor_uncond_inputs(
            noised,
            maskable,
            attention_mask=None,
            mask_token_id=99,
            pad_token_id=None,
            null_prompt_mode="pad",
        )
    except ValueError as exc:
        assert "pad_token_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for pad without pad_token_id")


def main() -> int:
    tests = [
        test_mask_replaces_q_ids_attn_none,
        test_empty_keeps_ids_synthesizes_attn_when_none,
        test_empty_respects_existing_attn_and_zeros_q,
        test_pad_replaces_q_with_pad_id,
        test_pad_requires_pad_token_id,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
