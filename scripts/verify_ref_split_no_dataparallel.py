#!/usr/bin/env python3
"""Smoke tests for ref-split GPU layout: no DataParallel on the trainable model."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

_REPO_ROOT = __file__.rsplit("/", 2)[0]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, f"{_REPO_ROOT}/src")

from unlearn_run_utils import (  # noqa: E402
    apply_disable_data_parallel,
    assert_no_data_parallel_when_ref_split,
)


def test_apply_disable_data_parallel_auto() -> None:
    training_args = SimpleNamespace(n_gpu=2, _n_gpu=2)
    with patch("torch.cuda.device_count", return_value=2):
        ok = apply_disable_data_parallel(training_args, "auto", "cuda:1")
    assert ok is True
    assert training_args._n_gpu == 1


def test_apply_disable_data_parallel_skips_when_colocated() -> None:
    training_args = SimpleNamespace(n_gpu=2, _n_gpu=2)
    with patch("torch.cuda.device_count", return_value=2):
        ok = apply_disable_data_parallel(training_args, "auto", None)
    assert ok is False
    assert training_args._n_gpu == 2


def test_assert_passes_without_dataparallel() -> None:
    trainer = SimpleNamespace(model=nn.Linear(2, 2), model_wrapped=nn.Linear(2, 2))
    assert_no_data_parallel_when_ref_split(trainer, "cuda:1")


def test_assert_raises_on_dataparallel() -> None:
    base = nn.Linear(2, 2)
    trainer = SimpleNamespace(model=nn.DataParallel(base), model_wrapped=base)
    try:
        assert_no_data_parallel_when_ref_split(trainer, "cuda:1")
    except RuntimeError as exc:
        assert "DataParallel" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for DataParallel + ref split")


def main() -> int:
    test_apply_disable_data_parallel_auto()
    test_apply_disable_data_parallel_skips_when_colocated()
    test_assert_passes_without_dataparallel()
    test_assert_raises_on_dataparallel()
    print("verify_ref_split_no_dataparallel: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
