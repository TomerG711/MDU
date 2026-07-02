"""Shared helpers for MDU unlearning runs: data loading, checkpoints, W&B."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import logging

import accelerate
import torch
import transformers

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from tofu_data import load_jsonl, load_tofu_hf  # noqa: E402

DEFAULT_WANDB_PROJECT = "unlearning-dllms-MDU"
_MDU_RUN_NAME_PLACEHOLDER = "./checkpoints/_mdu_run"

# Pre-TOFU instruct checkpoints (conditional anchor defaults)
DEFAULT_PRE_SFT_REF_LLAMA = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_PRE_SFT_REF_DREAM = "Dream-org/Dream-v0-Instruct-7B"


def resolve_ref_device(ref_device: str) -> Optional[str]:
    """
    Resolve where to place the frozen ref_model.

    - ``auto`` (default): ``cuda:1`` when 2+ CUDA devices are visible, else colocated
    - ``same`` / ``off`` / empty: keep ref_model on the trainable model device
    - ``cuda:N`` etc.: explicit device
    """
    import torch

    key = (ref_device or "auto").strip().lower()
    if key in ("same", "off", "none", "disabled", ""):
        return None
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return "cuda:1"
        return None
    return ref_device.strip()


def place_ref_model(ref_model, ref_device: str) -> Optional[str]:
    """Move ``ref_model`` if needed; return the device string or None if colocated."""
    target = resolve_ref_device(ref_device)
    if target is None:
        return None
    ref_model.to(target)
    return target


def null_anchor_source_key(null_anchor_source: str) -> str:
    return (null_anchor_source or "auto").strip().lower()


NULL_PROMPT_MODES = ("mask", "empty", "pad")


def normalize_null_prompt_mode(null_prompt_mode: str) -> str:
    key = (null_prompt_mode or "mask").strip().lower()
    if key not in NULL_PROMPT_MODES:
        raise ValueError(
            f"null_prompt_mode must be one of {NULL_PROMPT_MODES}, got {null_prompt_mode!r}"
        )
    return key


def null_prompt_mode_slug(null_prompt_mode: str) -> str:
    """Checkpoint/eval suffix when mode is not the default ``mask``."""
    mode = normalize_null_prompt_mode(null_prompt_mode)
    if mode == "mask":
        return ""
    return f"nullprompt_{mode}"


def build_null_anchor_uncond_inputs(
    noised: torch.Tensor,
    maskable_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    mask_token_id: int,
    pad_token_id: Optional[int],
    null_prompt_mode: str = "mask",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build uncond forward inputs for null-anchor KL.

    ``maskable_mask`` is True on answer positions (labels != -100); Q is ~maskable_mask.
    """
    mode = normalize_null_prompt_mode(null_prompt_mode)
    q_mask = ~maskable_mask
    if attention_mask is not None:
        q_mask = q_mask & attention_mask.bool()

    if mode == "mask":
        noised_u = torch.where(q_mask, mask_token_id, noised)
        attn_u = attention_mask
    elif mode == "empty":
        noised_u = noised
        base = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(noised, dtype=torch.long, device=noised.device)
        )
        attn_u = base.clone()
        attn_u[q_mask] = 0
    elif mode == "pad":
        if pad_token_id is None:
            raise ValueError("null_prompt_mode=pad requires tokenizer pad_token_id")
        noised_u = torch.where(q_mask, pad_token_id, noised)
        attn_u = attention_mask
    else:
        raise ValueError(f"unsupported null_prompt_mode: {mode}")

    return noised_u, attn_u


def build_null_anchor_anchor_inputs(
    noised: torch.Tensor,
    maskable_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    mask_token_id: int,
    pad_token_id: Optional[int],
    null_prompt_mode: str = "mask",
    anchor_resolved: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build anchor-side forward inputs for null-anchor KL.

  ``pre_sft_cond`` uses the same conditional inputs as the student; other anchors
    delegate to ``build_null_anchor_uncond_inputs``.
    """
    if anchor_resolved == "pre_sft_cond":
        return noised, attention_mask
    return build_null_anchor_uncond_inputs(
        noised,
        maskable_mask,
        attention_mask,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        null_prompt_mode=null_prompt_mode,
    )


def _default_pre_sft_ref_for_student(model_name_or_path: str) -> str:
    path = (model_name_or_path or "").lower()
    if "dream" in path:
        return DEFAULT_PRE_SFT_REF_DREAM
    return DEFAULT_PRE_SFT_REF_LLAMA


def resolve_ref_model_path(model_args, null_anchor_source: str, *, loss_type: str = "") -> str:
    """
    Resolve frozen ref checkpoint path.

    - ``pre_sft_cond``: explicit ``ref_model_name_or_path`` or backbone default instruct HF id
    - ``frozen_sft`` / ``npo``: explicit override or student ``model_name_or_path``
  """
    explicit = (getattr(model_args, "ref_model_name_or_path", None) or "").strip()
    if explicit:
        return explicit
    key = null_anchor_source_key(null_anchor_source)
    if key in ("pre_sft_cond", "pre_sft", "base", "base_instruct", "presftcond"):
        return _default_pre_sft_ref_for_student(
            getattr(model_args, "model_name_or_path", "") or ""
        )
    if loss_type == "npo" or key in ("frozen_sft", "frozen", "ref"):
        return getattr(model_args, "model_name_or_path", "")
    raise ValueError(
        f"resolve_ref_model_path: no ref path for null_anchor_source={null_anchor_source!r}"
    )


def resolve_null_anchor_uncond(
    *,
    loss_type: str,
    null_anchor_source: str,
    match_mode: str,
    null_anchor_traj_rollout: bool,
) -> Optional[str]:
    """
    Resolved uncond anchor for null-anchor KL.

    Returns ``frozen_sft``, ``pre_sft_cond``, ``trainable_cfg``, ``ema``, or ``None``.

    ``auto`` (upstream-compatible):
      - ``match_mode=random`` or traj rollout → frozen SFT ref
      - denoise trajectory modes (position/token_id/…) → trainable CFG (Q masked)

    ``pre_sft_cond`` is explicit opt-in only (never via ``auto``): frozen pre-SFT
    instruct ref with **conditional** inputs (same Q+A as student).
    """
    if loss_type != "null_anchor":
        return None
    key = null_anchor_source_key(null_anchor_source)
    if key in ("pre_sft_cond", "pre_sft", "base", "base_instruct", "presftcond"):
        return "pre_sft_cond"
    if key in ("ema", "ema_sft", "ema_cfg"):
        return "ema"
    if key in ("frozen_sft", "frozen", "ref"):
        return "frozen_sft"
    if key in ("trainable_cfg", "trainable", "cfg"):
        return "trainable_cfg"
    if null_anchor_traj_rollout:
        return "frozen_sft"
    if match_mode == "random":
        return "frozen_sft"
    return "trainable_cfg"


def null_anchor_uses_frozen_ref(
    *,
    loss_type: str,
    null_anchor_source: str,
    match_mode: str,
    null_anchor_traj_rollout: bool,
) -> bool:
    """Whether null-anchor uncond logits come from frozen SFT ref_model."""
    return resolve_null_anchor_uncond(
        loss_type=loss_type,
        null_anchor_source=null_anchor_source,
        match_mode=match_mode,
        null_anchor_traj_rollout=null_anchor_traj_rollout,
    ) == "frozen_sft"


def null_anchor_uses_pre_sft_cond(
    *,
    loss_type: str,
    null_anchor_source: str,
    match_mode: str,
    null_anchor_traj_rollout: bool,
) -> bool:
    """Whether null-anchor logits come from frozen pre-SFT ref with conditional inputs."""
    return resolve_null_anchor_uncond(
        loss_type=loss_type,
        null_anchor_source=null_anchor_source,
        match_mode=match_mode,
        null_anchor_traj_rollout=null_anchor_traj_rollout,
    ) == "pre_sft_cond"


def null_anchor_uses_ref_model(
    *,
    loss_type: str,
    null_anchor_source: str,
    match_mode: str,
    null_anchor_traj_rollout: bool,
) -> bool:
    """Whether null-anchor uses an external frozen ref_model (SFT-null or pre-SFT cond)."""
    resolved = resolve_null_anchor_uncond(
        loss_type=loss_type,
        null_anchor_source=null_anchor_source,
        match_mode=match_mode,
        null_anchor_traj_rollout=null_anchor_traj_rollout,
    )
    return resolved in ("frozen_sft", "pre_sft_cond")


def null_anchor_uses_ema(
    *,
    loss_type: str,
    null_anchor_source: str,
    match_mode: str,
    null_anchor_traj_rollout: bool,
) -> bool:
    """Whether null-anchor uncond logits come from an EMA copy of the trainable model."""
    return resolve_null_anchor_uncond(
        loss_type=loss_type,
        null_anchor_source=null_anchor_source,
        match_mode=match_mode,
        null_anchor_traj_rollout=null_anchor_traj_rollout,
    ) == "ema"


def needs_ref_model(data_args) -> bool:
    """True when a separate frozen ref_model checkpoint must be loaded."""
    if data_args.loss_type == "npo":
        return True
    if data_args.loss_type == "null_anchor":
        return null_anchor_uses_ref_model(
            loss_type=data_args.loss_type,
            null_anchor_source=getattr(data_args, "null_anchor_source", "auto"),
            match_mode=data_args.match_mode,
            null_anchor_traj_rollout=getattr(data_args, "null_anchor_traj_rollout", False),
        )
    return False


def needs_ema_model(data_args) -> bool:
    """True when a separate EMA anchor model must be loaded (updated each optimizer step)."""
    if data_args.loss_type != "null_anchor":
        return False
    return null_anchor_uses_ema(
        loss_type=data_args.loss_type,
        null_anchor_source=getattr(data_args, "null_anchor_source", "auto"),
        match_mode=data_args.match_mode,
        null_anchor_traj_rollout=getattr(data_args, "null_anchor_traj_rollout", False),
    )


def describe_forget_loss_path(data_args) -> str:
    """Human-readable forget-loss function selected by ``compute_loss``."""
    if data_args.match_mode == "random":
        if data_args.loss_type == "ga":
            return "_random_ga_loss"
        if data_args.loss_type == "npo":
            return "_random_npo_loss"
        if data_args.loss_type == "null_anchor":
            if getattr(data_args, "null_anchor_traj_rollout", False):
                return "_traj_rollout_na_loss"
            return "_random_sft_null_anchor_loss"
    return "_denoise_novel_ga_loss"


def describe_null_anchor_uncond(data_args) -> Optional[str]:
    """Resolved uncond anchor: ``frozen_sft``, ``trainable_cfg``, ``ema``, or ``None``."""
    return resolve_null_anchor_uncond(
        loss_type=data_args.loss_type,
        null_anchor_source=getattr(data_args, "null_anchor_source", "auto"),
        match_mode=data_args.match_mode,
        null_anchor_traj_rollout=getattr(data_args, "null_anchor_traj_rollout", False),
    )


def build_mdu_setup_summary(
    data_args,
    *,
    ref_loaded: bool,
    ref_device_placed: Optional[str],
    ema_loaded: bool = False,
    ref_model_name_or_path: Optional[str] = None,
) -> dict:
    """Single source of truth for logs, train_config, and replication audits."""
    uncond = describe_null_anchor_uncond(data_args)
    anchor_input_mode = (
        "conditional" if uncond == "pre_sft_cond" else "uncond" if uncond else None
    )
    return {
        "loss_type": data_args.loss_type,
        "match_mode": data_args.match_mode,
        "forget_loss_path": describe_forget_loss_path(data_args),
        "null_anchor_source": getattr(data_args, "null_anchor_source", "auto"),
        "null_anchor_uncond_resolved": uncond,
        "anchor_input_mode": anchor_input_mode,
        "null_anchor_tau": getattr(data_args, "null_anchor_tau", None),
        "null_anchor_ema_decay": getattr(data_args, "null_anchor_ema_decay", None),
        "null_anchor_traj_rollout": getattr(data_args, "null_anchor_traj_rollout", False),
        "null_prompt_mode": getattr(data_args, "null_prompt_mode", "mask"),
        "ref_model_loaded": ref_loaded,
        "ref_model_name_or_path": ref_model_name_or_path,
        "ema_model_loaded": ema_loaded,
        "ref_device_placed": ref_device_placed,
        "ref_device_arg": getattr(data_args, "ref_device", None),
    }


def format_mdu_setup_log(setup: dict) -> str:
    """One-line training setup summary (logged once at run start)."""
    parts = [
        f"forget={setup['forget_loss_path']}",
        f"match_mode={setup['match_mode']}",
        f"loss_type={setup['loss_type']}",
    ]
    if setup["null_anchor_uncond_resolved"] is not None:
        parts.append(f"anchor={setup['null_anchor_uncond_resolved']}")
        parts.append(f"null_anchor_source={setup['null_anchor_source']}")
        if setup.get("anchor_input_mode"):
            parts.append(f"anchor_input={setup['anchor_input_mode']}")
        if setup.get("null_anchor_tau") is not None:
            parts.append(f"τ={setup['null_anchor_tau']}")
        npm = setup.get("null_prompt_mode") or "mask"
        if npm != "mask" and setup["null_anchor_uncond_resolved"] != "pre_sft_cond":
            parts.append(f"null_prompt_mode={npm}")
    ref = "loaded" if setup["ref_model_loaded"] else "not_loaded"
    if setup["ref_device_placed"]:
        ref += f"@{setup['ref_device_placed']}"
    parts.append(f"ref_model={ref}")
    if setup.get("ref_model_name_or_path"):
        parts.append(f"ref_path={setup['ref_model_name_or_path']}")
    if setup.get("ema_model_loaded"):
        ema = "loaded"
        if setup["ref_device_placed"]:
            ema += f"@{setup['ref_device_placed']}"
        if setup.get("null_anchor_ema_decay") is not None:
            ema += f"(decay={setup['null_anchor_ema_decay']})"
        parts.append(f"ema_model={ema}")
    return "[mdu-setup] " + " ".join(parts)


def unwrap_trainer_model(model):
    """Unwrap HF Trainer / Accelerate wrappers before reading parameter tensors."""
    if model is None:
        return None
    try:
        from accelerate.utils import extract_model_from_parallel

        return extract_model_from_parallel(model)
    except Exception:
        return model


@torch.no_grad()
def update_ema_model(ema_model, model, decay: float) -> None:
    """EMA teacher update: θ_ema ← decay·θ_ema + (1−decay)·θ_student."""
    if ema_model is None or model is None:
        return
    student = unwrap_trainer_model(model)
    decay = float(decay)
    one_minus = 1.0 - decay
    for ema_p, student_p in zip(ema_model.parameters(), student.parameters()):
        ema_p.mul_(decay).add_(student_p.data.to(ema_p.device), alpha=one_minus)


class NullAnchorEMACallback(transformers.TrainerCallback):
    """Update EMA anchor weights after each optimizer step."""

    def __init__(self, ema_model, decay: float):
        self.ema_model = ema_model
        self.decay = float(decay)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        update_ema_model(self.ema_model, model, self.decay)
        return control


def assert_no_data_parallel_when_ref_split(
    trainer,
    ref_device_placed: Optional[str],
    *,
    log=None,
) -> None:
    """
    Fail fast if ref/EMA is on a separate GPU but HF Trainer wrapped the student in DataParallel.

    Call from ``on_train_begin`` (after ``_wrap_model``) or a smoke test.
    """
    if not ref_device_placed:
        return
    import torch.nn as nn

    log = log or logging.getLogger("unlearn_run_utils")
    for label, obj in (
        ("trainer.model", getattr(trainer, "model", None)),
        ("trainer.model_wrapped", getattr(trainer, "model_wrapped", None)),
    ):
        if obj is not None and isinstance(obj, nn.DataParallel):
            raise RuntimeError(
                f"[trainer] {label} is nn.DataParallel while ref/EMA is on {ref_device_placed}. "
                "Trainable model must stay on a single GPU. "
                "Set --disable_data_parallel yes (or auto) and ensure training_args.n_gpu == 1."
            )
    log.info(
        "[trainer] DataParallel guard OK: trainable model not wrapped "
        f"(ref/EMA on {ref_device_placed})"
    )


class RefSplitDataParallelGuardCallback(transformers.TrainerCallback):
    """Verify trainable model is not DataParallel when ref/EMA uses a split GPU."""

    def __init__(self, trainer, ref_device_placed: Optional[str]):
        self._trainer = trainer
        self._ref_device_placed = ref_device_placed

    def on_train_begin(self, args, state, control, **kwargs):
        assert_no_data_parallel_when_ref_split(
            self._trainer, self._ref_device_placed
        )
        return control


def copy_script_snapshot(src_path: str, dest_dir: str, dest_name: Optional[str] = None) -> dict:
    """Copy a script into a run/checkpoint directory; return path + sha256 fingerprint."""
    os.makedirs(dest_dir, exist_ok=True)
    name = dest_name or os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, name)
    shutil.copy2(src_path, dest_path)
    with open(dest_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"path": dest_path, "sha256": digest, "source": os.path.abspath(src_path)}


def apply_disable_data_parallel(
    training_args,
    disable: str,
    ref_device_placed: Optional[str],
) -> bool:
    """
    Force ``training_args._n_gpu = 1`` so HF Trainer skips ``nn.DataParallel``.

    ``disable``: ``auto`` (default) → disable when ref is on a separate GPU;
    ``yes``/``no`` to force.
    """
    import logging

    key = (disable or "auto").strip().lower()
    if key in ("no", "false", "0", "off"):
        return False
    if key in ("yes", "true", "1", "on"):
        do_disable = True
    else:
        do_disable = ref_device_placed is not None

    if not do_disable:
        return False

    import torch

    if torch.cuda.device_count() > 1 and training_args.n_gpu > 1:
        training_args._n_gpu = 1
        log = logging.getLogger("unlearn_run_utils")
        log.info(
            "[trainer] DataParallel disabled (_n_gpu=1); trainable model stays on cuda:0"
        )
        return True
    return False


def rows_to_messages(rows: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }
        for row in rows
    ]


def load_forget_retain_rows(data_args) -> tuple[list[dict], list[dict], dict]:
    """
    Load TOFU forget/retain rows from Hugging Face (like eval) or local JSONL.

    HF is used when ``tofu_split`` / ``retain_tofu_split`` are non-empty strings.
    """
    source: dict[str, Any] = {
        "hf_dataset": getattr(data_args, "hf_dataset", "locuslab/TOFU"),
        "hf_split": getattr(data_args, "hf_split", "train"),
    }
    tofu_split = getattr(data_args, "tofu_split", None)
    retain_tofu_split = getattr(data_args, "retain_tofu_split", None)

    if tofu_split:
        forget_rows = load_tofu_hf(
            tofu_split,
            dataset_name=source["hf_dataset"],
            hf_split=source["hf_split"],
        )
        source["forget"] = {"mode": "hf", "split": tofu_split}
    else:
        forget_rows = load_jsonl(data_args.tofu_path)
        source["forget"] = {"mode": "local", "path": data_args.tofu_path}

    retain_rows: list[dict] = []
    if getattr(data_args, "alpha", 1.0) != 0:
        if retain_tofu_split:
            retain_rows = load_tofu_hf(
                retain_tofu_split,
                dataset_name=source["hf_dataset"],
                hf_split=source["hf_split"],
            )
            source["retain"] = {"mode": "hf", "split": retain_tofu_split}
        else:
            retain_rows = load_jsonl(data_args.retain_path)
            source["retain"] = {"mode": "local", "path": data_args.retain_path}
    else:
        source["retain"] = {"mode": "none", "reason": "alpha=0"}

    source["forget_n"] = len(forget_rows)
    source["retain_n"] = len(retain_rows)
    return forget_rows, retain_rows, source


def _backbone_tag(model_name_or_path: str) -> str:
    path = model_name_or_path.lower()
    if "llada" in path:
        return "llada"
    if "dream" in path:
        return "dream"
    return os.path.basename(model_name_or_path.rstrip("/"))[:24] or "model"


def _anchor_tag(null_anchor_source: str) -> str:
    key = (null_anchor_source or "auto").strip().lower()
    if key in ("frozen_sft", "frozen", "ref"):
        return "frozen"
    if key in ("trainable_cfg", "trainable", "cfg"):
        return "cfg"
    if key in ("ema", "ema_sft", "ema_cfg"):
        return "ema"
    if key in ("pre_sft_cond", "pre_sft", "base", "base_instruct", "presftcond"):
        return "presftcond"
    return key.replace("_", "")


def default_checkpoint_name(
    *,
    model_name_or_path: str,
    tofu_split: Optional[str],
    loss_type: str,
    null_anchor_tau: float,
    match_mode: str,
    null_anchor_source: str = "auto",
    null_prompt_mode: str = "mask",
) -> str:
    backbone = _backbone_tag(model_name_or_path)
    split = (tofu_split or "local").replace("/", "_")
    if loss_type == "null_anchor":
        tau_s = f"{null_anchor_tau:g}".replace(".", "p")
        mode = match_mode.replace("/", "_")
        anchor = _anchor_tag(null_anchor_source)
        npm_slug = (
            ""
            if anchor == "presftcond"
            else null_prompt_mode_slug(null_prompt_mode)
        )
        npm_part = f"_{npm_slug}" if npm_slug else ""
        # Legacy: random + frozen + mask (pre-grid τ sweep names)
        if mode == "random" and anchor == "frozen" and not npm_part:
            return f"mdu_{backbone}_{split}_nullanchor_tau{tau_s}"
        return f"mdu_{backbone}_{split}_{mode}_{anchor}{npm_part}_tau{tau_s}"
    mode = match_mode.replace("/", "_")
    return f"mdu_{backbone}_{split}_{loss_type}_{mode}"


def resolve_run_directory(
    *,
    model_args,
    data_args,
    training_args,
) -> tuple[str, str]:
    """Set ``training_args.output_dir`` under ``checkpoints_root`` / ``checkpoint_name``."""
    name = (getattr(training_args, "checkpoint_name", None) or "").strip()
    if not name:
        name = default_checkpoint_name(
            model_name_or_path=model_args.model_name_or_path,
            tofu_split=getattr(data_args, "tofu_split", None),
            loss_type=data_args.loss_type,
            null_anchor_tau=data_args.null_anchor_tau,
            match_mode=data_args.match_mode,
            null_anchor_source=getattr(data_args, "null_anchor_source", "auto"),
            null_prompt_mode=getattr(data_args, "null_prompt_mode", "mask"),
        )
    root = getattr(training_args, "checkpoints_root", "./checkpoints")
    out_dir = os.path.join(root, name)
    training_args.output_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, name


def _json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _git_info() -> dict:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def build_train_config(
    *,
    model_args,
    data_args,
    training_args,
    data_source: dict,
    checkpoint_name: str,
    trainer_state: Optional[dict] = None,
) -> dict:
    cfg = {
        "checkpoint_name": checkpoint_name,
        "output_dir": training_args.output_dir,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "model_args": _json_safe(model_args),
        "data_args": _json_safe(data_args),
        "training_args": _json_safe(training_args),
        "data_source": data_source,
        "git": _git_info(),
    }
    if trainer_state is not None:
        cfg["trainer_state"] = trainer_state
    return cfg


def write_train_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_run_readme(path: str, *, checkpoint_name: str, output_dir: str) -> None:
    lines = [
        f"MDU unlearning checkpoint: {checkpoint_name}",
        f"Directory: {output_dir}",
        "",
        "Files:",
        "  train_config.json  — full run hyperparameters and data provenance",
        "  wandb_run.json     — W&B run link (entity, project, run_id, url)",
        "  wandb_run.txt      — one-line W&B URL",
        "  model weights + tokenizer — HuggingFace layout",
        "  unlearn_mdu_llada.py — training script snapshot for this run",
        "",
        "W&B linkage (when --report_to wandb):",
        "  Local → W&B: train_config.json + wandb_run.json; checkpoint_name in config/summary",
        "  W&B run display name: auto-generated unless --run_name is set",
        "",
        "Eval example:",
        f"  python scripts/eval_tofu_llada.py --model {output_dir} \\",
        "    --experiment mdu --run_id <id> --tofu_split forget10",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare_wandb_run_name(training_args) -> None:
    """
    Clear dllm's auto ``run_name`` (set to default ``output_dir`` at parse time).

    After ``resolve_run_directory`` updates ``output_dir``, the stale name would
    still be ``./checkpoints/_mdu_run`` and become the W&B display name. Leave
    ``run_name`` unset so W&B auto-names unless the user passed ``--run_name``.
    """
    rn = (getattr(training_args, "run_name", None) or "").strip()
    if not rn or rn == _MDU_RUN_NAME_PLACEHOLDER or rn.endswith("/_mdu_run"):
        training_args.run_name = None


def configure_wandb(training_args) -> None:
    """Apply default W&B project when ``report_to`` includes wandb."""
    report_to = training_args.report_to
    if isinstance(report_to, str):
        uses_wandb = report_to == "wandb" or (
            report_to != "none" and "wandb" in report_to.split(",")
        )
    else:
        uses_wandb = "wandb" in (report_to or [])
    if not uses_wandb:
        return
    project = getattr(training_args, "wandb_project", DEFAULT_WANDB_PROJECT)
    os.environ.setdefault("WANDB_PROJECT", project)
    # Do not inherit a stale WANDB_RUN_NAME from the shell; only pin when explicit.
    if getattr(training_args, "run_name", None):
        os.environ["WANDB_RUN_NAME"] = str(training_args.run_name)
    else:
        os.environ.pop("WANDB_RUN_NAME", None)


def uses_wandb_reporting(training_args) -> bool:
    report_to = training_args.report_to
    if isinstance(report_to, str):
        return report_to == "wandb" or (
            report_to != "none" and "wandb" in report_to.split(",")
        )
    return "wandb" in (report_to or [])


def publish_config_to_wandb(config: dict) -> None:
    """Push the full local train_config payload to the active W&B run."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    # Nested dict: reproducible from W&B Config tab (not just TrainingArguments).
    wandb.config.update(_json_safe(config), allow_val_change=True)
    summary = wandb.run.summary
    summary["checkpoint_name"] = config.get("checkpoint_name")
    summary["output_dir"] = config.get("output_dir")
    if config.get("wandb"):
        summary["wandb_run_id"] = config["wandb"].get("run_id")
    wandb.run.tags = tuple(
        sorted(
            set(
                list(wandb.run.tags or ())
                + [str(config.get("checkpoint_name", ""))]
                + [
                    f"loss_{config.get('data_args', {}).get('loss_type', '')}",
                    f"tau_{config.get('data_args', {}).get('null_anchor_tau', '')}",
                    str(config.get("data_args", {}).get("match_mode", "")),
                ]
            )
            - {""}
        )
    )


def persist_wandb_link_local(output_dir: str, config: Optional[dict] = None) -> Optional[dict]:
    """Write wandb_run.{json,txt} and embed wandb block in train_config when a run is active."""
    wandb_info = collect_wandb_run_info()
    if wandb_info is None:
        return None
    write_wandb_link(output_dir, wandb_info)
    if config is not None:
        config["wandb"] = wandb_info
        write_train_config(os.path.join(output_dir, "train_config.json"), config)
    return wandb_info


class WandbRunSyncCallback(transformers.TrainerCallback):
    """Bidirectional local <> W&B linking: full config on W&B, run id/url on disk early."""

    def __init__(
        self,
        *,
        model_args,
        data_args,
        training_args,
        data_source: dict,
        checkpoint_name: str,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.data_source = data_source
        self.checkpoint_name = checkpoint_name

    def _config(self, **extra) -> dict:
        cfg = build_train_config(
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            data_source=self.data_source,
            checkpoint_name=self.checkpoint_name,
        )
        cfg.update(extra)
        return cfg

    def on_train_begin(self, args, state, control, **kwargs):
        if not uses_wandb_reporting(self.training_args):
            return
        if not accelerate.PartialState().is_main_process:
            return
        config = self._config(status="running")
        publish_config_to_wandb(config)
        persist_wandb_link_local(self.training_args.output_dir, config)

    def on_train_end(self, args, state, control, **kwargs):
        if not uses_wandb_reporting(self.training_args):
            return
        if not accelerate.PartialState().is_main_process:
            return
        trainer_state = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
        }
        config = self._config(status="train_finished", trainer_state=trainer_state)
        publish_config_to_wandb(config)
        persist_wandb_link_local(self.training_args.output_dir, config)
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"train_config-{self.checkpoint_name}", type="config"
                )
                path = os.path.join(self.training_args.output_dir, "train_config.json")
                write_train_config(path, config)
                artifact.add_file(path)
                wandb.log_artifact(artifact)
        except Exception:
            pass


def maybe_add_wandb_callback(
    trainer,
    *,
    model_args,
    data_args,
    training_args,
    data_source: dict,
    checkpoint_name: str,
):
    if uses_wandb_reporting(training_args):
        trainer.add_callback(
            WandbRunSyncCallback(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                data_source=data_source,
                checkpoint_name=checkpoint_name,
            )
        )


def collect_wandb_run_info() -> Optional[dict]:
    try:
        import wandb
    except ImportError:
        return None
    if wandb.run is None:
        return None
    entity = wandb.run.entity
    project = wandb.run.project
    run_id = wandb.run.id
    name = wandb.run.name
    url = wandb.run.url or f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    return {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "name": name,
        "url": url,
    }


def write_wandb_link(output_dir: str, wandb_info: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "wandb_run.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(wandb_info, f, indent=2)
        f.write("\n")
    txt_path = os.path.join(output_dir, "wandb_run.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{wandb_info['url']}\n")


def copy_model_python_files(source_model_dir: str, dest_dir: str) -> list[str]:
    copied = []
    for src in glob.glob(os.path.join(source_model_dir, "*.py")):
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def save_final_checkpoint(
    trainer,
    *,
    model_args,
    data_args,
    training_args,
    data_source: dict,
    checkpoint_name: str,
    training_script_path: Optional[str] = None,
) -> str:
    """Save model, tokenizer, config manifest, and optional W&B link."""
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)
    trainer.processing_class.save_pretrained(output_dir)

    training_script_snapshot = None
    if training_script_path and os.path.isfile(training_script_path):
        training_script_snapshot = copy_script_snapshot(training_script_path, output_dir)

    copied = copy_model_python_files(model_args.model_name_or_path, output_dir)
    if not copied:
        print(
            f"[checkpoint] no *.py files copied from {model_args.model_name_or_path}",
            flush=True,
        )

    trainer_state = None
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            trainer_state = json.load(f)

    config = build_train_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        data_source=data_source,
        checkpoint_name=checkpoint_name,
        trainer_state=trainer_state,
    )
    config["status"] = "completed"
    config["copied_python_files"] = [os.path.basename(p) for p in copied]
    if training_script_snapshot is not None:
        config["training_script_snapshot"] = training_script_snapshot
    write_train_config(os.path.join(output_dir, "train_config.json"), config)
    write_run_readme(
        os.path.join(output_dir, "README.txt"),
        checkpoint_name=checkpoint_name,
        output_dir=output_dir,
    )

    wandb_info = collect_wandb_run_info()
    if wandb_info is not None:
        write_wandb_link(output_dir, wandb_info)
        config["wandb"] = wandb_info
        write_train_config(os.path.join(output_dir, "train_config.json"), config)
        publish_config_to_wandb(config)

    return output_dir
