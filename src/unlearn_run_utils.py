"""Shared helpers for MDU unlearning runs: data loading, checkpoints, W&B."""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import accelerate
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


def default_checkpoint_name(
    *,
    model_name_or_path: str,
    tofu_split: Optional[str],
    loss_type: str,
    null_anchor_tau: float,
    match_mode: str,
) -> str:
    backbone = _backbone_tag(model_name_or_path)
    split = (tofu_split or "local").replace("/", "_")
    if loss_type == "null_anchor":
        tau_s = f"{null_anchor_tau:g}".replace(".", "p")
        return f"mdu_{backbone}_{split}_nullanchor_tau{tau_s}"
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
) -> str:
    """Save model, tokenizer, config manifest, and optional W&B link."""
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)
    trainer.processing_class.save_pretrained(output_dir)

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
