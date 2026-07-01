#!/usr/bin/env bash
# One-step training smoke for null_prompt_mode=empty (all anchor paths).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-./.venv/bin/python}"
ACCELERATE="${ACCELERATE:-./.venv/bin/accelerate}"
UNLEARN="${REPO_ROOT}/src/unlearn_mdu_llada.py"
BASE="${LLADA_BASE_SFT:-./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU}"
CKPT_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"

export PYTHONPATH="${REPO_ROOT}/dllm:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

common_args=(
  --model_name_or_path "${BASE}"
  --checkpoints_root "${CKPT_ROOT}"
  --tofu_split forget10
  --retain_tofu_split retain_perturbed
  --hf_dataset locuslab/TOFU
  --loss_type null_anchor
  --null_prompt_mode empty
  --alpha 1.0
  --null_anchor_tau 0.25
  --null_anchor_eta 0.0
  --null_anchor_kl_dir forward
  --denoise_steps 128
  --max_new_tokens 128
  --max_steps 1
  --learning_rate 1e-5
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 8
  --save_strategy no
  --report_to none
)

run_smoke() {
  local name="$1"
  shift
  local log="${REPO_ROOT}/unlearn_logs/smoke_empty_${name}.log"
  mkdir -p "${REPO_ROOT}/unlearn_logs"
  echo "=== smoke: ${name} ===" | tee "${log}"
  "${ACCELERATE}" launch --num_processes 1 "${UNLEARN}" \
    --checkpoint_name "smoke_empty_${name}" \
    "${common_args[@]}" \
    "$@" >> "${log}" 2>&1
  if ! grep -q 'null_prompt_mode=empty' "${log}"; then
    echo "FAIL: missing null_prompt_mode=empty in ${log}" >&2
    exit 1
  fi
  echo "ok  ${name}"
}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_FROZEN:-0,1}"
run_smoke frozen_random \
  --match_mode random \
  --null_anchor_source frozen_sft \
  --ref_device auto \
  --disable_data_parallel auto

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_TRAINABLE:-0}"
run_smoke trainable_position \
  --match_mode position \
  --null_anchor_source trainable_cfg \
  --novel_percentile 100 \
  --ref_device same \
  --disable_data_parallel yes \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}'

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_FROZEN:-0,1}"
run_smoke ema_position \
  --match_mode position \
  --null_anchor_source ema \
  --null_anchor_ema_decay 0.999 \
  --novel_percentile 100 \
  --ref_device auto \
  --disable_data_parallel auto \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}'

echo "all smoke runs passed"
