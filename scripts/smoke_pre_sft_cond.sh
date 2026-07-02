#!/usr/bin/env bash
# One-step training smoke for null_anchor_source=pre_sft_cond (conditional pre-SFT ref).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-./.venv/bin/python}"
ACCELERATE="${ACCELERATE:-./.venv/bin/accelerate}"
UNLEARN="${REPO_ROOT}/src/unlearn_mdu_llada.py"
BASE="${LLADA_BASE_SFT:-./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU}"
PRE_SFT_REF="${LLADA_PRE_SFT_REF:-GSAI-ML/LLaDA-8B-Instruct}"
CKPT_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"

export PYTHONPATH="${REPO_ROOT}/dllm:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_FROZEN:-0,1}"

log="${REPO_ROOT}/unlearn_logs/smoke_pre_sft_cond.log"
mkdir -p "${REPO_ROOT}/unlearn_logs"

echo "=== smoke: pre_sft_cond position ===" | tee "${log}"
echo "ref=${PRE_SFT_REF}" | tee -a "${log}"

"${ACCELERATE}" launch --num_processes 1 "${UNLEARN}" \
  --model_name_or_path "${BASE}" \
  --ref_model_name_or_path "${PRE_SFT_REF}" \
  --checkpoints_root "${CKPT_ROOT}" \
  --checkpoint_name "smoke_pre_sft_cond_position" \
  --tofu_split forget10 \
  --retain_tofu_split retain_perturbed \
  --hf_dataset locuslab/TOFU \
  --loss_type null_anchor \
  --match_mode position \
  --null_anchor_source pre_sft_cond \
  --novel_percentile 100 \
  --alpha 1.0 \
  --null_anchor_tau 0.25 \
  --null_anchor_eta 0.0 \
  --null_anchor_kl_dir forward \
  --denoise_steps 128 \
  --max_new_tokens 128 \
  --max_steps 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --save_strategy no \
  --ref_device auto \
  --disable_data_parallel auto \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --report_to none \
  >> "${log}" 2>&1

for needle in 'anchor=pre_sft_cond' 'anchor_input=conditional' "ref_path=${PRE_SFT_REF}"; do
  if ! grep -qF "${needle}" "${log}"; then
    echo "FAIL: missing '${needle}' in ${log}" >&2
    exit 1
  fi
done

echo "ok smoke_pre_sft_cond"
