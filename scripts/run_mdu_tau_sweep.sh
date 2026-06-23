#!/usr/bin/env bash
# Sequential MDU τ sweep (LLaDA / TOFU forget10): unlearn → eval → next τ.
#
# Paper / repo config: README.md + configs/mdu_tofu.yaml
#   null_anchor, random masking, η=0, forward KL, α=1, denoise_steps=128
#   lr=1e-5, 9 epochs, forget10 + retain_perturbed (HF locuslab/TOFU)
#   paper microbatch 4×4; we default 2×8 (effective 16) for 2×A100 OOM safety
#
# Per τ, ONLY these change: --null_anchor_tau, --checkpoint_name, checkpoint/eval paths.
# Starting model is always LLADA_BASE_SFT (same SFT checkpoint for every τ).
#
# Usage:
#   cd /path/to/MDU
#   export WANDB_API_KEY=...    # or: wandb login
#   DRY_RUN=1 bash scripts/run_mdu_tau_sweep.sh
#   bash scripts/run_mdu_tau_sweep.sh
#   # overnight (recommended):
#   nohup bash scripts/run_mdu_tau_sweep.sh > /dev/null 2>&1 &
#   # master log is always written to SWEEP_MASTER_LOG (see below)
#
# Resume / skip:
#   SKIP_TAU_0=1            skip τ=0 (already done; default: 1)
#   SKIP_TAU_0P5=1          skip τ=0.5 (already done; default: 1)
#   START_FROM_TAU=0.25     begin at this τ (inclusive)
#   SKIP_UNLEARN_IF_CKPT=1  skip training when checkpoint weights exist (default: 1)
#   SKIP_EVAL_IF_DONE=1     skip eval splits already completed (default: 1)
#
# Env (optional):
#   LLADA_BASE_SFT, CHECKPOINTS_ROOT, EVAL_OUTPUTS_ROOT
#   EVAL_DATE, EVAL_RUN_VERSION, WANDB_PROJECT, WANDB=1
#   CUDA_DEVICES=0,1  REF_DEVICE=auto  DISABLE_DP=auto
#   PER_DEVICE_BATCH=2  GRAD_ACCUM=8
#   SWEEP_LOG_DIR, UNLEARN_LOG_DIR, SWEEP_MASTER_LOG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────

TAUS=(0 0.25 0.5 0.75 1)

LLADA_BASE_SFT="${LLADA_BASE_SFT:-./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"
EVAL_OUTPUTS_ROOT="${EVAL_OUTPUTS_ROOT:-./eval_outputs}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-./sweep_logs}"
UNLEARN_LOG_DIR="${UNLEARN_LOG_DIR:-./unlearn_logs}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
REF_DEVICE="${REF_DEVICE:-auto}"
NULL_ANCHOR_SOURCE="${NULL_ANCHOR_SOURCE:-auto}"
DISABLE_DP="${DISABLE_DP:-auto}"

LR="${LR:-1e-5}"
EPO="${EPO:-9}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unlearning-dllms-MDU}"

SKIP_TAU_0="${SKIP_TAU_0:-1}"
SKIP_TAU_0P5="${SKIP_TAU_0P5:-1}"
SKIP_UNLEARN_IF_CKPT="${SKIP_UNLEARN_IF_CKPT:-1}"
SKIP_EVAL_IF_DONE="${SKIP_EVAL_IF_DONE:-1}"
START_FROM_TAU="${START_FROM_TAU:-}"
DRY_RUN="${DRY_RUN:-0}"

EVAL_DATE="${EVAL_DATE:-$(date +%Y-%m-%d)}"
EVAL_RUN_VERSION="${EVAL_RUN_VERSION:-v1}"
EVAL_RUN_ID_TAU_0="${EVAL_RUN_ID_TAU_0:-2026-06-22_mdu_tau0_v1}"
EVAL_RUN_ID_TAU_0P5="${EVAL_RUN_ID_TAU_0P5:-2026-06-22_mdu_tau0p5_v1}"

# Eval hyperparams (must match mdu_tau0p5 run and eval_tofu_llada.py defaults)
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_STEPS="${EVAL_STEPS:-256}"
EVAL_MASK_SAMPLES="${EVAL_MASK_SAMPLES:-128}"
EVAL_SEED="${EVAL_SEED:-42}"

PYTHON="${PYTHON:-./.venv/bin/python}"
ACCELERATE="${ACCELERATE:-./.venv/bin/accelerate}"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_tofu_llada.py"
UNLEARN_SCRIPT="${REPO_ROOT}/src/unlearn_mdu_llada.py"

export PYTHONPATH="${REPO_ROOT}/dllm:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

mkdir -p "${SWEEP_LOG_DIR}" "${UNLEARN_LOG_DIR}" "${CHECKPOINTS_ROOT}"

# Master sweep log (tee stdout+stderr). Re-entry safe via SWEEP_LOGGING_ACTIVE.
SWEEP_MASTER_LOG="${SWEEP_MASTER_LOG:-${SWEEP_LOG_DIR}/mdu_tau_sweep_${EVAL_DATE}.log}"

# ── Logging (stderr so command-substitution pid capture stays clean) ──────────

log() { echo "[$(date -Iseconds)] $*" >&2; }

if [[ "${SWEEP_LOGGING_ACTIVE:-0}" != "1" && "${DRY_RUN}" != "1" ]]; then
  export SWEEP_LOGGING_ACTIVE=1
  exec > >(tee -a "${SWEEP_MASTER_LOG}") 2>&1
  log "Master sweep log: ${SWEEP_MASTER_LOG}"
fi

on_err() {
  local ec=$?
  log "FATAL: sweep aborted (exit ${ec}) at line ${BASH_LINENO[0]}"
  exit "${ec}"
}
trap on_err ERR

# ── Helpers ───────────────────────────────────────────────────────────────────

tau_slug() {
  python3 -c "import sys; t=float(sys.argv[1]); print(f'{t:g}'.replace('.', 'p'))" "$1"
}

checkpoint_name_for_tau() {
  echo "mdu_llada_forget10_nullanchor_tau$(tau_slug "$1")"
}

eval_experiment_for_tau() {
  echo "mdu_tau$(tau_slug "$1")"
}

eval_run_id_for_tau() {
  local tau="$1"
  if [[ "${tau}" == "0" ]]; then
    echo "${EVAL_RUN_ID_TAU_0}"
  elif [[ "${tau}" == "0.5" ]]; then
    echo "${EVAL_RUN_ID_TAU_0P5}"
  else
    echo "${EVAL_DATE}_mdu_tau$(tau_slug "${tau}")_${EVAL_RUN_VERSION}"
  fi
}

checkpoint_dir_for_tau() {
  echo "${CHECKPOINTS_ROOT}/$(checkpoint_name_for_tau "$1")"
}

eval_run_root_for_tau() {
  echo "${EVAL_OUTPUTS_ROOT}/$(eval_experiment_for_tau "$1")/$(eval_run_id_for_tau "$1")"
}

checkpoint_has_weights() {
  local ckpt="$1"
  [[ -f "${ckpt}/model.safetensors.index.json" ]] || [[ -f "${ckpt}/model.safetensors" ]]
}

eval_split_done() {
  local run_root="$1"
  local split="$2"
  local expected="$3"
  local meta="${run_root}/splits/${split}/split_meta.json"
  local details="${run_root}/splits/${split}/details.jsonl"
  [[ -f "${meta}" ]] || return 1
  python3 - <<PY || return 1
import json
meta = json.load(open("${meta}"))
if meta.get("status") != "completed":
    raise SystemExit(1)
PY
  [[ -f "${details}" ]] || return 1
  local n
  n=$(wc -l < "${details}")
  [[ "${n}" -ge "${expected}" ]]
}

all_evals_done() {
  local run_root="$1"
  eval_split_done "${run_root}" forget10 400 \
    && eval_split_done "${run_root}" retain_perturbed 400 \
    && eval_split_done "${run_root}" world_facts 117 \
    && eval_split_done "${run_root}" real_authors 100
}

should_skip_tau() {
  local tau="$1"
  [[ "${SKIP_TAU_0}" == "1" && "${tau}" == "0" ]] && return 0
  [[ "${SKIP_TAU_0P5}" == "1" && "${tau}" == "0.5" ]] && return 0
  return 1
}

skip_tau_reason() {
  local tau="$1"
  if [[ "${SKIP_TAU_0}" == "1" && "${tau}" == "0" ]]; then
    echo "SKIP_TAU_0=1"
  elif [[ "${SKIP_TAU_0P5}" == "1" && "${tau}" == "0.5" ]]; then
    echo "SKIP_TAU_0P5=1"
  else
    echo "skip flag"
  fi
}

past_start_tau() {
  local tau="$1"
  [[ -z "${START_FROM_TAU}" ]] && return 0
  python3 - <<PY
import sys
taus = [0, 0.25, 0.5, 0.75, 1]
t, start = float("${tau}"), float("${START_FROM_TAU}")
sys.exit(0 if taus.index(t) >= taus.index(start) else 1)
PY
}

wait_pid() {
  local pid="$1"
  local label="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] would wait for ${label} (pid ${pid})"
    return 0
  fi
  if ! wait "${pid}"; then
    log "ERROR: ${label} failed (pid ${pid})"
    exit 1
  fi
  log "${label} finished (pid ${pid})"
}

validate_prerequisites() {
  local missing=0
  for bin in "${PYTHON}" "${ACCELERATE}"; do
    if [[ ! -x "${bin}" ]]; then
      log "ERROR: missing executable: ${bin}"
      missing=1
    fi
  done
  for f in "${UNLEARN_SCRIPT}" "${EVAL_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
      log "ERROR: missing script: ${f}"
      missing=1
    fi
  done
  if ! checkpoint_has_weights "${LLADA_BASE_SFT}"; then
    log "ERROR: SFT base model has no weights: ${LLADA_BASE_SFT}"
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
}

print_sweep_plan() {
  log "════════════════════════════════════════════════════════════════"
  log "MDU τ sweep plan"
  log "════════════════════════════════════════════════════════════════"
  log "Repo:              ${REPO_ROOT}"
  log "DRY_RUN:           ${DRY_RUN}"
  log "τ values:          ${TAUS[*]}"
  log "SKIP_TAU_0:         ${SKIP_TAU_0}"
  log "SKIP_TAU_0P5:      ${SKIP_TAU_0P5}"
  log "START_FROM_TAU:    ${START_FROM_TAU:-<start>}"
  log ""
  log "── Training (identical except τ + checkpoint name) ──"
  log "Base model:        ${LLADA_BASE_SFT}"
  log "Checkpoints root:  ${CHECKPOINTS_ROOT}"
  log "loss_type:         null_anchor"
  log "match_mode:        random"
  log "alpha:             1.0"
  log "null_anchor_eta:   0.0"
  log "null_anchor_kl_dir: forward"
  log "denoise_steps:     128"
  log "max_new_tokens:    128"
  log "tofu_split:        forget10"
  log "retain_tofu_split: retain_perturbed"
  log "hf_dataset:        locuslab/TOFU"
  log "epochs / lr:       ${EPO} / ${LR}"
  log "batch × accum:     ${PER_DEVICE_BATCH} × ${GRAD_ACCUM} (effective $((PER_DEVICE_BATCH * GRAD_ACCUM)))"
  log "save_strategy:     no"
  log "Training GPUs:     CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} (ref_device=${REF_DEVICE}, null_anchor_source=${NULL_ANCHOR_SOURCE}, disable_dp=${DISABLE_DP})"
  log "Unlearn logs:      ${UNLEARN_LOG_DIR}/unlearn_<checkpoint>.log"
  log ""
  log "── W&B ──"
  log "WANDB:             ${WANDB}"
  log "WANDB_PROJECT:     ${WANDB_PROJECT}"
  log "run_name:          <checkpoint_name> per τ (W&B tags include tau_*)"
  log ""
  log "── Eval (identical across τ except model path + experiment/run_id) ──"
  log "Eval outputs root: ${EVAL_OUTPUTS_ROOT}"
  log "max_new_tokens:    ${EVAL_MAX_NEW_TOKENS}"
  log "steps:             ${EVAL_STEPS}"
  log "mask_samples:      ${EVAL_MASK_SAMPLES}"
  log "seed:              ${EVAL_SEED}"
  log "truth_ratio:       false"
  log "hf_dataset:        locuslab/TOFU"
  log "Eval schedule:     forget10(GPU0) ∥ retain_perturbed(GPU1); 1st done→WF, 2nd done→RA (pipelined)"
  log "Eval logs:         ${SWEEP_LOG_DIR}/eval_<checkpoint>_<split>.log"
  log "Post-eval:         eval_tofu_llada.py rebuilds summary.json + manifest.json per split"
  log ""
  log "── Per-τ paths ──"
  local tau
  for tau in "${TAUS[@]}"; do
    if should_skip_tau "${tau}"; then
      log "  τ=${tau}  SKIP ($(skip_tau_reason "${tau}"))"
      continue
    fi
    if ! past_start_tau "${tau}"; then
      log "  τ=${tau}  SKIP (before START_FROM_TAU)"
      continue
    fi
    log "  τ=${tau}"
    log "    ckpt:     $(checkpoint_dir_for_tau "${tau}")"
    log "    eval:     $(eval_run_root_for_tau "${tau}")"
    log "    wandb:    project=${WANDB_PROJECT} name=$(checkpoint_name_for_tau "${tau}")"
  done
  log "════════════════════════════════════════════════════════════════"
}

print_tau_plan() {
  local tau="$1"
  log "── τ=${tau} ──"
  log "  varies:  null_anchor_tau=${tau}, checkpoint=$(checkpoint_name_for_tau "${tau}")"
  log "  ckpt:    $(checkpoint_dir_for_tau "${tau}")"
  log "  eval:    $(eval_run_root_for_tau "${tau}")"
}

post_eval_validate() {
  local tau="$1"
  local run_root
  run_root="$(eval_run_root_for_tau "${tau}")"
  local summary="${run_root}/summary.json"
  local manifest="${run_root}/manifest.json"

  if [[ ! -f "${summary}" ]]; then
    log "ERROR: missing ${summary}"
    exit 1
  fi
  if [[ ! -f "${manifest}" ]]; then
    log "ERROR: missing ${manifest}"
    exit 1
  fi

  log "Post-eval OK τ=${tau}"
  log "  summary:  ${summary}"
  log "  manifest: ${manifest}"
  python3 - <<PY
import json
s = json.load(open("${summary}"))
print(json.dumps(s, indent=2))
PY
}

# ── Unlearning ────────────────────────────────────────────────────────────────

run_unlearn() {
  local tau="$1"
  local ckpt_name ckpt_dir log_file
  ckpt_name="$(checkpoint_name_for_tau "${tau}")"
  ckpt_dir="$(checkpoint_dir_for_tau "${tau}")"
  log_file="${UNLEARN_LOG_DIR}/unlearn_${ckpt_name}.log"

  if [[ "${SKIP_UNLEARN_IF_CKPT}" == "1" ]] && checkpoint_has_weights "${ckpt_dir}"; then
    log "Skip unlearn τ=${tau}: checkpoint exists at ${ckpt_dir}"
    return 0
  fi

  log "Unlearn τ=${tau} → ${ckpt_dir} (log: ${log_file})"
  local -a wandb_args=()
  if [[ "${WANDB}" == "1" ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT}"
    unset WANDB_RUN_NAME 2>/dev/null || true
    wandb_args=(
      --report_to wandb
      --wandb_project "${WANDB_PROJECT}"
      --run_name "${ckpt_name}"
    )
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] would run accelerate launch → ${log_file}"
    return 0
  fi

  "${ACCELERATE}" launch --num_processes 1 "${UNLEARN_SCRIPT}" \
    --model_name_or_path "${LLADA_BASE_SFT}" \
    --checkpoints_root "${CHECKPOINTS_ROOT}" \
    --checkpoint_name "${ckpt_name}" \
    --tofu_split forget10 \
    --retain_tofu_split retain_perturbed \
    --hf_dataset locuslab/TOFU \
    --loss_type null_anchor \
    --match_mode random \
    --alpha 1.0 \
    --null_anchor_tau "${tau}" \
    --null_anchor_eta 0.0 \
    --null_anchor_kl_dir forward \
    --denoise_steps 128 \
    --max_new_tokens 128 \
    --num_train_epochs "${EPO}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_strategy no \
    --ref_device "${REF_DEVICE}" \
    --null_anchor_source "${NULL_ANCHOR_SOURCE}" \
    --disable_data_parallel "${DISABLE_DP}" \
    "${wandb_args[@]}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  log "Unlearn started pid=${pid}"
  wait_pid "${pid}" "unlearn τ=${tau}"

  if ! checkpoint_has_weights "${ckpt_dir}"; then
    log "ERROR: no weights in ${ckpt_dir} after training (see ${log_file})"
    exit 1
  fi
  log "Unlearn τ=${tau} OK: ${ckpt_dir}"
}

# ── Eval (pipelined: F∥R, then WF/RA on first two GPU frees) ───────────────────

# Launch eval in current shell; caller reads $! immediately (never pid=$(...)).
run_eval_split() {
  local gpu="$1"
  local model="$2"
  local experiment="$3"
  local run_id="$4"
  local split="$5"
  local log_suffix="$6"

  local log_file="${SWEEP_LOG_DIR}/eval_${log_suffix}.log"
  log "Eval ${split} on physical GPU ${gpu} → ${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] eval ${split} (experiment=${experiment} run_id=${run_id})"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVAL_SCRIPT}" \
    --model "${model}" \
    --experiment "${experiment}" \
    --run_id "${run_id}" \
    --eval_outputs_root "${EVAL_OUTPUTS_ROOT}" \
    --tofu_split "${split}" \
    --hf_dataset locuslab/TOFU \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --steps "${EVAL_STEPS}" \
    --mask_samples "${EVAL_MASK_SAMPLES}" \
    --seed "${EVAL_SEED}" \
    > "${log_file}" 2>&1 &
}

# Remove one PID from a nameref array.
_eval_remove_pid() {
  local target="$1"
  local -n _arr=$2
  local kept=()
  local p
  for p in "${_arr[@]}"; do
    [[ "${p}" == "${target}" ]] || kept+=("${p}")
  done
  _arr=("${kept[@]}")
}

# Wait until one tracked PID exits; sets nameref $2 to that PID. Never use wait -n.
_eval_wait_any() {
  local -n _pids=$1
  local -n _finished=$2
  if ((${#_pids[@]} == 0)); then
    return 1
  fi
  local pid ec
  while true; do
    for pid in "${_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        ec=0
        wait "${pid}" 2>/dev/null || ec=$?
        if ((ec != 0)); then
          log "ERROR: eval pid ${pid} exited with status ${ec}"
          return 1
        fi
        _finished="${pid}"
        return 0
      fi
    done
    sleep 1
  done
}

# Track a launched eval in active_pids / pid_gpu / pid_split namerefs.
_eval_track() {
  local pid="$1" gpu="$2" split="$3"
  local -n _active=$4
  local -n _gpu_map=$5
  local -n _split_map=$6
  _active+=("${pid}")
  _gpu_map["${pid}"]="${gpu}"
  _split_map["${pid}"]="${split}"
}

# Pipelined eval: F@0 ∥ R@1; 1st finish→WF on that GPU; 2nd finish→RA on that GPU.
run_evals_pipelined() {
  local tau="$1" model="$2" experiment="$3" run_id="$4" slug="$5"
  local need_forget="$6" need_retain="$7" need_wf="$8" need_ra="$9"

  local -a active_pids=()
  declare -A pid_gpu=()
  declare -A pid_split=()

  _eval_launch() {
    local gpu="$1" split="$2" suffix="$3"
    run_eval_split "${gpu}" "${model}" "${experiment}" "${run_id}" "${split}" "${suffix}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      return 0
    fi
    local pid=$!
    _eval_track "${pid}" "${gpu}" "${split}" active_pids pid_gpu pid_split
    log "  started ${split} pid=${pid} gpu=${gpu}"
  }

  local seeds=$((need_forget + need_retain))

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] pipelined eval (seeds=${seeds} need_wf=${need_wf} need_ra=${need_ra})"
    [[ "${need_forget}" == "1" ]] && _eval_launch 0 forget10 "${slug}_forget10"
    [[ "${need_retain}" == "1" ]] && _eval_launch 1 retain_perturbed "${slug}_retain"
    [[ "${seeds}" -eq 0 && "${need_wf}" == "1" ]] && _eval_launch 0 world_facts "${slug}_world_facts"
    [[ "${seeds}" -eq 0 && "${need_ra}" == "1" ]] && _eval_launch 1 real_authors "${slug}_real_authors"
    return 0
  fi

  if [[ "${seeds}" -eq 0 ]]; then
    log "Eval pipeline: forget/retain skipped → WF/RA only"
    [[ "${need_wf}" == "1" ]] && _eval_launch 0 world_facts "${slug}_world_facts"
    [[ "${need_ra}" == "1" ]] && _eval_launch 1 real_authors "${slug}_real_authors"
  else
    log "Eval pipeline: forget∥retain; 1st done→WF, 2nd done→RA (max 1 job/GPU)"
    [[ "${need_forget}" == "1" ]] && _eval_launch 0 forget10 "${slug}_forget10"
    [[ "${need_retain}" == "1" ]] && _eval_launch 1 retain_perturbed "${slug}_retain"
  fi

  local completions=0
  while ((${#active_pids[@]} > 0)); do
    local finished gpu split
    if ! _eval_wait_any active_pids finished; then
      log "ERROR: eval wait failed τ=${tau}"
      exit 1
    fi
    gpu="${pid_gpu[${finished}]:-?}"
    split="${pid_split[${finished}]:-?}"
    _eval_remove_pid "${finished}" active_pids
    completions=$((completions + 1))
    log "Eval ${split} finished (gpu=${gpu}, completion #${completions})"

    if [[ "${completions}" -eq 1 && "${need_wf}" == "1" ]]; then
      _eval_launch "${gpu}" world_facts "${slug}_world_facts"
    elif [[ "${completions}" -eq 2 && "${need_ra}" == "1" ]]; then
      _eval_launch "${gpu}" real_authors "${slug}_real_authors"
    fi
  done
}

run_evals_for_tau() {
  local tau="$1"
  local model experiment run_id run_root slug
  model="$(checkpoint_dir_for_tau "${tau}")"
  experiment="$(eval_experiment_for_tau "${tau}")"
  run_id="$(eval_run_id_for_tau "${tau}")"
  run_root="$(eval_run_root_for_tau "${tau}")"
  slug="$(checkpoint_name_for_tau "${tau}")"

  if [[ ! -d "${model}" ]] && [[ "${DRY_RUN}" != "1" ]]; then
    log "ERROR: eval model not found: ${model}"
    exit 1
  fi

  if [[ "${SKIP_EVAL_IF_DONE}" == "1" ]] && all_evals_done "${run_root}"; then
    log "Skip eval τ=${tau}: all splits done under ${run_root}"
    post_eval_validate "${tau}"
    return 0
  fi

  log "Eval τ=${tau} model=${model} run_root=${run_root}"

  local need_forget=1 need_retain=1 need_wf=1 need_ra=1
  if [[ "${SKIP_EVAL_IF_DONE}" == "1" ]]; then
    eval_split_done "${run_root}" forget10 400 && need_forget=0
    eval_split_done "${run_root}" retain_perturbed 400 && need_retain=0
    eval_split_done "${run_root}" world_facts 117 && need_wf=0
    eval_split_done "${run_root}" real_authors 100 && need_ra=0
  fi

  if [[ "${need_forget}" == "0" && "${need_retain}" == "0" && "${need_wf}" == "0" && "${need_ra}" == "0" ]]; then
    log "All eval splits already done for τ=${tau}"
    post_eval_validate "${tau}"
    return 0
  fi

  run_evals_pipelined "${tau}" "${model}" "${experiment}" "${run_id}" "${slug}" \
    "${need_forget}" "${need_retain}" "${need_wf}" "${need_ra}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if ! all_evals_done "${run_root}"; then
      log "ERROR: eval incomplete for τ=${tau} under ${run_root}"
      exit 1
    fi
    post_eval_validate "${tau}"
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

validate_prerequisites
print_sweep_plan

for tau in "${TAUS[@]}"; do
  if ! past_start_tau "${tau}"; then
    log "Skip τ=${tau} (before START_FROM_TAU=${START_FROM_TAU})"
    continue
  fi
  if should_skip_tau "${tau}"; then
    log "Skip τ=${tau} ($(skip_tau_reason "${tau}"))"
    continue
  fi

  log ""
  log "════════════════════════════════════════════════════════════════"
  log "START τ=${tau}"
  log "════════════════════════════════════════════════════════════════"
  print_tau_plan "${tau}"

  run_unlearn "${tau}"
  run_evals_for_tau "${tau}"

  log "COMPLETE τ=${tau}"
done

log ""
log "MDU τ sweep finished successfully."
log "Master log: ${SWEEP_MASTER_LOG}"
log "If new τ completed: update eval_outputs/RESULTS.md from each run's summary.json."
