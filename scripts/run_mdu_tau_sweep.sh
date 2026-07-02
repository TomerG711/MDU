#!/usr/bin/env bash
# MDU τ sweep (LLaDA / TOFU forget10): unlearn → eval for each τ.
#
# Sweeps τ only. Set match_mode + null_anchor_source once per invocation
# (disk limit: ~20G per checkpoint — run 6 configs as separate jobs).
#
#   bash scripts/run_mdu_tau_sweep.sh
#   MATCH_MODE=token_id NULL_ANCHOR_SOURCE=frozen_sft bash scripts/run_mdu_tau_sweep.sh
#   MATCH_MODE=random NULL_ANCHOR_SOURCE=trainable_cfg bash scripts/run_mdu_tau_sweep.sh
#
# Full 30-run grid = 6 invocations × 5 τ. See docs/SWEEP_GRID.md.
#
# Env (per invocation):
#   MATCH_MODE=random|token_id|position     (default: random)
#   NULL_ANCHOR_SOURCE=frozen_sft|trainable_cfg|ema|pre_sft_cond  (default: frozen_sft)
#   REF_MODEL_NAME_OR_PATH=...   # optional ref override
#   NULL_ANCHOR_EMA_DECAY=0.999   # when NULL_ANCHOR_SOURCE=ema
#   NULL_PROMPT_MODE=mask|empty|pad  (default: mask)
#   TAUS="0 0.25 0.5 0.75 1"
#   NOVEL_PERCENTILE=100   # token_id/position (upstream)
#   GRADIENT_CHECKPOINTING=1  # reduce Pass-2 backward memory (position/token_id)
#   SKIP_UNLEARN_IF_CKPT=1  SKIP_EVAL_IF_DONE=1  START_FROM_TAU=

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────

read -r -a TAUS <<< "${TAUS:-0 0.25 0.5 0.75 1}"
MATCH_MODE="${MATCH_MODE:-random}"
NULL_ANCHOR_SOURCE="${NULL_ANCHOR_SOURCE:-frozen_sft}"

LLADA_BASE_SFT="${LLADA_BASE_SFT:-./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"
EVAL_OUTPUTS_ROOT="${EVAL_OUTPUTS_ROOT:-./eval_outputs}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-./sweep_logs}"
UNLEARN_LOG_DIR="${UNLEARN_LOG_DIR:-./unlearn_logs}"

CUDA_DEVICES_FROZEN="${CUDA_DEVICES_FROZEN:-0,1}"
CUDA_DEVICES_TRAINABLE="${CUDA_DEVICES_TRAINABLE:-0}"
REF_DEVICE_FROZEN="${REF_DEVICE_FROZEN:-auto}"
REF_DEVICE_TRAINABLE="${REF_DEVICE_TRAINABLE:-same}"
DISABLE_DP_FROZEN="${DISABLE_DP_FROZEN:-auto}"
DISABLE_DP_TRAINABLE="${DISABLE_DP_TRAINABLE:-yes}"

LR="${LR:-1e-5}"
EPO="${EPO:-9}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NOVEL_PERCENTILE="${NOVEL_PERCENTILE:-100}"
NULL_ANCHOR_EMA_DECAY="${NULL_ANCHOR_EMA_DECAY:-0.999}"
NULL_PROMPT_MODE="${NULL_PROMPT_MODE:-mask}"
REF_MODEL_NAME_OR_PATH="${REF_MODEL_NAME_OR_PATH:-}"
LLADA_PRE_SFT_REF="${LLADA_PRE_SFT_REF:-GSAI-ML/LLaDA-8B-Instruct}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unlearning-dllms-MDU}"

SKIP_UNLEARN_IF_CKPT="${SKIP_UNLEARN_IF_CKPT:-1}"
SKIP_EVAL_IF_DONE="${SKIP_EVAL_IF_DONE:-1}"
START_FROM_TAU="${START_FROM_TAU:-}"
DRY_RUN="${DRY_RUN:-0}"

EVAL_DATE="${EVAL_DATE:-$(date +%Y-%m-%d)}"
EVAL_RUN_VERSION="${EVAL_RUN_VERSION:-v1}"
# Legacy eval run_ids (random + frozen_sft only)
EVAL_RUN_ID_TAU_0="${EVAL_RUN_ID_TAU_0:-2026-06-22_mdu_tau0_v1}"
EVAL_RUN_ID_TAU_0P5="${EVAL_RUN_ID_TAU_0P5:-2026-06-22_mdu_tau0p5_v1}"

EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_STEPS="${EVAL_STEPS:-256}"
EVAL_MASK_SAMPLES="${EVAL_MASK_SAMPLES:-128}"
EVAL_SEED="${EVAL_SEED:-42}"

PYTHON="${PYTHON:-./.venv/bin/python}"
ACCELERATE="${ACCELERATE:-./.venv/bin/accelerate}"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_tofu_llada.py"
UNLEARN_SCRIPT="${REPO_ROOT}/src/unlearn_mdu_llada.py"

export PYTHONPATH="${REPO_ROOT}/dllm:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${SWEEP_LOG_DIR}" "${UNLEARN_LOG_DIR}" "${CHECKPOINTS_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────

log() { echo "[$(date -Iseconds)] $*" >&2; }

# ── Helpers ───────────────────────────────────────────────────────────────────

tau_slug() {
  python3 -c "import sys; t=float(sys.argv[1]); print(f'{t:g}'.replace('.', 'p'))" "$1"
}

anchor_slug() {
  case "$1" in
    frozen_sft|frozen|ref) echo "frozen" ;;
    trainable_cfg|trainable|cfg) echo "cfg" ;;
    ema|ema_sft|ema_cfg) echo "ema" ;;
    pre_sft_cond|pre_sft|base|base_instruct|presftcond) echo "presftcond" ;;
    *) echo "$1" ;;
  esac
}

ema_decay_slug() {
  python3 -c "import sys; d=float(sys.argv[1]); print(f'{d:g}'.replace('.', 'p'))" "$1"
}

# EMA checkpoints/evals include decay when not the legacy default (0.999 → plain "ema").
effective_anchor_slug() {
  local anchor="$1"
  local base; base="$(anchor_slug "${anchor}")"
  if [[ "${base}" != "ema" ]]; then
    echo "${base}"
    return
  fi
  if [[ "${NULL_ANCHOR_EMA_DECAY}" == "0.999" ]]; then
    echo "ema"
  else
    echo "ema$(ema_decay_slug "${NULL_ANCHOR_EMA_DECAY}")"
  fi
}

null_prompt_slug() {
  case "$1" in
    mask|"") echo "" ;;
    empty|pad) echo "nullprompt_${1}" ;;
    *)
      log "ERROR: NULL_PROMPT_MODE must be mask, empty, or pad (got: $1)"
      exit 1
      ;;
  esac
}

ANCHOR_SLUG="$(effective_anchor_slug "${NULL_ANCHOR_SOURCE}")"
NULL_PROMPT_SLUG="$(null_prompt_slug "${NULL_PROMPT_MODE}")"
SWEEP_NAME_SUFFIX="${ANCHOR_SLUG}"
if [[ -n "${NULL_PROMPT_SLUG}" ]]; then
  SWEEP_NAME_SUFFIX="${SWEEP_NAME_SUFFIX}_${NULL_PROMPT_SLUG}"
fi
SWEEP_MASTER_LOG="${SWEEP_MASTER_LOG:-${SWEEP_LOG_DIR}/mdu_tau_sweep_${MATCH_MODE}_${SWEEP_NAME_SUFFIX}_${EVAL_DATE}.log}"

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

is_legacy_random_frozen() {
  [[ "$1" == "random" && "$(anchor_slug "$2")" == "frozen" && "${NULL_PROMPT_MODE}" == "mask" ]]
}

null_prompt_name_part() {
  local slug; slug="$(null_prompt_slug "${NULL_PROMPT_MODE}")"
  if [[ -n "${slug}" ]]; then
    echo "_${slug}"
  fi
}

checkpoint_name_for_run() {
  local tau="$1" match="$2" anchor="$3"
  local slug; slug="$(tau_slug "${tau}")"
  local npm_part; npm_part="$(null_prompt_name_part)"
  if is_legacy_random_frozen "${match}" "${anchor}"; then
    echo "mdu_llada_forget10_nullanchor_tau${slug}"
  else
    echo "mdu_llada_forget10_${match}_$(effective_anchor_slug "${anchor}")${npm_part}_tau${slug}"
  fi
}

eval_experiment_for_run() {
  local tau="$1" match="$2" anchor="$3"
  local npm_part; npm_part="$(null_prompt_name_part)"
  if is_legacy_random_frozen "${match}" "${anchor}"; then
    echo "mdu_tau$(tau_slug "${tau}")"
  else
    echo "mdu_${match}_$(effective_anchor_slug "${anchor}")${npm_part}"
  fi
}

eval_run_id_for_run() {
  local tau="$1" match="$2" anchor="$3"
  if is_legacy_random_frozen "${match}" "${anchor}"; then
    if [[ "${tau}" == "0" ]]; then
      echo "${EVAL_RUN_ID_TAU_0}"
    elif [[ "${tau}" == "0.5" ]]; then
      echo "${EVAL_RUN_ID_TAU_0P5}"
    else
      echo "${EVAL_DATE}_mdu_tau$(tau_slug "${tau}")_${EVAL_RUN_VERSION}"
    fi
  else
    echo "${EVAL_DATE}_tau$(tau_slug "${tau}")_${EVAL_RUN_VERSION}"
  fi
}

checkpoint_dir_for_run() {
  echo "${CHECKPOINTS_ROOT}/$(checkpoint_name_for_run "$1" "$2" "$3")"
}

eval_run_root_for_run() {
  echo "${EVAL_OUTPUTS_ROOT}/$(eval_experiment_for_run "$1" "$2" "$3")/$(eval_run_id_for_run "$1" "$2" "$3")"
}

configure_gpus_for_anchor() {
  local anchor="$1"
  case "$(anchor_slug "${anchor}")" in
    frozen|ema|presftcond)
      export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_FROZEN}"
      export REF_DEVICE="${REF_DEVICE_FROZEN}"
      export DISABLE_DP="${DISABLE_DP_FROZEN}"
      ;;
    cfg)
      export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_TRAINABLE}"
      export REF_DEVICE="${REF_DEVICE_TRAINABLE}"
      export DISABLE_DP="${DISABLE_DP_TRAINABLE}"
      ;;
    *)
      log "ERROR: unknown anchor source: ${anchor}"
      exit 1
      ;;
  esac
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
  if [[ "$(anchor_slug "${NULL_ANCHOR_SOURCE}")" == "presftcond" ]]; then
    local ref_path="${REF_MODEL_NAME_OR_PATH:-${LLADA_PRE_SFT_REF}}"
    if [[ -d "${ref_path}" || -f "${ref_path}/model.safetensors" || -f "${ref_path}/model.safetensors.index.json" ]]; then
      if ! checkpoint_has_weights "${ref_path}"; then
        log "ERROR: pre_sft_cond ref has no local weights: ${ref_path}"
        missing=1
      fi
    else
      log "pre_sft_cond ref will load from Hugging Face: ${ref_path}"
    fi
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
  log "match_mode:        ${MATCH_MODE}"
  log "null_anchor_src:   ${NULL_ANCHOR_SOURCE}"
  log "null_prompt_mode:  ${NULL_PROMPT_MODE}"
  if [[ "$(anchor_slug "${NULL_ANCHOR_SOURCE}")" == "ema" ]]; then
    log "ema_decay:         ${NULL_ANCHOR_EMA_DECAY} (slug: ${ANCHOR_SLUG})"
  fi
  log "τ values:          ${TAUS[*]}  (${#TAUS[@]} runs)"
  log ""
  log "── Training ──"
  log "Base model:        ${LLADA_BASE_SFT}"
  log "novel_percentile:  ${NOVEL_PERCENTILE} (token_id/position only)"
  log "epochs / lr:       ${EPO} / ${LR}"
  log "batch × accum:     ${PER_DEVICE_BATCH} × ${GRAD_ACCUM}"
  log "grad checkpoint:   ${GRADIENT_CHECKPOINTING}"
  log "GPUs (this run):   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} ref_device=${REF_DEVICE:-?}"
  log ""
  log "── Per-τ paths ──"
  local tau
  for tau in "${TAUS[@]}"; do
    if ! past_start_tau "${tau}"; then
      log "  τ=${tau}  SKIP (before START_FROM_TAU)"
      continue
    fi
    log "  τ=${tau}"
    log "    ckpt: $(checkpoint_dir_for_run "${tau}" "${MATCH_MODE}" "${NULL_ANCHOR_SOURCE}")"
    log "    eval: $(eval_run_root_for_run "${tau}" "${MATCH_MODE}" "${NULL_ANCHOR_SOURCE}")"
  done
  log "════════════════════════════════════════════════════════════════"
}

post_eval_validate() {
  local tau="$1" match="$2" anchor="$3"
  local run_root
  run_root="$(eval_run_root_for_run "${tau}" "${match}" "${anchor}")"
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
  log "Post-eval OK τ=${tau} match=${match} anchor=${anchor}"
}

# ── Unlearning ────────────────────────────────────────────────────────────────

run_unlearn() {
  local tau="$1" match="$2" anchor="$3"
  local ckpt_name ckpt_dir log_file
  ckpt_name="$(checkpoint_name_for_run "${tau}" "${match}" "${anchor}")"
  ckpt_dir="$(checkpoint_dir_for_run "${tau}" "${match}" "${anchor}")"
  log_file="${UNLEARN_LOG_DIR}/unlearn_${ckpt_name}.log"

  if [[ "${SKIP_UNLEARN_IF_CKPT}" == "1" ]] && checkpoint_has_weights "${ckpt_dir}"; then
    log "Skip unlearn τ=${tau}: ckpt exists at ${ckpt_dir}"
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

  local -a traj_args=()
  if [[ "${match}" != "random" ]]; then
    traj_args=(--novel_percentile "${NOVEL_PERCENTILE}")
  fi

  local -a gc_args=()
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    gc_args=(
      --gradient_checkpointing
      --gradient_checkpointing_kwargs '{"use_reentrant":false}'
    )
  fi

  local -a ref_args=()
  if [[ -n "${REF_MODEL_NAME_OR_PATH}" ]]; then
    ref_args=(--ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}")
  elif [[ "$(anchor_slug "${anchor}")" == "presftcond" ]]; then
    ref_args=(--ref_model_name_or_path "${LLADA_PRE_SFT_REF}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] would run accelerate launch → ${log_file}"
    return 0
  fi

  "${ACCELERATE}" launch --num_processes 1 "${UNLEARN_SCRIPT}" \
    --model_name_or_path "${LLADA_BASE_SFT}" \
    "${ref_args[@]}" \
    --checkpoints_root "${CHECKPOINTS_ROOT}" \
    --checkpoint_name "${ckpt_name}" \
    --tofu_split forget10 \
    --retain_tofu_split retain_perturbed \
    --hf_dataset locuslab/TOFU \
    --loss_type null_anchor \
    --match_mode "${match}" \
    --null_anchor_source "${anchor}" \
    --null_anchor_ema_decay "${NULL_ANCHOR_EMA_DECAY}" \
    --null_prompt_mode "${NULL_PROMPT_MODE}" \
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
    --disable_data_parallel "${DISABLE_DP}" \
    "${traj_args[@]}" \
    "${gc_args[@]}" \
    "${wandb_args[@]}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  wait_pid "${pid}" "unlearn τ=${tau} match=${match} anchor=${anchor}"

  if ! checkpoint_has_weights "${ckpt_dir}"; then
    log "ERROR: no weights in ${ckpt_dir} (see ${log_file})"
    exit 1
  fi
}

# ── Eval (pipelined) ──────────────────────────────────────────────────────────

run_eval_split() {
  local gpu="$1"
  local model="$2"
  local experiment="$3"
  local run_id="$4"
  local split="$5"
  local log_suffix="$6"

  local log_file="${SWEEP_LOG_DIR}/eval_${log_suffix}.log"
  log "Eval ${split} on GPU ${gpu} → ${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
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

_eval_track() {
  local pid="$1" gpu="$2" split="$3"
  local -n _active=$4
  local -n _gpu_map=$5
  local -n _split_map=$6
  _active+=("${pid}")
  _gpu_map["${pid}"]="${gpu}"
  _split_map["${pid}"]="${split}"
}

run_evals_pipelined() {
  local tau="$1" match="$2" anchor="$3"
  local model experiment run_id slug
  model="$(checkpoint_dir_for_run "${tau}" "${match}" "${anchor}")"
  experiment="$(eval_experiment_for_run "${tau}" "${match}" "${anchor}")"
  run_id="$(eval_run_id_for_run "${tau}" "${match}" "${anchor}")"
  slug="$(checkpoint_name_for_run "${tau}" "${match}" "${anchor}")"

  local need_forget=1 need_retain=1 need_wf=1 need_ra=1
  local run_root
  run_root="$(eval_run_root_for_run "${tau}" "${match}" "${anchor}")"

  if [[ "${SKIP_EVAL_IF_DONE}" == "1" ]]; then
    eval_split_done "${run_root}" forget10 400 && need_forget=0
    eval_split_done "${run_root}" retain_perturbed 400 && need_retain=0
    eval_split_done "${run_root}" world_facts 117 && need_wf=0
    eval_split_done "${run_root}" real_authors 100 && need_ra=0
  fi

  if [[ "${need_forget}" == "0" && "${need_retain}" == "0" && "${need_wf}" == "0" && "${need_ra}" == "0" ]]; then
    log "Skip eval: all splits done under ${run_root}"
    post_eval_validate "${tau}" "${match}" "${anchor}"
    return 0
  fi

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
  }

  local seeds=$((need_forget + need_retain))
  if [[ "${DRY_RUN}" == "1" ]]; then
    [[ "${need_forget}" == "1" ]] && _eval_launch 0 forget10 "${slug}_forget10"
    [[ "${need_retain}" == "1" ]] && _eval_launch 1 retain_perturbed "${slug}_retain"
    return 0
  fi

  if [[ "${seeds}" -eq 0 ]]; then
    [[ "${need_wf}" == "1" ]] && _eval_launch 0 world_facts "${slug}_world_facts"
    [[ "${need_ra}" == "1" ]] && _eval_launch 1 real_authors "${slug}_real_authors"
  else
    [[ "${need_forget}" == "1" ]] && _eval_launch 0 forget10 "${slug}_forget10"
    [[ "${need_retain}" == "1" ]] && _eval_launch 1 retain_perturbed "${slug}_retain"
  fi

  local completions=0
  while ((${#active_pids[@]} > 0)); do
    local finished gpu split
    if ! _eval_wait_any active_pids finished; then
      exit 1
    fi
    gpu="${pid_gpu[${finished}]:-?}"
    split="${pid_split[${finished}]:-?}"
    _eval_remove_pid "${finished}" active_pids
    completions=$((completions + 1))
    log "Eval ${split} finished (gpu=${gpu}, #${completions})"
    if [[ "${completions}" -eq 1 && "${need_wf}" == "1" ]]; then
      _eval_launch "${gpu}" world_facts "${slug}_world_facts"
    elif [[ "${completions}" -eq 2 && "${need_ra}" == "1" ]]; then
      _eval_launch "${gpu}" real_authors "${slug}_real_authors"
    fi
  done

  if ! all_evals_done "${run_root}"; then
    log "ERROR: eval incomplete under ${run_root}"
    exit 1
  fi
  post_eval_validate "${tau}" "${match}" "${anchor}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

validate_prerequisites
configure_gpus_for_anchor "${NULL_ANCHOR_SOURCE}"
print_sweep_plan

for tau in "${TAUS[@]}"; do
  if ! past_start_tau "${tau}"; then
    log "Skip τ=${tau} (before START_FROM_TAU=${START_FROM_TAU})"
    continue
  fi

  log ""
  log "════════════════════════════════════════════════════════════════"
  log "START τ=${tau}  match=${MATCH_MODE}  anchor=${NULL_ANCHOR_SOURCE}"
  log "════════════════════════════════════════════════════════════════"

  run_unlearn "${tau}" "${MATCH_MODE}" "${NULL_ANCHOR_SOURCE}"
  run_evals_pipelined "${tau}" "${MATCH_MODE}" "${NULL_ANCHOR_SOURCE}"

  log "COMPLETE τ=${tau}"
done

log ""
log "MDU τ sweep finished (match=${MATCH_MODE} anchor=${NULL_ANCHOR_SOURCE})."
log "Master log: ${SWEEP_MASTER_LOG}"
