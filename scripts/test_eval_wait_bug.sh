#!/usr/bin/env bash
# Validate eval-wait fix for run_mdu_tau_sweep.sh
#
# Root causes:
#   1. pid=$(run_eval_split ...) — background job starts in $(...) subshell;
#      parent cannot wait on it.
#   2. finished=$(_eval_wait_any ...) — wait runs inside $(...) subshell too.
#   3. wait -n — reaps unrelated children (e.g. tee from master log).
#
# Usage: bash scripts/test_eval_wait_bug.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${ROOT}/sweep_logs}"
mkdir -p "${SWEEP_LOG_DIR}"

run_one() {
  local mode="$1"
  local log="${SWEEP_LOG_DIR}/test_eval_wait_${mode}.log"
  : > "${log}"

  (
    exec > >(tee -a "${log}") 2>&1
    echo "[test] MODE=${mode} start"

    _job() { sleep "$1"; }

    _remove_pid() {
      local target="$1"
      local -n _arr=$2
      local kept=()
      local p
      for p in "${_arr[@]}"; do
        [[ "${p}" == "${target}" ]] || kept+=("${p}")
      done
      _arr=("${kept[@]}")
    }

    _wait_broken() {
      local -n _pids=$1
      local -n _out=$2
      local finished=""
      if ! wait -n -p finished; then
        return 1
      fi
      _out="${finished}"
    }

    _wait_fixed() {
      local -n _pids=$1
      local -n _out=$2
      local pid ec
      while true; do
        for pid in "${_pids[@]}"; do
          if ! kill -0 "${pid}" 2>/dev/null; then
            ec=0
            wait "${pid}" 2>/dev/null || ec=$?
            ((ec == 0)) || return 1
            _out="${pid}"
            return 0
          fi
        done
        sleep 0.2
      done
    }

    _launch_broken() {
      local secs="$1"
      local -n _active=$2
      local pid
      pid=$(_job "${secs}" & echo $!)
      _active+=("${pid}")
      echo "[test] broken launch pid=${pid} sleep=${secs}"
    }

    _launch_fixed() {
      local secs="$1"
      local -n _active=$2
      _job "${secs}" &
      local pid=$!
      _active+=("${pid}")
      echo "[test] fixed launch pid=${pid} sleep=${secs}"
    }

    local -a active=()
    local completions=0 finished

    if [[ "${mode}" == "broken" ]]; then
      _launch_broken 2 active
      _launch_broken 4 active
      while ((${#active[@]} > 0)); do
        if ! _wait_broken active finished; then
          echo "[test] FAIL broken: wait -n failed at completion ${completions}"
          exit 1
        fi
        if [[ ! " ${active[*]} " =~ " ${finished} " ]]; then
          echo "[test] FAIL broken: wait -n reaped untracked pid=${finished} (active: ${active[*]})"
          exit 1
        fi
        _remove_pid "${finished}" active
        completions=$((completions + 1))
        echo "[test] broken completion #${completions} pid=${finished}"
        [[ "${completions}" -eq 1 ]] && _launch_broken 1 active
        [[ "${completions}" -eq 2 ]] && _launch_broken 1 active
      done
      echo "[test] UNEXPECTED broken passed"
      exit 1
    else
      _launch_fixed 2 active
      _launch_fixed 4 active
      while ((${#active[@]} > 0)); do
        if ! _wait_fixed active finished; then
          echo "[test] FAIL fixed: wait failed at completion ${completions}"
          exit 1
        fi
        _remove_pid "${finished}" active
        completions=$((completions + 1))
        echo "[test] fixed completion #${completions} pid=${finished}"
        [[ "${completions}" -eq 1 ]] && _launch_fixed 1 active
        [[ "${completions}" -eq 2 ]] && _launch_fixed 1 active
      done
      echo "[test] PASS fixed: ${completions} completions (pipelined 2∥4 → 1 → 1)"
    fi
  )
}

MODE=${1:-all}
FAIL=0

if [[ "${MODE}" == "all" || "${MODE}" == "broken" ]]; then
  echo "=== Reproducing broken pattern ==="
  if run_one broken; then
    echo "broken test: unexpected success"
    FAIL=1
  else
    echo "broken test: OK (failed as expected)"
  fi
fi

if [[ "${MODE}" == "all" || "${MODE}" == "fixed" ]]; then
  echo "=== Validating fixed pattern ==="
  if run_one fixed; then
    echo "fixed test: OK"
  else
    echo "fixed test: FAIL"
    FAIL=1
  fi
fi

# Extra: broken launch + fixed wait still fails (orphaned jobs)
if [[ "${MODE}" == "all" || "${MODE}" == "orphan" ]]; then
  echo "=== Orphan launch + fixed wait (should fail) ==="
  orphan_log="${SWEEP_LOG_DIR}/test_eval_wait_orphan.log"
  : > "${orphan_log}"
  if (
    exec > >(tee -a "${orphan_log}") 2>&1
    _job() { sleep 1; }
    _wait_fixed() {
      local -n _pids=$1
      local -n _out=$2
      local pid ec
      while true; do
        for pid in "${_pids[@]}"; do
          if ! kill -0 "${pid}" 2>/dev/null; then
            ec=0
            wait "${pid}" 2>/dev/null || ec=$?
            ((ec == 0)) || return 1
            _out="${pid}"
            return 0
          fi
        done
        sleep 0.2
      done
    }
    active=()
    pid= finished=
    pid=$(_job 1 & echo $!)
    active+=("${pid}")
    echo "[test] orphan pid=${pid}"
    if _wait_fixed active finished; then
      echo "[test] UNEXPECTED orphan wait succeeded"
      exit 1
    fi
    echo "[test] PASS orphan: fixed wait correctly fails on subshell-launched job"
  ); then
    echo "orphan test: OK"
  else
    echo "orphan test: FAIL"
    FAIL=1
  fi
fi

exit "${FAIL}"
