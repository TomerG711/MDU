#!/usr/bin/env bash
# MDU runner for both backbones / both benchmarks.
#
# Usage:
#   bash run_main.sh tofu_llada   <tau>   <output_dir>
#   bash run_main.sh tofu_dream   <tau>   <output_dir>
#   bash run_main.sh rwku_dream   <tau>   <output_dir>   <subject>
#
# Hyperparameters (learning rate, num_train_epochs) are taken from
# environment variables LR and EPO; pick the values from the paper's
# experiment-setup section (see configs/*.yaml for the values we used).
#
# Examples:
#   LR=8e-6 EPO=9 bash run_main.sh tofu_llada 0.5 ./outputs/llada_tofu_tau0p5
#   LR=1e-5 EPO=3 bash run_main.sh rwku_dream 0.5 ./outputs/dream_rwku_tau0p5_SK 1_Stephen_King

set -uo pipefail
PRESET=${1:?usage: tofu_llada / tofu_dream / rwku_dream}
TAU=${2:?tau (e.g. 0, 0.25, 0.5, 0.75, 1)}
OUT=${3:?output_dir}
SUBJECT=${4:-}            # only required for rwku_dream

LR=${LR:?LR env var required}
EPO=${EPO:?EPO env var required}

# >>> EDIT THESE PATHS FOR YOUR SETUP <<<
LLADA_BASE_SFT=${LLADA_BASE_SFT:-/path/to/LLaDA-TOFU-SFT-1000ep}
DREAM_BASE_SFT=${DREAM_BASE_SFT:-/path/to/Dream-TOFU-SFT-300ep}
DREAM_BASE=${DREAM_BASE:-Dream-org/Dream-v0-Instruct-7B}
TOFU_FORGET_PATH=${TOFU_FORGET_PATH:-./data/tofu/forget10.json}
TOFU_RETAIN_PATH=${TOFU_RETAIN_PATH:-./data/tofu/retain_perturbed.json}
RWKU_DATA_ROOT=${RWKU_DATA_ROOT:-./data/rwku/dream_subset}
RETAIN_PLACEHOLDER=${RETAIN_PLACEHOLDER:-./data/rwku_retain_placeholder.jsonl}

case "$PRESET" in
    tofu_llada)
        accelerate launch --num_processes 1 src/unlearn_mdu_llada.py \
            --model_name_or_path "$LLADA_BASE_SFT" \
            --tofu_path "$TOFU_FORGET_PATH" --retain_path "$TOFU_RETAIN_PATH" \
            --output_dir "$OUT" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
            --save_strategy no \
            --alpha 1.0 \
            --null_anchor_tau "$TAU" --null_anchor_eta 0.0 \
            --null_anchor_kl_dir forward \
            --denoise_steps 128 --max_new_tokens 128 \
            --novel_percentile 100
        ;;

    tofu_dream)
        accelerate launch --num_processes 1 src/unlearn_mdu_dream.py \
            --model_name_or_path "$DREAM_BASE_SFT" \
            --tofu_path "$TOFU_FORGET_PATH" --retain_path "$TOFU_RETAIN_PATH" \
            --output_dir "$OUT" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
            --save_strategy no \
            --alpha 1.0 \
            --null_anchor_tau "$TAU" --null_anchor_eta 0.0 \
            --null_anchor_kl_dir forward \
            --denoise_steps 128 --max_new_tokens 128 \
            --novel_percentile 100
        ;;

    rwku_dream)
        [ -z "$SUBJECT" ] && { echo "rwku_dream requires <subject> as the 4th argument"; exit 1; }
        FORGET="$RWKU_DATA_ROOT/$SUBJECT/forget.jsonl"
        accelerate launch --num_processes 1 src/unlearn_mdu_dream.py \
            --model_name_or_path "$DREAM_BASE" \
            --tofu_path "$FORGET" --retain_path "$RETAIN_PLACEHOLDER" \
            --output_dir "$OUT" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --save_strategy no \
            --alpha 0.0 --no_retain \
            --null_anchor_tau "$TAU" --null_anchor_eta 0.0 \
            --null_anchor_kl_dir forward \
            --denoise_steps 128 --max_new_tokens 128 \
            --novel_percentile 100
        ;;

    *)
        echo "Unknown preset: $PRESET"
        echo "Use: tofu_llada / tofu_dream / rwku_dream"
        exit 1
        ;;
esac
