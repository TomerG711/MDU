#!/usr/bin/env bash
# MDU runner for both backbones / both benchmarks.
#
# Usage:
#   bash run_main.sh tofu_llada   <tau>   [checkpoint_name]
#   bash run_main.sh tofu_dream   <tau>   [checkpoint_name]
#   bash run_main.sh rwku_dream   <tau>   [checkpoint_name]   <subject>
#
# Checkpoints are written under ./checkpoints/<checkpoint_name>/ (auto-named if omitted).
# TOFU data loads from Hugging Face by default (locuslab/TOFU).
#
# W&B (optional):
#   export WANDB_API_KEY=...   # or run: wandb login
#   WANDB=1 bash run_main.sh tofu_llada 0.5
#   WANDB_PROJECT=my-project WANDB=1 bash run_main.sh ...
#
#   REF_DEVICE=auto|same|cuda:1   # frozen ref_model GPU (default: auto → cuda:1 if 2 GPUs)
#   REF_DEVICE=same bash run_main.sh tofu_llada 0.5   # disable split, colocate both models
#   MATCH_MODE=random|token_id|position   # default: random
#   NOVEL_PERCENTILE=100                    # for token_id/position (upstream TOFU)
#   NULL_ANCHOR_SOURCE=auto|frozen_sft|trainable_cfg  # uncond anchor (default: auto = upstream)
#   CUDA_DEVICES=0     # single GPU when NULL_ANCHOR_SOURCE=trainable_cfg (no ref load)
#   DISABLE_DP=auto|yes|no        # disable HF DataParallel (default: auto when ref is split)
#   CUDA_VISIBLE_DEVICES=0,1      # required for ref split (train on 0, ref on 1)
#
# Examples:
#   LR=1e-5 EPO=9 bash run_main.sh tofu_llada 0.5
#   LR=1e-5 EPO=5 WANDB=1 bash run_main.sh tofu_dream 0.5 mdu_dream_tau0p5

set -uo pipefail
PRESET=${1:?usage: tofu_llada / tofu_dream / rwku_dream}
TAU=${2:?tau (e.g. 0, 0.25, 0.5, 0.75, 1)}
CKPT_NAME=${3:-}
SUBJECT=${4:-}            # only required for rwku_dream

LR=${LR:?LR env var required}
EPO=${EPO:?EPO env var required}

# >>> EDIT THESE PATHS FOR YOUR SETUP <<<
LLADA_BASE_SFT=${LLADA_BASE_SFT:-./checkpoints/LLaDA-8B-Instruct-full-SFT-TOFU}
DREAM_BASE_SFT=${DREAM_BASE_SFT:-/path/to/Dream-TOFU-SFT-300ep}
DREAM_BASE=${DREAM_BASE:-Dream-org/Dream-v0-Instruct-7B}
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-./checkpoints}
REF_DEVICE=${REF_DEVICE:-auto}
NULL_ANCHOR_SOURCE=${NULL_ANCHOR_SOURCE:-auto}
DISABLE_DP=${DISABLE_DP:-auto}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
WANDB_PROJECT=${WANDB_PROJECT:-unlearning-dllms-MDU}
RWKU_DATA_ROOT=${RWKU_DATA_ROOT:-./data/rwku/dream_subset}
RETAIN_PLACEHOLDER=${RETAIN_PLACEHOLDER:-./data/rwku_retain_placeholder.jsonl}

WANDB_ARGS=()
if [ "${WANDB:-0}" = 1 ]; then
    WANDB_ARGS=(--report_to wandb --wandb_project "$WANDB_PROJECT")
fi

CKPT_ARGS=(--checkpoints_root "$CHECKPOINTS_ROOT")
if [ -n "$CKPT_NAME" ]; then
    CKPT_ARGS+=(--checkpoint_name "$CKPT_NAME")
fi

MDU_ARGS=(
    --loss_type null_anchor
    --match_mode "${MATCH_MODE:-random}"
    --alpha 1.0
    --null_anchor_tau "$TAU"
    --null_anchor_eta 0.0
    --null_anchor_kl_dir forward
    --denoise_steps 128
    --max_new_tokens 128
    --tofu_split forget10
    --retain_tofu_split retain_perturbed
    --hf_dataset locuslab/TOFU
    --ref_device "$REF_DEVICE"
    --null_anchor_source "$NULL_ANCHOR_SOURCE"
    --disable_data_parallel "$DISABLE_DP"
)

# Upstream TOFU uses novel_percentile=100 for trajectory modes (token_id/position)
if [ "${MATCH_MODE:-random}" != "random" ]; then
    MDU_ARGS+=(--novel_percentile "${NOVEL_PERCENTILE:-100}")
fi

case "$PRESET" in
    tofu_llada)
        accelerate launch --num_processes 1 src/unlearn_mdu_llada.py \
            --model_name_or_path "$LLADA_BASE_SFT" \
            "${CKPT_ARGS[@]}" \
            "${MDU_ARGS[@]}" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
            --save_strategy no \
            "${WANDB_ARGS[@]}"
        ;;

    tofu_dream)
        accelerate launch --num_processes 1 src/unlearn_mdu_dream.py \
            --model_name_or_path "$DREAM_BASE_SFT" \
            "${CKPT_ARGS[@]}" \
            "${MDU_ARGS[@]}" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
            --save_strategy no \
            "${WANDB_ARGS[@]}"
        ;;

    rwku_dream)
        [ -z "$SUBJECT" ] && { echo "rwku_dream requires <subject> as the 4th argument"; exit 1; }
        FORGET="$RWKU_DATA_ROOT/$SUBJECT/forget.jsonl"
        accelerate launch --num_processes 1 src/unlearn_mdu_dream.py \
            --model_name_or_path "$DREAM_BASE" \
            --tofu_path "$FORGET" \
            --retain_path "$RETAIN_PLACEHOLDER" \
            --tofu_split "" \
            --retain_tofu_split "" \
            "${CKPT_ARGS[@]}" \
            --num_train_epochs "$EPO" --learning_rate "$LR" \
            --per_device_train_batch_size 4 --save_strategy no \
            --alpha 0.0 \
            --loss_type null_anchor --match_mode random \
            --null_anchor_tau "$TAU" --null_anchor_eta 0.0 \
            --null_anchor_kl_dir forward \
            --denoise_steps 128 --max_new_tokens 128 \
            "${WANDB_ARGS[@]}"
        ;;

    *)
        echo "Unknown preset: $PRESET"
        echo "Use: tofu_llada / tofu_dream / rwku_dream"
        exit 1
        ;;
esac
