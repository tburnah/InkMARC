#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "usage: $0 NAME INIT_CHECKPOINT STEPS EPOCHS HEAD_LR LAMBDA_XY LAMBDA_HEATMAP XY_TARGET [FREEZE_NECK]" >&2
  exit 2
fi

NAME="$1"
INIT_CHECKPOINT="$2"
STEPS="$3"
EPOCHS="$4"
HEAD_LR="$5"
LAMBDA_XY="$6"
LAMBDA_HEATMAP="$7"
XY_TARGET="$8"
FREEZE_NECK="${9:-true}"

PY="${PY:-/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python}"
CONFIG="${CONFIG:-configs/touch_live_da3_small_geometry_original_stride4_stable_w5_p1000.yaml}"
MANIFEST="${MANIFEST:-/data/baseAVIs_432x672/manifest_original_no307_no321_frames_stride2_stable_w3.json}"
ROOT="${ROOT:-outputs/location_tuning_experiments_stride2_no307_no321}"
OUT="${ROOT}/${NAME}"

mkdir -p "${OUT}"

"${PY}" scripts/train_student.py \
  --config "${CONFIG}" \
  --override \
    data.manifest_path="${MANIFEST}" \
    output_dir="${OUT}" \
    training.init_checkpoint="${INIT_CHECKPOINT}" \
    training.epochs="${EPOCHS}" \
    training.max_steps_per_epoch="${STEPS}" \
    training.val_every=999 \
    model.train_touch=false \
    model.train_state=false \
    model.train_location=true \
    model.freeze_backbone=true \
    model.freeze_neck="${FREEZE_NECK}" \
    model.freeze_depth_head=true \
    model.freeze_state_head=true \
    model.freeze_location_head=false \
    loss.lambda_touch=0.0 \
    loss.lambda_state=0.0 \
    loss.lambda_xy="${LAMBDA_XY}" \
    loss.lambda_heatmap="${LAMBDA_HEATMAP}" \
    loss.xy_target="${XY_TARGET}" \
    optimizer.head_lr="${HEAD_LR}"

"${PY}" scripts/evaluate_touch_location_report.py \
  --config "${CONFIG}" \
  --checkpoint "${OUT}/latest.pt" \
  --split test \
  --output-dir "${OUT}/test_report_latest" \
  --raw-threshold 0.28 \
  --hyst-on 0.60 \
  --hyst-off 0.10 \
  --hyst-min-on 1 \
  --hyst-min-off 1 \
  --override \
    data.manifest_path="${MANIFEST}" \
    output_dir="${OUT}" \
    model.train_touch=true \
    model.train_state=false \
    model.train_location=true
