#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/touch_live_da3_small_geometry_original_stride4_stable_w5_p1000.yaml}"
MANIFEST="${MANIFEST:-/data/baseAVIs_432x672/manifest_original_no307_no321_frames_stride2_stable_w3.json}"
BASE_RUN="${BASE_RUN:-outputs/touch_live_da3_small_geometry_original_stride2_stable_w3_no307_no321_freshsplit}"
START_CKPT="${START_CKPT:-$BASE_RUN/best_val_loss.pt}"
LOCATION_OUT="${LOCATION_OUT:-outputs/touch_live_da3_small_geometry_original_stride2_stable_w3_no307_no321_freshsplit_xy_head_only}"
TOUCH_OUT="${TOUCH_OUT:-outputs/touch_live_da3_small_geometry_original_stride2_stable_w3_no307_no321_freshsplit_touch_head_only_from_best_xy}"
EPOCHS="${EPOCHS:-6}"

cd "$ROOT"

echo "Stage 1: location head only"
"$PYTHON" scripts/train_student.py \
  --config "$CONFIG" \
  --override \
    data.manifest_path="$MANIFEST" \
    output_dir="$LOCATION_OUT" \
    training.init_checkpoint="$START_CKPT" \
    training.epochs="$EPOCHS" \
    training.resume=null \
    training.copy_manifest=true \
    model.train_touch=false \
    model.train_state=false \
    model.train_location=true \
    model.freeze_backbone=true \
    model.freeze_neck=true \
    model.freeze_depth_head=true \
    model.freeze_state_head=true \
    model.freeze_location_head=false \
    loss.lambda_touch=0.0 \
    loss.lambda_state=0.0 \
    loss.lambda_xy=1.0 \
    loss.lambda_heatmap=0.5 \
    optimizer.backbone_lr=0.0 \
    optimizer.head_lr=0.0002

echo "Stage 2: touch head only, initialized from Stage 1 best_xy.pt"
"$PYTHON" scripts/train_student.py \
  --config "$CONFIG" \
  --override \
    data.manifest_path="$MANIFEST" \
    output_dir="$TOUCH_OUT" \
    training.init_checkpoint="$LOCATION_OUT/best_xy.pt" \
    training.epochs="$EPOCHS" \
    training.resume=null \
    training.copy_manifest=true \
    model.train_touch=true \
    model.train_state=false \
    model.train_location=true \
    model.freeze_backbone=true \
    model.freeze_neck=true \
    model.freeze_depth_head=true \
    model.freeze_state_head=false \
    model.freeze_location_head=true \
    loss.lambda_touch=2.0 \
    loss.lambda_state=0.0 \
    loss.lambda_xy=0.0 \
    loss.lambda_heatmap=0.0 \
    optimizer.backbone_lr=0.0 \
    optimizer.head_lr=0.0002

echo "Done."
echo "Location best: $LOCATION_OUT/best_xy.pt"
echo "Touch best:    $TOUCH_OUT/best_touch.pt"
