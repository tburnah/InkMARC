#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/touch_live_da3_small_geometry_original_stride4_stable_w5_p1000.yaml}"
MANIFEST="${MANIFEST:-/data/baseAVIs_432x672/manifest_original_no307_no321_frames_stride2_stable_w3.json}"
BASE_CKPT="${BASE_CKPT:-outputs/touch_live_da3_small_geometry_original_stride2_stable_w3_no307_no321_freshsplit_touch_head_only_from_best_xy/best_touch.pt}"
ROOT_OUT="${ROOT_OUT:-outputs/location_tuning_experiments_stride2_no307_no321}"
MAX_STEPS="${MAX_STEPS:-2000}"

cd "$ROOT"
mkdir -p "$ROOT_OUT"

run_location_variant() {
  local name="$1"
  local epochs="$2"
  local head_lr="$3"
  local lambda_xy="$4"
  local lambda_heatmap="$5"
  local xy_target="$6"
  local out_dir="$ROOT_OUT/$name"
  local report_dir="$out_dir/test_report"
  local hyst_dir="$out_dir/hysteresis_test_sweep"

  echo "=== train $name ==="
  "$PYTHON" scripts/train_student.py \
    --config "$CONFIG" \
    --override \
      data.manifest_path="$MANIFEST" \
      output_dir="$out_dir" \
      training.init_checkpoint="$BASE_CKPT" \
      training.epochs="$epochs" \
      training.max_steps_per_epoch="$MAX_STEPS" \
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
      loss.lambda_xy="$lambda_xy" \
      loss.lambda_heatmap="$lambda_heatmap" \
      loss.xy_target="$xy_target" \
      optimizer.backbone_lr=0.0 \
      optimizer.head_lr="$head_lr"

  echo "=== evaluate $name ==="
  "$PYTHON" scripts/evaluate_touch_location_report.py \
    --config "$CONFIG" \
    --checkpoint "$out_dir/best_xy.pt" \
    --split test \
    --output-dir "$report_dir" \
    --raw-threshold 0.28 \
    --hyst-on 0.60 \
    --hyst-off 0.10 \
    --hyst-min-on 1 \
    --hyst-min-off 1 \
    --override \
      data.manifest_path="$MANIFEST" \
      output_dir="$out_dir" \
      model.train_touch=true \
      model.train_state=false \
      model.train_location=true

  "$PYTHON" scripts/evaluate_touch_hysteresis.py \
    --predictions "$report_dir/touch_location_frame_predictions.csv" \
    --output-dir "$hyst_dir" \
    --on-thresholds 0.10:0.80:0.05 \
    --off-thresholds 0.05:0.50:0.05 \
    --min-on-samples 1,2 \
    --min-off-samples 1,2,3
}

run_location_variant "probe_xy_touch_only_lr1e-4_hm0p5_e2_s${MAX_STEPS}" 2 0.0001 1.0 0.5 touch
run_location_variant "probe_xy_touch_only_lr5e-5_hm0p5_e2_s${MAX_STEPS}" 2 0.00005 1.0 0.5 touch
run_location_variant "probe_xy_touch_only_lr1e-4_hm1p0_e2_s${MAX_STEPS}" 2 0.0001 1.0 1.0 touch
