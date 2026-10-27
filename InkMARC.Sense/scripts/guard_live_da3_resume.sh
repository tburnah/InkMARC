#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 PID OUTPUT_DIR" >&2
  exit 2
fi

PID="$1"
OUT="$(realpath "$2")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python"
LOG="$OUT/tmux_resume.log"
MANIFEST="/data/baseAVIs_432x672/manifest_texture_v3_balanced_frames_stable_w5_p1000_no307_depth_outliers.json"
CONFIG="$ROOT/configs/touch_live_da3_small_original_stride4_stable_w5_p1000.yaml"

epoch_rows() {
  if [ -f "$OUT/metrics.csv" ]; then
    awk 'END { if (NR > 0) print NR - 1; else print 0 }' "$OUT/metrics.csv"
  else
    echo 0
  fi
}

mkdir -p "$OUT"
echo "[$(date)] guard watching PID $PID" >> "$LOG"

while kill -0 "$PID" 2>/dev/null; do
  rows="$(epoch_rows)"
  echo "[$(date)] current run alive; metrics rows=$rows" >> "$LOG"
  sleep 60
done

rows="$(epoch_rows)"
echo "[$(date)] watched PID exited; metrics rows=$rows" >> "$LOG"

if [ "$rows" -lt 10 ]; then
  echo "[$(date)] resuming from $OUT/latest.pt" >> "$LOG"
  exec "$PY" "$ROOT/scripts/train_student.py" \
    --config "$CONFIG" \
    --override \
    "output_dir=$OUT" \
    "training.epochs=10" \
    "training.batch_size=8" \
    "training.num_workers=4" \
    "training.resume=$OUT/latest.pt" \
    "data.manifest_path=$MANIFEST" \
    "data.variant_kind_filter=null" \
    >> "$LOG" 2>&1
fi

echo "[$(date)] training already complete; no resume needed" >> "$LOG"
