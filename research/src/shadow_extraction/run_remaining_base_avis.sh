#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/baseAVIs"
FOCUSED_DIR="/data/LocationData/focused"
TEXTURES_DIR="/home/t/tburnah/src/shadow_extraction/textures"
SCRIPT="/home/t/tburnah/src/shadow_extraction/test_shadow_extraction.py"
REFERENCE="/home/t/tburnah/src/shadow_extraction/TagsBorderless.png"
CONDA_SH="/home/t/tburnah/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="opencv_cuda"
START_AFTER="participant_209_T4b_1775613418638.avi"
REFERENCE_SCALE="0.35"
PROGRESS_INTERVAL="100"
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: run_remaining_base_avis.sh [--dry-run] [--force]

Runs shadow extraction for every /data/baseAVIs participant_*.avi after:
  participant_209_T4b_1775613418638.avi

Outputs one AVI per texture into:
  /data/LocationData/focused

Options:
  --dry-run   Print planned jobs without running them.
  --force     Re-run jobs even if all expected texture outputs already exist.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda setup: $CONDA_SH" >&2
  exit 1
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

mkdir -p "$FOCUSED_DIR"

mapfile -t TEXTURES < <(
  find "$TEXTURES_DIR" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) \
    -printf '%f\n' | sort
)

if [[ ${#TEXTURES[@]} -eq 0 ]]; then
  echo "No texture files found in $TEXTURES_DIR" >&2
  exit 1
fi

mapfile -t INPUTS < <(
  find "$BASE_DIR" -maxdepth 1 -type f -name 'participant_*.avi' -printf '%f\n' | sort -V
)

start_seen=0
planned=()
for name in "${INPUTS[@]}"; do
  if [[ "$start_seen" -eq 0 ]]; then
    if [[ "$name" == "$START_AFTER" ]]; then
      start_seen=1
    fi
    continue
  fi
  planned+=("$name")
done

if [[ "$start_seen" -eq 0 ]]; then
  echo "Start marker not found in $BASE_DIR: $START_AFTER" >&2
  exit 1
fi

echo "Textures: ${#TEXTURES[@]}"
echo "Jobs after $START_AFTER: ${#planned[@]}"

job_complete() {
  local input_name="$1"
  local stem="${input_name%.avi}"
  local texture_name
  local texture_stem
  for texture_name in "${TEXTURES[@]}"; do
    texture_stem="${texture_name%.*}"
    if [[ ! -s "$FOCUSED_DIR/${stem}_${texture_stem}.avi" ]]; then
      return 1
    fi
  done
  return 0
}

for input_name in "${planned[@]}"; do
  input_path="$BASE_DIR/$input_name"
  stem="${input_name%.avi}"
  output_prefix="$FOCUSED_DIR/$stem.avi"

  if [[ "$FORCE" -eq 0 ]] && job_complete "$input_name"; then
    echo "[SKIP] $input_name (all texture outputs exist)"
    continue
  fi

  echo "[RUN] $input_name"
  echo "      output prefix: $output_prefix"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  PYTHONUNBUFFERED=1 python "$SCRIPT" \
    --video "$input_path" \
    --reference "$REFERENCE" \
    --outdir "$FOCUSED_DIR/" \
    --fast \
    --cuda \
    --minimal-outputs \
    --surface-textures-dir "$TEXTURES_DIR" \
    --reference-scale "$REFERENCE_SCALE" \
    --output-video "$output_prefix" \
    --video-progress-interval "$PROGRESS_INTERVAL"
done

echo "[DONE] Remaining base AVI processing complete."
