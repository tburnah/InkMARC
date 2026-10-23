#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/baseAVIs_480x747"
CLEANED_DIR="/data/LocationData/cleaned"
TEXTURES_DIR="/home/t/tburnah/src/shadow_extraction/textures"
REFERENCE="/home/t/tburnah/src/shadow_extraction/TagsBorderless.png"
EXTRACTOR="/home/t/tburnah/src/shadow_extraction/shadow_extract_v3"
PY="/home/t/tburnah/miniconda3/envs/opencv_cuda/bin/python"
DEPTH_ROOT="/home/t/tburnah/src/depth_model/Depth-Anything-V2"
DEPTH_CHECKPOINT="/home/t/tburnah/src/depth_model/depth_anything_v2_vits.pth"
START_AFTER=""
DRY_RUN=0
FORCE=0
TIMING=0
PIPELINE_VIDEO_FORMAT="mkv-ffv1"
WRITE_PREVIEW_MP4=0

usage() {
  cat <<'EOF'
Usage: run_cleaned_base_avis_v3.sh [options]

Runs shadow_extract_v3 for every /data/baseAVIs_480x747/participant_*.avi,
using only each video's verified _corrected.h5 geometry and writing one
final-overlay video per texture into:
  /data/LocationData/cleaned

Options:
  --dry-run             Print planned jobs without running them.
  --force               Re-run jobs even if all expected texture outputs exist.
  --timing              Pass --timing to shadow_extract_v3.
  --pipeline-video-format NAME
                        Generated pipeline video format: legacy, mkv-ffv1. Default: mkv-ffv1.
  --write-preview-mp4   Also write browser-compatible preview MP4s for final overlays.
  --start-after NAME    Skip inputs through NAME, then process later sorted AVIs.
  -h, --help            Show this help.
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
    --timing)
      TIMING=1
      shift
      ;;
    --pipeline-video-format)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --pipeline-video-format" >&2
        exit 2
      fi
      PIPELINE_VIDEO_FORMAT="$2"
      if [[ "$PIPELINE_VIDEO_FORMAT" != "legacy" && "$PIPELINE_VIDEO_FORMAT" != "mkv-ffv1" ]]; then
        echo "--pipeline-video-format must be one of: legacy, mkv-ffv1" >&2
        exit 2
      fi
      shift 2
      ;;
    --write-preview-mp4)
      WRITE_PREVIEW_MP4=1
      shift
      ;;
    --start-after)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --start-after" >&2
        exit 2
      fi
      START_AFTER="$2"
      shift 2
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

for required in "$BASE_DIR" "$TEXTURES_DIR" "$REFERENCE" "$EXTRACTOR" "$PY" "$DEPTH_ROOT" "$DEPTH_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$CLEANED_DIR"

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

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "No participant_*.avi files found in $BASE_DIR" >&2
  exit 1
fi

planned=()
if [[ -n "$START_AFTER" ]]; then
  start_seen=0
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
else
  planned=("${INPUTS[@]}")
fi

job_complete() {
  local input_name="$1"
  local stem="${input_name%.avi}"
  local ext="avi"
  if [[ "$PIPELINE_VIDEO_FORMAT" == "mkv-ffv1" ]]; then
    ext="mkv"
  fi
  local texture_name
  local texture_stem
  for texture_name in "${TEXTURES[@]}"; do
    texture_stem="${texture_name%.*}"
    if [[ ! -s "$CLEANED_DIR/${stem}_${texture_stem}.${ext}" ]]; then
      return 1
    fi
  done
  return 0
}

echo "Input dir: $BASE_DIR"
echo "Output dir: $CLEANED_DIR"
echo "Textures: ${#TEXTURES[@]}"
echo "Planned input AVIs: ${#planned[@]}"
echo "Pipeline video format: $PIPELINE_VIDEO_FORMAT"
if [[ "$WRITE_PREVIEW_MP4" -eq 1 ]]; then
  echo "Preview MP4: yes"
fi
if [[ -n "$START_AFTER" ]]; then
  echo "Start after: $START_AFTER"
fi

for input_name in "${planned[@]}"; do
  input_path="$BASE_DIR/$input_name"

  if [[ "$FORCE" -eq 0 ]] && job_complete "$input_name"; then
    echo "[SKIP] $input_name (all texture outputs exist)"
    continue
  fi

  echo "[RUN] $input_name"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  args=(
    --video "$input_path"
    --reference "$REFERENCE"
    --outdir "$CLEANED_DIR"
    --surface-texture-dir "$TEXTURES_DIR"
    --corrected-sidecar-geometry-only
    --final-overlay-only
    --pipeline-video-format "$PIPELINE_VIDEO_FORMAT"
    --raw-depth-python "$PY"
    --depth-anything-root "$DEPTH_ROOT"
    --depth-model-version v2
    --depth-checkpoint "$DEPTH_CHECKPOINT"
    --depth-encoder vits
    --depth-device cuda
  )
  if [[ "$TIMING" -eq 1 ]]; then
    args+=(--timing)
  fi
  if [[ "$WRITE_PREVIEW_MP4" -eq 1 ]]; then
    args+=(--write-preview-mp4)
  fi

  PYTHONUNBUFFERED=1 "$EXTRACTOR" "${args[@]}"
done

echo "[DONE] Cleaned v3 base AVI processing complete."
