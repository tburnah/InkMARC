#!/usr/bin/env bash
set -euo pipefail

COMPONENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_DIR="/data/baseAVIs_432x672"
OUT_DIR=""
TEXTURES_DIR="${COMPONENT_ROOT}/textures"
REFERENCE="${COMPONENT_ROOT}/TagsBorderless.png"
EXTRACTOR="${COMPONENT_ROOT}/shadow_extract_v3"
PY="/home/t/tburnah/miniconda3/envs/opencv_cuda/bin/python"
DEPTH_ROOT="/home/t/tburnah/src/depth_model/Depth-Anything-V2"
DEPTH_CHECKPOINT="/home/t/tburnah/src/depth_model/depth_anything_v2_vits.pth"
DEPTH_DEVICE="cuda"
PIPELINE_VIDEO_FORMAT="mp4"
FINAL_OVERLAY_VIDEO_CODEC="h264-high"
GRADIENT_MASK_THRESHOLD=16
TILT_GRADIENT_MASK_THRESHOLD=8
WRITE_PREVIEW_MP4=0
START_AFTER=""
ONLY=""
LIMIT=0
DRY_RUN=0
FORCE=0
TIMING=0

usage() {
  cat <<'EOF'
Usage: run_all_base_avis_texture_replacement_v3.sh [options]

Runs shadow_extract_v3 for participant_*.avi files, replacing the board with
every texture in the textures directory using fast final-overlay production
mode. Geometry is read only from the verified _corrected.h5 sidecar produced
by run_tag_sheet_matching_v3.sh; tag matching is not repeated here.

Defaults:
  Input AVIs:    /data/baseAVIs_432x672
  Textures:      <InkMARC.Augment>/textures
  Output videos: <base-dir>/texture_replacement_v3
  Output format: high-quality H.264 MP4
  Preview MP4s:  disabled

Options:
  --base-dir DIR               Input AVI directory.
  --out-dir DIR                Output directory.
  --textures-dir DIR           Directory of texture images.
  --only NAME                  Process one AVI filename or full path.
  --limit N                    Process only the first N planned AVIs.
  --start-after NAME           Skip inputs through NAME, then process later sorted AVIs.
  --force                      Re-run even when all expected outputs already exist.
  --dry-run                    Print planned commands without running.
  --timing                     Pass --timing to shadow_extract_v3.
  --depth-device DEVICE        cuda, mps, or cpu. Default: cuda.
  --pipeline-video-format NAME mp4, legacy, or mkv-ffv1. Default: mp4.
  --final-overlay-video-codec NAME
                               lossless, h264-high, or mjpeg-high. Default: h264-high.
  --gradient-mask-threshold N  Legacy raw-depth difference threshold. Default: 16.
  --tilt-gradient-mask-threshold N
                               Tilt raw-depth difference threshold. Default: 8.
  --write-preview-mp4          Also write lower-quality browser preview MP4s.
  --no-preview-mp4             Do not write preview MP4s. Default.
  -h, --help                   Show this help.
EOF
}

need_value() {
  local name="$1"
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $name" >&2
    exit 2
  fi
  printf '%s\n' "$2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      BASE_DIR="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --textures-dir)
      TEXTURES_DIR="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --only)
      ONLY="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --limit)
      LIMIT="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --start-after)
      START_AFTER="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --timing)
      TIMING=1
      shift
      ;;
    --depth-device)
      DEPTH_DEVICE="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --pipeline-video-format)
      PIPELINE_VIDEO_FORMAT="$(need_value "$1" "${2:-}")"
      if [[ "$PIPELINE_VIDEO_FORMAT" != "mp4" &&
            "$PIPELINE_VIDEO_FORMAT" != "legacy" &&
            "$PIPELINE_VIDEO_FORMAT" != "mkv-ffv1" ]]; then
        echo "--pipeline-video-format must be one of: mp4, legacy, mkv-ffv1" >&2
        exit 2
      fi
      shift 2
      ;;
    --final-overlay-video-codec)
      FINAL_OVERLAY_VIDEO_CODEC="$(need_value "$1" "${2:-}")"
      if [[ "$FINAL_OVERLAY_VIDEO_CODEC" != "lossless" &&
            "$FINAL_OVERLAY_VIDEO_CODEC" != "h264-high" &&
            "$FINAL_OVERLAY_VIDEO_CODEC" != "mjpeg-high" ]]; then
        echo "--final-overlay-video-codec must be one of: lossless, h264-high, mjpeg-high" >&2
        exit 2
      fi
      shift 2
      ;;
    --gradient-mask-threshold)
      GRADIENT_MASK_THRESHOLD="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --tilt-gradient-mask-threshold)
      TILT_GRADIENT_MASK_THRESHOLD="$(need_value "$1" "${2:-}")"
      shift 2
      ;;
    --write-preview-mp4)
      WRITE_PREVIEW_MP4=1
      shift
      ;;
    --no-preview-mp4)
      WRITE_PREVIEW_MP4=0
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

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$BASE_DIR/texture_replacement_v3"
fi

for numeric in "$LIMIT" "$GRADIENT_MASK_THRESHOLD" "$TILT_GRADIENT_MASK_THRESHOLD"; do
  if [[ ! "$numeric" =~ ^[0-9]+$ ]]; then
    echo "Expected numeric value, got: $numeric" >&2
    exit 2
  fi
done

for required in "$BASE_DIR" "$TEXTURES_DIR" "$REFERENCE" "$EXTRACTOR" "$PY" "$DEPTH_ROOT" "$DEPTH_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

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
if [[ -n "$ONLY" ]]; then
  only_name="${ONLY##*/}"
  if [[ "$only_name" != *.avi ]]; then
    only_name="${only_name}.avi"
  fi
  if [[ ! -f "$BASE_DIR/$only_name" ]]; then
    echo "Requested --only video not found in $BASE_DIR: $only_name" >&2
    exit 1
  fi
  planned=("$only_name")
elif [[ -n "$START_AFTER" ]]; then
  start_name="${START_AFTER##*/}"
  start_seen=0
  for name in "${INPUTS[@]}"; do
    if [[ "$start_seen" -eq 0 ]]; then
      if [[ "$name" == "$start_name" ]]; then
        start_seen=1
      fi
      continue
    fi
    planned+=("$name")
  done
  if [[ "$start_seen" -eq 0 ]]; then
    echo "Start marker not found in $BASE_DIR: $start_name" >&2
    exit 1
  fi
else
  planned=("${INPUTS[@]}")
fi

if [[ "$LIMIT" -gt 0 && "${#planned[@]}" -gt "$LIMIT" ]]; then
  planned=("${planned[@]:0:$LIMIT}")
fi

if [[ ${#planned[@]} -eq 0 ]]; then
  echo "No videos planned." >&2
  exit 1
fi

output_ext="avi"
if [[ "$PIPELINE_VIDEO_FORMAT" == "mkv-ffv1" ]]; then
  output_ext="mkv"
elif [[ "$FINAL_OVERLAY_VIDEO_CODEC" == "h264-high" ]]; then
  output_ext="mp4"
fi

job_complete() {
  local input_name="$1"
  local stem="${input_name%.avi}"
  local texture_name texture_stem
  for texture_name in "${TEXTURES[@]}"; do
    texture_stem="${texture_name%.*}"
    if [[ ! -s "$OUT_DIR/${stem}_${texture_stem}.${output_ext}" ]]; then
      return 1
    fi
    if [[ "$WRITE_PREVIEW_MP4" -eq 1 &&
          ! -s "$OUT_DIR/${stem}_${texture_stem}_preview.mp4" ]]; then
      return 1
    fi
  done
  return 0
}

echo "Input dir: $BASE_DIR"
echo "Output dir: $OUT_DIR"
echo "Textures dir: $TEXTURES_DIR"
echo "Textures: ${#TEXTURES[@]}"
echo "Planned input AVIs: ${#planned[@]}"
echo "Pipeline video format: $PIPELINE_VIDEO_FORMAT"
echo "Preview MP4: $([[ "$WRITE_PREVIEW_MP4" -eq 1 ]] && echo yes || echo no)"
echo "Legacy threshold: $GRADIENT_MASK_THRESHOLD"
echo "Tilt threshold: $TILT_GRADIENT_MASK_THRESHOLD"

for input_name in "${planned[@]}"; do
  input_path="$BASE_DIR/$input_name"

  if [[ "$FORCE" -eq 0 ]] && job_complete "$input_name"; then
    echo "[SKIP] $input_name (all texture outputs exist)"
    continue
  fi

  args=(
    --video "$input_path"
    --reference "$REFERENCE"
    --outdir "$OUT_DIR"
    --surface-texture-dir "$TEXTURES_DIR"
    --corrected-sidecar-geometry-only
    --fast
    --pipeline-video-format "$PIPELINE_VIDEO_FORMAT"
    --final-overlay-video-codec "$FINAL_OVERLAY_VIDEO_CODEC"
    --raw-depth-python "$PY"
    --depth-anything-root "$DEPTH_ROOT"
    --depth-model-version v2
    --depth-checkpoint "$DEPTH_CHECKPOINT"
    --depth-encoder vits
    --depth-device "$DEPTH_DEVICE"
    --gradient-mask-threshold "$GRADIENT_MASK_THRESHOLD"
    --tilt-gradient-mask-threshold "$TILT_GRADIENT_MASK_THRESHOLD"
  )
  if [[ "$TIMING" -eq 1 ]]; then
    args+=(--timing)
  fi
  if [[ "$WRITE_PREVIEW_MP4" -eq 1 ]]; then
    args+=(--write-preview-mp4)
  fi

  echo "[RUN] $input_name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '      '
    printf '%q ' "$EXTRACTOR" "${args[@]}"
    printf '\n'
    continue
  fi

  PYTHONUNBUFFERED=1 "$EXTRACTOR" "${args[@]}"
done

echo "[DONE] Texture replacement batch complete."
