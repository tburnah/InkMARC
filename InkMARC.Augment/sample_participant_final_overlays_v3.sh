#!/usr/bin/env bash
set -euo pipefail

COMPONENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_DIR="/data/baseAVIs_432x672"
OUT_ROOT="${COMPONENT_ROOT}/output/final_overlay_samples"
REFERENCE="${COMPONENT_ROOT}/TagsBorderless.png"
EXTRACTOR="${COMPONENT_ROOT}/shadow_extract_v3"
TEXTURE="${COMPONENT_ROOT}/textures/flat_white_512.png"
PY="/home/t/tburnah/miniconda3/envs/opencv_cuda/bin/python"
DEPTH_ROOT="/home/t/tburnah/src/depth_model/Depth-Anything-V2"
DEPTH_CHECKPOINT="/home/t/tburnah/src/depth_model/depth_anything_v2_vits.pth"
DEPTH_DEVICE="cuda"
REFERENCE_SCALE="0.35"
DRY_RUN=0
TIMING=0

usage() {
  cat <<'EOF'
Usage: sample_participant_final_overlays_v3.sh PARTICIPANT [options]

Finds all matching videos in /data/baseAVIs_432x672 and writes final-overlay
PNGs for frames at 1/4, 1/2, and 3/4 through each video.

PARTICIPANT can be "209" or "participant_209".

Options:
  --base-dir DIR          Input AVI directory. Default: /data/baseAVIs_432x672
  --out-root DIR          Output root directory.
                          Default: ./output/final_overlay_samples
  --texture IMAGE         Surface texture image. Default: textures/flat_white_512.png
  --reference-scale N     Reference scale. Default: 0.35
  --depth-device DEVICE   cuda, mps, or cpu. Default: cuda
  --timing                Pass --timing to shadow_extract_v3.
  --dry-run               Print planned commands without running.
  -h, --help              Show this help.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PARTICIPANT="$1"
shift
PARTICIPANT="${PARTICIPANT#participant_}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --texture)
      TEXTURE="$2"
      shift 2
      ;;
    --reference-scale)
      REFERENCE_SCALE="$2"
      shift 2
      ;;
    --depth-device)
      DEPTH_DEVICE="$2"
      shift 2
      ;;
    --timing)
      TIMING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

for required in "$BASE_DIR" "$REFERENCE" "$EXTRACTOR" "$TEXTURE" "$PY" "$DEPTH_ROOT" "$DEPTH_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mapfile -t INPUTS < <(
  find "$BASE_DIR" -maxdepth 1 -type f -name "participant_${PARTICIPANT}_*.avi" -printf '%f\n' | sort -V
)

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "No videos found for participant_${PARTICIPANT} in $BASE_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "Participant: participant_${PARTICIPANT}"
echo "Input dir: $BASE_DIR"
echo "Output root: $OUT_ROOT"
echo "Texture: $TEXTURE"
echo "Videos: ${#INPUTS[@]}"

frame_count_for_video() {
  local video="$1"
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$video"
}

for input_name in "${INPUTS[@]}"; do
  input_path="$BASE_DIR/$input_name"
  stem="${input_name%.avi}"
  outdir="$OUT_ROOT/$stem"
  frame_count="$(frame_count_for_video "$input_path")"
  if [[ ! "$frame_count" =~ ^[0-9]+$ ]] || [[ "$frame_count" -le 0 ]]; then
    echo "[WARN] Could not determine frame count for $input_name; skipping" >&2
    continue
  fi

  q1=$(( frame_count / 4 ))
  q2=$(( frame_count / 2 ))
  q3=$(( (frame_count * 3) / 4 ))
  last=$(( frame_count - 1 ))
  for var in q1 q2 q3; do
    if [[ "${!var}" -gt "$last" ]]; then
      printf -v "$var" '%s' "$last"
    fi
  done

  args=(
    --video "$input_path"
    --reference "$REFERENCE"
    --outdir "$outdir"
    --frames "$q1" "$q2" "$q3"
    --reference-scale "$REFERENCE_SCALE"
    --surface-texture "$TEXTURE"
    --final-overlay-only
    --write-colormatch
    --raw-depth-python "$PY"
    --depth-anything-root "$DEPTH_ROOT"
    --depth-model-version v2
    --depth-checkpoint "$DEPTH_CHECKPOINT"
    --depth-encoder vits
    --depth-device "$DEPTH_DEVICE"
  )
  if [[ "$TIMING" -eq 1 ]]; then
    args+=(--timing)
  fi

  echo "[RUN] $input_name frames=$q1,$q2,$q3/$last"
  echo "      out: $outdir"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '      '
    printf '%q ' "$EXTRACTOR" "${args[@]}"
    printf '\n'
    continue
  fi

  rm -rf "$outdir"
  mkdir -p "$outdir"
  PYTHONUNBUFFERED=1 "$EXTRACTOR" "${args[@]}"
  find "$outdir" -maxdepth 1 -type f ! -name '*_final_overlay.png' -delete
done

echo "[DONE] Final-overlay samples complete."
