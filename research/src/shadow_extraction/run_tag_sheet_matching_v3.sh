#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/baseAVIs_432x672"
OUT_DIR=""
REFERENCE="/home/t/tburnah/src/shadow_extraction/TagsBorderless.png"
EXTRACTOR="/home/t/tburnah/src/shadow_extraction/shadow_extract_v3"
ONLY=""
START_AFTER=""
LIMIT=0
DRY_RUN=0
FORCE=0
TIMING=0

usage() {
  cat <<'EOF'
Usage: run_tag_sheet_matching_v3.sh [options]

Runs tag-sheet matching as a standalone first stage for participant_*.avi
videos. Each successful job produces:
  <video-dir>/<stem>_corrected.h5
  <out-dir>/<stem>_v3_tag_sheet_overlay.mp4

Review all overlay MP4s before running shadow extraction. The extraction batch
uses --corrected-sidecar-geometry-only and will fail if a corrected sidecar is
missing or unusable.

Options:
  --base-dir DIR       Input AVI directory. Default: /data/baseAVIs_432x672.
  --out-dir DIR        Review MP4 directory. Default: <base-dir>/tag_sheet_review_v3.
  --only NAME          Process one AVI filename or full path.
  --limit N            Process only the first N planned AVIs.
  --start-after NAME   Skip inputs through NAME, then process later sorted AVIs.
  --force              Re-run jobs whose two outputs already exist.
  --dry-run            Print planned commands without running them.
  --timing             Print pipeline timing information.
  -h, --help           Show this help.
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
  OUT_DIR="$BASE_DIR/tag_sheet_review_v3"
fi
if [[ ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "--limit must be a non-negative integer" >&2
  exit 2
fi
for required in "$BASE_DIR" "$REFERENCE" "$EXTRACTOR"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
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

echo "Input dir: $BASE_DIR"
echo "Review dir: $OUT_DIR"
echo "Planned input AVIs: ${#planned[@]}"

for input_name in "${planned[@]}"; do
  input_path="$BASE_DIR/$input_name"
  stem="${input_name%.avi}"
  corrected_sidecar="$BASE_DIR/${stem}_corrected.h5"
  review_video="$OUT_DIR/${stem}_v3_tag_sheet_overlay.mp4"

  if [[ "$FORCE" -eq 0 && -s "$corrected_sidecar" && -s "$review_video" ]]; then
    echo "[SKIP] $input_name (corrected sidecar and review MP4 exist)"
    continue
  fi

  args=(
    --video "$input_path"
    --reference "$REFERENCE"
    --outdir "$OUT_DIR"
    --surface-tracking-only
  )
  if [[ "$TIMING" -eq 1 ]]; then
    args+=(--timing)
  fi

  echo "[RUN] $input_name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '      '
    printf '%q ' "$EXTRACTOR" "${args[@]}"
    printf '\n'
    continue
  fi

  "$EXTRACTOR" "${args[@]}"
  if [[ ! -s "$corrected_sidecar" || ! -s "$review_video" ]]; then
    echo "[ERROR] Expected stage-1 outputs were not produced for $input_name" >&2
    exit 1
  fi
done

echo "[DONE] Tag-sheet matching complete. Review the MP4s in $OUT_DIR."
