#!/usr/bin/env bash
set -euo pipefail

WIDTH=""
HEIGHT=""
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: resize_copy_media.sh [options] --size WIDTHxHEIGHT INPUT_DIR OUTPUT_DIR

Mirrors files from INPUT_DIR into OUTPUT_DIR. Existing outputs are skipped by
default. Video and image files are resized to WIDTHxHEIGHT with ffmpeg; other
files are copied unchanged, which keeps sidecars such as .h5 files alongside
the resized videos.

Options:
  --size WIDTHxHEIGHT   Required output media size, for example 480x747.
  --force               Recreate outputs even if they already exist.
  --dry-run             Print planned work without writing files.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --size" >&2
        exit 2
      fi
      if [[ ! "$2" =~ ^([0-9]+)x([0-9]+)$ ]]; then
        echo "--size must be WIDTHxHEIGHT, for example 480x747" >&2
        exit 2
      fi
      WIDTH="${BASH_REMATCH[1]}"
      HEIGHT="${BASH_REMATCH[2]}"
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
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$WIDTH" || -z "$HEIGHT" ]]; then
  echo "--size is required" >&2
  usage >&2
  exit 2
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but was not found on PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

is_video() {
  case "${1,,}" in
    *.avi|*.mp4|*.mov|*.mkv|*.m4v|*.wmv|*.webm) return 0 ;;
    *) return 1 ;;
  esac
}

is_image() {
  case "${1,,}" in
    *.png|*.jpg|*.jpeg|*.bmp|*.tif|*.tiff|*.webp) return 0 ;;
    *) return 1 ;;
  esac
}

resize_video() {
  local src="$1"
  local dst="$2"
  local ext="${dst##*.}"
  ext="${ext,,}"

  local common_args=(
    -hide_banner -loglevel error -nostdin -y
    -i "$src"
    -map 0:v:0
    -map '0:a?'
    -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos"
  )

  case "$ext" in
    avi)
      ffmpeg "${common_args[@]}" -c:v mjpeg -q:v 3 -c:a copy "$dst"
      ;;
    mp4|mov|mkv|m4v)
      ffmpeg "${common_args[@]}" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a copy "$dst"
      ;;
    webm)
      ffmpeg "${common_args[@]}" -c:v libvpx-vp9 -crf 24 -b:v 0 -c:a libopus "$dst"
      ;;
    *)
      ffmpeg "${common_args[@]}" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a copy "$dst"
      ;;
  esac
}

resize_image() {
  local src="$1"
  local dst="$2"
  ffmpeg -hide_banner -loglevel error -nostdin -y \
    -i "$src" \
    -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos" \
    "$dst"
}

copy_file() {
  local src="$1"
  local dst="$2"
  cp -p "$src" "$dst"
}

processed=0
skipped=0

while IFS= read -r -d '' src; do
  rel="${src#"$INPUT_DIR"/}"
  dst="$OUTPUT_DIR/$rel"
  if [[ "$dst" == *.* ]]; then
    tmp="${dst%.*}.tmp.$$.${dst##*.}"
  else
    tmp="${dst}.tmp.$$"
  fi
  mkdir -p "$(dirname "$dst")"

  if [[ "$FORCE" -eq 0 && -e "$dst" ]]; then
    echo "[SKIP] $rel"
    skipped=$((skipped + 1))
    continue
  fi

  if is_video "$src"; then
    echo "[VIDEO] $rel -> ${WIDTH}x${HEIGHT}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      rm -f "$tmp"
      resize_video "$src" "$tmp"
      touch -r "$src" "$tmp"
      mv -f "$tmp" "$dst"
    fi
  elif is_image "$src"; then
    echo "[IMAGE] $rel -> ${WIDTH}x${HEIGHT}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      rm -f "$tmp"
      resize_image "$src" "$tmp"
      touch -r "$src" "$tmp"
      mv -f "$tmp" "$dst"
    fi
  else
    echo "[COPY] $rel"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      copy_file "$src" "$dst"
    fi
  fi

  processed=$((processed + 1))
done < <(find "$INPUT_DIR" -type f -print0 | sort -z)

echo "[DONE] processed=$processed skipped=$skipped output=$OUTPUT_DIR"
