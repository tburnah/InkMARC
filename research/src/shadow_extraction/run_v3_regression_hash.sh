#!/usr/bin/env bash
set -euo pipefail

VIDEO="${VIDEO:-/data/baseAVIs_480x747/participant_212_T5a_1777506635273.avi}"
REFERENCE="${REFERENCE:-TagsBorderless.png}"
OUTDIR="${OUTDIR:-/tmp/shadow_v3_regression_hash}"
FRAMES="${FRAMES:-89 90 91 219 220 222 230}"
BASELINE="${BASELINE:-}"
AVI_BASELINE="${AVI_BASELINE:-}"
HASH_AVI="${HASH_AVI:-0}"
FULL_VIDEO="${FULL_VIDEO:-0}"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

run_args=(
  --video "$VIDEO"
  --reference "$REFERENCE"
  --outdir "$OUTDIR"
  --write-overlay
)

if [[ "$FULL_VIDEO" == "0" ]]; then
  # shellcheck disable=SC2206
  frame_args=($FRAMES)
  run_args+=(--frames "${frame_args[@]}")
fi

./shadow_extract_v3 "${run_args[@]}"

(
  cd "$OUTDIR"
  shopt -s nullglob
  png_files=(*_v3_tag_sheet_overlay.png)
  if [[ ${#png_files[@]} -gt 0 ]]; then
    sha256sum "${png_files[@]}" | sort > sha256.txt
  else
    : > sha256.txt
  fi
)

if [[ -s "$OUTDIR/sha256.txt" ]]; then
  cat "$OUTDIR/sha256.txt"
fi

if [[ -n "$BASELINE" ]]; then
  diff -u "$BASELINE" "$OUTDIR/sha256.txt"
fi

if [[ "$HASH_AVI" != "0" ]]; then
  shopt -s nullglob
  avi_files=("$OUTDIR"/*.avi)
  if [[ ${#avi_files[@]} -eq 0 ]]; then
    echo "[WARN] HASH_AVI requested but no AVI files were written in $OUTDIR" >&2
  else
    : > "$OUTDIR/avi_frame_sha256.txt"
    for avi in "${avi_files[@]}"; do
      name="$(basename "$avi")"
      hash="$(
        ffmpeg -hide_banner -loglevel error -i "$avi" \
          -map 0:v:0 -f rawvideo -pix_fmt bgr24 - 2>/dev/null | sha256sum | awk '{print $1}'
      )"
      printf '%s  %s\n' "$hash" "$name" >> "$OUTDIR/avi_frame_sha256.txt"
    done
    sort -o "$OUTDIR/avi_frame_sha256.txt" "$OUTDIR/avi_frame_sha256.txt"
    cat "$OUTDIR/avi_frame_sha256.txt"
    if [[ -n "$AVI_BASELINE" ]]; then
      diff -u "$AVI_BASELINE" "$OUTDIR/avi_frame_sha256.txt"
    fi
  fi
fi
