#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

BASE_DIR="/data/baseAVIs_432x672"
OUT_ROOT=""
REFERENCE="/home/t/tburnah/src/shadow_extraction/TagsBorderless.png"
EXTRACTOR="/home/t/tburnah/src/shadow_extraction/shadow_extract_v3"
TEXTURE="/home/t/tburnah/src/shadow_extraction/textures/flat_white_512.png"
PY="/home/t/tburnah/miniconda3/envs/opencv_cuda/bin/python"
DEPTH_ROOT="/home/t/tburnah/src/depth_model/Depth-Anything-V2"
DEPTH_CHECKPOINT="/home/t/tburnah/src/depth_model/depth_anything_v2_vits.pth"
DEPTH_DEVICE="cuda"
REFERENCE_SCALE="0.35"
DRY_RUN=0
FORCE=0
TIMING=0
MANIFEST=""

usage() {
  cat <<'EOF'
Usage: sample_longest_participant_final_overlays_v3.sh [options]

For each participant in /data/baseAVIs_432x672, selects that participant's
longest AVI, then writes three final-overlay PNGs sampled at 1/4, 1/2, and
3/4 through that video.

By default outputs are written flat into:
  /data/baseAVIs_432x672/longest_video_final_overlay_samples

Output files are named:
  participant_<ID>_q1_final_overlay.png
  participant_<ID>_q2_final_overlay.png
  participant_<ID>_q3_final_overlay.png

Options:
  --base-dir DIR          Input AVI directory. Default: /data/baseAVIs_432x672
  --out-root DIR          Output directory. Default: <base-dir>/longest_video_final_overlay_samples
  --manifest FILE         Manifest with frame_count values. Default: <base-dir>/manifest_participant_splits.json,
                          falling back to <base-dir>/manifest.json, then ffprobe.
  --texture IMAGE         Surface texture image. Default: textures/flat_white_512.png
  --reference-scale N     Reference scale. Default: 0.35
  --depth-device DEVICE   cuda, mps, or cpu. Default: cuda
  --force                 Re-run a participant even if its three PNGs already exist.
  --timing                Pass --timing to shadow_extract_v3.
  --dry-run               Print planned commands without running.
  -h, --help              Show this help.
EOF
}

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
    --manifest)
      MANIFEST="$2"
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
    --force)
      FORCE=1
      shift
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

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$BASE_DIR/longest_video_final_overlay_samples"
fi

if [[ -z "$MANIFEST" ]]; then
  if [[ -f "$BASE_DIR/manifest_participant_splits.json" ]]; then
    MANIFEST="$BASE_DIR/manifest_participant_splits.json"
  elif [[ -f "$BASE_DIR/manifest.json" ]]; then
    MANIFEST="$BASE_DIR/manifest.json"
  fi
fi

for required in "$BASE_DIR" "$REFERENCE" "$EXTRACTOR" "$TEXTURE" "$PY" "$DEPTH_ROOT" "$DEPTH_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
WORK_ROOT="$OUT_ROOT/.work"

frame_count_for_video() {
  local video="$1"
  local count
  count="$(
    ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$video" || true
  )"
  if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
    printf '%s\n' "$count"
    return 0
  fi

  count="$(
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$video" || true
  )"
  if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
    printf '%s\n' "$count"
    return 0
  fi

  return 1
}

selection_from_manifest() {
  local manifest="$1"
  jq -r --arg base "$BASE_DIR" '
    def participant_id:
      if .participant_id then .participant_id
      else (.stem | capture("^participant_(?<id>[0-9]+)_").id)
      end;

    [
      .items[]
      | select(.stem | test("^participant_[0-9]+_"))
      | select((.frame_count // 0) > 0)
      | {
          id: participant_id,
          frames: .frame_count,
          stem: .stem,
          path: ((.output_video_path // "") as $p | if $p == "" then ($base + "/" + .stem + ".avi") else $p end)
        }
      | select(.path | startswith($base + "/"))
    ]
    | sort_by(.id, .frames)
    | group_by(.id)[]
    | max_by(.frames)
    | [.id, .frames, .stem, .path]
    | @tsv
  ' "$manifest"
}

selection_from_ffprobe() {
  declare -A best_frames=()
  declare -A best_name=()
  declare -A best_path=()
  local input_path input_name id frame_count

  while IFS= read -r input_path; do
    input_name="${input_path##*/}"
    if [[ ! "$input_name" =~ ^participant_([0-9]+)_.*\.avi$ ]]; then
      continue
    fi
    id="${BASH_REMATCH[1]}"
    if ! frame_count="$(frame_count_for_video "$input_path")"; then
      echo "[WARN] Could not determine frame count for $input_name; skipping" >&2
      continue
    fi
    if [[ -z "${best_frames[$id]+x}" ]] || [[ "$frame_count" -gt "${best_frames[$id]}" ]]; then
      best_frames[$id]="$frame_count"
      best_name[$id]="${input_name%.avi}"
      best_path[$id]="$input_path"
    fi
  done < <(find "$BASE_DIR" -maxdepth 1 -type f -name 'participant_*.avi' | sort -V)

  for id in "${!best_frames[@]}"; do
    printf '%s\t%s\t%s\t%s\n' "$id" "${best_frames[$id]}" "${best_name[$id]}" "${best_path[$id]}"
  done | sort -t$'\t' -k1,1n
}

participant_done() {
  local participant="$1"
  [[ -s "$OUT_ROOT/participant_${participant}_q1_final_overlay.png" ]] &&
    [[ -s "$OUT_ROOT/participant_${participant}_q2_final_overlay.png" ]] &&
    [[ -s "$OUT_ROOT/participant_${participant}_q3_final_overlay.png" ]]
}

mapfile -t SELECTED < <(
  if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    selection_from_manifest "$MANIFEST"
  else
    selection_from_ffprobe
  fi
)

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "No participant videos found in $BASE_DIR" >&2
  exit 1
fi

echo "Input dir: $BASE_DIR"
echo "Output dir: $OUT_ROOT"
if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
  echo "Manifest: $MANIFEST"
else
  echo "Manifest: none; using ffprobe frame counts"
fi
echo "Texture: $TEXTURE"
echo "Participants: ${#SELECTED[@]}"

summary_tsv="$OUT_ROOT/selected_longest_videos.tsv"
if [[ "$DRY_RUN" -eq 0 ]]; then
  printf 'participant\tframes\tstem\tvideo_path\tq1\tq2\tq3\n' > "$summary_tsv"
fi

for row in "${SELECTED[@]}"; do
  IFS=$'\t' read -r participant frame_count stem input_path <<<"$row"
  if [[ ! -f "$input_path" ]]; then
    echo "[WARN] Missing selected video for participant_${participant}: $input_path; skipping" >&2
    continue
  fi

  if [[ ! "$frame_count" =~ ^[0-9]+$ ]] || [[ "$frame_count" -le 0 ]]; then
    if ! frame_count="$(frame_count_for_video "$input_path")"; then
      echo "[WARN] Could not determine frame count for $input_path; skipping" >&2
      continue
    fi
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

  if [[ "$DRY_RUN" -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$participant" "$frame_count" "$stem" "$input_path" "$q1" "$q2" "$q3" >> "$summary_tsv"
  fi

  if [[ "$FORCE" -eq 0 ]] && participant_done "$participant"; then
    echo "[SKIP] participant_${participant} (three PNGs already exist)"
    continue
  fi

  outdir="$WORK_ROOT/participant_${participant}"
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

  echo "[RUN] participant_${participant}: $stem frames=$q1,$q2,$q3/$last"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '      '
    printf '%q ' "$EXTRACTOR" "${args[@]}"
    printf '\n'
    continue
  fi

  rm -rf "$outdir"
  mkdir -p "$outdir"
  PYTHONUNBUFFERED=1 "$EXTRACTOR" "${args[@]}"

  declare -a frames=("$q1" "$q2" "$q3")
  declare -a labels=("q1" "q2" "q3")
  for i in "${!frames[@]}"; do
    frame="${frames[$i]}"
    label="${labels[$i]}"
    printf -v prefix 'frame_%06d' "$frame"
    exact="$outdir/${prefix}_final_overlay.png"
    if [[ -f "$exact" ]]; then
      match=("$exact")
    else
      match=("$outdir/${prefix}"_*_final_overlay.png)
    fi
    if [[ ${#match[@]} -ne 1 || ! -f "${match[0]}" ]]; then
      echo "[WARN] Expected one final overlay for participant_${participant} ${prefix}; found ${#match[@]}" >&2
      continue
    fi
    cp -f "${match[0]}" "$OUT_ROOT/participant_${participant}_${label}_final_overlay.png"
  done

done

if [[ "$DRY_RUN" -eq 0 ]]; then
  rm -rf "$WORK_ROOT"
fi

echo "[DONE] Longest-video participant samples complete."
