# Data Contract

## InkMARC LocationData Format

The primary real-data path uses `/data/LocationData` with the existing InkMARC indexed dataset:

- `indices/manifest_<version>.json`
- `indices/index_<version>.jsonl`
- `indices/splits_<version>.json`
- HDF5 files containing `corners`, `labels`, `label_mask`, and `timestamps_ns`
- AVI video variants referenced by the manifest/index metadata

Each sample is transformed as:

```text
raw video frame + surface corners
-> perspective warp to canonical RGB surface crop
-> float32 tensor in [0, 1]
-> shape 3,320,480
```

The HDF5 label schema is:

- `labels[0]`: x tablet coordinate
- `labels[1]`: y tablet coordinate
- `labels[2]`: pressure
- `labels[3]`: tx
- `labels[4]`: ty

`label_mask[0]` and `label_mask[1]` determine whether xy is valid. `label_mask[2]` determines whether pressure/touch supervision exists.

State labels are derived as:

- `0` background: no valid xy
- `1` hover: valid xy and pressure <= threshold
- `2` touch: valid xy and pressure > threshold

Coordinates are normalized by the existing transform using tablet dimensions `44800 x 29600`.

## Generic Manifest Format

JSONL and CSV manifests are supported. Required fields:

- `image_path`: path to the canonical warped RGB image
- `state`: integer label, `0` background, `1` hover, `2` touch
- `touch`: `1` for touch frames, `0` otherwise
- `valid`: `1` for hover or touch, `0` for background
- `x_norm`: normalized x coordinate in `[0, 1]`
- `y_norm`: normalized y coordinate in `[0, 1]`
- `has_xy`: `1` when `x_norm/y_norm` are meaningful

Optional fields:

- `prev_image_path`
- `prev2_image_path`
- `timestamp`
- `prev_timestamp`
- `pressure`
- `participant_id`
- `sequence_id`
- `frame_idx`
- `split`
- `transition`

## Coordinate Convention

`x_norm=0` is the left edge of the canonical warped surface and `x_norm=1` is the right edge. `y_norm=0` is the top edge and `y_norm=1` is the bottom edge. Coordinates represent the interaction point on the canonical surface, not raw camera pixels.

## Label Meanings

- Background: no valid interaction. Use `state=0`, `valid=0`, `touch=0`, `has_xy=0`.
- Hover: stylus or hand is present but not touching. Use `state=1`, `valid=1`, `touch=0`, `has_xy=1`.
- Touch: contact is active. Use `state=2`, `valid=1`, `touch=1`, `has_xy=1`.

If a background frame has a meaningful supervised coordinate for a special task, set `has_xy=1`; otherwise leave `has_xy=0` so the location loss is masked.

## Missing XY

When `has_xy=0`, `x_norm` and `y_norm` may be `0`, but they are ignored by the xy loss. Do not put sentinel values outside `[0, 1]` in production manifests.

## Split Strategy

Use participant-held-out validation to test person generalization. Use sequence-held-out validation to catch temporal and scene leakage. Maintain a transition-focused validation set enriched for hover-to-touch, touch-to-hover, and entering/leaving-frame cases.
