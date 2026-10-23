# Model Readiness

## Purpose

`LocTouchNetLite` predicts a normalized interaction point and hierarchical interaction state from a warped canonical RGB surface frame.

## Input Contract

- `image`: `float32`, `[B, 3, 320, 480]`
- RGB channel order
- Values in `[0, 1]`
- Android preprocessing should not apply ImageNet normalization. The model applies ImageNet mean/std internally.

## Output Contract

- `xy`: `[B, 2]`, normalized `[0, 1]` as `[x, y]`
- `heatmap_logits`: `[B, 1, 40, 60]` by default
- `heatmap_prob`: spatial softmax over the heatmap
- `loc_conf`: `[B]` heatmap peak confidence
- `valid_logit`: background vs valid interaction
- `touch_logit`: conditional touch vs hover given valid interaction
- `state_probs`: `[B, 3]` for background, hover, touch
- `state_logits`: `log(state_probs + eps)` for Android softmax consumers

Final contact probability is `state_probs[:, 2]`, not `sigmoid(touch_logit)` alone.

## Architecture Summary

The model uses an internal ImageNet normalization layer, a shared MobileNetV3-small-style backbone, a heatmap head, DSNT soft-argmax for location, and two state heads. The valid and touch heads consume both a global pooled token and a heatmap-weighted local token near the predicted interaction point, plus `xy` and `loc_conf`.

The default config disables temporal inputs but includes `temporal.enabled` and `temporal.num_motion_maps` fields so motion features can be added without changing the public config shape.

## Loss Summary

`LocTouchLoss` combines valid BCE, touch BCE masked to valid frames, state CE, masked xy SmoothL1, and optional heatmap, pressure, and transition losses. Background frames do not contribute to xy loss unless `has_xy=1`.

## Training Data Requirements

Real training now uses the existing InkMARC indexed LocationData pipeline by default. The adapter reads HDF5 labels/corners, loads the matching AVI frame, warps it to the canonical surface, and maps the resulting sample to `image`, `valid`, `touch`, `state`, `xy`, `has_xy`, and `heatmap_target`.

Generic JSONL/CSV manifests are still supported for future prewarped datasets.

## ONNX Export Status

The export path uses opset 17 by default, input name `image`, dynamic batch axes, and outputs `xy`, `loc_conf`, `valid_logit`, `touch_logit`, and `state_logits`. `heatmap_logits` can be included through config.

Checkpoint export is supported with `--checkpoint`. The current trained readiness artifact is `ml/artifacts/loctouch_lite_trained.onnx`, exported from `ml/artifacts/checkpoints/best.pt`, and passed ONNX checker plus ONNX Runtime parity.

## Current Real-Data Smoke Result

Using `/data/LocationData`, `index_version=focused`, `require_labels=false`, 32768 deterministic capped train samples, and 4096 deterministic capped validation samples:

- Split strategy: existing InkMARC focused split. It is sequence-held-out with zero train/val sequence overlap, but it is not participant-held-out; train and val both contain participants 209 and 210.
- Best validation-loss checkpoint: `ml/artifacts/checkpoints/best.pt`, validation loss `2.2320`.
- Selected checkpoint for Android benchmark: `ml/artifacts/checkpoints/selected.pt`, copied from best calibrated touch-F1 checkpoint.
- Selected checkpoint metrics: validation loss `3.0726`, state accuracy `0.6631`, valid accuracy `0.9102`, xy normalized L2 `0.1681`, xy pixel error `74.4 px`.
- Touch at threshold `0.5`: precision `0.7472`, recall `0.4105`, F1 `0.5299`.
- Calibrated touch threshold: `0.06`, precision `0.6424`, recall `0.7876`, F1 `0.7076`.
- Mean final touch probability: `0.4081` for true touch vs `0.0963` for non-touch.
- Calibration JSON: `ml/artifacts/calibration/loctouch_lite_thresholds.json`.

## AOI Diagnostics

AOI diagnostics on `selected.pt` over a 1024-sample validation subset with calibrated threshold `0.06` showed:

- Normal touch F1 `0.682`.
- AOI-only with 48/64/96 px patches preserved touch signal (`0.700`, `0.693`, `0.673` F1), while 32 px was too small (`0.361` F1).
- AOI-masked performance dropped as radius increased: 32 px `0.642`, 48 px `0.591`, 64 px `0.554`, 96 px `0.519`.
- Zeroing the local token dropped touch F1 to `0.531` and state accuracy to `0.224`.
- Zeroing the global token had little effect on touch F1 (`0.677`) or state accuracy (`0.657`).

Interpretation: the current touch decision depends strongly on local AOI features. The larger failure mode is likely hover-vs-touch supervision/class balance/hard examples rather than the touch head ignoring the local token.

## Known Limitations

- Synthetic overfit only proves pipeline wiring, not real-world performance.
- Default lightweight backbone is intended for readiness tests; enable torchvision MobileNetV3-small or tune width after latency measurement.
- Temporal and pressure heads are placeholders at the interface/loss level.
- Touch/contact threshold calibration is still required; `0.5` is not currently the best threshold on the focused validation slice.

## Checklist Before Real Training

- Prepare JSONL/CSV manifests following `data_contract.md`.
- Confirm the selected InkMARC `index_version` and splits are the intended real-data split.
- Use participant-held-out and sequence-held-out validation splits.
- Add transition-focused validation sequences.
- Run the smoke tests and ONNX checks after dependency installation.
- Train a tiny subset until it overfits, then expand to full data.
- Evaluate Android ONNX output parity on device-side sample tensors.
