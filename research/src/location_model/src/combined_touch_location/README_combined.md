# Combined Touch + Location Model

This folder trains a single mobile-friendly model that replaces the separate runtime touch classifier plus location model. It uses one shared MobileNetV3 backbone with two heads:

- `state_logits`: three-class background/hover/touch classifier output.
- `xy`: normalized `(x, y)` location in `[0, 1]`.

The old `touch_model` dataset is intentionally not used. That dataset has binary touch labels but no paired location coordinates, so it is not suitable for training a combined touch/location model. This implementation uses only the existing `location_model` indexed dataset, HDF5 labels, and video-frame pipeline.

## Data Source

Input preprocessing matches the existing location pipeline:

```text
raw video frame + surface corners
-> perspective warp to canonical RGB surface crop
-> float32 tensor in [0, 1]
-> shape B,3,H,W
```

Default input shape:

```text
B,3,320,480
```

The indexed dataset must provide the existing HDF5 fields:

```text
corners:    4,2
labels:     5
label_mask: 5
```

The label schema is:

```text
labels[0] = x tablet coordinate
labels[1] = y tablet coordinate
labels[2] = pressure
labels[3] = tx
labels[4] = ty
```

State labels are derived from xy validity plus pressure:

```text
background = no valid xy
hover      = valid xy and pressure <= --touch-pressure-threshold
touch      = valid xy and pressure >  --touch-pressure-threshold
```

The default pressure threshold is `0.0`.

## Masking

All frames contribute to state classification loss. Background frames do not contribute to location losses.

The training loop uses:

```text
loc_mask = xy_mask
hm_mask  = heatmap_mask
```

So coordinate and heatmap losses are applied to hover/touch samples with valid xy supervision.

## Model

The first implementation matches the existing phase2 location model style:

```text
torchvision.models.mobilenet_v3_large(weights=None).features
```

The model also supports:

```text
--backbone mobilenet_v3_small
```

Location head:

```text
Conv2d(backbone_channels -> 256, kernel_size=1)
ReLU
Conv2d(256 -> 1, kernel_size=1)
resize to hm_h,hm_w
spatial softmax
DSNT expected x,y
```

State head:

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(backbone_channels -> 256)
ReLU
Dropout(0.1)
Linear(256 -> 3)
```

The model does not apply softmax internally. Training uses `CrossEntropyLoss`; app inference should apply softmax externally.

## App Inference

ONNX default outputs:

```text
state_logits: B,3
xy:          B,2
```

Runtime logic:

```text
state_prob = softmax(state_logits)
state = argmax(state_prob)
if state in {hover, touch}:
    use xy
else:
    ignore xy
```

## Train

Run from the `location_model` root:

```bash
python src/combined_touch_location/train_combined.py \
  --dataset-root /path/to/location_dataset \
  --version v1 \
  --out-dir runs/combined_touch_location \
  --epochs 10 \
  --batch-size 32
```

Useful options:

```bash
--touch-pressure-threshold 0.0
--lambda-state 1.0
--lambda-xy 1.0
--lambda-hm 1.0
--amp
--backbone mobilenet_v3_large
--recalibrate-bn-before-eval
--bn-recalibration-max-batches 200
```

The trainer writes:

```text
checkpoint_epoch_XXX.pt
checkpoint_best.pt
training_config.json
metrics_history.json
```

`checkpoint_best.pt` is selected by validation `xy_mae_px` on hover/touch valid-location samples, not by total loss.

To continue a run, pass the checkpoint and set `--epochs` to the final epoch number you want:

```bash
python src/combined_touch_location/train_combined.py \
  --dataset-root /path/to/location_dataset \
  --version v1 \
  --out-dir runs/combined_touch_location \
  --resume runs/combined_touch_location/checkpoint_best.pt \
  --epochs 40
```

## Export ONNX

Run from the `location_model` root:

```bash
python src/combined_touch_location/export_combined_onnx.py \
  --checkpoint runs/combined_touch_location/checkpoint_best.pt \
  --onnx-path runs/combined_touch_location/combined_touch_location.onnx
```

Default ONNX input:

```text
image: B,3,320,480
```

Default ONNX outputs:

```text
state_logits: B,3
xy:          B,2
```

To also export heatmap outputs:

```bash
python src/combined_touch_location/export_combined_onnx.py \
  --checkpoint runs/combined_touch_location/checkpoint_best.pt \
  --onnx-path runs/combined_touch_location/combined_touch_location_debug.onnx \
  --include-heatmap
```
