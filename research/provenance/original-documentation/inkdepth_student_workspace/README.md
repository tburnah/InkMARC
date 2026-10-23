# InkDepth Student

PyTorch scaffold for training a fast single-frame mobile student model for smartphone stylus tracking. The model uses Depth-Anything-V2-Large pseudo-depth as auxiliary supervision while keeping touch/state and location heads connected to shared RGB features directly.

This is not a general-purpose Depth-Anything clone. The intended comparisons are:

1. RGB-only touch/location baseline
2. Depth-supervised multitask model
3. Task-finetuned model with lower depth weight

## Install

```bash
cd /home/t/tburnah/inkdepth_student
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable imports from notebooks or external scripts:

```bash
pip install -e .
```

No `setup.py` is required for the included scripts because they add the project root to `PYTHONPATH` automatically.

## Data Configuration

Default config: `configs/inkdepth_mobilenetv3.yaml`

The config points to:

```yaml
data:
  manifest_path: /data/baseAVIs_432x672/manifest.json
```

`StylusDataset` supports common manifest shapes:

- a list of frame records
- a dict with `frames`
- a dict with `videos`, each containing `frames`

Frame records can include fields such as `image_path`, `frame_path`, `video_path`, `frame_index`, `depth_path`, `depth_key`, `touch`, `state`, `x`, and `y`. If labels live outside the manifest, set `data.labels.format` to `json`, `csv`, or `h5`.

The H5 label reader is intentionally a placeholder adapter in `inkdepth/datasets/stylus_dataset.py`. It currently expects simple datasets named `video_id`, `frame_index`, and optional `touch`, `state`, `x`, `y`. Replace `_load_external_h5_labels` if your H5 schema differs.

For `/data/baseAVIs_432x672`, the dataset also supports the resized InkMARC-style layout directly:

- video-level records under manifest key `items`
- resized AVI at `output_video_path`
- paired H5 at `output_h5_path`
- per-video Zarr depth at `/data/baseAVIs_432x672/depth/<stem>.zarr`
- `labels[:,0]`, `labels[:,1]`, `labels[:,2]` interpreted as tablet x, tablet y, and pressure
- `corners` interpreted as the per-frame surface quadrilateral in original image coordinates

Create participant-level splits without modifying the original manifest:

```bash
python scripts/create_participant_splits.py \
  --manifest /data/baseAVIs_432x672/manifest.json \
  --base-dir /data/baseAVIs_432x672 \
  --out /data/baseAVIs_432x672/manifest_participant_splits.json \
  --seed 1337 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --test-frac 0.1
```

The generated manifest adds `participant_id`, `split`, `depth_path`, and `depth_key` per video. Splits are participant-exclusive.

Images are returned as `[3,672,432]` float tensors in `[0,1]`. Depth is returned as `[1,672,432]`.

The real dataset path uses strict validity flags:

- `has_depth` gates depth losses
- `has_state` gates state loss
- `has_touch` gates binary touch loss
- `valid_xy` gates xy and heatmap losses
- missing H5 files produce `has_labels=false`, `has_state=false`, `has_touch=false`, `valid_xy=false`, while still allowing depth supervision when Zarr depth exists

The default coordinate mode is `surface_to_image` with `surface_width=44800` and `surface_height=29600`. Raw stylus coordinates are normalized on the physical surface, placed into the same canonical `480x320` rectangle used by the C# viewer, projected through the H5 `corners` quadrilateral with a perspective homography, then normalized to image-space `[0,1]` for the heatmap and xy losses. The current resized H5 files appear to store corners in resized-video coordinates, so `corners_coordinate_mode=resized` is the default. Use `coordinate_mode=auto` for auditing only; location training will fail fast with `auto`.

## Audits

Dataset label audit:

```bash
python scripts/audit_dataset_labels.py \
  --manifest /data/baseAVIs_432x672/manifest_participant_splits.json \
  --output-dir outputs/audits
```

Split audit:

```bash
python scripts/audit_participant_splits.py \
  --manifest /data/baseAVIs_432x672/manifest_participant_splits.json \
  --output-dir outputs/audits
```

Visual samples:

```bash
python scripts/visualize_dataset_samples.py \
  --manifest /data/baseAVIs_432x672/manifest_participant_splits.json \
  --split train \
  --num-samples 50 \
  --output-dir outputs/debug_samples
```

Useful visual filters include `--task T5a`, `--participant-id 319`, `--state touch`, `--touch-only`, and `--valid-xy-only`.

Black-frame exclusions:

```bash
python scripts/add_black_frame_exclusions.py \
  --manifest /data/baseAVIs_432x672/manifest_texture_replacement_v3_no_missing_h5_participant_splits.json \
  --out /data/baseAVIs_432x672/manifest_texture_replacement_v3_no_missing_h5_no_black_participant_splits.json \
  --variant-kind texture_replacement_v3 \
  --black-max 2 \
  --workers 8 \
  --ffmpeg-threads 1
```

The generated manifest stores compact `exclude_frame_ranges` per affected video. `StylusDataset` skips those frame indices when expanding video-level items. Increase `--black-max` if the bad frames are near-black rather than exactly black. Use `--method raw --black-mean ...` only if FFmpeg's faster `blackdetect` path misses frames you can see visually.

## Participant-Level CV Statistics

Use `scripts/evaluate_participant_cv_statistics.py` to compute InkMARC-Sense statistical summaries from out-of-fold participant-level metrics. The script consumes, or can generate, held-out fold prediction CSVs and then performs tests over participants rather than individual frames, because neighbouring frames are temporally correlated and are not independent observations.

The contact-vs-non-contact location comparison and the Raw Threshold-vs-Hysteresis touch comparison are paired within participant. Mann-Whitney is therefore not used; when paired differences are not approximately normal, Wilcoxon signed-rank is the non-parametric paired alternative.

Example:

```bash
python scripts/evaluate_participant_cv_statistics.py \
  --fold-root outputs/crossval/final_combined_live_da3_geometry_heatmap_touch_presence \
  --output-dir outputs/cv_participant_statistics \
  --generate-missing-predictions \
  --config configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml
```

## Inspect A Batch

Real data:

```bash
python scripts/inspect_batch.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --forward \
  --override \
    data.manifest_path=/data/baseAVIs_432x672/manifest_participant_splits.json \
    training.batch_size=2 \
    training.num_workers=0 \
    data.frame_stride=120 \
    data.max_frames_per_video=2 \
    data.max_videos_per_split=4 \
    model.pretrained=false \
    model.width_mult=0.25
```

Synthetic smoke path:

```bash
python scripts/inspect_batch.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --synthetic \
  --forward \
  --override training.batch_size=2 data.synthetic.length=4 model.width_mult=0.25
```

Strict real-manifest smoke test:

```bash
python scripts/inspect_batch.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --all-splits \
  --forward \
  --compute-loss \
  --check-missing-h5 \
  --save-panel outputs/debug_samples/inspect_panel.jpg \
  --override \
    data.manifest_path=/data/baseAVIs_432x672/manifest_participant_splits.json \
    training.batch_size=2 \
    training.num_workers=0 \
    data.frame_stride=120 \
    data.max_frames_per_video=2 \
    data.max_videos_per_split=4 \
    model.pretrained=false \
    model.width_mult=0.25
```

## Train

Depth pretrain:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override training.phase=depth_pretrain output_dir=outputs/depth_pretrain
```

Multitask training:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override training.phase=multitask output_dir=outputs/multitask
```

Train with strict masks:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override data.manifest_path=/data/baseAVIs_432x672/manifest_participant_splits.json
```

Small real-data smoke run:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override \
    data.manifest_path=/data/baseAVIs_432x672/manifest_participant_splits.json \
    data.frame_stride=120 \
    data.max_frames_per_video=2 \
    data.max_videos_per_split=4 \
    training.epochs=1 \
    training.batch_size=2 \
    training.num_workers=0 \
    training.save_visuals=true \
    training.max_visuals=2 \
    model.pretrained=false \
    model.width_mult=0.25 \
    output_dir=outputs/real_smoke_test
```

Task finetune from a checkpoint:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override \
    training.phase=task_finetune \
    training.resume=outputs/multitask/best_val_loss.pt \
    output_dir=outputs/task_finetune
```

RGB-only baseline:

```bash
python scripts/train_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --override \
    model.use_depth_features_for_tasks=false \
    model.train_depth=false \
    loss.lambda_depth_ssi=0.0 \
    loss.lambda_depth_grad=0.0 \
    output_dir=outputs/rgb_baseline
```

For binary touch/no-touch, use `model.state_mode=binary`. For 3-class background/hover/touch, use:

```bash
--override model.state_mode=multiclass model.num_states=3
```

## Validate

```bash
python scripts/validate_student.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --checkpoint outputs/multitask/best_val_loss.pt \
  --save-visuals
```

Debug panels are saved under `outputs/.../visuals/`. Each panel contains RGB, teacher depth, student depth, absolute depth error, and heatmap with predicted xy overlay.

## Export ONNX

```bash
python scripts/export_student_onnx.py \
  --config configs/inkdepth_mobilenetv3.yaml \
  --checkpoint outputs/task_finetune/best_val_loss.pt \
  --output outputs/task_finetune/inkdepth_student.onnx
```

The export uses fixed input shape `[1,3,672,432]` and includes:

- `touch_logits` for binary mode, or `state_logits` for multiclass mode
- `xy`
- `depth_lowres` unless `--no-depth` is passed

The script runs ONNX Runtime numerical validation against PyTorch on a dummy input when `onnxruntime` is installed. ONNX Runtime quantization is intentionally left for a later deployment step.

## Outputs

Checkpoints:

- `latest.pt`
- `best_val_loss.pt`
- `best_touch.pt`
- `best_xy.pt`

Logs:

- TensorBoard: `outputs/.../tb`
- CSV: `outputs/.../metrics.csv`
- grouped validation JSON/CSV:
  - `metrics_epoch_<N>.json`
  - `metrics_latest.json`
  - `metrics_by_video.csv`
  - `metrics_by_participant.csv`
  - `metrics_by_task.csv`

Training copies the split manifest into the run output directory by default for reproducibility.

## Notes

- No temporal modeling is implemented yet; this is a single-frame model.
- Geometric augmentation is intentionally avoided until RGB, labels, and pseudo-depth can be transformed consistently.
- MobileNetV3-Large is the only backbone in this first implementation. Set `model.width_mult` lower for faster smoke tests or smaller deployment candidates.

## DA3-Small Live Backbone

DA3-small is integrated as a live, frozen depth backbone for touch/location modeling. It is loaded directly with:

```python
DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
```

Use the config `configs/touch_live_da3_small_original_stride4_stable_w5_p1000.yaml` to train the DA3-backed touch/location head without precomputing zarr depth. The DA3 repo must be available locally at `/home/t/tburnah/src/depth_model/Depth-Anything-3`, or override `model.da3_repo_root`.

With `data.surface_geometry.enabled=true`, the current live-DA3 geometry flow is:

```text
Inputs
  RGB image frame
  4 surface corner points from H5 labels
        |
        v
StylusDataset Surface Geometry
  corners -> 3 geometry maps at input resolution:
    1. surface mask
    2. surface u-coordinate map
    3. surface v-coordinate map
        |
        v
InkDepthDA3Student
  input image tensor from dataset:
    RGB: 3 channels
    geometry: 3 channels
  live frozen DA3-small:
    RGB -> normalized depth map
  student tensor:
    RGB + DA3 depth + geometry = 7 channels
        |
        v
InkDepthStudent MobileNetV3-Large Backbone
  extracts multi-scale features from the 7-channel tensor
        |
        v
FPN Neck
  combines backbone features:
    p4, p8, p16
        |
        +------------------------+
        |                        |
        v                        v
Location Head              Task Feature Pooling
  p4 features              global pooled p4/p8/p16 features
  -> heatmap logits        + heatmap-weighted local p4 features
  -> soft-argmax xy        + 5 DA3 depth stats from DA3 depth
        |                        |
        v                        v
XY Output                  Task Vector
  x, y image-normalized          |
  location                       +--> Binary Touch Head
                                |    touch/contact vs no-touch
                                |
                                +--> Optional Presence Head
                                     when train_state=true:
                                     background vs interaction
                                     interaction = hover or touch
```

For the geometry DA3 configs, `use_depth_features_for_tasks=false`, so the learned student `DepthHead` decoder features are not used by the task heads. DA3 depth is used as an input channel and for the heatmap-conditioned depth statistics.

For surface-plane XY experiments, use `configs/location_surface_xy_live_da3_small_geometry_original_stride4_stable_w5_p1000.yaml`. That config sets `data.labels.coordinate_mode=surface_pixels`, `model.location_head_type=direct`, and `loss.lambda_heatmap=0.0`, so the location head predicts normalized surface-plane coordinates instead of an image-space heatmap location. Validation reports `xy_surface` in tablet/surface units.

```bash
python scripts/run_live_da3_experiment.py --dry-run
```

### Export DA3-Small Depth ONNX

For Android deployment, export the frozen DA3-small depth model as a fixed-shape, depth-only ONNX graph. DA3 requires dimensions divisible by 14.

Fastest square input:

```bash
python scripts/export_da3_small_onnx.py \
  --repo-root /home/t/tburnah/src/depth_model/Depth-Anything-3 \
  --model-id depth-anything/DA3-SMALL \
  --height 322 \
  --width 322 \
  --output outputs/onnx/da3_small_depth_322x322.onnx
```

Shape matching this repo's live DA3 preprocessing from `data.input_size: [672, 432]`,
`da3_process_res: 322`, and `lower_bound_resize`:

```bash
python scripts/export_da3_small_onnx.py \
  --repo-root /home/t/tburnah/src/depth_model/Depth-Anything-3 \
  --model-id depth-anything/DA3-SMALL \
  --from-frame-size 672 432 \
  --process-res 322 \
  --process-res-method lower_bound_resize \
  --output outputs/onnx/da3_small_depth_504x322.onnx
```

The ONNX input is the already-resized, ImageNet-normalized DA3 tensor with shape
`[1,1,3,H,W]`; keep image resize and normalization in the Android preprocessing path.
