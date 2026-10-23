# LocTouchNetLite ML Scaffold

This scaffold prepares a PyTorch and ONNX training path for a fast mobile model that predicts interaction location and state from a canonical warped RGB surface image.

## Input

- `image`: float32 tensor `[B, 3, 320, 480]`
- RGB, values in `[0, 1]`
- ImageNet normalization is inside the model graph and exported to ONNX.

## Outputs

`LocTouchNetLite(image)` returns `xy`, `heatmap_logits`, `heatmap_prob`, `loc_conf`, `valid_logit`, `touch_logit`, `state_logits`, and `state_probs`.

`touch_logit` is the conditional touch-vs-hover logit given a valid interaction. The final unconditional contact probability is `state_probs[:, 2]`.

## Smoke Commands

Use the Python environment that has torch installed, for example:

```bash
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m pytest ml/tests
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.training.overfit_tiny --config ml/configs/loctouch_lite.yaml
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.export.export_onnx --config ml/configs/loctouch_lite.yaml --out ml/artifacts/loctouch_lite.onnx
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.export.check_onnx --onnx ml/artifacts/loctouch_lite.onnx
```

For a quick training smoke run:

```bash
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.training.train --config ml/configs/loctouch_lite.yaml --dry-run
```

To continue from the current best checkpoint and export the trained model:

```bash
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.training.train --config ml/configs/loctouch_lite.yaml --resume ml/artifacts/checkpoints/best.pt --learning-rate 0.001 --max-steps 600
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.export.export_onnx --config ml/configs/loctouch_lite.yaml --checkpoint ml/artifacts/checkpoints/best.pt --out ml/artifacts/loctouch_lite_trained.onnx
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python -m ml.inkloc.export.check_onnx --onnx ml/artifacts/loctouch_lite_trained.onnx --config ml/configs/loctouch_lite.yaml --checkpoint ml/artifacts/checkpoints/best.pt
```

## Real Data

The default config now uses the existing InkMARC indexed LocationData pipeline:

```yaml
data:
  type: inkmarc
  dataset_root: /data/LocationData
  inkmarc_src: /home/t/tburnah/inkmarc/src
  index_version: v1
```

This reads `indices/manifest_v1.json`, `indices/index_v1.jsonl`, and `indices/splits_v1.json`, loads the matching raw video frame, warps it with the stored surface corners, and derives background/hover/touch from xy validity plus pressure.

The default config uses `index_version: focused` for faster real-data iteration. Switch to `v1` for full-dataset training once the loop is stable.

The generic JSONL/CSV manifest loader is still available by setting `data.type` away from `inkmarc` and setting `data.manifest`.

## Notes

The default config uses a lightweight MobileNetV3-style backbone for fast CPU readiness tests. Set `model.use_torchvision: true` to use torchvision's MobileNetV3-small implementation when that dependency is available.

The first InkMARC dataset construction can take tens of seconds because the upstream loader filters labeled samples across the HDF5 index.
