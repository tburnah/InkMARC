# InkMARC Depth (InkDepth) — distilled relative-depth proxy

Thesis §7.1.2 (pp. 116–117) describes a smaller depth model distilled from
Depth Anything 3 Small. The thesis names it “InkMARC Depth”; InkDepth is
retained as the author's associated name. This component is the
MobileNetV2 proxy, separate from earlier students in `src/depth_model/`.

`inkdepth/models/depth_proxy.py` defines `DepthProxyMobileNet` and the
fixed-resolution decoder. `configs/da3_depth_proxy_504x322.yaml` preserves
MobileNetV2, width multiplier 1.0, fixed `[504,322]` height/width and the
original loss/optimisation settings. Input tensors are `[B,3,504,322]` RGB;
the output is a single-channel relative-depth map at that resolution.

## Training and deployment source

- `scripts/generate_da3_proxy_depth_cache.py` and
  `generate_da3_proxy_depth_cache_npy_xz.py`: existing teacher-cache tools.
- `scripts/train_da3_depth_proxy.py`: RGB/teacher-depth cache training.
- `scripts/export_depth_proxy_onnx.py`: floating-point ONNX export.
- `scripts/quantize_depth_proxy_qnn_qdq.py`: existing QNN-oriented QDQ tool.
- `scripts/evaluate_depth_proxy_student_pipeline.py`: proxy plus downstream
  sensing ONNX evaluation.

These sources are preserved for inspection; no training, export or
quantisation was run during archive preparation.

## Existing artifact family

Through this project's `outputs` link,
`da3_depth_proxy_504x322_mnv2_mixed_full/` contains:

- `best_val_loss.pt`;
- `depth_proxy_504x322.onnx` and its existing JSON export metadata;
- `depth_proxy_504x322_qnn_qdq_u16a_u8w.onnx` and its existing JSON metadata.

The metadata links the floating-point export to `best_val_loss.pt` and
records unsigned 16-bit activations, unsigned 8-bit weights and 512 calibration
frames for the quantised artifact. The files are copied byte for byte,
including the historical path strings in metadata. Hashes are in
[models.csv](../provenance/models.csv). This identifies an implementation/
artifact match, not a fresh validation of thesis measurements.

Training needs the original RGB/depth caches and manifests; mobile execution
needs the separately provided QNN runtime/hardware. The unsigned-16 QDQ
export was not executed as a generic CPU smoke test.
