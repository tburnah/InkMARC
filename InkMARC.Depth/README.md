# InkMARC-Depth — distilled relative-depth proxy

Thesis §7.1.2 (pp. 116–117) describes a smaller depth model distilled from
Depth Anything 3 Small. Formerly called InkDepth / “InkMARC Depth”, this
component is now named **InkMARC-Depth**. It is the MobileNetV2 proxy extracted
from the sensing workspace, separate from the earlier students in
`research/src/depth_model/` at the repository root.

`inkmarc_depth/model.py` defines `InkMARCDepth` and the
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
- [Sense pipeline evaluation](../InkMARC.Sense/scripts/evaluate_depth_proxy_student_pipeline.py):
  proxy plus downstream sensing ONNX evaluation, retained with its sensing utilities.

These sources are preserved for inspection; no training, export or
quantisation was run during archive preparation.

## Setup and compatibility

From the repository root:

```sh
python -m pip install -r InkMARC.Depth/requirements.txt
python -m pip install -e ./InkMARC.Depth -e ./InkMARC.Sense
cd InkMARC.Depth
python scripts/train_da3_depth_proxy.py --help
python scripts/export_depth_proxy_onnx.py --help
```

The model package `inkmarc_depth` can be installed on its own for inference.
Training/cache/quantisation scripts share dataset and configuration helpers
from the adjacent `InkMARC.Sense` checkout. Run those scripts from this
component directory so configuration paths such as `outputs/...` resolve
correctly. Original datasets, teacher sources and weights remain external.

`DepthProxyMobileNet` remains an alias of `InkMARCDepth`, and the Sense
compatibility module `inkdepth.models.depth_proxy` forwards to this package.
Factories accept both model names. Parameter names and model artifact names
are unchanged. The `outputs` link resolves to
`../research/inkdepth_student/outputs`, shared with Sense.

For a checkout without symlink support, use the scripts' explicit checkpoint,
output and configuration override arguments with that shared directory.

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
[models.csv](../research/provenance/models.csv). This identifies an implementation/
artifact match, not a fresh validation of thesis measurements.

Training needs the original RGB/depth caches and manifests; mobile execution
needs the separately provided QNN runtime/hardware. The unsigned-16 QDQ
export was not executed as a generic CPU smoke test.

## Copyright and third-party rights

See [COPYRIGHT](COPYRIGHT), [repository terms](../LICENSE) and the
[research third-party notices](../research/THIRD_PARTY_NOTICES.md).
