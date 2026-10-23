# Preliminary touch-state classification

This historical project supports thesis §6.1 (pp. 102–104), the stand-alone
touch/no-touch model development associated with RQ3.1. Preserve its model
names and variants as records of the research progression.

`src/Model*.py` contains AutoKeras, TensorFlow/Keras and PyTorch experiments.
The ResNet18 + 256-unit/dropout/sigmoid architecture is identifiable in
`src/Model15.py` and `train19.py`; other trainers include `train1.py`,
`train2.py` and `train18torch.py`. Evaluation, inference, TFRecord conversion,
ONNX export and quantisation utilities remain under their original names.

Inputs are RGB imagery and binary labels loaded from historical TFRecord,
NumPy or H5 paths. Outputs are trained classifiers, evaluation values and
deployment exports. These scripts use different frameworks and data
conventions; a single shared environment or preprocessing recipe is not
inferred. Some scripts perform training or file writes at module scope.

TensorFlow/Keras, AutoKeras, PyTorch/torchvision, scikit-learn, tf2onnx and
ONNX tools appear across variants. No complete historical dependency lock
or universally applicable invocation was found. Existing model binaries and
research examples remain held for author selection; no model was regenerated.

`imu_odometry/` is retained as an ancillary sensor-window research baseline.
Its original README documents its own input columns and entry points; it is
not assigned a direct thesis-reported experiment here.

See [the chapter index](../../docs/contribution-index.md) and
[dataset requirements](../../docs/datasets.md).

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../../COPYRIGHT), [reserved-rights terms](../../LICENSE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
