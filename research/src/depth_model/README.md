# Relative-depth model development — historical prototypes

This project preserves the earlier relative-depth student models, teacher-label
generation and comparison utilities. It records research progression alongside
the later [InkMARC Depth / InkDepth proxy](../../../InkMARC.Depth/README.md)
associated with thesis §7.1.2. An exact thesis experiment is not inferred for
every historical variant.

`models/student_depth_net*.py` defines the architecture families.
`train_student*.py` preserves their training and export implementations.
`build_teacher_dataset.py` and `generate_depth_teacher_labels.py` prepare
pseudo-depth data; `compare_*` and other inspection tools compare predictions.

Inputs include RGB/pseudo-depth pairs and manifests under
`/data/depth/pseudo_depth_dataset*`, research video/TFRecords, and third-party
teacher weights. Outputs include checkpoints, ONNX models and generated
previews. For example, `train_student_v7_edgefocus.py` uses the architecture
defined in `models/student_depth_net_v6.py`; filenames are not rewritten.

PyTorch/torchvision, NumPy, Pillow, OpenCV and tqdm appear throughout; some
preparation utilities also use TensorFlow or Zarr. The locally used Depth
Anything checkouts are pinned in [third-party provenance](../../provenance/third-party.json)
and must be separately provided at their established paths when needed.

Earlier weights, per-recording Zarr data and preview images remain untouched
in the source workspace and are held for selection. No historical model has
been converted, regenerated or promoted to the thesis-final proxy by renaming.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../../COPYRIGHT), [reserved-rights terms](../../LICENSE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
