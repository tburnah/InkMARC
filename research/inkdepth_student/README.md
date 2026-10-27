# Integrated sensing development — historical student experiments

This earlier `inkdepth` source snapshot preserves depth-supervised,
multitask and temporal-touch development related to the Chapter 7 sensing
work. It remains distinct from the later
[InkMARC-Sense workspace](../../InkMARC.Sense/README.md).

The package contains `datasets/stylus_dataset.py`, model/heads/losses,
training/validation and configuration/checkpoint/export helpers.
`scripts/train_student.py`, `validate_student.py`, `export_student_onnx.py`
and temporal/feature-based touch utilities retain their original interfaces.
Its `configs/` directory preserves historical settings without modification.

Inputs include recorded RGB, H5 labels/geometry and pseudo-depth according
to configuration. Outputs include touch/state logits, coordinates, depth
predictions, checkpoints and evaluation artifacts. Setup definitions are in
`requirements.txt` and `pyproject.toml`; external datasets are described in
[dataset requirements](../docs/datasets.md).

Use an environment separate from the current workspace when installing this
snapshot: both export the `inkdepth` package name. Its original README is
retained under `provenance/original-documentation/`.

`outputs/` is a historical shared storage location. The selectively copied
files there include later workspace artifacts; their placement does not mean
this older source snapshot created them. Use [the model index](../docs/models.md)
to identify the corresponding source family. The root Sense and Depth components' relative
`outputs` links point here, preserving that shared-output relationship within
the archive.

`reference/` contains InkMARC-Clean source excerpts associated with §5.3.3;
it is not the complete desktop application.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../COPYRIGHT), [reserved-rights terms](../LICENSE) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
