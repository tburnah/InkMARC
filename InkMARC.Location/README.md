# InkMARC-Location — preliminary stylus-tip localisation

Moved from `research/src/location_model`. InkMARC-Location is the suite folder
name for these Chapter 6 experiments; individual phase/model names are retained.

This historical project contains the localisation experiments associated with
thesis §6.2 (pp. 105–110), supporting RQ2.1. It retains regression, explicit
geometry, heatmap–DSNT, temporal and history-conditioned implementations.

The [chapter index](../research/docs/contribution-index.md) maps E01–E11 to code
families. Source `phase` identifiers are preserved and are not equated with
thesis experiment numbers. Exact historical run/checkpoint selection remains
separate from those family-level associations.

Main entry points are `src/train_phase*.py`; model definitions are in
`src/models/`, transforms/loaders in `src/data/`, and evaluation in `src/eval/`
and `src/evaluate*.py`. `scripts/` contains indexing, geometry/data inspection
and deployment-evaluation tools. Earlier surface-bound and coordinate
experiments at the project root remain intact.

Most indexed models consume video frames, H5 labels and surface corners and
use a canonical 480×320 RGB surface view. Outputs depend on the variant:
coordinates, location heatmaps, interaction outputs or surface bounds.
PyTorch/torchvision, OpenCV, NumPy, HDF5 and ONNX tooling are common.

The expected `/data/LocationData/indices` tree is external and was absent as
a live index at staging. See [data contracts](../research/docs/datasets.md).
Most trainers expose their own arguments; they were not run during archiving.
Do not import a historical trainer as a passive inspection step.

`src/combined_touch_location/README_combined.md` describes an additional shared
MobileNetV3 state/location experiment. It remains distinct from Chapter 7's
depth-and-geometry-aware InkMARC-Sense implementation. Historical checkpoints
and exports in this project remain indexed locally for author selection.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../research/COPYRIGHT), [reserved-rights terms](../research/LICENSE) and
[third-party notices](../research/THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
