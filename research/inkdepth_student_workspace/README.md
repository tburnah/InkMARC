# InkMARC-Sense — final sensing model

This historical `inkdepth_student_workspace` contains the integrated sensing
implementation described in thesis Chapter 7 (pp. 113–129), supporting RQ2
(stylus-tip location) and RQ3 (touch/hover). The `inkdepth` executable package
name is preserved. “Final sensing model” identifies the chapter's architecture;
individual artifact-to-result associations remain separately indexed.

## Model and interfaces

`inkdepth/models/inkdepth_student.py` defines the shared MobileNetV3-Large
feature extractor, feature pyramid and task paths. `heads.py` defines the
heatmap location/soft-argmax and classification heads. The Chapter 7 family
combines RGB, relative depth and three surface mask/U/V channels. Presence
and touch use global features plus heatmap-weighted visual/depth evidence.

`inkdepth/datasets/stylus_dataset.py` loads frames, H5 labels/geometry and
configuration-dependent depth, target validity and surface maps. The source
YAML convention uses `[672,432]` for input height/width. Outputs include
image-normalised xy, a location heatmap, touch logits and optional
presence/state logits; model variants can also return depth.

`InkDepthDA3Student` provides live frozen DA3 depth for development. The
[InkMARC Depth / InkDepth proxy](DEPTH_PROXY.md) is a separate MobileNetV2
depth provider for the two-model deployment path. Relative depth is not a
calibrated physical-distance measurement.

## Important entry points

| Purpose | Files under `scripts/` |
| --- | --- |
| Model training/validation | `train_student.py`, `validate_student.py` |
| Splits/filtering/data preparation | `create_participant_splits.py`, `create_stable_pressure_manifest.py`, `create_texture_manifest.py`, related manifest tools |
| Touch/location evaluation | `evaluate_touch_location_report.py`, `evaluate_touch_location_by_participant.py` |
| Cross-validation/statistics | `run_participant_crossval.py`, `evaluate_participant_cv_statistics.py`, `evaluate_presence_cv_threshold.py` |
| Deployment/inference tools | `export_student_onnx.py`, `extract_deployable_student_checkpoint.py`, `evaluate_depth_proxy_student_pipeline.py`, ONNX/DA3 parity and benchmark scripts |
| Historical alternatives | Temporal touch, depth features/fusion, RGB-only, direct-location and ablation tools |

The preserved configurations include
`final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml`,
`inkmarc_rgbd_geometry_export_best_strict.yaml` and
`inkmarc_rgbd_geometry_presence_touch_location_export_best_state.yaml`.
They are separate historical configurations, not interchangeable labels.

## Setup and use

`requirements.txt` and `pyproject.toml` preserve the declared setup
(Python >=3.10). Use this project root as the working directory, and a
separate environment from the earlier snapshot. See
[dependencies](../docs/dependencies.md) for optional utility and DA3 imports.
No dependency was updated during archiving.

Existing scripts accept `--config` and, where implemented, dotted
`--override KEY=VALUE` arguments. [Execution notes](../docs/reproducibility.md)
describe those interfaces and output behaviour. Original research data,
manifests, teacher sources/weights and mobile runtimes are separate inputs.

The `outputs` link is internal to this archive and exposes selected existing
checkpoints/exports under their original relative names. See
[the model index](../docs/models.md) for artifact roles and hashes.
`reference/` contains [InkMARC-Clean excerpts](reference/README.md).

The archive includes ancillary experiments without claiming that every
configuration or file produced a reported thesis result. Original README
content is retained in provenance for historical context.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../COPYRIGHT), [reserved-rights terms](../LICENSE) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
