# Thesis contribution index

References use printed thesis pages. In the supplied 245-page PDF, main-text
printed page p corresponds to PDF page p + 25. The thesis PDF is not included
in this candidate. This index maps implementation families; it does not
reconstruct experimental results or infer checkpoint identity from scores.

## Chapters 3 and 5: research tools and evidence construction

Section 3.1 (p. 51) defines the InkMARC suite. Dataset roles in Chapter 5 are
D0 / Probe (§5.2.1), D1 / Train (§5.2.2), D2 / Cue (§5.2.3), D3 / Capture
(§5.3), and D3a / Augment (§5.3.5). See [terminology](terminology-map.md).

| Thesis section | Archive source | Match |
| --- | --- | --- |
| §5.3.3, pp. 91–92: ground-truth reconstruction/export | `inkdepth_student_workspace/reference/`, earlier `inkdepth_student/reference/`, `src/shadow_extraction/MainViewModel.cs` | InkMARC.Clean namespaces, strip extraction, frame/H5 and geometry handling |
| §5.3.4, p. 92: retained source data/filtering | `scripts/create_stable_pressure_manifest.py`, `create_balanced_frame_manifest.py` and related utilities in the InkDepth projects | Preparation implementation family; retain each historical configuration |
| §5.3.5, pp. 92–96: surface appearance | `src/shadow_extraction/shadow_pipeline_v3.cpp`, CLI, depth helper and batch scripts | Direct workflow match for registration/review, foreground separation and shadow transfer |
| §5.3.6, pp. 96–97: derived dataset/partitions | `create_texture_manifest.py`, `create_participant_splits.py`, manifest and audit utilities | D3/D3a preparation and partition tooling |

## Chapter 6: preliminary sensing

Section 6.1 (pp. 102–104) concerns stand-alone touch/no-touch classification
and supports RQ3.1. `src/touch_model/` retains the NAS/AutoKeras,
EfficientNet, ResNet and deployment experiment families. The ResNet18
architecture is present in `src/Model15.py` and `train19.py`.

Section 6.2 (pp. 105–110) concerns stand-alone localisation and supports
RQ2.1. E01–E11 are thesis experiment identifiers, not source phase numbers.

| Thesis label | Code relative to `src/location_model/` | Association |
| --- | --- | --- |
| E01 Rectified direct regression | `src/train_phase1_v2_location.py`, location-only model/transform family | Family match |
| E02 Raw image + corners | `src/train_phase1a.py`, `src/models/model_v3.py` | Family match |
| E03 Heatmap–DSNT baseline | `src/train_phase2.py`, `src/models/model_phase2.py` | Family match |
| E04/E05 Larger/full-size heatmaps | Heatmap–DSNT trainer's heatmap-size options | Configuration-dependent family match |
| E06 Touch-restricted heatmap | `src/train_phase2a.py` and touch-restricted option | Family match |
| E07 Touch-restricted short context | `train_phase2c.py`, `train_phase2d.py`, `train_phase6.py` | Provisional variant association |
| E08 Causal smoothing | `src/train_phase3.py`, `src/eval/eval_phase3.py`, `src/eval/temporal_stabilizer.py` | Family match |
| E09 Clip-based temporal model | `src/train_phase4_tsm.py`, `src/models/model_phase4_tsm.py` | Family match |
| E10 Transformer temporal model | `src/train_phase5.py` | Family match |
| E11 History-conditioned coordinates | `src/train_phase2e.py`, `src/data/dataset_history.py`, associated evaluation tools | Three-position-history implementation match |

Exact historical commands and selected checkpoints for these experiment IDs
remain an author selection item. No E-number was inserted into source code.

## Chapter 7: InkMARC-Sense and InkMARC Depth

The current `inkdepth_student_workspace/` contains the integrated sensing
family supporting RQ2 (stylus location) and RQ3 (touch/hover).

| Thesis section | Source relative to the current workspace |
| --- | --- |
| §7.1.1, p. 115: surface mask and U/V encoding | `inkdepth/datasets/stylus_dataset.py`; `scripts/audit_surface_geometry_maps.py` |
| §7.1.2, pp. 116–117: distilled DA3 depth proxy | `inkdepth/models/depth_proxy.py`; `configs/da3_depth_proxy_504x322.yaml`; training/cache/export/quantisation utilities |
| §7.1.3, pp. 117–118: shared MobileNetV3/FPN | `inkdepth/models/inkdepth_student.py` |
| §7.1.4, p. 118: heatmap and soft-argmax | `inkdepth/models/heads.py`; location path in `inkdepth_student.py` |
| §7.1.5, pp. 118–121: presence and touch heads | Heatmap-weighted feature/depth pooling and separate heads in `inkdepth_student.py` |
| §7.2, p. 121: implementation settings | Original YAML configurations, model definitions and selected export metadata |
| §7.3: evaluation | `evaluate_touch_location_report.py`, `evaluate_touch_location_by_participant.py`, `run_participant_crossval.py`, `evaluate_participant_cv_statistics.py`, `evaluate_presence_cv_threshold.py` |

`InkDepthDA3Student` supplies live frozen DA3 depth during development;
`DepthProxyMobileNet` and `evaluate_depth_proxy_student_pipeline.py` provide
the smaller proxy path. See [model provenance](models.md) for the distinction
between component identity and individual historical checkpoint selection.

## Chapter 8 and supplementary material

Chapter 8 (pp. 130–159) describes InkMARC-Surface: a live gravity-constrained
tracker using camera imagery and smartphone orientation. The complete phone
implementation is outside this snapshot. Offline tag-sheet registration in
`shadow_extraction/` is indexed under Chapter 5.

`ml/`, combined touch/location experiments, historical depth students and
ancillary inertial-odometry work remain supplementary research sources.
No direct reported-experiment association is asserted for those folders.
