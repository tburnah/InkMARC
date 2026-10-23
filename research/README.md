# InkMARC research archive

This archive accompanies Tavish Burnah's Massey University thesis,
*Passive Stylus Interaction on Ordinary Surfaces: Using Smartphone Video
See-Through Interaction, Machine Learning, and InkMARC*. InkMARC means
**Ink Mobile Augmented Reality Capture**.

The archive preserves research implementations for dataset preparation and
surface-appearance augmentation (Chapter 5), preliminary touch-state and
stylus-tip localisation experiments (Chapter 6), and InkMARC-Sense with its
distilled relative-depth proxy (Chapter 7). Original project paths, package
names, experimental variants and executable identifiers are retained.

## Navigate by thesis contribution

| Thesis contribution | Start here | Implementation status |
| --- | --- | --- |
| Chapter 5, §5.3.3: ground-truth reconstruction and export | [InkMARC-Clean reference sources](inkdepth_student_workspace/reference/README.md) | Desktop source excerpts |
| Chapter 5, §5.3.5: surface-appearance augmentation | [InkMARC-Augment](src/shadow_extraction/README.md) | Offline C++/Python workflow; historical versions retained |
| Chapter 5: D3/D3a manifests, filtering and splits | [Dataset preparation](docs/datasets.md) | Utilities distributed across the original projects |
| Chapter 6, §6.1: preliminary touch classification | [Touch experiments](src/touch_model/README.md) | Historical stand-alone experiments |
| Chapter 6, §6.2: preliminary localisation, E01–E11 | [Localisation experiments](src/location_model/README.md) | Historical experiment families |
| Chapter 7: final sensing architecture | [InkMARC-Sense](inkdepth_student_workspace/README.md) | Integrated sensing implementation and selected artifacts |
| Chapter 7, §7.1.2: distilled relative depth | [InkMARC Depth / InkDepth](inkdepth_student_workspace/DEPTH_PROXY.md) | MobileNetV2 DA3 proxy and existing exports |
| Chapter 8: live surface tracking | [Suite terminology and scope](docs/terminology-map.md) | Live tracker described in thesis; complete implementation not in this snapshot |

See the [detailed chapter index](docs/contribution-index.md),
[model index](docs/models.md), [execution notes](docs/reproducibility.md),
and [dependencies](docs/dependencies.md).

## Research progression and component relationships

The preliminary classifiers and localisation models investigate separate
sensing tasks. InkMARC-Sense combines image-space tip localisation,
interaction presence and touch state using a shared backbone. Its inputs
combine RGB, estimated relative depth and surface mask/U/V channels.
InkMARC Depth is the smaller depth proxy described in Chapter 7; earlier
depth students remain in a separate historical project.

InkMARC-Clean reconstructs aligned recordings. InkMARC-Augment uses reviewed
geometry, foreground separation and shadow transfer to construct D3a surface
variants. The live InkMARC-Surface tracker described in Chapter 8 supplies
runtime surface geometry; offline AprilTag sheet registration belongs to the
Chapter 5 preparation/augmentation workflow.

## Directory structure

```text
docs/                         chapter index, models, data and execution notes
provenance/                   copy hashes, source versions and change records
src/touch_model/              preliminary touch experiments and ancillary IMU work
src/location_model/           preliminary localisation and combined experiments
src/depth_model/              historical relative-depth student development
src/shadow_extraction/        InkMARC-Augment and preparation utilities
ml/                           supplementary LocTouchNetLite experiments
inkdepth_student/             earlier multitask student source snapshot
inkdepth_student/outputs/     selectively retained existing model artifacts
inkdepth_student_workspace/   InkMARC-Sense and DA3 proxy development workspace
```

The workspace `outputs` link points to `../inkdepth_student/outputs` within
this archive. Run `ml` modules from the archive root; run other project
scripts from their documented project roots. The two `inkdepth` packages are
historical snapshots and should use separate environments when installed.

## Models and data

Selected sensing, initialization, cross-validation and depth-proxy artifacts
are indexed with byte sizes and SHA-256 hashes in
[provenance/models.csv](provenance/models.csv). Their historical names are
retained. A component match does not by itself identify the exact checkpoint
behind a thesis table; unconfirmed run associations remain labelled as such.

Raw recordings, H5 research labels, participant-level predictions, large
manifests, generated depth caches, example photographs and texture assets
are not included. [Dataset notes](docs/datasets.md) explain their roles.
Capture, Cue, Train, Probe, Prepare, the complete Clean application, the live
Surface application and the integrated phone application are described in the
suite index where their complete sources are outside this snapshot.

## Preservation and practical scope

Research source, configurations and copied model binaries are preserved
byte for byte. Documentation and ignore-rule replacements are listed in
[the change log](provenance/CHANGELOG.md); replaced documents are retained
under `provenance/original-documentation/`.

Historical scripts contain VM paths and require separately supplied data,
upstream dependencies or mobile hardware. Some entry points train or export
models when run. Consult the [execution notes](docs/reproducibility.md)
before using them. Verification results are recorded in
[provenance/verification.json](provenance/verification.json); these checks do
not reproduce the thesis experiments.

This is a local publication candidate. Original contributions are all rights reserved; model distribution
remains subject to its upstream terms and author review. See [licensing](docs/licensing.md).

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](COPYRIGHT), [reserved-rights terms](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
