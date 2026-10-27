# InkMARC — source code accompanying the thesis

This repository contains the software and model-development code accompanying
Tavish Burnah's PhD thesis in Computer Science at Massey University, New Zealand:

**Passive Stylus Interaction on Ordinary Surfaces: Using Smartphone Video
See-Through Interaction, Machine Learning, and InkMARC**

**InkMARC** stands for **Ink Mobile Augmented Reality Capture**. The suite
supports research into using a passive stylus on ordinary physical surfaces
through smartphone video see-through interaction. It includes instrumented
data collection, recording preparation and annotation, surface-appearance
augmentation, learned stylus sensing, and mobile surface tracking.

The repository contains research implementations, experimental variants,
configuration files and selected trained models. The component guide below
connects these sources to the thesis; component READMEs provide more detailed
entry points and requirements.

## Find the code by thesis contribution

Thesis component names use hyphens, such as **InkMARC-Sense**; source folders
use dots, such as `InkMARC.Sense`.

| Thesis contribution | Component and source | Purpose |
| --- | --- | --- |
| Chapter 5: exploratory collection, D0 | [InkMARC-Probe](InkMARC.Probe/README.md) | Stylus, writing-exercise and camera collection; formerly OcuInk / OcuInk Train |
| Chapter 5: prompted collection, D2 | [InkMARC-Cue](InkMARC.Cue/README.md) | Prompted touch/no-touch recordings |
| Chapter 5: instrumented collection, D3 | [InkMARC-Capture](InkMARC.Capture/README.md) | Android camera recording, AprilTag detection and USB stylus ground truth |
| Chapter 5: reconstruction and preparation | [InkMARC-Clean](InkMARC.Clean/README.md) | Recording inspection, geometry correction and paired video/HDF5 export |
| Chapter 5: surface-appearance augmentation, D3a | [InkMARC-Augment](InkMARC.Augment/README.md) | Surface replacement with foreground preservation and shadow/illumination transfer |
| Chapter 6: preliminary touch sensing | [InkMARC-Touch](InkMARC.Touch/README.md) | Stand-alone touch-state classifiers and associated training/evaluation tools |
| Chapter 6: preliminary location sensing | [InkMARC-Location](InkMARC.Location/README.md) | Stylus-tip localisation experiments, including regression, heatmap and temporal models |
| Chapter 7: integrated sensing | [InkMARC-Sense](InkMARC.Sense/README.md) | Shared model for stylus-tip location, interaction presence and touch state |
| Chapter 7: distilled relative depth | [InkMARC-Depth](InkMARC.Depth/README.md) | Compact depth proxy distilled from Depth Anything 3, previously called InkDepth / InkMARC Depth |
| Chapter 8: mobile surface tracking | [InkMARC-Surface](InkMARC.Surface/) | Android surface-tracking and interaction source |

Capture and Clean provide aligned recordings and reference labels. Augment
produces surface-appearance variants of those recordings. Depth supplies
estimated relative depth to the sensing pipeline, while Sense predicts
stylus location, presence and touch. Surface maintains the writing-surface
geometry used to map image-space interaction into surface coordinates.

Touch and Location group the preliminary Chapter 6 experiments. Their folder
names are descriptive suite names; individual experimental variants retain
their original identifiers. The [detailed experiment index](research/docs/contribution-index.md)
maps the thesis's E01–E11 location experiments to implementation families.
These associations do not establish that every included variant or checkpoint
produced a reported thesis result.

## Supporting applications and earlier experiments

| Source | Purpose |
| --- | --- |
| [InkMARC.Prepare](InkMARC.Prepare/README.md) | Earlier video/ink alignment, frame processing and HDF5 export |
| [InkMARC.Label](InkMARC.Label/README.md) | Touch/location annotation, surface geometry and dataset export |
| [InkMARC.Deform](InkMARC.Deform/README.md) | Historical instrumented pen/exercise and pressure-related collection tools |
| [InkMARC.Evaluate](InkMARC.Evaluate/README.md) | Smartphone camera and ONNX touch-inference experiments |
| [InkMARC.Test](InkMARC.Test/README.md) | Desktop prediction and annotation experiments; an application rather than a unit-test suite |
| [InkMARC.Models](InkMARC.Models/README.md) | Shared ink, exercise and camera data types |
| [Earlier student experiments](research/inkdepth_student/README.md) | Historical multitask sensing snapshot and shared model artifact storage |
| [Earlier depth experiments](research/src/depth_model/README.md) | Depth-student development preceding InkMARC-Depth |
| [Supplementary sensing experiments](research/ml/README.md) | LocTouchNetLite and related experiments |

InkMARC-Train is the thesis's D1 collection application name. Its exact
association with the historical collection versions in this repository remains
unconfirmed; Deform is therefore retained under its existing name.

The [component map](docs/component-layout.md) records the source reorganisation,
important files and compatibility names. Historical import inventories retain
their original paths; the [path mapping](docs/component-layout.json) locates
those files in the current layout.

## Getting started

Clone the repository with its native AprilTag dependency:

```sh
git clone --recurse-submodules https://github.com/tburnah/InkMARC.git
cd InkMARC
```

For an existing clone, run `git submodule update --init --recursive`.
Select the component that corresponds to the part of the thesis you want to
inspect or run. Each component has its own dependencies and working directory;
there is no single solution or build command for the whole suite.

| Component family | Development environment |
| --- | --- |
| Sense and Depth | Python 3.10 or later, PyTorch/torchvision and the component requirements |
| Preliminary and supplementary models | Experiment-specific Python environments; see the relevant README |
| Augment | C++17, OpenCV, HDF5 and Python helpers; build scripts retain local dependency paths |
| Capture and Surface | Separate Android Gradle projects; open the selected folder in Android Studio |
| Collection, preparation and annotation applications | Component-specific .NET, WPF or MAUI SDKs and platform workloads |

For Sense and Depth, use a dedicated Python environment and install from the
repository root:

```sh
python -m pip install -r InkMARC.Sense/requirements.txt
python -m pip install -e ./InkMARC.Depth -e ./InkMARC.Sense
```

Then run scripts from the relevant component directory, following its README.
Sense retains the historical `inkdepth` Python namespace and compatible model
aliases. Install the earlier `research/inkdepth_student` snapshot in a separate
environment because it uses the same namespace.

See the [application dependency inventory](docs/dependencies.md),
[research dependencies](research/docs/dependencies.md) and
[execution notes](research/docs/reproducibility.md) for further requirements.
Data roots, teacher sources, weights and hardware settings must be supplied
for the chosen experiment; historical machine paths may need overrides.

## Models, data and reproducibility

Selected checkpoints and exports are documented in the
[model index](research/docs/models.md), with sizes and SHA-256 hashes in the
[model inventory](research/provenance/models.csv). Sense and Depth share the
artifacts in `research/inkdepth_student/outputs` through relative `outputs`
links. If a checkout does not support symbolic links, supply explicit paths
to that shared directory through the scripts' arguments or configuration
overrides.

Raw participant recordings, research label datasets, large manifests and
generated depth caches are not included. The [dataset notes](research/docs/datasets.md)
describe the inputs and their roles. Dataset-dependent training and evaluation
require those inputs and the corresponding split/configuration records.

The component reorganisation was checked through file-preservation audits,
Python and shell syntax checks, and tests that load existing Sense and Depth
checkpoints and run inference on synthetic inputs. After installing the Python
requirements, run the compatibility tests from the repository root:

```sh
python -m unittest discover -s InkMARC.Depth/tests -v
```

These checks do not reproduce the thesis experiments or establish mobile
runtime performance. The [verification record](docs/component-layout.md#verification)
states what was checked and the environment limitations. Original import
records and run inventories remain in [research provenance](research/provenance/README.md).

## Referencing this repository

When referring to this software in research, cite the accompanying thesis and
identify this repository, the component used and the Git commit or release
version. Record the model filename and configuration when using a supplied
checkpoint so the implementation can be distinguished from other experimental
variants.

## Copyright and licensing

Copyright (c) 2024–2026 Tavish Burnah. **All rights reserved in original
contributions, subject to prior valid licence grants and third-party rights.**

See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT) and the
[licensing history](docs/licensing/README.md) for the applicable scope.
Third-party code, adapted material and model parameters retain their own terms;
see the [repository notices](THIRD_PARTY_NOTICES.md),
[research notices](research/THIRD_PARTY_NOTICES.md) and
[model-rights records](research/provenance/model-licensing.json).
