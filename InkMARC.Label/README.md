# InkMARC.Label

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Historical touch/location labelling, geometry review and dataset export. Preserve this workflow separately from the Chapter 5 InkMARC-Clean project and VM reference snapshots.

## Source and inputs

Recorded videos and annotation/project data; touch/location labels, geometry and HDF5/video export. Existing ONNX bounds prediction is part of the historical workflow.

Start with:

- [ViewModels/LocationLabellingViewModel.cs](ViewModels/LocationLabellingViewModel.cs)
- [ViewModels/TouchLabellingViewModel.cs](ViewModels/TouchLabellingViewModel.cs)
- [Services/SessionManager.cs](Services/SessionManager.cs)
- [Services/BoundsPredictor.cs](Services/BoundsPredictor.cs)

## Build scope

Windows WPF; target `net9.0-windows`. Retain the sibling InkMARC.Models project.

Open [InkMARC.Label.sln](InkMARC.Label.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Label.csproj](InkMARC.Label.csproj) and the [dependency inventory](../docs/dependencies.md).

Uses external FFmpeg and native OpenCV/HDF5 dependencies. A historical Python tracking integration and model/data paths remain in source; no replacement environment or model is generated.

Project XML and source entry points were inspected. Packages were not restored;
the application was not built or run. No model was trained, exported or changed.

## Rights and archive status

See the [root license](../LICENSE), [copyright scope](../COPYRIGHT),
[historical licensing record](../docs/licensing/README.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Existing prior grants and
third-party permissions remain effective; this README grants no new general
reuse permission.

The [contribution index](../docs/contribution-index.md) and
[integration plan](../docs/integration-plan.md) distinguish existing source,
the incorporated VM research and the two pending desktop contributions.

Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original
contributions, subject to prior valid license grants and third-party rights.
