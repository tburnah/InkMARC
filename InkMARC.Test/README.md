# InkMARC.Test

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Historical desktop prediction and annotation research application. The directory name does not identify an automated unit-test project; no exact thesis experiment/run association is inferred.

## Source and inputs

Images/video and an existing ONNX prediction path; interactive frame display, annotation and data-save operations. Referenced model/data paths remain unchanged.

Start with:

- [MainViewViewModel.cs](MainViewViewModel.cs)
- [ImagePredict.cs](ImagePredict.cs)
- [DataSave.cs](DataSave.cs)

## Build scope

Windows WPF; target `net9.0-windows`. Retain the sibling InkMARC.Models project.

Open [InkMARC.Test.sln](InkMARC.Test.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Test.csproj](InkMARC.Test.csproj) and the [dependency inventory](../docs/dependencies.md).

Declared dependencies include native OpenCV, HDF5, ML.NET/TensorFlow, ONNX Runtime, ImageSharp and Windows shell components. Exact package terms and missing external-input requirements must be checked before execution.

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
