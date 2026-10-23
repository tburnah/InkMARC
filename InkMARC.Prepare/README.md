# InkMARC.Prepare

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Historical InkMARC-Prepare video/ink alignment and preparation workflow. Its role relates to Chapter 5 preparation; exact experiment-version identity is not asserted.

## Source and inputs

Video and recorded ink-point data; frame extraction, timestamp alignment, model prediction and HDF5 export. Source/configured model and data paths remain historical.

Start with:

- [MainViewViewModel.cs](MainViewViewModel.cs)
- [DataSave.cs](DataSave.cs)
- [Utilities.cs](Utilities.cs)

## Build scope

Windows WPF; target `net9.0-windows`. Retain the sibling InkMARC.Models project.

Open [../IncMARC.Prepare.sln](../IncMARC.Prepare.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Prepare.csproj](InkMARC.Prepare.csproj) and the [dependency inventory](../docs/dependencies.md).

The root solution is spelled `IncMARC.Prepare.sln`; this historical spelling is preserved. Model paths, native OpenCV/HDF5 dependencies and input data must be checked on the intended Windows machine.

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
