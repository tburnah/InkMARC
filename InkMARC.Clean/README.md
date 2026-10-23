# InkMARC.Clean

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

InkMARC-Clean: reconstruction, inspection and export associated with Chapter 5, section 5.3.3. This repository contains a project and solution, beyond the reference excerpts preserved in the VM archive. No exact thesis run or desktop revision identity is inferred.

## Source and inputs

Existing video/session inputs and frame geometry/label metadata; inspection and dataset exports include HDF5 metadata and video handling. Research recordings and labels are not supplied as example inputs.

Start with:

- [ViewModels/MainViewModel.cs](ViewModels/MainViewModel.cs)
- [Services/SessionManager.cs](Services/SessionManager.cs)
- [Services/VideoService.cs](Services/VideoService.cs)
- [AttributeIndices.cs](AttributeIndices.cs)

## Build scope

Windows WPF; target `net8.0-windows7.0`, x64 and unsafe blocks enabled.

Open [InkMARC.Clean.sln](InkMARC.Clean.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Clean.csproj](InkMARC.Clean.csproj) and the [dependency inventory](../docs/dependencies.md).

VideoService invokes external FFmpeg. Inspect original native HDF5/OpenCV requirements, resource references and data paths before execution.

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
