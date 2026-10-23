# InkMARC.Deform

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Historical instrumented pen/exercise collection and hand/pressure interaction tooling. Association with the exact thesis Probe or Train application version remains unconfirmed; this is not a replacement for desktop InkMARC-Capture.

## Source and inputs

Digitizer/exercise interaction and capture utilities; ink/exercise records and collection outputs. Keep participant data outside the public archive.

Start with:

- [ViewModel/DrawingPageViewModel.cs](ViewModel/DrawingPageViewModel.cs)
- [Utilities/DataUtilities.cs](Utilities/DataUtilities.cs)
- [Utilities/CameraExtractor.cs](Utilities/CameraExtractor.cs)
- [MauiProgram.cs](MauiProgram.cs)

## Build scope

MAUI; declared Android target `net9.0-android35.0` with conditional `net9.0-windows10.0.19041.0` on Windows. Original package versions include .NET 8-era MAUI packages; they have not been updated.

Open [../InkMARC.Deform.sln](../InkMARC.Deform.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Deform.csproj](InkMARC.Deform.csproj) and the [dependency inventory](../docs/dependencies.md).

Retain the sibling InkMARC.Models reference. Original signing settings and machine-specific paths remain unchanged. The tracked strong-name file requires a release-handling decision; no key material is reproduced here. The current component notice reserves rights in original contributions; the historical MIT text and prior valid permissions are preserved.

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
