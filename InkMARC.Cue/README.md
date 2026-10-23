# InkMARC.Cue

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

InkMARC-Cue: prompted touch/no-touch recording, matching the Chapter 5, section 5.2.3 D2 collection role. This version is distinct from the latest InkMARC-Capture project pending incorporation from desktop.

## Source and inputs

Camera input and touch/no-touch/music prompts; the Android recording path produces session-named MP4 files. Participant recordings are not publication examples.

Start with:

- [InkMARC.Cue/MainPage.xaml.cs](InkMARC.Cue/MainPage.xaml.cs)
- [InkMARC.Cue/Platforms/Android/CameraPreviewHandler.cs](InkMARC.Cue/Platforms/Android/CameraPreviewHandler.cs)
- [InkMARC.Cue/SessionIDUtilities.cs](InkMARC.Cue/SessionIDUtilities.cs)

## Build scope

MAUI; declared targets `net9.0-android` and `net9.0-ios`. Android-specific camera recording is present; declared targets are not a claim of tested platform support.

Open [InkMARC.Cue.sln](InkMARC.Cue.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Cue/InkMARC.Cue.csproj](InkMARC.Cue/InkMARC.Cue.csproj) and the [dependency inventory](../docs/dependencies.md).

Requires the corresponding MAUI/device workloads and camera permissions. Existing music and font assets have separate rights; see the root notices before redistribution.

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
