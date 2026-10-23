# InkMARC.Evaluate

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Historical mobile camera/ONNX touch-inference and calibration experiments. No exact Chapter 6 experiment identity is asserted. This application does not represent the final Chapter 8 InkMARC-Surface/integrated implementation, which is pending incorporation from desktop.

## Source and inputs

Live camera images, existing ONNX models and touch/no-touch sampling; application inference/calibration output. No participant session was run during this review.

Start with:

- [InkMARC.Evaluate/MainPage.xaml.cs](InkMARC.Evaluate/MainPage.xaml.cs)
- [InkMARC.Evaluate/Platforms/Android/OnnxModelRunner.cs](InkMARC.Evaluate/Platforms/Android/OnnxModelRunner.cs)
- [InkMARC.Evaluate/IModelRunner.cs](InkMARC.Evaluate/IModelRunner.cs)

## Build scope

MAUI; declared target `net9.0-android35.0`. Other platform source files are retained without implying enabled or tested targets.

Open [InkMARC.Evaluate.sln](InkMARC.Evaluate.sln) from the cloned repository layout.

The declared target and package versions are recorded in [InkMARC.Evaluate/InkMARC.Evaluate.csproj](InkMARC.Evaluate/InkMARC.Evaluate.csproj) and the [dependency inventory](../docs/dependencies.md).

Existing Android ONNX assets and `onnxruntime-android-1.21.1.aar` are preserved. They are distinct from the VM-selected artifacts and need their own model/runtime provenance review. Native runtime notices are retained centrally.

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
