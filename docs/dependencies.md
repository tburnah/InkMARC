# Declared .NET dependencies

This inventory comes from the eight existing project XML files. Versions and
targets were preserved; no package restore, upgrade or build was performed.
The [machine-readable inventory](repository-dependencies.json) includes target
conditions, exact package references and shared project references.

| Project | Declared target(s) | Direct package references |
| --- | --- | --- |
| [InkMARC.Clean/InkMARC.Clean.csproj](../InkMARC.Clean/InkMARC.Clean.csproj) | `net8.0-windows7.0` | CommunityToolkit.Mvvm 8.4.0; HDF.PInvoke.NETStandard 1.10.502; MaterialDesignThemes 5.3.0; Microsoft.Extensions.DependencyInjection 10.0.0; Microsoft.Extensions.Hosting 10.0.0; Microsoft.Extensions.Logging 10.0.0; Microsoft.ML.OnnxRuntime 1.23.2; Microsoft.ML.OnnxRuntime.Managed 1.23.2; OpenCvSharp4 4.11.0.20250507; OpenCvSharp4.Extensions 4.11.0.20250507; OpenCvSharp4.runtime.win 4.11.0.20250507; OpenCvSharp4.Windows 4.11.0.20250507; OpenCvSharp4.WpfExtensions 4.11.0.20250507; System.Drawing.Common 10.0.0; WindowsAPICodePack 8.0.12 |
| [InkMARC.Cue/InkMARC.Cue/InkMARC.Cue.csproj](../InkMARC.Cue/InkMARC.Cue/InkMARC.Cue.csproj) | `net9.0-android;net9.0-ios` | Microsoft.Extensions.DependencyInjection 9.0.3; Microsoft.Maui.Controls 9.0.50; Microsoft.Extensions.Logging.Debug 9.0.3; Plugin.Maui.Audio 3.0.1; TagLibSharp 2.3.0 |
| [InkMARC.Deform/InkMARC.Deform.csproj](../InkMARC.Deform/InkMARC.Deform.csproj) | `net9.0-android35.0; $(TargetFrameworks);net9.0-windows10.0.19041.0` | CommunityToolkit.Maui 7.0.1; CommunityToolkit.Mvvm 8.2.2; Microsoft.Extensions.Logging.Debug 8.0.0; Microsoft.Maui.Controls 8.0.7 |
| [InkMARC.Evaluate/InkMARC.Evaluate/InkMARC.Evaluate.csproj](../InkMARC.Evaluate/InkMARC.Evaluate/InkMARC.Evaluate.csproj) | `net9.0-android35.0` | Microsoft.Extensions.DependencyInjection 9.0.3; Microsoft.Maui.Controls 9.0.50; Microsoft.Extensions.Logging.Debug 9.0.3; Microsoft.Maui.Graphics 9.0.50; Microsoft.ML.OnnxRuntime 1.21.0; Plugin.Maui.Audio 3.0.1; TagLibSharp 2.3.0 |
| [InkMARC.Label/InkMARC.Label.csproj](../InkMARC.Label/InkMARC.Label.csproj) | `net9.0-windows` | CommunityToolkit.Mvvm 8.4.0; HDF5-CSharp 1.19.1; MaterialDesignThemes 5.2.1; Microsoft.Extensions.DependencyInjection 9.0.8; Microsoft.Extensions.Hosting 9.0.8; Microsoft.Extensions.Logging 9.0.8; Microsoft.ML.OnnxRuntime 1.22.1; Microsoft.ML.OnnxRuntime.Managed 1.22.1; OpenCvSharp4 4.11.0.20250507; OpenCvSharp4.Extensions 4.11.0.20250507; OpenCvSharp4.runtime.win 4.11.0.20250507; OpenCvSharp4.Windows 4.11.0.20250507; OpenCvSharp4.WpfExtensions 4.11.0.20250507; System.Drawing.Common 9.0.6; WindowsAPICodePack 8.0.6 |
| [InkMARC.Models/InkMARC.Models.csproj](../InkMARC.Models/InkMARC.Models.csproj) | `net8.0` | None declared |
| [InkMARC.Prepare/InkMARC.Prepare.csproj](../InkMARC.Prepare/InkMARC.Prepare.csproj) | `net9.0-windows` | CommunityToolkit.Mvvm 8.4.0; HDF5-CSharp 1.19.0; MaterialDesignThemes 5.1.0; Microsoft.ML.OnnxRuntime 1.20.1; Microsoft.ML.OnnxRuntime.Managed 1.20.1; OpenCvSharp4 4.10.0.20241108; OpenCvSharp4.Extensions 4.10.0.20241108; OpenCvSharp4.Windows 4.10.0.20241108; SixLabors.ImageSharp 3.1.6; SixLabors.ImageSharp.Drawing 2.1.5; System.Drawing.Common 9.0.0 |
| [InkMARC.Test/InkMARC.Test.csproj](../InkMARC.Test/InkMARC.Test.csproj) | `net9.0-windows` | CommunityToolkit.Mvvm 8.4.0; HDF5-CSharp 1.19.1; MaterialDesignThemes 5.2.1; Microsoft.ML.OnnxRuntime 1.21.0; Microsoft.ML.OnnxRuntime.Managed 1.21.0; Microsoft.ML.OnnxTransformer 4.0.2; Microsoft.ML.TensorFlow 4.0.2; Microsoft.ML.TensorFlow.Redist 0.14.0; OpenCvSharp4 4.10.0.20241108; OpenCvSharp4.Extensions 4.10.0.20241108; OpenCvSharp4.Windows 4.10.0.20241108; OpenCvSharp4.WpfExtensions 4.10.0.20241108; SixLabors.ImageSharp 3.1.7; SixLabors.ImageSharp.Drawing 2.1.5; System.Drawing.Common 9.0.3; WindowsAPICodePack 8.0.6 |

Clean, Prepare, Label and Test are Windows WPF projects. Cue, Deform and
Evaluate require their declared MAUI/platform workloads; Models is a shared
library. Native OpenCV/HDF5/runtime dependencies, external FFmpeg calls,
historical model/data paths and signing settings also affect execution.
Installed SDKs and available workloads have not been represented as a tested
environment. Do not modernise or restore packages merely to make this review
pass.

The project name InkMARC.Test denotes an application, not an automated test
suite. No application/device tests or research-data exports were performed.

See [exact-version package license evidence](licensing/nuget-license-evidence.json)
and [third-party notices](../THIRD_PARTY_NOTICES.md). A package may have native,
transitive or bundled dependencies beyond these direct declarations. The
imported VM tree retains its separate [Python/C++ dependencies](../research/docs/dependencies.md)
and [execution notes](../research/docs/reproducibility.md). Run those projects
from their documented directories within `research/`; no environment was
combined, restored or upgraded during the import.

Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original
contributions, subject to prior valid license grants and third-party rights.
