# InkMARC.Probe

InkMARC Probe is the desktop stylus and camera data-collection application
formerly named **OcuInk / OcuInk Train**. The author confirmed its thesis name
when importing it into the InkMARC suite. It records writing exercises,
stylus input and camera video.

Open [InkMARC.Probe.sln](InkMARC.Probe.sln). This folder sits at the repository
root alongside the other InkMARC applications and contains:

- `InkMARC.Probe/`: the MAUI application, including Windows desktop support.
- `InkMARC.Probe.Models/`: this application's original data model library.
- `Camera.MAUI/`: its bundled camera component, retained with its own notices.

The local models remain separate from the suite's root `InkMARC.Models`
library so this migration does not change other applications' data types.

## Build

The application retains its .NET 8 targets: `net8.0-android34.0`, with
`net8.0-windows10.0.19041.0` added on Windows. Use a matching .NET 8 MAUI
development environment and the required Windows/Android SDKs. The camera
project also declares `net8.0` and `net8.0-maccatalyst`; those declarations
are preserved and do not imply all platforms have been tested.

For the Windows desktop target, from this directory:

```powershell
dotnet build .\InkMARC.Probe\InkMARC.Probe.csproj -f net8.0-windows10.0.19041.0 -p:AppxPackageSigningEnabled=false
```

The strong-name key reference now resolves to `../InkMARC.Probe.snk`
inside this directory. Packaged releases still require appropriate local
certificate/signing configuration; the historical certificate settings
have been retained.

Validation on 27 October 2026 used the installed .NET SDK 10.0.302. The
models library and Windows application compiled successfully. This newer
toolchain required explicitly supplying `MauiVersion=8.0.7` and restoring
the Windows targets and the plain `net8.0` models separately:

```powershell
dotnet restore .\InkMARC.Probe\InkMARC.Probe.csproj -p:TargetFrameworks=net8.0-windows10.0.19041.0 -p:MauiVersion=8.0.7
dotnet restore .\InkMARC.Probe.Models\InkMARC.Probe.Models.csproj
dotnet build .\InkMARC.Probe\InkMARC.Probe.csproj --no-restore -f net8.0-windows10.0.19041.0 -p:TargetFrameworks=net8.0-windows10.0.19041.0 -p:MauiVersion=8.0.7 -p:AppxPackageSigningEnabled=false
```

The Windows build reports nullability, MVVM, XAML binding, strong-name and
runtime-identifier warnings. Installation, camera/stylus operation and
Android execution were not tested.

## Migration and compatibility

Imported from source commit `33238557ea4995a26a80f55e48cea7041770ab88` of
`tavish-burnah/OcuInkTrain`. Application/project names, C#/XAML namespaces,
drawing type names, visible titles and repository metadata were renamed.
The original licenses and binary assets were preserved. The import does
not upgrade target frameworks, packages or collection behavior.

The application ID is now `com.tavishburnah.inkmarcprobe`. Existing OcuInk
installations and their private application data are not automatically
migrated into this new application identity. Recorded JSON field names
and video/data filename conventions are unchanged.

See the [import record](../docs/provenance/probe-import.json) for provenance
and validation results. The original source repository is retained locally
as a backup; its Git metadata is not nested inside this repository.

## Notices

The source repository's [Apache-2.0 license](LICENSE), application
[MIT notice](InkMARC.Probe/LICENSE.md), and camera
[MIT notice](Camera.MAUI/LICENSE.md) are retained unchanged. This import
does not reinterpret or replace those existing notices or third-party rights.
