# InkMARC-Clean — ground-truth reconstruction and export reference sources

These C# excerpts relate to thesis §5.3.3 (pp. 91–92): decoding recorded
diagnostic information, reconstructing frame-aligned stylus/surface
measurements, inspection and aligned AVI/HDF5 export.

The sources retain `InkMARC.Clean` namespaces. `VideoFileFrameSource.cs`
handles video and diagnostic-strip extraction; `FrameProcessor.cs`,
`GeometryHelper.cs`, `Hdf5SessionFrameSource.cs`, `VideoService.cs`,
`MainViewModel.cs` and `H5LabelStatscs.cs` retain their original roles.

These are reference excerpts, not a standalone buildable Clean release.
The complete C# solution, application types and project dependencies
are outside this snapshot. Preserve these sources as context for the
data contract; no application build or export was run during staging.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../../COPYRIGHT), [reserved-rights terms](../../LICENSE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
