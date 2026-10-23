# InkMARC.Models

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [component rights notice](LICENSE.md); prior grants and third-party terms remain effective.

Shared ink, drawing-line, exercise, size and camera data types used by the historical .NET tools. This is not the Python model-architecture directory or a collection of trained weights.

## Source and inputs

Shared data objects/interfaces consumed by the applications; no standalone executable or training pipeline.

Start with:

- [Primatives/InkMARCPoint.cs](Primatives/InkMARCPoint.cs)
- [Primatives/InkMARCDrawingLine.cs](Primatives/InkMARCDrawingLine.cs)
- [Primatives/ExerciseData.cs](Primatives/ExerciseData.cs)
- [Interfaces/IAdvancedDrawingLine.cs](Interfaces/IAdvancedDrawingLine.cs)

## Build scope

Class library; target `net8.0`, referenced from sibling application projects.

The declared target and package versions are recorded in [InkMARC.Models.csproj](InkMARC.Models.csproj) and the [dependency inventory](../docs/dependencies.md).

Keep the original directory, namespace and historical `Primatives` spelling so project references and types remain intact.

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
