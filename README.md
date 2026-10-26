# InkMARC research archive

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See [LICENSE](LICENSE) for scope, prior grants and third-party exceptions.

InkMARC means **Ink Mobile Augmented Reality Capture**. This repository
contains research tools supporting Tavish Burnah's Massey University PhD
thesis, *Passive Stylus Interaction on Ordinary Surfaces: Using Smartphone
Video See-Through Interaction, Machine Learning, and InkMARC*.

The existing .NET projects preserve data-collection, preparation, labelling,
cleaning and evaluation implementations. The imported [VM research archive](research/README.md)
under `research/` adds offline augmentation, preliminary sensing,
InkMARC-Sense and InkMARC Depth. Its frozen Phase B files are preserved intact.

**Archive completion is pending.** The latest InkMARC-Capture application is
now incorporated as [InkMARC.Capture](InkMARC.Capture/README.md). The Chapter
8 InkMARC-Surface/integrated application source remains stored on the
author's desktop and pending incorporation. Existing repository applications
and older VM implementations do not replace that final desktop project.

## Existing applications and thesis navigation

| Source | Purpose and contribution | Coverage |
| --- | --- | --- |
| [InkMARC.Clean](InkMARC.Clean/README.md) | Reconstruction, inspection, geometry and paired video/HDF5 export; Chapter 5, §5.3.3 | Project and solution present, rather than only the VM reference excerpts |
| [InkMARC.Cue](InkMARC.Cue/README.md) | Prompted touch/no-touch recording; Chapter 5, §5.2.3, D2 | Project and solution present; device build not verified here |
| [InkMARC.Capture](InkMARC.Capture/README.md) | Android camera, AprilTag and USB-stylus collection; Chapter 5, §5.3.1, D3 | Gradle project and native AprilTag submodule present; device build not verified here |
| [InkMARC.Prepare](InkMARC.Prepare/README.md) | Earlier video/ink alignment, frame processing and HDF5 export | Historical preparation workflow; original solution spelling retained |
| [InkMARC.Deform](InkMARC.Deform/README.md) | Instrumented pen/exercise collection and pressure-related data | Historical collection application; exact Probe/Train version association remains unconfirmed |
| [InkMARC.Label](InkMARC.Label/README.md) | Touch/location annotation, geometry and dataset export | Historical labelling workflow, preserved separately from Clean |
| [InkMARC.Evaluate](InkMARC.Evaluate/README.md) | Smartphone camera/ONNX touch-inference experiments | Historical evaluation application; not the final Chapter 8 application |
| [InkMARC.Test](InkMARC.Test/README.md) | Desktop video/image prediction and annotation experiments | Historical WPF application, not an automated unit-test suite |
| [InkMARC.Models](InkMARC.Models/README.md) | Shared ink, exercise and camera data types | Shared library; retain sibling project references |
| [InkMARC-Augment](research/src/shadow_extraction/README.md) | Offline surface-appearance augmentation; Chapter 5, §5.3.5 | Imported VM workflow; distinct from Chapter 8 live tracking |
| [Touch experiments](research/src/touch_model/README.md) and [localisation experiments](research/src/location_model/README.md) | Preliminary sensing; Chapter 6, §§6.1–6.2 | Historical implementation families preserved |
| [InkMARC-Sense](research/inkdepth_student_workspace/README.md) | Integrated location/presence/touch sensing; Chapter 7 | VM implementation, configurations and selected model artifacts imported |
| [InkMARC Depth / InkDepth](research/inkdepth_student_workspace/DEPTH_PROXY.md) | Distilled DA3 MobileNetV2 depth proxy; Chapter 7, §7.1.2 | Existing proxy source and selected exports imported |

The [contribution index](docs/contribution-index.md) distinguishes the
existing applications, incorporated VM components and pending desktop components.
These are implementation-family associations, not claims that a particular
repository commit or model produced a particular reported result.

## VM research integration

The combined layout keeps every existing project in place and includes
the frozen Phase B publication tree intact beneath `research/`:

```text
InkMARC.Clean/                existing .NET projects remain in place
InkMARC.Cue/
InkMARC.Capture/             Android D3 collection project
InkMARC.Deform/
InkMARC.Evaluate/
InkMARC.Label/
InkMARC.Models/
InkMARC.Prepare/
InkMARC.Test/
docs/                        repository index, licensing and import provenance
research/                    imported VM archive; frozen files preserved
  src/                       Augment and historical sensing/depth experiments
  ml/                        supplementary sensing experiments
  inkdepth_student/          earlier source and selected existing models
  inkdepth_student_workspace/ InkMARC-Sense and InkMARC Depth
  docs/                      VM-specific documentation and third-party notices
  provenance/                preserved Phase B hashes and inventories
applications/                proposed future desktop project; not added yet
  InkMARC/                   Chapter 8 application and original Surface layout
```

See the [import review](docs/research-import.md),
[integration record](docs/integration-plan.md) and
[preserved file mapping](docs/integration-plan.json). Keeping the VM
tree together preserves relative imports, its internal model-output symlink,
component documentation and checksum scope. The existing .NET solutions,
shared library and historical VM copies will remain separate.

The nested [VM contribution index](research/docs/contribution-index.md),
[model index](research/docs/models.md), [execution notes](research/docs/reproducibility.md)
and [dependencies](research/docs/dependencies.md) retain their Phase B scope.
Use this root README and the [combined contribution index](docs/contribution-index.md)
for current archive coverage, including the existing Clean project and the
pending Chapter 8 desktop addition. The nested README's statements about applications
outside its snapshot describe that VM snapshot, not the whole combined repository.

The import contains 646 regular files and one internal symlink (about 356 MiB).
All selected research source, configurations and models remain unchanged.
Private review records and withheld research data were not imported.

## Build and practical requirements

Targets differ between projects: Clean uses .NET 8 Windows/WPF; Prepare,
Label and Test use .NET 9 Windows/WPF; Cue targets .NET 9 Android/iOS;
Deform and Evaluate declare .NET 9 Android targets, with an additional
conditional Windows target in Deform; Models targets .NET 8.
See each component README, the [declared .NET dependency inventory](docs/dependencies.md)
and the [VM execution notes](research/docs/reproducibility.md). Run VM entry
points from their documented roots within `research/`; the VM archive root
is `research/`, while each original Python project retains its own root.

Clone with:

```sh
git clone https://github.com/tburnah/InkMARC.git
cd InkMARC
```

Open the selected existing solution on a machine with its matching SDK,
platform workloads and native dependencies. There is no single root solution
for the whole suite. Historical machine paths, signing settings, research
inputs and external model references are preserved. Project files were
inspected without restoring packages, building applications or running
research-data operations. No device or full experiment reproduction is claimed.

Raw participant recordings and data are not supplied by the VM publication
candidate. Existing repository ONNX files, the Android runtime archive, music,
fonts and signing material retain their historical state and require their
own release review; they are not cleared by the VM-only review.

## Copyright and licensing

Copyright (c) 2024–2026 Tavish Burnah. **All rights reserved in original
contributions, subject to prior valid license grants and third-party rights.**
This revision grants no new general reuse license. See [LICENSE](LICENSE),
[copyright scope](COPYRIGHT), [licensing history](docs/licensing/README.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
The [VM rights notices](research/COPYRIGHT),
[VM third-party credits](research/THIRD_PARTY_NOTICES.md) and
[model-rights records](research/provenance/model-licensing.json) also apply
within the imported tree. The root reservation does not revoke prior valid
grants or restrict third-party rights in any imported material.

The repository previously carried a CC BY-NC 4.0 root license and an MIT
notice in Deform. Current component notices now state all rights reserved;
the historical license texts are preserved. Permissions validly granted for previously licensed
material continue to apply; this update does not revoke them. The former
README's additional Apache claim is retained in the historical record,
rather than asserted as the current blanket license.

The VM research import is complete. The author has approved this VM archive
and documentation update for GitHub; the desktop components will follow in
a separate update. See the [publication record](docs/provenance/vm-publication.json).
The complete archive is not ready for final GitHub publication until the
Chapter 8 desktop component is reviewed and incorporated and remaining rights,
asset and release decisions are resolved. This VM update does not claim
that the complete archive is finished or that documented upstream rights
questions have been resolved.
