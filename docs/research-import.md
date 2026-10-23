# VM research import review

The frozen Phase B `public/` tree has been imported into `research/` in the
local InkMARC clone. The existing .NET suite remains in place. All imported
files retain their bytes, modes and original relative paths; no implementation
was merged, reconstructed, reformatted or deduplicated.

The [fresh import provenance](provenance/research-import.json) records the
destination base commit, source snapshot identity, documentation changes and
verification results. The [preflight file mapping](integration-plan.json)
is retained as historical evidence; its planned-state label predates this
import. The [integration record](integration-plan.md) describes the implemented
layout and preservation procedure.

## Imported coverage

| Material | Destination and navigation |
| --- | --- |
| InkMARC-Augment and preparation utilities | [Offline surface appearance](../research/src/shadow_extraction/README.md) |
| Preliminary touch/localisation | [Touch](../research/src/touch_model/README.md), [localisation](../research/src/location_model/README.md) and [experiment mapping](../research/docs/contribution-index.md) |
| InkMARC-Sense | [Current sensing workspace](../research/inkdepth_student_workspace/README.md) |
| InkMARC Depth / InkDepth | [Distilled depth proxy](../research/inkdepth_student_workspace/DEPTH_PROXY.md) |
| Historical and supplementary sources | [Earlier student](../research/inkdepth_student/README.md), [depth prototypes](../research/src/depth_model/README.md), [LocTouchNetLite](../research/ml/README.md) |
| Selected existing models and exports | [Model index](../research/docs/models.md), [artifact hashes](../research/provenance/models.csv), [rights records](../research/provenance/model-licensing.json) |
| Dependencies, inputs and practical execution | [Dependencies](../research/docs/dependencies.md), [data](../research/docs/datasets.md), [execution notes](../research/docs/reproducibility.md) |
| Snapshot evidence and original notices | [Copy manifest](../research/provenance/copied-files.json), [Phase B verification](../research/provenance/verification.json), [third-party credits](../research/THIRD_PARTY_NOTICES.md) |

646 regular files and one internal symlink were copied: 373,457,272 bytes
in regular files, about 356 MiB. All 26 selected model binaries retain their
recorded hashes. The link at `research/inkdepth_student_workspace/outputs`
still points to `../inkdepth_student/outputs`, inside the imported tree.
Copies are independent files, rather than hard links to the frozen source.

## Fresh verification and retained evidence

All 646 imported regular-file hashes, file modes, symlink target and the
preflight mapping matched. The 645 entries in the Phase B checksum manifest
passed from inside `research/`; the checksum file itself was checked against
its independent preflight hash. Copied-file and model-rights manifests also
matched their recorded artifact hashes.

The actual combined ignore rules allow every imported file/link and exclude
the checked private/generated example paths. Authored repository navigation
links resolve. Private review directories, original Git histories, raw
participant media, withheld checkpoints and generated research caches were
not added by the import.

Original selected VM inputs and Git research state were rechecked, and the
full frozen staging-tree inventory remained unchanged. Existing .NET source,
project definitions, binaries and assets were preserved. Repository-level
documentation changes are listed separately in the import record.

Phase B's syntax/configuration checks, five ONNX schema checks and six passing
synthetic tests remain historical evidence for the unchanged imported bytes.
No training, evaluation run, package restore, application build, conversion,
quantisation, model execution or data regeneration was performed during the
import. This review does not claim full experiment or device reproduction.
The earlier privacy-pattern checks retain their stated limits; adding existing
model binaries locally does not establish their redistribution clearance.

## Current availability, rights and release state

The nested VM READMEs and manifests intentionally retain the scope of the
Phase B snapshot. The [combined contribution index](contribution-index.md)
adds the existing full Clean project and other .NET applications and is the
current navigation/availability overlay for the combined repository.

The latest InkMARC-Capture (§5.3.1) and Chapter 8 InkMARC-Surface/integrated
application source are stored on the author's desktop and remain **pending
incorporation from desktop**. Older VM or existing .NET applications do not
substitute for those implementations. Recommended destinations and required
preservation, dependency, sensitive-file, licensing, provenance and index
checks remain in the [desktop incorporation plan](integration-plan.md#pending-desktop-projects).

Original contributions remain all rights reserved, subject to prior valid
grants and third-party terms. Root and imported notices are both retained;
the imported files' historical copyright notices have not been rewritten.
Existing model, asset, signing-material and release decisions remain as
documented. No new release clearance is claimed.

The VM import is complete. The [complete archive status](archive-status.json)
remains pending until the two desktop components are reviewed and incorporated
and remaining release decisions are resolved. The author has separately
approved committing and pushing the combined VM archive and documentation
to `tburnah/InkMARC` on `main`; the desktop projects will follow in a later
push. The [publication record](provenance/vm-publication.json) records that
scope separately from the historical local import verification.

Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original
contributions, subject to prior grants and third-party rights.
