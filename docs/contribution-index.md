# Thesis contribution index

References use the thesis's printed page numbers. Component associations
describe roles; they do not establish exact experiment runs or final desktop
revision identity. The imported OcuInk application is renamed InkMARC.Probe
following the author's explicit confirmation of its thesis name; other
existing project directories and executable names remain unchanged.

| Thesis contribution | Repository source or confirmed location | Coverage and pending work |
| --- | --- | --- |
| Chapter 5, §5.2.1: exploratory tools and earlier stream preparation | [Probe](../InkMARC.Probe/README.md) is the author-confirmed former OcuInk desktop collection application; [Prepare](../InkMARC.Prepare/README.md) supplies an earlier alignment/export family; [Deform](../InkMARC.Deform/README.md) preserves instrumented collection | Probe imported and renamed; no exact experiment-run association asserted and Deform remains a separate application |
| Chapter 5, §5.2.2: InkMARC-Train/D1 collection | No exact application-version association asserted | Thesis Train is a collection tool, not a label for all Python training scripts |
| Chapter 5, §5.2.3: InkMARC-Cue/D2 | [Cue](../InkMARC.Cue/README.md), prompted touch/no-touch recording | Preserve this repository version; no final desktop Capture claim |
| Chapter 5, §5.3.1: InkMARC-Capture/D3 | [Capture](../InkMARC.Capture/README.md), Android camera, AprilTag and USB-stylus collection | Gradle Android project and AprilTag submodule incorporated; device build not verified here |
| Chapter 5, §5.3.3, pp. 91–92: InkMARC-Clean | [Clean](../InkMARC.Clean/README.md), project/solution and reconstruction/export source; [Label](../InkMARC.Label/README.md) retained as a separate historical labelling workflow | VM reference excerpts remain historical evidence rather than replacements for this project |
| Chapter 5, §5.3.5, pp. 92–96: InkMARC-Augment | [Offline augmentation](../research/src/shadow_extraction/README.md) | VM source imported intact; registration/separation/shadow transfer |
| Chapter 6, §§6.1–6.2, pp. 102–110: preliminary sensing | [Touch](../research/src/touch_model/README.md) and [localisation](../research/src/location_model/README.md); [Evaluate](../InkMARC.Evaluate/README.md) and [Test](../InkMARC.Test/README.md) remain supporting historical tools | Imported implementation families retain the [E01–E11 mapping](../research/docs/contribution-index.md); no exact supporting-tool run association asserted |
| Chapter 7: InkMARC-Sense | [Integrated sensing workspace](../research/inkdepth_student_workspace/README.md) and [selected model index](../research/docs/models.md) | Source, configurations and selected existing artifacts imported; application integration remains a desktop addition |
| Chapter 7, §7.1.2, pp. 116–117: InkMARC Depth / InkDepth | [Current-workspace depth proxy](../research/inkdepth_student_workspace/DEPTH_PROXY.md) | MobileNetV2 DA3 proxy imported; [earlier depth students](../research/src/depth_model/README.md) retained separately |
| Chapter 8, pp. 130–159: InkMARC-Surface and integrated application | Latest implementation stored on the author's desktop | **Pending incorporation from desktop**, proposed `applications/InkMARC/`, preserving Surface's actual project location |

The imported [VM contribution index](../research/docs/contribution-index.md)
contains the detailed Chapter 6 and Chapter 7 mapping. Its original files
and documentation remain byte-identical to the frozen Phase B snapshot.
That nested index describes the VM snapshot's scope; this repository-level
index also covers the existing .NET projects and desktop incorporation status.

Do not substitute Deform, Cue, Evaluate, offline augmentation or reconstructed
code for the pending Chapter 8 desktop implementation. Do not deduplicate or merge
Clean's VM excerpts into this repository's project merely because names match.

The [import review](research-import.md) and [provenance record](provenance/research-import.json)
record how these sources were combined without modifying their research
implementations. [Archive status](archive-status.json) keeps the Chapter 8
desktop contribution and complete archive pending final release review.

Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original
contributions, subject to prior grants and third-party rights.
