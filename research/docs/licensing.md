# Copyright and licensing scope

**Copyright (c) 2026 Tavish Burnah. All rights reserved.**

At the author's direction, [LICENSE](../LICENSE) reserves rights in original
InkMARC code, configurations and documentation. This is not an open-source
license and grants no general reuse permission. Root and component COPYRIGHT
notices state the scope without editing executable research files.

Third-party material remains under its own terms. In particular, the
MobileNetV4 adapted portions carry Duo Li's MIT notice, upstream software
notices and model cards are excluded from the reservation, and pretrained
model portions are not claimed as exclusively author-owned.

## Credits and obligations

[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) identifies adapted code,
imported dependencies, depth teachers, pretrained backbones, optional
excluded components and the applicable obligation summaries.
[The dependency register](third-party/DEPENDENCIES.md) and
[evidence manifest](../provenance/license-evidence.json) provide project
credits, source links, observed versions, unmodified notice files and hashes.

The [file-rights map](../provenance/file-rights.json) identifies mixed-origin
source files. The [model licensing map](../provenance/model-licensing.json)
covers each staged checkpoint/export. Model bytes and configuration values
are unchanged; a JSON sidecar records terms rather than rewriting binaries.

## Model release scope still requiring provenance

The official DA3-SMALL card declares Apache-2.0. Depth Anything V2 uses
Apache-2.0 for Small weights and CC-BY-NC-4.0 for Base/Large/Giant weights.
Those declarations are recorded separately from source-code terms.

TorchVision explicitly states that pretrained models can have their own
license/data conditions: see [its weight documentation](https://docs.pytorch.org/vision/0.20/models.html).
This review does not infer unrestricted redistribution of fine-tuned weights
from TorchVision's BSD source license. The exact historical initialization
artifacts and downstream permission scope remain model-release items.
Similarly, distillation through teacher-generated targets is recorded without
inventing a rule that all student models inherit, or avoid, teacher terms.

The current candidate is a local research archive, not a cleared binary
application distribution. Data/example/asset consent and rights are separate;
they are not granted by the copyright notice. No original project, model,
Git history or remote was changed by this licensing documentation update.
