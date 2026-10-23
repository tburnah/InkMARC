# Third-party code and model notices

Original InkMARC contributions are Copyright (c) 2026 Tavish Burnah,
all rights reserved, subject to [LICENSE](LICENSE). Upstream rights and
permissions remain intact. This document distinguishes material present in
the candidate from libraries and tools that users obtain separately.

## Adapted material present in the research source

**MobileNetV4 PyTorch implementation — Duo Li.** Copyright (c) 2024 Duo Li,
MIT License. The historical Large and Medium block specifications in
`src/depth_model/models/student_depth_net_v6.py` and
`student_depth_net_v8.py` explicitly credit
[d-li14/mobilenetv4.pytorch](https://github.com/d-li14/mobilenetv4.pytorch).
The Small/common MobileNetV4 block family in `student_depth_net_v3.py`
is conservatively included in this attribution because it shares that
implementation family; the precise historical copied portions/revision are
not established. No exclusive author ownership is asserted for these portions.

Preserve [Duo Li's full MIT notice](docs/third-party/mobilenetv4.pytorch/LICENSE)
with copies or substantial portions of the adapted material. The notice is
also present in [the model-directory sidecar](src/depth_model/models/THIRD_PARTY_NOTICES.md).
The upstream revision inspected now is recorded in
[license evidence](provenance/license-evidence.json); it is not presented as
a recovered historical adaptation revision. Existing source comments remain.

## Depth estimators and model sources

- **Depth Anything 3 / DA3-SMALL**: credit the Depth Anything 3 authors and
  ByteDance Ltd. and/or its affiliates. The recorded source checkout is
  Apache-2.0, and the official DA3-SMALL model card declares Apache-2.0.
  The code is an external dependency and the standalone teacher weights
  are not bundled. The custom proxy was trained using teacher outputs;
  that relationship is recorded without asserting that teacher terms
  automatically transfer to, or disappear from, a student.
- **Depth Anything V2**: credit Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang
  Xu, Jiashi Feng and Hengshuang Zhao (Depth Anything V2). The recorded
  code checkout is Apache-2.0. Small model weights declare Apache-2.0;
  Base/Large/Giant weights declare CC-BY-NC-4.0. Code and weight licenses
  are separate. These standalone teacher weights are not bundled.
- **TorchVision pretrained backbones**: credit TorchVision/PyTorch and the
  relevant MobileNet, ResNet or EfficientNet implementations/providers.
  TorchVision source is BSD-3-Clause. Its official documentation expressly
  distinguishes pretrained weight/data terms from the code license.
  ImageNet initialization is present or supported in the archived source;
  a software license does not establish unrestricted rights in all weights.
- **Keras / TensorFlow pretrained applications**: historical trainers use
  application backbones and ImageNet initialization. Keras/TensorFlow code
  is Apache-2.0; the exact weight artifact/provider and applicable data terms
  must be retained for any released historical model. Those standalone
  preliminary weights remain outside the current candidate.

Original model-card snapshots, their observed revisions, full license texts
and links appear under [docs/third-party](docs/third-party/README.md).
[The model licensing map](provenance/model-licensing.json) covers all 26
staged artifacts and distinguishes original contributions, initialization,
teacher relationships and unresolved release terms. No blanket all-rights-
reserved restriction is imposed on upstream model portions.

## External libraries and tools

[The dependency register](docs/third-party/DEPENDENCIES.md) credits every
identified direct Python dependency, C# reference dependency and the relevant
native/media tools. Full retained copyright/license notices are indexed
in [license-evidence.json](provenance/license-evidence.json).

Calling or importing a separately installed library is not the same as
redistributing its implementation. This source/model candidate contains no
installed environment, linked executable, DLL/shared library, Python wheel,
QNN runtime or CUDA toolkit. A later binary/application package needs its
own exact dependency and bundled-code notice inventory.

## Upstream branches outside this snapshot

The pinned OpenCV/contrib and Depth Anything source trees are not vendored.
If those trees are redistributed later, retain their own file-level notices
and nested attributions, not just their root license.

The excluded DA3 source tree credits Meta DINO/DINOv2, timm, MoGe and other
components in individual files; some references identify inspiration rather
than copied code. Its optional streaming SALAD submodule has a GPL-3.0
license text, retained here for reference only. SALAD source, binaries and
weights are not distributed in this candidate. Copying a GPL license text
alone does not license the original research code under the GPL.

FFmpeg is invoked externally for media processing. Its effective license
depends on build options (LGPL-2.1-or-later, or GPL when GPL parts are enabled).
No FFmpeg binary is bundled. [Its own legal guidance](https://ffmpeg.org/legal.html)
governs any later binary/linking distribution.

## Required preservation when distributing covered material

| Terms | Obligations for material actually covered by those terms |
| --- | --- |
| MIT | Retain copyright and permission notice with copies/substantial portions. |
| BSD-2-Clause / BSD-3-Clause | Retain copyright, conditions and disclaimer; reproduce them with binary distributions; BSD-3 also prohibits implied endorsement. |
| Apache-2.0 | Supply license; retain applicable copyright/attribution notices; preserve relevant NOTICE content when supplied; mark modified upstream files; do not remove the upstream grants. Original separable work or original modifications can have separate terms subject to these conditions. |
| CC-BY-NC-4.0 | Attribution and license link, retained supplied notices and modification indications; licensed sharing/adaptation is noncommercial; do not add restrictions that prevent exercising the licensed rights. Other rights are not automatically cleared. |
| MPL-2.0 | Preserve notices and provide covered source as required when distributing covered files/executables; this file-level license is not an all-rights-reserved grant for modified MPL code. |
| GPL / LGPL | Exact component, linking and distribution determine source/relinking obligations. Preserve the relevant license and rights; an all-rights-reserved notice cannot override a covered work's grants. No covered runtime/source is bundled here merely because its license is retained. |

These summaries are navigation aids. The full component terms control.
Attribution is not a substitute for permissions or model/data provenance.
