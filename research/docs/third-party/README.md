# Third-party attribution evidence

This directory contains upstream copyright/license notices and model-card
snapshots. Its third-party contents are excluded from the author reserved-rights
notice. Source bytes and model binaries are not modified by these sidecars.

- [Archive notices and obligations](../../THIRD_PARTY_NOTICES.md)
- [Dependency credits](DEPENDENCIES.md)
- [Evidence origins, revisions and hashes](../../provenance/license-evidence.json)
- [Per-artifact model terms](../../provenance/model-licensing.json)

## Previously recorded upstream source notices

| Project | Recorded checkout | Notice |
| --- | --- | --- |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | `e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1` | [license](Depth-Anything-V2/LICENSE) |
| [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | `41736238f5bced4debf3f2a12375d2466874866d` | [license](Depth-Anything-3/LICENSE) |
| [opencv](https://github.com/opencv/opencv) | `31b0eeea0b44b370fd0712312df4214d4ae1b158` | [license](opencv/LICENSE) |
| [opencv_contrib](https://github.com/opencv/opencv_contrib) | `0e5254ebf54d2aed6e7eaf6660bf3b797cf50a02` | [license](opencv_contrib/LICENSE) |
| [salad](https://github.com/serizba/salad) | `6aede13a3f6c25750bf7fde10209c06cb73060bb` | [license](salad/LICENSE) |

## MobileNetV4 adaptation

[MIT notice](mobilenetv4.pytorch/LICENSE), Copyright (c) 2024 Duo Li.
The inspected current revision is recorded separately from the unknown
historical adaptation revision. See the source-directory attribution sidecar.

## Model cards

| Model | Declared license | Observed card |
| --- | --- | --- |
| [depth-anything/DA3-SMALL](https://huggingface.co/api/resolve-cache/models/depth-anything/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3/README.md?%2Fdepth-anything%2FDA3-SMALL%2Fresolve%2Fe08cab65ca0ec38e7826075418411ab90cab4da3%2FREADME.md=&etag=%22efbcb8b72e26241e56330cd76d6df4ca475f897f%22) | apache-2.0 | [snapshot](model-cards/DA3-SMALL.md) |
| [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/api/resolve-cache/models/depth-anything/Depth-Anything-V2-Small/03876f8651c73a60fe4c2c48294e09fcb6838fcf/README.md?%2Fdepth-anything%2FDepth-Anything-V2-Small%2Fresolve%2F03876f8651c73a60fe4c2c48294e09fcb6838fcf%2FREADME.md=&etag=%2261f929ab2f782e1bd9bc15b1e9efb0120ac4091a%22) | apache-2.0 | [snapshot](model-cards/DA-V2-Small.md) |
| [depth-anything/Depth-Anything-V2-Base](https://huggingface.co/api/resolve-cache/models/depth-anything/Depth-Anything-V2-Base/a4e71a6c2ce52fe50df0f212066b0d4a87be9b5e/README.md?%2Fdepth-anything%2FDepth-Anything-V2-Base%2Fresolve%2Fa4e71a6c2ce52fe50df0f212066b0d4a87be9b5e%2FREADME.md=&etag=%222144d0e3c3e0bcce096906c305c4235c941379c5%22) | cc-by-nc-4.0 | [snapshot](model-cards/DA-V2-Base.md) |
| [depth-anything/Depth-Anything-V2-Large](https://huggingface.co/api/resolve-cache/models/depth-anything/Depth-Anything-V2-Large/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/README.md?%2Fdepth-anything%2FDepth-Anything-V2-Large%2Fresolve%2Fcbbb86a30ce19b5684b7a05155dc7e6cbc7685b9%2FREADME.md=&etag=%223352e63c21ce4961dedebbb6aae70fe0977318b3%22) | cc-by-nc-4.0 | [snapshot](model-cards/DA-V2-Large.md) |

## Additional full terms

- [CC BY-NC 4.0](licenses/CC-BY-NC-4.0.txt)
- [MPL 2.0](licenses/MPL-2.0.txt)

Native-library notices are indexed with their exact package identifiers
and hashes in the evidence manifest. They are not licenses for the original
research code. Preserved upstream NOTICE/third-party notice files travel
with their corresponding package license snapshots.
