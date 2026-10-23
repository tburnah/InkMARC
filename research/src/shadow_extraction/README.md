# InkMARC-Augment — surface-appearance augmentation

This historical `shadow_extraction` project implements the offline workflow
described in thesis §5.3.5 (pp. 92–96): reviewed sheet geometry, hand/stylus
foreground separation, and shadow/illumination transfer to replacement
surfaces. It constructs D3a appearance variants from D3 recordings.

## Important files

- `shadow_pipeline_v3.cpp` / `.hpp`: registration, foreground handling,
  photometric transfer and compositing.
- `shadow_extract_v3_main.cpp`: command-line modes and defaults.
- `scripts/depth_anything_v2_raw_depth.py`: relative-depth and foreground
  processing; the documented production default uses Depth Anything V2 Small.
- `run_tag_sheet_matching_v3.sh`: corrected H5 and geometry-review overlays.
- `run_all_base_avis_texture_replacement_v3.sh`: appearance replacement after
  geometry review.
- `resize_videos_and_h5.py`: aligned frame/H5 resizing utility.

[SHADOW_EXTRACTION_PIPELINE.md](SHADOW_EXTRACTION_PIPELINE.md) preserves the
detailed implementation description. Its output-example links refer to
generated visualisations withheld from this candidate. v2, the earlier C++
trial and the Python implementation are retained as historical sources.

Inputs are registered video/H5 recordings, the reference tag sheet, replacement
textures, teacher code and weights. Outputs include corrected H5, review
overlays and appearance-replaced videos. Data/assets, generated media and
compiled executables are not included pending review.

Build definitions use C++17, local OpenCV and HDF5 libraries. The existing
scripts document their original include/library paths; no portable rebuild
is claimed. See [execution notes](../../docs/reproducibility.md).

This offline registration workflow is indexed under Chapter 5. Chapter 8's
InkMARC-Surface refers to the separate live phone tracker. `MainViewModel.cs`
is an InkMARC-Clean reference excerpt.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../../COPYRIGHT), [reserved-rights terms](../../LICENSE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
