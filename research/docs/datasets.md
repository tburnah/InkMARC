# Dataset and input requirements

Research recordings, labels, frame-level manifests, per-participant prediction
tables and generated depth caches are not included in this public candidate.
Their detailed local inventory remains outside `public/` for author review.

| Thesis dataset | Collection/preparation role | Related implementation inputs |
| --- | --- | --- |
| D0 | InkMARC-Probe, followed by Prepare | Earlier synchronized video/stylus streams |
| D1 | InkMARC-Train on a non-emissive Wacom surface | Preliminary touch images, TFRecords or NumPy arrays |
| D2 | InkMARC-Cue prompted imagery | Complementary touch imagery and proxy training inputs |
| D3 | InkMARC-Capture, processed by Clean | Aligned AVI and HDF5, timestamps, labels, corners and validity masks |
| D3a | InkMARC-Augment surface derivatives | Texture variants linked to original recording labels and partitions |

## Historical VM locations

`location_model` and `ml` expect indexed data under `/data/LocationData`,
including `indices/manifest_<version>.json`, `index_<version>.jsonl` and
`splits_<version>.json`. That live index tree was not present when staged;
the directory contains a data ZIP. No archive was extracted or data rebuilt.

Preliminary touch scripts reference `/data/TouchData` and older `/data/`
TFRecord/NumPy paths. Several inputs currently exist only as compressed
`.xz` archives. The archive does not silently substitute those paths.

The current sensing configurations reference `/data/baseAVIs_432x672`,
diagnostic manifests under `outputs/diagnostic_manifests/`, and generated
depth caches. Some required manifests are large and contain participant/
recording paths. They are withheld, rather than replaced by synthetic files.

Earlier relative-depth students use `/data/depth/pseudo_depth_dataset` and
`/data/depth/pseudo_depth_dataset_320`: RGB/depth pairs plus `manifest.csv`.
Dedicated model files found beside those datasets remain in the local model
selection inventory. Third-party teacher weights are separately supplied.

## Data contracts

The indexed localisation pipeline uses `corners`, `labels`, `label_mask`
and timestamps. Its five stylus labels represent x, y, pressure, tx and ty.
Most preliminary localisation inputs are canonical RGB surface views with
tensor shape `[B,3,320,480]`.

Current sensing loads frames and H5 geometry through `StylusDataset`.
Configuration-dependent masks identify valid depth, interaction state,
touch and location targets. Consult the preserved configuration and dataset
code for the selected run rather than mixing conventions across projects.

The Chapter 7 proxy code uses `[B,3,504,322]` in NCHW order and predicts a
single-channel map. The sensing code's YAML input size is `[672,432]`
(height, width). These are code tensor conventions; this document does not
change any dimensions or preprocessing.

No small research example is supplied until its release and provenance are
approved. Synthetic input utilities remain in source for lightweight checks.
