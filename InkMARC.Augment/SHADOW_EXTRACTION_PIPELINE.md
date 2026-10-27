# Shadow extraction and texture-replacement pipeline (v3)

This document describes the current `shadow_extract_v3` implementation and the
two batch scripts used for production. It distinguishes the reviewed production
path from diagnostic, experimental, and older code that is still present in the
repository.

The most important correction to the older nine-step description is that the
current workflow has two separate passes:

1. **Surface tracking and review** computes tag-sheet geometry and writes it to
   a corrected H5 sidecar.
2. **Shadow extraction and texture replacement** trusts that reviewed geometry,
   estimates foreground from relative depth, extracts a photometric shadow
   field, and paints that field onto each replacement texture.

In compact form:

```text
raw video + reference sheet + original H5
    -> hard-tag detection and semi-tag recovery
    -> refined per-frame sheet homography
    -> corrected H5 + review overlay
    -> manual geometry review

raw video + reference sheet + reviewed corrected H5 + texture(s)
    -> relative depth
    -> expected surface-depth field
    -> foreground depth residual and hand/stylus mask
    -> aligned reference and photometric difference
    -> clean shadow evidence
    -> projected texture with foreground restored
    -> transferable shadow factor
    -> final overlay
```

A complete worked visual example for frame 110, including the internal mask,
texture-compositing, and shadow-factor images, is available in
[`output/304_T5a_frame110_complete_visuals/README.md`](output/304_T5a_frame110_complete_visuals/README.md).

## Canonical current stage list

The full process is best described as the following 18 stages.

### Pass 1: surface registration, correction, and review

1. **Load the inputs and build the reference model.**
   The inputs are the raw video, `TagsBorderless.png`, and the video's original
   H5 sidecar. The reference image is rotated 90 degrees counter-clockwise and
   optionally scaled. The pipeline detects the reference AprilTags and builds
   blue/green masks and an expected 6x6 payload-cell pattern for every tag.

2. **Run multi-pass hard AprilTag detection on each raw frame.**
   Detection uses AprilTag 36h11 with AprilTag corner refinement. If the first
   detection pass is sparse, the frame is retried with histogram equalization,
   CLAHE, normalization, brightening, and a reduced-resolution pass. A
   successfully decoded tag with four detected corners is a **hard tag**.

3. **Fit the ordinary hard-tag homography when evidence is sufficient.**
   With at least three reference/frame tag IDs in common, all matched tag
   corners are fitted with a RANSAC homography. This is the normal geometry
   path and does not enter the semi-tag refinement path.

4. **Create a temporal or sparse geometry seed when fewer tags decode.**
   With one or two hard tags, the previous successful homography is preferred
   when it agrees with the visible hard tag(s). Otherwise a homography is
   seeded from the one hard tag or the two-tag corner set. With no hard tags,
   the previous-frame homography acts only as a motion prior.

5. **Recover semi-tags (`SoftTag` in the code).**
   A semi-tag is not an independently decoded AprilTag. It is a known reference
   tag whose expected location is supported by partial image evidence. The
   current recovery sources are:

   - rejected AprilTag quadrilaterals that have plausible geometry and match
     the expected blue/green tag pattern;
   - blue-mask contours near the seed-projected tag location;
   - a seed-projected tag whose blue/green pattern remains visible after a
     small translation search (`seed` support);
   - a seed-projected quad containing enough blue tag material
     (`seed-color` support).

   Candidate evidence is filtered using visible fraction, pattern agreement,
   geometry, distance from the seed, spatial spread, and reprojection error.
   Seed-derived semi-tags are useful support, but are not independent proof of
   the seed geometry.

6. **Optionally validate partial tag payloads.**
   The experimental `--partial-cell-pattern` path compares visible 6x6 payload
   cells with every expected reference tag and accepts an identity only when it
   has adequate cells, score, and separation from the second-best identity.
   It uses partial-tag centres instead of trusting damaged outer corners.
   `--center-consensus` is another experimental centre-based fit that excludes
   seed-derived semi-tags. Neither option is enabled by the production batch
   script.

7. **Refine and validate sparse geometry using the whole sheet.**
   Sparse fits can be adjusted using the visible green sheet boundaries, blue
   tag edges, the blue/green tag pattern, and a small dense visual alignment
   search. Guards reject changes that damage hard-tag/support-point
   reprojection, move the projected sheet implausibly far, or disagree with
   usable sheet edges. When tags are completely occluded, a previous pose may
   be held only when semi-tag or sheet-edge evidence supports it.

8. **Choose the frame geometry and project the board mask.**
   Full-video processing compares direct and temporally seeded candidates in
   fallback frames. The selected reference-to-frame homography projects the
   four reference-image corners to form the per-frame board quadrilateral and
   board mask.

9. **Write and review the corrected H5 geometry.**
   Successful projected sheet corners replace the original H5 `corners` values
   while the other H5 data and known attributes are preserved. The pass writes
   `<video_stem>_corrected.h5` beside the video and a numbered
   `<video_stem>_v3_tag_sheet_overlay.mp4` in the review directory. Repeated
   unchanged corner runs produce warnings. The operational process requires
   review of every overlay before pass 2; the filename is treated as the
   indication that the geometry has been reviewed, rather than a separate
   machine-readable approval flag.

### Pass 2: foreground separation, shadow extraction, and painting

10. **Load only the reviewed sheet geometry.**
    The production texture-replacement script passes
    `--corrected-sidecar-geometry-only`. Each homography and board mask is
    reconstructed from the corrected H5 corners. AprilTag and semi-tag matching
    are deliberately not repeated in this pass.

11. **Estimate a relative-depth image from the raw frame.**
    Depth Anything V2 Small (`vits`) is run per frame by default. Its
    floating-point prediction is min/max-normalized per frame into an 8-bit
    image. Despite artifact names such as `_raw_depth`, this is **relative,
    normalized depth**, not calibrated metric depth. Optional DA V3 support
    remains available and uses 2nd-to-98th-percentile normalization, but it is
    not the production default. In fast production the depth, residual, and
    mask are streamed to C++ instead of being retained as diagnostic videos.

12. **Estimate the depth field expected for the flat sheet.**
    The word `gradient` in the artifact names means a background depth trend,
    not an image-gradient operator such as Sobel. The corrected-H5 surface mask
    is constructed before this fit. The primary model samples the strip at the
    left of the relative-depth image only on rows where that strip is reliably
    inside the surface. It takes a robust row profile, extends the nearest
    reliable profile value through the surface's sloping top and bottom ends,
    smooths the extended profile vertically, estimates a robust global
    left-to-right slope from the reliable rows, and extrapolates that tilted
    surface field across the frame. This prevents off-surface depth from making
    dark horizontal marks at the top or bottom of the expected-depth image and
    false boundary responses in its residual. A legacy model simply repeats
    each row's mean from the first three columns across the image.

13. **Subtract the expected depth field and construct a foreground candidate.**
    Only the positive residual
    `relative_depth - expected_surface_depth` is retained. It represents pixels
    that stand out from the sheet in the model's relative-depth convention.
    The residual is restricted to the H5 sheet polygon and thresholded. The
    Python depth stage first creates a conservative component-selected
    candidate from the robust tilted field at the base threshold (default 8).
    C++ then uses confidently visible blue/green sheet pixels as negative
    evidence: it raises the threshold until two consecutive levels remove no
    more than 3% under the existing sheet-color exclusion while retaining at
    least 99.5% of the cleaned base candidate. The first qualifying level is
    Stage 01. The default search stops at 32, requires useful sheet-color
    coverage, protects the stylus AOI when available, and limits sudden
    threshold decreases in sequential video. If these guards cannot be met,
    the conservative base candidate is retained. Non-fast diagnostic mode can
    still fall back to the legacy depth field, including its dynamic
    histogram-gap threshold (base default 16) and temporal vertical gate.

    The sheet-guided selection is enabled by default. Its principal controls
    are `--no-sheet-guided-depth-threshold`,
    `--sheet-guided-max-threshold`,
    `--sheet-guided-mismatch-percent`, and
    `--sheet-guided-min-retention-percent`.

14. **Select and refine the hand/stylus foreground mask.**
    The conservative base component is selected using, in order of
    availability, the H5 stylus location, temporal proximity to the previous
    component, and depth residual strength/area. Sheet-guided rethresholding
    keeps only stronger pixels within that selected component, so it cannot
    acquire a different foreground object. C++ then reconnects a missed stylus tip using H5
    position and tilt priors together with local RGB, edge, and temporal
    evidence. The mask is clipped to the board. If depth yields no mask (or an
    optional union mode is enabled), a color-difference mask supplies a
    fallback. Visible blue/green sheet pixels are removed by default, including
    from depth-derived masks; holes can be filled, and a nearby yellow stylus
    body can be added.

15. **Warp the reference and establish the photometric baseline.**
    The reference sheet is warped into the raw frame using the reviewed
    homography. In a tag-detecting run, the top-left common hard tag can supply
    median observed green and blue colors, which repaint those color classes in
    the aligned reference.

    There is an important production-path caveat: corrected-sidecar-only
    geometry contains no current-frame tag detections, so the top-left-tag
    color-match attempt has no common tag and leaves the aligned reference
    unchanged. The older flat-space cell-quality/color-matching implementation
    is still present in `shadow_pipeline_v3.cpp`, but it has no call site in the
    v3 pipeline. Therefore **flat-space quality matching is not an active
    production stage**, and tag-based reference color matching is active only
    in modes that also carry frame tag detections.

16. **Construct and clean the shadow-evidence difference map.**
    The pipeline computes the absolute BGR difference between the raw frame and
    aligned reference inside the board, then zeros the hand/stylus region. To
    prevent printed structure and registration boundaries from becoming
    shadows, blue tag regions and the surface border are filled from
    neighbouring valid surface pixels and locally blurred. Pixels outside the
    surface and inside the foreground mask remain zero. This produces
    `_aligned_ref_masked_difference_clean`.

17. **Project the replacement texture and restore foreground objects.**
    Each input texture is cover-scaled and centre-cropped to reference-sheet
    dimensions, then warped through the same homography. It replaces the board
    only. A compositor keep-mask, derived from the hand/stylus mask, copies the
    original hand and stylus back over the texture. The keep-mask is dilated,
    optionally eroded to reduce green/blue halos, clipped to the board, and
    hole-filled. A further Lab-space edge cleanup suppresses residual
    green/blue chroma around the foreground boundary.

18. **Build the transferable shadow factor and paint the final overlay.**
    The clean difference supplies a soft shadow-support mask. In linear RGB,
    the raw/aligned-reference luminance ratio estimates shadow darkness and the
    normalized channel ratio estimates a weak color tint. These fields are
    spatially smoothed, clipped, and combined into a per-channel multiplicative
    factor whose neutral value is 1. The requested `shadow_strength` scales its
    distance from 1. The factor is multiplied into the projected-texture
    composite in linear RGB and converted back to sRGB, producing the final
    overlay. The raw frame remains unchanged outside the board and the restored
    foreground remains substantially neutral because foreground pixels were
    removed from the shadow evidence.

## What relative depth does—and does not do

Relative depth is used to identify elevated foreground objects so that the hand
and stylus are excluded from the shadow estimate and preserved in the final
composite. It does **not** directly estimate the shadow. The shadow is still
photometric: it comes from the registered raw/reference color and luminance
difference after foreground exclusion.

The depth-related diagnostic artifacts are:

| Artifact | Meaning |
| --- | --- |
| `_raw_depth` | Per-frame normalized relative-depth prediction |
| `_raw_depth_left_tilt_gradient` | Robust expected depth field for the tilted sheet |
| `_raw_depth_left_tilt_gradient_difference` | Positive residual from the tilted field |
| `_raw_depth_left_gradient` | Legacy row-wise expected depth field |
| `_raw_depth_gradient_difference` | Positive residual from the legacy field |
| `_raw_depth_active_gradient_difference` | Residual chosen for thresholding |
| `_raw_depth_gradient_mask` | Selected binary foreground component |

These images are normally available in diagnostic frame runs. Fast production
streams only what later stages require.

## Tag terminology

- **Hard tag:** a decoded AprilTag ID with four detected corners.
- **Semi-tag / soft tag:** a reference tag identity and projected quad supported
  by partial pattern, rejected-quad, or color evidence. The C++ type and logs use
  `SoftTag`; “semi-tag” is the useful process-level name.
- **Seed-supported semi-tag:** a semi-tag whose location starts from the current
  temporal/sparse homography. It adds confidence and local correction but is
  not fully independent geometry evidence.
- **Partial-cell validation:** an experimental identity check using visible 6x6
  payload cells and tag centres.
- **Sheet refinement:** global geometry checks and small corrections using green
  sheet boundaries and dense blue/green agreement, rather than tag decoding.

## Production commands and hand-off

Pass 1 is orchestrated by:

```bash
./run_tag_sheet_matching_v3.sh
```

It produces the corrected H5 sidecars and review MP4s. After those overlays are
reviewed, pass 2 is orchestrated by:

```bash
./run_all_base_avis_texture_replacement_v3.sh
```

The second script runs with `--corrected-sidecar-geometry-only --fast`, uses
Depth Anything V2 Small (`vits`), and writes one final-overlay video per video/texture
pair. It intentionally does not revisit the tag-matching decision.

## Mapping from the older nine-step description

| Older description | Current interpretation |
| --- | --- |
| 1. Raw image | Raw video frame plus reference, original/corrected H5, and texture inputs |
| 2. AprilTag surface registration | Pass-1 hard tags, temporal seeding, semi-tags, whole-sheet refinement, corrected H5, and review |
| 3. Reference colour matching | Reference warp; top-left-tag color match only when frame detections are available; no-op in corrected-only production |
| 4. Flat-space quality and colour matching | Older implementation retained but not called by v3 production |
| 5. Hand and stylus mask generation | Relative depth, expected surface-depth field, residual thresholding/component selection, H5/RGB/edge stylus recovery, and mask cleanup |
| 6. Texture projection | Texture fitting, homographic projection, foreground restoration, and chroma-edge cleanup |
| 7. Clean difference map construction | Board-restricted raw/reference difference, foreground exclusion, tag/border inpainting, and blur |
| 8. Transferable Shadow Model | Linear-RGB darkness/tint factor estimated under the clean-difference support mask |
| 9. Overlay Painting | Shadow-strength scaling and multiplication into every projected texture |

## Source-of-truth implementation files

- [`shadow_pipeline_v3.cpp`](shadow_pipeline_v3.cpp): registration, semi-tags, H5 correction, masks,
  photometric shadow factor, texture projection, and overlay composition.
- [`shadow_extract_v3_main.cpp`](shadow_extract_v3_main.cpp): command-line modes and defaults.
- [`scripts/depth_anything_v2_raw_depth.py`](scripts/depth_anything_v2_raw_depth.py): DA V2/V3 relative-depth inference,
  expected surface-depth fields, residuals, thresholding, and component
  selection. DA V2 Small is the current default; DA V3 remains optional.
- [`run_tag_sheet_matching_v3.sh`](run_tag_sheet_matching_v3.sh): reviewed pass-1 batch workflow.
- [`run_all_base_avis_texture_replacement_v3.sh`](run_all_base_avis_texture_replacement_v3.sh): corrected-geometry-only
  production pass-2 workflow.
