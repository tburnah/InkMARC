# InkMARC session annotation

Copy the matching session files from the phone:

- `Movies/InkMARC/InkMARC_<timestamp>.mp4`
- `Downloads/InkMARC/InkMARC_<timestamp>_sensors.csv`

Then run:

```powershell
python tools/annotate_session.py `
  InkMARC_<timestamp>.mp4 `
  InkMARC_<timestamp>_sensors.csv
```

The default output is `InkMARC_<timestamp>_annotated.mp4`. The script uses camera-frame
`video_time_ns` values to select the synchronized tracked quad. FFmpeg is used for H.264 output
when available, with OpenCV's MP4 writer as a fallback.

Camera-frame rows also contain tracker diagnostics for offline failure analysis:

- `tracker_state`, `tracker_motion_model`, and `tracker_held`
- active, candidate, stable, and inlier track counts plus `tracker_inlier_ratio`
- reprojection error, reference/current feature coverage, transform condition, and confidence
- `tracker_rejection_reason`, including rejected transform models and geometry gates

Sensor rows leave these tracker-only columns empty.

## ARCore one-to-one comparison capture

`InkMarcActivity.ENABLE_ARCORE_COMPARISON` enables the ARCore path, the fallback tracker, and
ARCore dataset recording together. Recording begins when the ARCore session begins—not when a
plane appears—so it preserves the entire no-plane, depth-only, and plane-tracking sequence. Use an
ARCore-supported device, tap once to establish the shared placement, then perform the test
movement. The exported capture is in
`Downloads/InkMARCCaptures/<capture-id>/` and contains `session.mp4` plus `planes.ndjson`.

The status banner changes to **recording dataset** when full-session capture begins. When the run
is complete, tap **Finish & save
capture** and wait for **exported to Downloads** before leaving the app. The button stops the
dataset and exports it while InkMARC remains open; it does not start a second capture afterward.

Each frame in `planes.ndjson` records ARCore camera state, horizontal-plane count and point-cloud
size, alongside the fallback state, quadrilateral, feature counts, inliers, confidence, and a
direct outcome: `both_tracking`, `arcore_only`, `fallback_only`, or `neither_tracking`.

To make an Excel-friendly per-frame log and a compact result summary:

```powershell
python tools/analyze_arcore_comparison.py `
  Downloads/InkMARCCaptures/<capture-id>/planes.ndjson
```

To create a presentation-ready replay with the original camera video plus both logged overlays,
run:

```powershell
python tools/render_arcore_comparison_video.py `
  Downloads/InkMARCCaptures/<capture-id>/session.mp4 `
  Downloads/InkMARCCaptures/<capture-id>/planes.ndjson
```

It writes `arcore_fallback_overlay.mp4` next to the source video. White is ARCore, orange is the
fallback tracker, and the status panel displays the ARCore state, plane count, fallback confidence,
and the per-frame availability outcome. The default `1080x2400` SurfaceView matches the current
test device. Use `--rotation ccw` if the camera background is the wrong way around, or pass the
captured device's screen resolution with `--view-size WIDTHxHEIGHT` when using a different device.
The renderer selects the closest ARCore timestamp rather than assuming video and render frames have
the same rate. If a fixed residual delay is still visible, use `--offset-ms N` (a positive value
advances the overlays by `N` milliseconds).

This mode intentionally does not allow the fallback tracker to seed from ARCore planes or stop
when ARCore loses tracking. The systems share only the user-selected initial placement.

The frame log records ARCore camera state, its first tracked-plane timestamp, point-cloud size,
every plane, the active placement source (`depth` or `plane`), and the fallback state/quad. This
makes the no-plane and depth-only phases measurable rather than treating the white rectangle as
evidence of a plane.

If ARCore cannot create or resume a session (for example, an unsupported device or unavailable
Google Play Services for AR), InkMARC automatically releases ARCore and starts the Camera2 fallback
tracker. It writes an `arcore_failover_<timestamp>.json` event in the app's external
`failover/` directory and then records the normal Camera2 MP4/CSV session. That capture is
fallback-only and is not labelled as an ARCore comparison.

## Offline tracker experiments

Use the synchronized `tracker_state=seeded` rows as repeatable placements and compare image-only
trackers over the remainder of each interval:

```powershell
python tools/experiment_video_tracker.py `
  InkMARC_<timestamp>.mp4 `
  InkMARC_<timestamp>_sensors.csv `
  InkMARC_<timestamp>_tracking_experiment.mp4
```

The output video draws the phone result in yellow, grid-balanced sparse LK in green, the
LK-plus-SIFT correction hybrid in blue, and direct seed-frame SIFT matches in magenta. A matching
CSV contains each experimental quad, support/error information, rejection reason, and an
independent gradient-patch registration score. Add `--ecc` to include the much slower cyan direct
ECC experiment; it is disabled by default because it is both expensive and unreliable on the
low-texture wood recording.

For the current LK + adaptive SIFT keyframe + immutable-origin loop-closure experiment, run:

```powershell
python tools/experiment_keyframe_tracker.py `
  InkMARC_<timestamp>.mp4 `
  InkMARC_<timestamp>_sensors.csv `
  InkMARC_<timestamp>_keyframe_loop_experiment.mp4
```

The video draws the recorded phone quad in yellow, pure LK in green, and the keyframe tracker in
blue. The matching result CSV records every experimental quad and whether a frame used LK, a local
keyframe correction/promotion, or an origin correction/promotion. Tune descriptor, promotion, and
origin cadence with `--descriptor-interval`, `--promotion-interval`, and `--origin-interval`.

On `InkMARC_20260921_124821`, the default settings retained a valid quad on 629/646 active frames.
Against 162 frames with a confident independent seed-frame SIFT match, mean quad IoU was 0.914 and
mean corner error was 14.2 px, compared with 0.810 and 32.9 px for pure LK. Attempt 4 still received
no usable reference correction, so this experiment should not yet be treated as the final phone
tracker.

### Verified keyframe gallery

The next iteration retains the immutable origin plus up to four verified intermediate viewpoints.
Normal matching uses a tight search around the LK prediction; a wider origin search is reserved for
scheduled loop closure or recovery:

```powershell
python tools/experiment_gallery_tracker.py `
  InkMARC_<timestamp>.mp4 `
  InkMARC_<timestamp>_sensors.csv `
  InkMARC_<timestamp>_gallery_experiment.mp4
```

Tune it with `--descriptor-interval`, `--promotion-interval`, `--max-keyframes`, and
`--origin-interval`. The result CSV includes the gallery size and the winning reference's patch
similarity, inlier ratio, and combined quality score.

On `InkMARC_20260921_124821`, the default gallery retained a valid quad on 629/646 active frames and
raised mean gradient-patch registration from 0.102 to 0.123. Against the same 162 confident direct
seed-frame SIFT matches, mean quad IoU rose from 0.914 to 0.948 and mean corner error fell from
14.2 px to 9.1 px. Attempt 4 remained unchanged at 0.841 IoU because the strongly oblique view did
not produce another valid descriptor correction. This Python implementation takes several minutes
for the short recording, so descriptor scheduling and recovery frequency need optimization before
porting it to the phone.
