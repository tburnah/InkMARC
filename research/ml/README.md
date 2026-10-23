# Integrated touch and localisation — supplementary experiments

LocTouchNetLite is a supplementary PyTorch/ONNX implementation that predicts
stylus location and interaction state from a canonical RGB surface view.
It is retained as integrated-sensing development alongside Chapters 6–7,
without assigning an exact thesis experiment or final-checkpoint identity.

`inkloc/models/loctouch_lite.py` defines the lightweight backbone, heatmap/DSNT
location path and state outputs. `inkloc/training/` contains training,
evaluation, calibration, decision sweeps and inspection utilities;
`inkloc/export/` contains ONNX tools. Temporal variants and their experiment
YAML files remain under their historical paths.

The basic input is float RGB `[B,3,320,480]`; output dictionaries contain
coordinates, heatmaps/confidence and validity/touch/state logits or
probabilities. Optional temporal inputs depend on configuration.
`docs/data_contract.md` describes indexed and generic manifests.

Run module commands from the archive root so `ml.inkloc` resolves.
Real-data operation additionally requires the location-project source via
`data.inkmarc_src` and the external indexed dataset. Dependencies include
PyTorch, NumPy, Pillow, YAML, OpenCV/HDF5 through the adapter and ONNX tools;
tests use pytest. See [dependency notes](../docs/dependencies.md).

Existing synthetic shape/loss/temporal tests are retained. Their actual
staging outcome is recorded in [verification](../provenance/verification.json).
No training or export command was executed during staging.

Historical `docs/model_readiness.md` and the original README are retained as
provenance. Their previously recorded outcomes are not fresh archive tests.
Model/calibration selection and participant-derived outputs remain outside
this public candidate; scientific experiment YAMLs are selectively included.

## Copyright and third-party rights

Copyright (c) 2026 Tavish Burnah. All rights reserved for original contributions.
See [copyright scope](../COPYRIGHT), [reserved-rights terms](../LICENSE) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Upstream code/adapted portions
and model parameters retain their applicable third-party terms.
