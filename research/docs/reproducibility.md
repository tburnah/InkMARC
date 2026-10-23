# Execution and reproducibility notes

Start with the component README and original configuration. Historical paths,
arguments and numerical operations are preserved. The examples below are
documentation of available interfaces, not claims that the thesis experiments
have been rerun. Dataset-dependent evaluation needs the approved original
data, split manifests, checkpoints and upstream dependencies together.

## Environments and working directories

The two InkDepth snapshots use the same `inkdepth` import name. Use separate
environments for editable installations and run scripts from the intended
snapshot. The existing `requirements.txt` and `pyproject.toml` describe its
declared setup. Historical README copies are retained in provenance, while
staged component READMEs are the archive navigation entry points.

`ml` uses the `ml.inkloc` namespace: run its modules/tests from this archive's
root. Its real-data adapter additionally requires the location project source
through `data.inkmarc_src`, alongside the indexed dataset.

## Existing lightweight tests

These selected tests use fresh in-memory models and synthetic tensors. They
do not train models or use research checkpoints:

```bash
python -B -m pytest -q -p no:cacheprovider       ml/tests/test_model_shapes.py ml/tests/test_losses.py       ml/tests/test_inkmarc_temporal.py
```

The staging record specifies which tests actually ran and their outcome.
The broader test suite includes overfit/export work and was not selected
automatically. Legacy files with `test` in their names can process data or
execute experiments at module scope; they are not all passive unit tests.

## Current sensing interfaces

From `inkdepth_student_workspace/`, the following interfaces exist:

```text
scripts/train_student.py --config CONFIG --override KEY=VALUE ...
scripts/validate_student.py --config CONFIG --checkpoint CHECKPOINT --split SPLIT --override KEY=VALUE ...
scripts/evaluate_touch_location_report.py --config CONFIG --checkpoint CHECKPOINT --output-dir OUTPUT --override KEY=VALUE ...
scripts/evaluate_depth_proxy_student_pipeline.py --config CONFIG --proxy-onnx PROXY --student-onnx STUDENT --output-dir OUTPUT --override KEY=VALUE ...
```

`CONFIG`, `CHECKPOINT`, `PROXY`, `STUDENT` and `OUTPUT` are placeholders,
not literal arguments. The first interface trains; evaluation writes output
reports. Export and quantisation tools are retained for inspection but were
not executed during archival preparation.

The existing override parser accepts dotted keys, including
`data.manifest_path`, `model.da3_repo_root` and `output_dir`. Supply a reviewed
manifest and a new output directory when executing. The supplied configs
are historical experiment definitions; changing runtime paths through their
existing override interface is distinct from editing those definitions.

## Augmentation interfaces

`src/shadow_extraction/build_shadow_v3.sh` documents the original C++17 build
and exact local library paths. `run_tag_sheet_matching_v3.sh` creates corrected
H5 geometry and review overlays. After manual review,
`run_all_base_avis_texture_replacement_v3.sh` uses corrected-sidecar geometry
for foreground/shadow transfer. Input AVI/H5, `TagsBorderless.png`, textures,
teacher source and weights are required and are not included here.

Do not run the batch or regression scripts as a read-only inspection step:
they create outputs, and the regression script clears its configured output
directory. Review their existing arguments first. No such script was run
during staging.

## Archive checks versus research reproduction

Hash checks establish copied-file identity, not scientific reproducibility.
Syntax and selected synthetic checks do not establish mobile inference
parity or reproduce reported experimental outcomes. The per-file manifest,
selected model index and verification record define the checks performed.

[Configuration input references](../provenance/config-inputs.json) records which inspected inputs are present in the candidate and which remain external, without editing the configurations.
