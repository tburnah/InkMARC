# InkMARC component names and source locations

The 2026-10-27 reorganisation aligns the research implementations with the
suite names used in the thesis at `D:\workspaces\thesis\Tavish_Thesis`.
These are implementation-family matches, not new claims about which exact
checkpoint produced a thesis result.

| Component | Previous location | Current source and thesis association |
| --- | --- | --- |
| InkMARC-Augment | `research/src/shadow_extraction/` | [InkMARC.Augment](../InkMARC.Augment/README.md): `shadow_pipeline_v3.cpp`, `shadow_extract_v3_main.cpp`, geometry review and texture-replacement scripts; Chapter 5 surface augmentation |
| InkMARC-Sense | `research/inkdepth_student_workspace/` | [InkMARC.Sense](../InkMARC.Sense/README.md): `inkdepth/models/inkdepth_student.py`, heads, datasets, training and evaluation; Chapter 7 integrated location/presence/touch sensing |
| InkMARC-Depth | Proxy model and tools within that workspace | [InkMARC.Depth](../InkMARC.Depth/README.md): `inkmarc_depth/model.py`, `configs/da3_depth_proxy_504x322.yaml`, teacher-cache, training, export and quantisation tools; Chapter 7 distilled DA3 MobileNetV2 depth proxy |
| InkMARC-Touch | `research/src/touch_model/` | [InkMARC.Touch](../InkMARC.Touch/README.md): `src/Model15.py`, `train19.py` and other standalone classifier variants; Chapter 6 preliminary touch experiments |
| InkMARC-Location | `research/src/location_model/` | [InkMARC.Location](../InkMARC.Location/README.md): `src/train_phase*.py`, `src/models/`, data transforms and evaluation; Chapter 6 preliminary localisation experiments |

Touch and Location are descriptive suite component names introduced for the
folder layout. Their individual experimental variants keep their original
identifiers. The [E01–E11 mapping](../research/docs/contribution-index.md)
distinguishes thesis experiment numbers from source phase numbers.

InkMARC-Augment is offline surface-appearance processing; `InkMARC.Surface`
contains the separate Android surface-tracking application. The older
`research/src/depth_model/` depth students and `research/inkdepth_student/`
multitask snapshot remain historical projects. The latter is not the final
distilled proxy merely because its folder contains “depth”.

## Source names and compatibility

The current sensing classes are `InkMARCSense` and `InkMARCSenseDA3`; the
depth model is `InkMARCDepth`. Distribution names are `inkmarc-sense` and
`inkmarc-depth`. The depth package is `inkmarc_depth`; Sense retains the
`inkdepth` import namespace to support saved objects and existing scripts.

`InkDepthStudent`, `InkDepthDA3Student` and `DepthProxyMobileNet` remain aliases.
Factories accept both old and new configuration names. Sense's historical
`inkdepth.models.depth_proxy` import forwards to the extracted Depth package.
The earlier student snapshot also uses `inkdepth` and must be installed in a
separate environment from Sense.

Depth's model package can be installed independently. Its research tools
reuse Sense's dataset/configuration helpers from the adjacent checkout.
The combined proxy-plus-sensing evaluator remains in Sense because it uses
Sense's ONNX pipeline and evaluation helpers. See both component READMEs
for installation and working-directory instructions.

Both `InkMARC.Sense/outputs` and `InkMARC.Depth/outputs` point to
`research/inkdepth_student/outputs`. Existing binaries, output metadata and
checkpoint filenames are retained. Clones without symlink support need
explicit paths/overrides pointing to that shared directory.

## Historical records

[component-layout.json](component-layout.json) maps original paths to current
paths: apply exact file moves before directory moves. Original import
inventories, historical run configurations and model hashes under
`research/provenance/` and `docs/provenance/` retain their original scope.
They describe the pre-reorganisation snapshot; current source renames and
documentation edits are intentional changes after that import.

Thesis files, dataset paths, teacher weights and selected model binaries were
not rewritten. Dataset-dependent training, full experiments and device builds
are outside the verification performed for this layout change.

## Verification

- All 286 relocated regular files are accounted for; no original file is missing.
- All 32 shared model artifacts/metadata files and historical provenance records
  retain their original bytes.
- Four compatibility tests pass, including strict loading of existing Depth and
  Sense checkpoints and inference on synthetic inputs. Run them from the root:
  `python -m unittest discover -s InkMARC.Depth/tests -v`.
- Syntax checks pass for 203 Python files in the new components, the two updated
  supplementary ML modules, and all 10 Augment shell scripts.
- Depth imports independently of Sense. Current navigation links resolve.
- Help checks pass for the Depth and Sense exporters and the compressed-NPY
  teacher-cache script. Five dataset-dependent entry-point checks stop at the
  local environment's missing `zarr` dependency, already listed in requirements.
  Full training, quantisation, ONNX export and device builds were not run.
