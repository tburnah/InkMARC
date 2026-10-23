# Staging change log

Research executable source, configurations and model binaries were not edited.

| Staged path | Action | Original document retained |
| --- | --- | --- |
| `.gitignore` | add_documentation_or_ignore | — |
| `README.md` | add_documentation_or_ignore | — |
| `docs/contribution-index.md` | add_documentation_or_ignore | — |
| `docs/datasets.md` | add_documentation_or_ignore | — |
| `docs/dependencies.md` | add_documentation_or_ignore | — |
| `docs/known-issues.md` | add_documentation_or_ignore | — |
| `docs/licensing.md` | add_documentation_or_ignore | — |
| `docs/reproducibility.md` | add_documentation_or_ignore | — |
| `docs/terminology-map.md` | add_documentation_or_ignore | — |
| `inkdepth_student/README.md` | replace_documentation_or_ignore | `provenance/original-documentation/inkdepth_student/README.md` |
| `inkdepth_student/reference/README.md` | add_documentation_or_ignore | — |
| `inkdepth_student_workspace/.gitignore` | replace_documentation_or_ignore | `provenance/original-documentation/inkdepth_student_workspace/.gitignore` |
| `inkdepth_student_workspace/DEPTH_PROXY.md` | add_documentation_or_ignore | — |
| `inkdepth_student_workspace/README.md` | replace_documentation_or_ignore | `provenance/original-documentation/inkdepth_student_workspace/README.md` |
| `inkdepth_student_workspace/reference/README.md` | add_documentation_or_ignore | — |
| `ml/README.md` | replace_documentation_or_ignore | `provenance/original-documentation/ml/README.md` |
| `provenance/README.md` | add_documentation_or_ignore | — |
| `src/depth_model/README.md` | add_documentation_or_ignore | — |
| `src/location_model/README.md` | add_documentation_or_ignore | — |
| `src/shadow_extraction/README.md` | add_documentation_or_ignore | — |
| `src/touch_model/README.md` | add_documentation_or_ignore | — |

The workspace `outputs` symlink was replaced in staging with
`../inkdepth_student/outputs`, preserving the shared-output relationship
inside the archive. All other original symlinks are inventoried locally
without following or copying their targets.

Machine-generated copy/model/repository/exclusion manifests, environment
metadata and verification records are archival additions. The sibling
`review-only/tools/` scripts construct and check the candidate; they are
outside the proposed public root.

Forty-six small historical JSON run configurations were subsequently selected
and copied byte for byte under `provenance/run-configurations/`. Their original
source hashes and locations are in `copied-files.json`; generated run payloads
remain excluded. Root ignore rules explicitly retain these settings. The final
`public-files.json` and `SHA256SUMS` inventory and hash the complete candidate.


## Copyright and attribution update — 24 October 2026

At the author's explicit direction, original contributions are now all rights
reserved. Root/component COPYRIGHT notices, reserved-rights LICENSE, third-party
credits, full upstream permission notices, model-card snapshots and file/model
rights maps were added. README/licensing text was updated. No executable source,
configuration value, original project or model binary was edited. Previous
"license pending" statements describe the earlier Phase B state only.
The complete before/after additions and documentation hashes are recorded in
`licensing-changes.json` and the final checksum inventory.
