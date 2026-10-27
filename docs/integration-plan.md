# Integrating the VM publication candidate

> Historical import record. The current layout is documented in the
> [component map](component-layout.md); original inventories and hashes retain
> their import-time paths.

**Status: VM research imported locally and verified.** The local branch is based on
`715d1ce7e2b0e9432cf2c1e29246c51a9d1eecc7` of `tburnah/InkMARC`.
The Phase B staging tree and all original research repositories remain
unchanged. The import itself performed no remote writes or history rewrite;
the author subsequently approved the VM archive publication separately.
See the [import review](research-import.md) and
[fresh provenance record](provenance/research-import.json).

## Implemented placement

Only the separately staged **`public/` contents** were copied into `research/`
in this repository, preserving their entire internal structure. Existing
.NET directories, solution names and sibling references remain in place.

This creates a single research archive with a central thesis index while
preserving the .NET suite and VM source layouts. The VM root README, license,
ignore rules, dependency notices and provenance then have their own `research/`
scope; they do not overwrite the repository-level equivalents.

| Staged publication path | Imported repository path | Preservation reason |
| --- | --- | --- |
| `src/` | `research/src/` | Keep historical model and offline augmentation projects together |
| `ml/` | `research/ml/` | Retain package imports and separate experiment identity |
| `inkdepth_student/` | `research/inkdepth_student/` | Preserve earlier source and selected model locations |
| `inkdepth_student_workspace/` | `research/inkdepth_student_workspace/` | Preserve sensing/proxy source, configurations and scripts |
| `docs/` | `research/docs/` | Keep VM-specific dependency notices and chapter mapping intact |
| `provenance/` | `research/provenance/` | Preserve original manifests, hashes and historical documents |
| Root README, rights files and ignore rules | Same filenames under `research/` | Avoid root-document collisions and retain snapshot evidence |

The existing link
`inkdepth_student_workspace/outputs -> ../inkdepth_student/outputs`
remains within `research/` when both projects move together. Run VM entry
points from their documented roots within `research/`; for example, the
archival instruction “run ml from the archive root” refers to `research/`,
not to the containing .NET repository. Historical absolute paths are not
silently rewritten.

## Concrete preflight results

The [preserved preflight mapping](integration-plan.json) contains SHA-256 hashes
and destinations for all **646 regular files and one symlink**: 373,457,272
bytes in regular files (about 356 MiB). The largest staged file is 24,706,552
bytes, below [GitHub's 100 MiB limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github);
no Git LFS migration is needed for individual file sizes. Large-file
distribution can still be chosen separately.

No planned `research/` destination collided with the cloned tree. An isolated
layout preview combined the clone's existing ignore rules with all staged
ignore files: none of the 647 proposed file/link paths was ignored. This was
checked before the import; the actual destination was checked again afterward.
The mapping JSON's `planned_not_imported` status records its original preflight
state; the current import status is in the separate
[import record](provenance/research-import.json). No byte-identical
C# source files were found between the staged selection and the existing
repository, so no identity-based source consolidation is proposed.

Do not import the staging directory's private `review-only/`, Phase B reports,
local review folders, original `.git` directories, raw research datasets,
withheld checkpoints or participant-level outputs. Existing historical assets
already in the clone are a separate release-review scope.

## Import and documentation procedure used

1. The destination commit, branch, existing dirty state, file hashes and index
   were recorded privately. An absent `research/` destination was required.
   The copy retained symlinks and file modes and created independent files,
   preserving existing local documentation and application source.
2. Destination hashes were checked against the preflight mapping and the
   preserved Phase B checksum manifest. The checksum file's own hash, all
   copied entries and the internal symlink were checked separately. Original
   Phase B manifests and verification records remain unchanged.
3. The separate [repository-level import record](provenance/research-import.json)
   names the source snapshot, mapping prefix, base commit, copy time,
   documentation changes and fresh destination checks. Private paths,
   repository-state captures and review tooling remain outside this repository.
4. The root README, combined contribution index, component navigation and
   dependency/licensing links now point to incorporated source. The root
   index and [archive status](archive-status.json) supply current availability
   information without editing frozen nested documentation.
5. Both root and VM notices and model-rights records are retained. Clean's
   VM excerpts and this repository's project remain distinct historical
   sources. Prior valid grants and third-party rights remain effective.
6. Fresh destination integrity, links, ignore rules and exclusion checks
   passed. Historical Phase B syntax, ONNX-schema and synthetic-test results
   are retained as historical evidence, not rerun or presented as destination
   application testing. No research or model-generation workflow was executed.

To recheck the preserved snapshot's checksum manifest from its correct root:

```sh
cd research
sha256sum --check --strict provenance/SHA256SUMS
```

The entire `research/` tree is the unchanged imported snapshot. Future nested
documentation changes need their own successor change record; source,
configuration and model preservation constraints remain in force.

## Desktop project status

The latest Capture source is incorporated as `InkMARC.Capture/`, preserving
its standalone Android Gradle project and AprilTag dependency. The Chapter 8
Surface/application source remains on the author's desktop and pending
incorporation; its recommended future destination is `applications/InkMARC/`,
preserving its original solution layout and Surface's actual location.

Before inclusion, inventory and hash the original source and dirty repository
state; document actual dependencies, SDKs, hardware and build requirements;
exclude sensitive recordings, credentials and signing material; inspect
third-party/model/asset licensing; record private origin and public copy
provenance; and add verified links to the incorporated Capture (§5.3.1) and
pending Chapter 8 source in the contribution index. Preserve desktop originals
and do not reconstruct or substitute older VM applications.

## Review and publication gates

The VM import was completed independently of the desktop additions. The
complete archive must remain pending until the **Chapter 8 desktop component
has been reviewed and incorporated**. Existing model initialization/redistribution
questions, repository assets/binaries and tracked signing material need their
own handling decisions before the completed archive is released. The import
did not delete tracked historical material, rewrite Git history, revoke valid
prior licenses or enable Git LFS. The author has separately approved committing
and pushing the current combined VM archive and documentation to `main`;
the Chapter 8 desktop addition will follow in a later push. See the
[VM publication record](provenance/vm-publication.json). This approval does
not mark the complete archive finished.

Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original
contributions, subject to prior grants and third-party rights.
