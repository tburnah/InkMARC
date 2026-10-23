# Archive provenance

`repositories.json` records source repository identities and working-tree
status counts. Only the current workspace has first-party Git metadata;
its working source includes uncommitted and untracked implementations.
Original histories remain in place and were not rewritten or copied into
this publication candidate.

`copied-files.json` records each copied file's logical source location,
original SHA-256, final staged SHA-256, role and change classification.
`models.csv` indexes selected original model artifacts. Model files are
copied directly, without deserialisation, training, export or conversion.

`run-configurations/` retains small historical JSON run settings separately
from generated run payloads. `SHA256SUMS` covers every regular public file except
itself; `public-files.json` inventories content except itself and that checksum
file. `symlinks.json` separately records the one directory link.

`modifications.json` and `CHANGELOG.md` list documentation/ignore changes.
Original versions of replaced documents remain byte-identical under
`original-documentation/`; their historical instructions and relative links
may refer to the old workspace. Use the current component README to navigate.

`symlinks.json` records the internal shared-output link.
`exclusion-summary.json` records categories omitted from the public tree.
`third-party.json` pins external checkouts and their notices.
`verification.json` records actual archival checks and selected tests.

Complete absolute source locations, Git status paths, detailed exclusion and
held-checkpoint inventories, thesis PDF fingerprint and verification tooling
remain in the sibling `review-only/` directory. Do not publish that directory
as part of this root.
