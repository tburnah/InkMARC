# Thesis names and historical project names

InkMARC is the umbrella research programme and software suite. Thesis names
describe roles and do not imply one repository per tool. Executable names,
package imports, checkpoint keys and serialized paths are preserved.

| Thesis name | Role and thesis anchor | Source in this archive |
| --- | --- | --- |
| InkMARC-Probe | D0 exploratory collection, §5.2.1 | Complete application outside this snapshot |
| InkMARC-Prepare | Alignment/export of earlier streams, §5.2.1 | Complete application outside this snapshot |
| InkMARC-Train | D1 non-emissive Wacom collection application, §5.2.2 | Complete application outside this snapshot; not a new name for the Python trainers |
| InkMARC-Cue | D2 prompted touch/no-touch capture, §5.2.3 | Complete application outside this snapshot |
| InkMARC-Capture | D3 instrumented Android recording, §5.3.1 | Complete application outside this snapshot |
| InkMARC-Clean | Reconstruction, inspection and export, §5.3.3 | C# reference excerpts in the InkDepth projects and shadow folder |
| InkMARC-Augment | Surface appearance and shadow transfer, §5.3.5 | Historical folder `src/shadow_extraction/` |
| Preliminary touch/localisation | Stand-alone model development, Chapter 6 | `src/touch_model/`, `src/location_model/` |
| InkMARC-Sense | Integrated location/presence/touch, Chapter 7 | Historical package `inkdepth` in the current workspace |
| InkMARC Depth / InkDepth | Distilled DA3 MobileNetV2 proxy, §7.1.2 | `inkdepth/models/depth_proxy.py` in the current workspace |
| InkMARC-Surface | Live camera/orientation surface tracker, Chapter 8 | Complete live implementation outside this snapshot |
| InkMARC application | Smartphone integration/deployment | Complete application outside this snapshot |

The supplied thesis calls the custom proxy “InkMARC Depth” in §7.1; the
archive also retains the author's name “InkDepth.” The dedicated historical
`src/depth_model/` contains earlier depth students; it is kept separately
from the MobileNetV2 proxy. `inkdepth_student/` is an earlier multitask
sensing source snapshot, while `inkdepth_student_workspace/` contains the
later integrated and proxy tools. Neither directory nor package is renamed.
