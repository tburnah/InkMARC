# Thesis names and historical project names

InkMARC is the umbrella research programme and software suite. Thesis names
describe roles and do not imply one repository per tool. Root component names now follow the suite terminology. Historical imports,
checkpoint keys and serialized artifact paths remain compatible. Paths beginning
with `InkMARC.` below are relative to the repository root.

| Thesis name | Role and thesis anchor | Source in this archive |
| --- | --- | --- |
| InkMARC-Probe | D0 exploratory collection, §5.2.1 | Complete application outside this snapshot |
| InkMARC-Prepare | Alignment/export of earlier streams, §5.2.1 | Complete application outside this snapshot |
| InkMARC-Train | D1 non-emissive Wacom collection application, §5.2.2 | Complete application outside this snapshot; not a new name for the Python trainers |
| InkMARC-Cue | D2 prompted touch/no-touch capture, §5.2.3 | Complete application outside this snapshot |
| InkMARC-Capture | D3 instrumented Android recording, §5.3.1 | Complete application outside this snapshot |
| InkMARC-Clean | Reconstruction, inspection and export, §5.3.3 | C# reference excerpts in the InkDepth projects and shadow folder |
| InkMARC-Augment | Surface appearance and shadow transfer, §5.3.5 | Historical folder `InkMARC.Augment/` |
| Preliminary touch/localisation | Stand-alone model development, Chapter 6 | `InkMARC.Touch/`, `InkMARC.Location/` |
| InkMARC-Sense | Integrated location/presence/touch, Chapter 7 | Historical package `inkdepth` in the current workspace |
| InkMARC-Depth | Distilled DA3 MobileNetV2 proxy, §7.1.2 | `InkMARC.Depth/inkmarc_depth/model.py` |
| InkMARC-Surface | Live camera/orientation surface tracker, Chapter 8 | Complete live implementation outside this snapshot |
| InkMARC application | Smartphone integration/deployment | Complete application outside this snapshot |

The custom proxy is now consistently called **InkMARC-Depth**. The earlier
`research/src/depth_model/` students remain separate from that MobileNetV2
proxy. `research/inkdepth_student/` is an earlier multitask sensing snapshot;
`InkMARC.Sense/` contains the later integrated sensing tools, while
`InkMARC.Depth/` owns the proxy implementation and its training/export tools.
See the [component map](../../docs/component-layout.md) for original paths,
new source names and checkpoint compatibility.
