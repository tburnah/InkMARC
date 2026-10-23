# Dependencies and upstream versions

Dependency definitions are preserved. No environment was installed, updated
or repaired during staging. Declared lower bounds are not a historical lock.
[observed-environments.json](../provenance/observed-environments.json)
records selected package versions read from existing VM package metadata;
it is not proof of the environment used for every research experiment.

| Project | Definitions and observed code dependencies |
| --- | --- |
| Both InkDepth snapshots | `requirements.txt`, `pyproject.toml`; Python >=3.10, PyTorch/torchvision, NumPy, OpenCV, Pillow, YAML, tqdm, Zarr, h5py, pandas, TensorBoard, ONNX/ONNX Runtime |
| Current optional utilities | scikit-learn for feature classifiers/fusion; SciPy for statistical tests; Matplotlib for plots; upstream DA3 and its dependencies for live depth |
| Preliminary touch | TensorFlow/Keras, AutoKeras, PyTorch/torchvision, scikit-learn, tf2onnx and ONNX tools, depending on entry point; no single project-wide dependency file found |
| Preliminary location and `ml` | PyTorch/torchvision, OpenCV, NumPy, h5py, YAML, ONNX tools; pytest for existing tests; no complete historical lock found |
| Historical depth students | PyTorch/torchvision, NumPy, Pillow, OpenCV, tqdm, Depth Anything V2; some preparation scripts also import TensorFlow or Zarr |
| Shadow/augmentation | C++17 compiler, OpenCV libraries including aruco/objdetect, HDF5 and compression libraries; Python/OpenCV/PyTorch for depth; FFmpeg for media workflows |
| Clean references | C# desktop source excerpts, OpenCvSharp and application-specific types; full solution/project definitions are outside this snapshot |

## Third-party source

Pinned checkout revisions, upstream URLs and unmodified license notices are
recorded in [third-party.json](../provenance/third-party.json). Full upstream
source and build trees are not vendored. Retrieve the recorded revision into
the established project location when required, particularly
`src/depth_model/Depth-Anything-V2` or `Depth-Anything-3`.

Historical absolute DA3 roots can be supplied through existing CLI/config
overrides. DA3's frozen wrapper calls `from_pretrained` for
`depth-anything/DA3-SMALL`; model download/cache availability is a separate
requirement. Upstream teacher weights are not bundled.

The preserved DA3 declaration specifies `numpy<2`; the inspected VM
environments contain NumPy 2.x. Shadow build scripts also contain exact
local compiler/include/library paths. These facts are documented for setup
review; the archive applies no dependency or build changes.
