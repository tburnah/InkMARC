# export_to_onnx.py
import torch
from train_bounds import ResNetRegressor  # uses your class definition

ckpt_path = "checkpoints/best_model.pt"
onnx_path = "bounds_resnet18_448.onnx"

device = torch.device("cpu")
model = ResNetRegressor(pretrained=False, out_dim=8).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=True)
model.eval()

dummy = torch.randn(1, 3, 448, 448, device=device)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17
)
print(f"Exported ONNX -> {onnx_path}")
