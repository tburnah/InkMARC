import onnx

# Load model
model = onnx.load("resnet18_simplified.onnx")

# Convert to a lower opset version if needed
model.opset_import[0].version = 11

# Save the modified model
onnx.save(model, "resnet18_simplified_opset11.onnx")
