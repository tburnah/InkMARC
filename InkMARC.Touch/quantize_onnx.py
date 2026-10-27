from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import onnx


class NumpyDataReader(CalibrationDataReader):
    def __init__(self, images, input_name):
        self.input_name = input_name
        self.data = [img[np.newaxis, ...] for img in images]  # Adds batch dim
        self.data_iter = iter(self.data)

    def get_next(self):
        try:
            return {self.input_name: next(self.data_iter)}
        except StopIteration:
            return None


# Load calibration data (10 samples is enough for this)
images = np.load("train_images.npy").astype(np.float32) / 255.0
images = np.transpose(images, (0, 3, 1, 2))  # NHWC to NCHW
images = images[:10]

# Normalize to match training
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
images = (images - mean) / std

# Load model to get input name
model = onnx.load("resnet50_pytorch.onnx")
input_name = model.graph.input[0].name

# Set up data reader
calib_reader = NumpyDataReader(iter(images), input_name)

quantize_static(
    model_input="/data/resnet50_pytorch.onnx",
    model_output="/data/resnet50_quant_static.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8
)

print("Quantized using static quantization with QOperator format.")
