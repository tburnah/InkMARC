from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
import os

class NpyDataReader(CalibrationDataReader):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")]
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self._load_all_batches())
        return next(self.enum_data, None)

    def _load_all_batches(self):
        for file_path in self.data_files:
            arr = np.load(file_path)  # shape: (N, 448, 448, 3)
            print(f"{file_path} loaded with shape {arr.shape}")

            if arr.ndim == 4 and arr.shape[-1] == 3:
                for i in range(arr.shape[0]):
                    sample = arr[i]
                    sample = sample.astype(np.float32) / 255.0  # 🔥 Normalize here
                    sample = np.transpose(sample, (2, 0, 1))    # HWC → CHW
                    sample = np.expand_dims(sample, axis=0)    # Add batch dimension
                    yield {"input": sample}
            else:
                raise ValueError(f"Unexpected shape in file {file_path}: {arr.shape}")

quantize_static(
    model_input='resnet18_pytorch_20250528_195205.onnx',
    model_output='resnet18_preprocessed_quantized_20250528_195205.onnx',
    calibration_data_reader=NpyDataReader("/data/tmp"),
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True
)
