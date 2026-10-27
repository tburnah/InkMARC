import autokeras as ak
import h5py
import numpy as np
import tensorflow as tf
import os
import subprocess
import gc

# Load dataset from your preprocessed HDF5 files
with h5py.File('/data/train.h5', 'r') as f:
    x_train = np.array(f["Images"], dtype=np.float32) / 255.0
    y_train = np.array(f["Attributes"], dtype=np.int32)

with h5py.File('/data/validate.h5', 'r') as f:
    x_val = np.array(f["Images"], dtype=np.float32) / 255.0
    y_val = np.array(f["Attributes"], dtype=np.int32)

gc.collect()

def get_available_gpu_memory():
    """Query total and free GPU memory from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        total_memory, free_memory = map(int, result.stdout.strip().split("\n")[0].split(", "))
        return total_memory, free_memory
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return None, None

# 2) Dynamically limit GPU memory usage (optional)
total_gpu_mem, free_gpu_mem = get_available_gpu_memory()
if total_gpu_mem and free_gpu_mem:
    memory_limit = int(free_gpu_mem * 0.8)
    print(f"Setting GPU memory limit to: {memory_limit} MB")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
            print(f"Successfully set memory limit to {memory_limit} MB")
        except RuntimeError as e:
            print(f"Error setting GPU memory limit: {e}")

# Check if AutoKeras model directory exists
model_dir = "./autokeras_model"
overwrite = not os.path.exists(model_dir)

# AutoKeras automatically searches for the best CNN architecture
clf = ak.ImageClassifier(
    directory=model_dir,
    overwrite=overwrite,
    max_trials=10  # Try more architectures if you have enough resources
)

# Train and validate the classifier
clf.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# Export the best model
model = clf.export_model()
model.save("best_handwriting_model.keras")

print(model.summary())
