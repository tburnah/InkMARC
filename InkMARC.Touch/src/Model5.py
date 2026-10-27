import autokeras as ak
import tensorflow as tf
import h5py
import numpy as np
import os

# Mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Function to load batches from HDF5 file efficiently
def h5_dataset(file_path, batch_size):
    def generator():
        with h5py.File(file_path, "r") as f:
            images = f["Images"]
            labels = f["Attributes"]
            dataset_size = len(images)

            for i in range(0, dataset_size, batch_size):
                x_batch = images[i:i + batch_size] / 255.0
                y_batch = labels[i:i + batch_size]
                yield np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.int32)

    # Get shape info from one example
    with h5py.File(file_path, "r") as f:
        image_shape = f["Images"][0].shape
        label_shape = ()

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)

# Set paths and batch size
train_h5 = '/data/train.h5'
val_h5 = '/data/validate.h5'
batch_size = 4  # Keep small to avoid OOM

# Create datasets
train_data = h5_dataset(train_h5, batch_size)
val_data = h5_dataset(val_h5, batch_size)

# AutoKeras setup with resume capability
model_dir = "./autokeras_model"
overwrite = not os.path.exists(model_dir)

clf = ak.ImageClassifier(
    directory=model_dir,
    overwrite=overwrite,
    max_trials=10,
    objective='val_accuracy'
)

# Train the model
clf.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
)

# Save best model
model = clf.export_model()
model.save("best_model.keras")

print("Model training completed and saved.")
print(model.summary())
