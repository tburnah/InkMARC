import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tf2onnx  # For ONNX conversion
import onnx

# 🔹 Paths to HDF5 datasets
train_h5 = "/data/train.h5"
val_h5 = "/data/validate.h5"
batch_size = 8  # Adjust based on available memory

# 🔹 Function to create HDF5 dataset loader
def h5_dataset(file_path, batch_size):
    """Fast HDF5 data streaming with in-memory shuffling for each epoch."""
    with h5py.File(file_path, "r") as f:
        dataset_size = len(f["Images"])
        image_shape = f["Images"][0].shape  # Example: (448, 448, 3)

    def load_batch(indices):
        """Helper function to load a batch of images and labels (sorted indices)."""
        with h5py.File(file_path, "r") as f:
            sorted_indices = np.sort(indices)  # ✅ Fix: Ensure increasing order
            images = f["Images"][sorted_indices] / 255.0  # Normalize
            labels = f["Attributes"][sorted_indices]
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def generator():
        """Generator that shuffles indices in memory for each epoch."""
        indices = np.arange(dataset_size)
        while True:
            np.random.shuffle(indices)  # ✅ Shuffle once per epoch
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i + batch_size]
                yield load_batch(batch_indices)  # ✅ Load batch with sorted indices

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)  # ✅ Preload batches for speed


# 🔹 Create streaming datasets
train_data = h5_dataset(train_h5, batch_size)
val_data = h5_dataset(val_h5, batch_size)

def f1_metric(y_true, y_pred):
    """Computes the F1 score as a Keras metric."""
    y_true = K.cast(y_true, 'float32')  # Convert to float32 to match y_pred
    y_pred = K.round(y_pred)  # Convert probabilities to 0 or 1
    
    tp = K.sum(y_true * y_pred)  # True positives
    fp = K.sum((1 - y_true) * y_pred)  # False positives
    fn = K.sum(y_true * (1 - y_pred))  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# 🔹 Define Model Architecture
base_model = EfficientNetB7(
    include_top=False,  # Remove classifier head
    input_shape=(448, 448, 3),
    weights="imagenet"
)

base_model.trainable = False  # Freeze EfficientNet for feature extraction

seq_model = Sequential([
    base_model,  # EfficientNet backbone
    GlobalAveragePooling2D(),  # Flatten feature maps
    Dense(1),  # Fully connected layer
    Activation("sigmoid")  # Binary classification
])

# 🔹 Compile Model
seq_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=['accuracy', f1_metric]  # Add F1-score as a metric
)

seq_model.summary()

# 🔹 Train Model (with streaming HDF5 dataset)
epochs = 10
seq_model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# 🔹 Fine-tune (Unfreeze EfficientNet and retrain)
base_model.trainable = True  # Enable training on backbone
seq_model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

seq_model.fit(
    train_data,
    epochs=5,  # Shorter fine-tuning phase
    validation_data=val_data
)

# 🔹 Save Model
keras_model_path = "trained_efficientnetb7.keras"
onnx_model_path = "trained_efficientnetb7.onnx"

seq_model.save(keras_model_path)
print(f"Keras model saved as: {keras_model_path}")

# Define an input signature for ONNX conversion
input_signature = [tf.TensorSpec([None, 448, 448, 3], dtype=tf.float32, name="input_layer")]

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    seq_model,
    input_signature=input_signature,  # Explicitly define input signature
    opset=13,
    output_path="trained_efficientnetb7.onnx"  # Save ONNX model
)

print("ONNX model saved as: trained_efficientnetb7.onnx")