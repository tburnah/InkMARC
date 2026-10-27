import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tf2onnx  # For ONNX conversion
import onnx

# Enable Mixed Precision Training to Reduce Memory Usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Restrict TensorFlow GPU Memory Usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Paths to TFRecord datasets
TRAIN_TFRECORD = "train.tfrecord"
VALID_TFRECORD = "validate.tfrecord"
batch_size = 2  # Lower batch size to reduce memory usage

# Feature description for parsing TFRecord
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(proto):
    """Parses a single TFRecord example."""
    parsed = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, (448, 448, 3))  # Ensure correct shape
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

def load_tfrecord_dataset(tfrecord_file, batch_size):
    """Loads a TFRecord dataset efficiently."""
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=None)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Reduce shuffle buffer
    return dataset

# Create streaming datasets
train_data = load_tfrecord_dataset(TRAIN_TFRECORD, batch_size)
val_data = load_tfrecord_dataset(VALID_TFRECORD, batch_size)

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

# Define Model Architecture
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
    Activation("sigmoid", dtype=tf.float32)  # Ensure correct precision
])

# Compile Model
seq_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=['accuracy', f1_metric]  # Add F1-score as a metric
)

seq_model.summary()

# Train Model (with optimized TFRecord dataset)
epochs = 10
seq_model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# Fine-tune (Unfreeze EfficientNet and retrain)
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

# Save Model
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