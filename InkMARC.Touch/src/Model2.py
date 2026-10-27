import tensorflow as tf
import numpy as np
import autokeras as ak
import glob
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from tensorflow.keras.layers import Concatenate
from autokeras.engine.block import Block
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Enable Multi-GPU Training
strategy = tf.distribute.MirroredStrategy()

# Set Image Dimensions and Training Parameters
IMAGE_SHAPE = (448, 448, 3)
BATCH_SIZE = 32  # Increased batch size for better training
EPOCHS = 150  # More epochs for robust search
MAX_TRIALS = 100  # Extensive model search

# Locate TFRecord Files
train_files = sorted(glob.glob("/data/train_data_part*.tfrecord"))
val_files = sorted(glob.glob("/data/val_data_part*.tfrecord"))
if not train_files or not val_files:
    raise ValueError("No training or validation TFRecord files found!")

# Data Augmentation Layer
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Parse TFRecords

def parse_example(example_proto):
    feature_description = {
        'prev_image': tf.io.FixedLenFeature([], tf.string),
        'prev_touch': tf.io.FixedLenFeature([], tf.int64),
        'curr_image': tf.io.FixedLenFeature([], tf.string),
        'curr_touch': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    prev_image = tf.io.decode_raw(example['prev_image'], tf.float16)
    prev_image = tf.reshape(prev_image, IMAGE_SHAPE)
    curr_image = tf.io.decode_raw(example['curr_image'], tf.float16)
    curr_image = tf.reshape(curr_image, IMAGE_SHAPE)
    prev_image = augmentation(prev_image)  # Apply augmentation
    curr_image = augmentation(curr_image)
    prev_touch = tf.cast(example['prev_touch'], tf.float32) / 1.0  # Normalize
    curr_touch = tf.cast(example['curr_touch'], tf.float32)
    return (prev_image, curr_image, prev_touch), curr_touch

# Load and Preprocess Dataset

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # Optimized pipeline
    return dataset

print("Loading datasets into memory...")
train_data = load_dataset(train_files)
val_data = load_dataset(val_files)

# Define Custom Reshape Block
class ReshapeBlock(Block):
    def build(self, hp, inputs=None):
        if isinstance(inputs, list):
            inputs = inputs[0]
        return layers.Reshape((1,))(inputs)
    def get_config(self):
        return super().get_config()

# Model Definition
with strategy.scope():
    prev_image_input = ak.ImageInput(shape=IMAGE_SHAPE, name="prev_image")
    prev_features = ak.Normalization()(prev_image_input)
    prev_features = ak.ConvBlock(separable=True)(prev_features)
    prev_features = ak.DenseBlock(num_layers=5)(prev_features)
    prev_features = ak.Flatten()(prev_features)
    
    curr_image_input = ak.ImageInput(shape=IMAGE_SHAPE, name="curr_image")
    curr_features = ak.Normalization()(curr_image_input)
    curr_features = ak.ConvBlock(separable=True)(curr_features)
    curr_features = ak.DenseBlock(num_layers=5)(curr_features)
    curr_features = ak.Flatten()(curr_features)
    
    merged_image_features = ak.Merge()([prev_features, curr_features])
    prev_classification_input = ak.Input(shape=(1,), name="prev_classification")
    prev_classification_features = ReshapeBlock()(prev_classification_input)
    prev_classification_features = ak.DenseBlock()(prev_classification_features)
    
    merged_features = ak.Merge()([merged_image_features, prev_classification_features])
    output = ak.ClassificationHead(
        loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2.0),
        metrics=["accuracy", "AUC"]
    )(merged_features)
    
    auto_model = ak.AutoModel(
        inputs=[prev_image_input, curr_image_input, prev_classification_input],
        outputs=output,
        max_trials=MAX_TRIALS,
        overwrite=True
    )

# Learning Rate Scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True
)
auto_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# Train Model
print("Starting Auto-Keras binary classification training...")
auto_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# Save Best Model
best_model = auto_model.export_model()
best_model.save("autokeras_best_model_optimized.keras")
print("✅ Optimized model saved as autokeras_best_model_optimized.keras")
