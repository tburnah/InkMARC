import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------------
# Settings
# ---------------------
train_tfrecord = "/data/train.tfrecord"
val_tfrecord = "/data/validate.tfrecord"
model_path = "resNet50_initial_finetuned.keras"
batch_size = 8

# ---------------------
# Feature description for TFRecords
# ---------------------
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(proto):
    parsed = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, (448, 448, 3))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

def load_dataset(path, batch_size):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------
# Custom Metric
# ---------------------
def f1_metric(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), 'float32')
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

# ---------------------
# Load model
# ---------------------
model = load_model(model_path, custom_objects={'f1_metric': f1_metric})
model.summary()

# ---------------------
# Load datasets
# ---------------------
train_dataset = load_dataset(train_tfrecord, batch_size)
val_dataset = load_dataset(val_tfrecord, batch_size)

# ---------------------
# Count label distribution
# ---------------------
def count_labels(tfrecord_path):
    touch_count = 0
    no_touch_count = 0
    for record in tf.data.TFRecordDataset(tfrecord_path):
        parsed = tf.io.parse_single_example(record, feature_description)
        label = parsed['label'].numpy()
        if label == 1:
            touch_count += 1
        else:
            no_touch_count += 1
    return touch_count, no_touch_count

train_touch, train_no_touch = count_labels(train_tfrecord)
val_touch, val_no_touch = count_labels(val_tfrecord)

print(f"\n📊 Training Set Label Distribution:")
print(f"Touch: {train_touch}")
print(f"No Touch: {train_no_touch}")

print(f"\n📊 Validation Set Label Distribution:")
print(f"Touch: {val_touch}")
print(f"No Touch: {val_no_touch}")

# ---------------------
# Evaluate model
# ---------------------
print("\n🔍 Evaluating on training set...")
train_steps = (train_touch + train_no_touch) // batch_size
train_metrics = model.evaluate(train_dataset, steps=train_steps)

print("\n🔍 Evaluating on validation set...")
val_steps = (val_touch + val_no_touch) // batch_size
val_metrics = model.evaluate(val_dataset, steps=val_steps)

print("\n✅ Evaluation Complete")
for name, value in zip(model.metrics_names, train_metrics):
    print(f"Train {name}: {value:.4f}")
for name, value in zip(model.metrics_names, val_metrics):
    print(f"Val {name}: {value:.4f}")

from sklearn.metrics import confusion_matrix, classification_report

# Get all predictions
true_labels = []
predicted_labels = []

for images, labels in val_dataset:
    preds = model.predict(images)
    preds = (preds >= 0.5).astype(int).flatten()
    predicted_labels.extend(preds)
    true_labels.extend(labels.numpy())

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["No Touch", "Touch"]))
