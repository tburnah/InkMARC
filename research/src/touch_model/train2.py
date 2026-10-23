import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ----------------------------------------
# TFRecord Config
# ----------------------------------------
TRAIN_TFRECORD = "/data/train.tfrecord"
VALID_TFRECORD = "/data/validate.tfrecord"
batch_size = 2
max_samples = None
max_validation = None

# ----------------------------------------
# TFRecord Parsing
# ----------------------------------------
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

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def load_tfrecord_dataset(tfrecord_file, batch_size, is_training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=None)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1024)
        if max_samples is not None:
            dataset = dataset.take(max_samples)
    else:
        dataset = dataset.shuffle(1024)
        if max_validation is not None:
            dataset = dataset.take(max_validation)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ----------------------------------------
# Custom F1 Metric
# ----------------------------------------
def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), 'float32')
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# ----------------------------------------
# MobileNetV3 Model
# ----------------------------------------
def create_model(input_shape=(448, 448, 3)):
    base_model = MobileNetV3Large(input_shape=input_shape, include_top=False, weights="imagenet")
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# ----------------------------------------
# Main Execution
# ----------------------------------------
if __name__ == "__main__":
    print("Loading TFRecord datasets...")
    train_data = load_tfrecord_dataset(TRAIN_TFRECORD, batch_size=batch_size, is_training=True)
    val_data = load_tfrecord_dataset(VALID_TFRECORD, batch_size=batch_size, is_training=False)
    num_train_samples = sum(1 for _ in tf.data.TFRecordDataset(TRAIN_TFRECORD))
    num_val_samples = sum(1 for _ in tf.data.TFRecordDataset(VALID_TFRECORD))
    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size

    print("Creating and compiling model...")
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", f1_metric])
    model.summary()

    print("Training model...")
    model.fit(
        train_data,
        epochs=10,  # Adjust as needed
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps
    )

    # Save the trained model
    model_path = "model2.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # ----------------------------------------
    # Evaluation and Confusion Matrix
    # ----------------------------------------
    print("Evaluating on validation set...")
    results = model.evaluate(val_data, steps=validation_steps)
    print(f"\nValidation Results:\nLoss={results[0]:.4f}, Accuracy={results[1]:.4f}, F1 Score={results[2]:.4f}")

    print("Generating predictions for confusion matrix...")
    y_true = []
    y_pred = []

    for images, labels in val_data.take(validation_steps):
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
