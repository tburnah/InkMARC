import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Set this to the directory with your .png images
image_folder = "/data/sample_images/NoTouch"
model_path = "efficientnetb7_initial_finetuned.keras"

# Custom metric function (needed for loading the model)
def f1_metric(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), 'float32')
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

# Load the fine-tuned model
model = load_model(model_path, custom_objects={'f1_metric': f1_metric})

# Ensure image preprocessing matches training
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448))
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Accepted image extensions
valid_extensions = (".png", ".jpg", ".jpeg")

# Iterate over PNG files and perform inference
for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(valid_extensions):
        image_path = os.path.join(image_folder, file_name)
        input_tensor = preprocess_image(image_path)
        prediction = model.predict(input_tensor)[0][0]  # Binary prediction
        label = "Positive" if prediction >= 0.5 else "Negative"
        print(f"{file_name}: {prediction:.4f} ({label})")
