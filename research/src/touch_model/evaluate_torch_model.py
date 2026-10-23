import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# ---------------------
# Settings
# ---------------------
# onnx_model_path = "/data/resnet50_pytorch.onnx"
#onnx_model_path = "/data/resnet50_quant_static.onnx"
#onnx_model_path = "resnet18_preprocessed_quantized_20250528_195205.onnx"
onnx_model_path = "/home/t/tburnah/TouchModel/resnet18_pytorch_20250528_195205.onnx"
batch_size = 8

import glob
import os

def load_all_datasets(image_pattern, label_pattern):
    image_files = sorted(glob.glob(image_pattern))
    label_files = sorted(glob.glob(label_pattern))

    assert len(image_files) == len(label_files), "Mismatch in number of image and label files"

    images_list = []
    labels_list = []

    for img_file, lbl_file in zip(image_files, label_files):
        print(f"Loading: {os.path.basename(img_file)} and {os.path.basename(lbl_file)}")
        images_list.append(np.load(img_file))
        labels_list.append(np.load(lbl_file))

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return images, labels

train_images, train_labels = load_all_datasets("/data/TouchData/train_images*.npy", "/data/TouchData/train_labels*.npy")
val_images, val_labels = load_all_datasets("/data/TouchData/val_images*.npy", "/data/TouchData/val_labels*.npy")
test_images, test_labels = load_all_datasets("/data/TouchData/test_images*.npy", "/data/TouchData/test_labels*.npy")

print(f"\nTraining Set Label Distribution:")
print(f"Touch: {np.sum(train_labels == 1)}")
print(f"No Touch: {np.sum(train_labels == 0)}")

print(f"\nValidation Set Label Distribution:")
print(f"Touch: {np.sum(val_labels == 1)}")
print(f"No Touch: {np.sum(val_labels == 0)}")

# ---------------------
# Load ONNX Model
# ---------------------
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------------
# Preprocessing
# ---------------------
def preprocess_images(images):
    # Normalize like torchvision
    images = images.astype(np.float32) / 255.0
    images = np.transpose(images, (0, 3, 1, 2))  # NHWC → NCHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return ((images - mean) / std).astype(np.float32)

# ---------------------
# Inference
# ---------------------
def evaluate_model(images, labels):
    true_labels = []
    predicted_labels = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        inputs = preprocess_images(batch_imgs)
        preds = session.run([output_name], {input_name: inputs})[0]
        preds = (preds >= 0.5).astype(int).flatten()
        predicted_labels.extend(preds)
        true_labels.extend(batch_labels)

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=["No Touch", "Touch"]))

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

# ---------------------
# Run Evaluation
# ---------------------
# print("\nEvaluating on training set...")
# evaluate_model(train_images, train_labels)

# print("\nEvaluating on validation set...")
# evaluate_model(val_images, val_labels)
print("\nEvaluating on test set...")
evaluate_model(test_images, test_labels)