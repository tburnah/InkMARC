import tensorflow as tf
import numpy as np
import os

# -------- Settings --------
TFRECORDS = {
    "test": "/data/test_split.tfrecord"
    # "train": "/data/train_split.tfrecord",
    # "val": "/data/validate_split.tfrecord"
}
output_dir = "/data"
os.makedirs(output_dir, exist_ok=True)

# -------- TFRecord Parser --------
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(proto):
    parsed = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, (448, 448, 3))
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

# -------- Process and Save --------
def convert_to_numpy(tfrecord_path, prefix):
    images = []
    labels = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)  # Already binary (0 or 1)

    np.save(os.path.join(output_dir, f"{prefix}_images.npy"), images)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels)
    print(f"✅ Saved {prefix}_images.npy and {prefix}_labels.npy — shape: {images.shape}, {labels.shape}")

# -------- Run conversion --------
# convert_to_numpy(TFRECORDS["train"], "train")
# convert_to_numpy(TFRECORDS["val"], "val")
convert_to_numpy(TFRECORDS["test"], "test")

