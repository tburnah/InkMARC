#!/usr/bin/env python3
"""
Train an AutoKeras ImageClassifier on merged NumPy splits without blowing up RAM,
and print validation Accuracy and F1 at the end of each epoch.

Key features:
- Streams from disk using memmaps (no huge RAM allocations).
- Handles three on-disk cases for images:
    1) Proper .npy header -> np.load(..., mmap_mode='r')
    2) Raw binary (no .npy header) -> infer dtype by file size and open np.memmap
    3) Object-array .npy (pickled) -> explicit error with fix suggestion (to avoid RAM blowups)
- Normalizes inputs to [0,1] on-the-fly in tf.data pipeline.
- Prints Accuracy & F1 on the validation set at the end of each epoch.
"""

import argparse
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all,1=INFO,2=WARNING,3=ERROR
import tensorflow as tf
import autokeras as ak
from tensorflow.keras import backend as K

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# -----------------------------
# Hyperparameters
# -----------------------------
IMAGE_SHAPE = (448, 448, 3)  # (H, W, C)
BATCH_SIZE  = 16
EPOCHS      = 150
MAX_TRIALS  = 100

# -----------------------------
# Custom F1 loss (as before)
# -----------------------------
class F1Score(tf.keras.metrics.Metric):
    """Binary F1 with threshold 0.5, works with AutoKeras/Tuner."""
    def __init__(self, name="F1", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=self.dtype)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=self.dtype)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # handle sigmoid (N,1), softmax (N,2), or flat (N,)
        if y_pred.shape.rank is not None and y_pred.shape.rank == 2 and y_pred.shape[-1] == 2:
            p1 = y_pred[:, 1]
        elif y_pred.shape.rank is not None and y_pred.shape.rank == 2 and y_pred.shape[-1] == 1:
            p1 = tf.squeeze(y_pred, axis=-1)
        else:
            p1 = tf.reshape(y_pred, [-1])
        y_hat = tf.cast(p1 >= 0.5, tf.int32)

        tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_hat == 1), self.dtype))
        fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_hat == 1), self.dtype))
        fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_hat == 0), self.dtype))

        self.tp.assign_add(tp); self.fp.assign_add(fp); self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall    = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)

def f1_loss(y_true, y_pred):
    """Binary F1 loss: 1 - F1(y_true, round(y_pred))."""
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum((1 - y_pred) * y_true)
    precision = tp / (tp + fp + K.epsilon())
    recall    = tp / (tp + fn + K.epsilon())
    f1 = 2.0 * (precision * recall) / (precision + recall + K.epsilon())
    return 1.0 - f1

# -----------------------------
# Robust image openers
# -----------------------------
def _infer_raw_dtype(filesize_bytes: int, n: int, h: int, w: int, c: int):
    """Infer dtype by exact byte-size match among common dtypes."""
    n_elems = n * h * w * c
    for dt in (np.uint8, np.float16, np.float32):
        if filesize_bytes == n_elems * np.dtype(dt).itemsize:
            return dt
    raise ValueError(
        f"Cannot infer raw dtype for file size {filesize_bytes} with shape {(n,h,w,c)}; "
        f"expected one of uint8/float16/float32 raw layouts."
    )

def _open_images_any(path: str, n_items: int):
    """
    Open an on-disk image array of shape (N,H,W,C) as a NumPy array-like that supports slicing.
    Tries:
      1) Proper .npy memmap via np.load(..., mmap_mode='r')
      2) Raw memmap (infer dtype from file size)
      3) If object-array .npy, raise with a helpful message
    Returns a memmap/ndarray-like; does NOT load everything in RAM.
    """
    H, W, C = IMAGE_SHAPE

    # (1) Try proper .npy header (fast path)
    try:
        arr = np.load(path, mmap_mode="r")
        if getattr(arr, "dtype", None) == object:
            # It IS a .npy, but holds pickled objects (slow & RAM-hungry to use)
            raise RuntimeError(
                f"{path} is an object-array .npy (pickled). "
                f"Please convert to a numeric array on disk (e.g., uint8) first."
            )
        if arr.shape != (n_items, H, W, C):
            raise ValueError(f"{path}: shape {arr.shape} != expected {(n_items, H, W, C)}")
        return arr  # memmapped numeric .npy
    except RuntimeError:
        # re-raise object array case -> caller will show message
        raise
    except Exception:
        # Fall through to raw memmap detection
        pass

    # (2) Try raw memmap (no .npy header): infer dtype from file size
    sz = os.path.getsize(path)
    dtype = _infer_raw_dtype(sz, n_items, H, W, C)
    mm = np.memmap(path, dtype=dtype, mode="r", shape=(n_items, H, W, C))
    return mm  # slicing returns views, still no big RAM copies

def _open_memmaps(merged_dir):
    """
    Open images as read-only (proper .npy or raw memmap) and labels fully (small).
    Validates shapes match IMAGE_SHAPE and lengths match labels.
    """
    P = lambda n: os.path.join(merged_dir, n)

    # Labels: small, okay to load fully
    ytr = np.load(P("train_labels.npy")).astype(np.int32).reshape(-1)
    yva = np.load(P("val_labels.npy")).astype(np.int32).reshape(-1)

    # Images: tolerant open (proper .npy or raw memmap)
    Xtr = _open_images_any(P("train_images.npy"), len(ytr))
    Xva = _open_images_any(P("val_images.npy"),   len(yva))

    # Final sanity
    assert Xtr.shape[0] == ytr.shape[0], "Train image/label count mismatch"
    assert Xva.shape[0] == yva.shape[0], "Val image/label count mismatch"
    assert Xtr.shape[1:] == IMAGE_SHAPE, f"Train image shape {Xtr.shape[1:]} != {IMAGE_SHAPE}"
    assert Xva.shape[1:] == IMAGE_SHAPE, f"Val image shape {Xva.shape[1:]} != {IMAGE_SHAPE}"

    return Xtr, ytr, Xva, yva

# -----------------------------
# tf.data (streaming) pipeline
# -----------------------------
def _make_dataset_from_memmap(X_mm, y, batch_size, shuffle=True, seed=42, prefetch=2):
    """
    Build a streaming tf.data.Dataset from a NumPy memmap + labels.

    Pipeline:
      - Range of indices [0..N)
      - Optional shuffle of indices
      - Batch indices (not images yet)
      - In map(): use tf.numpy_function to fetch batch slices from memmap
      - Normalize to float32 [0,1] on the fly
      - Prefetch to overlap I/O and compute

    Keeps memory usage near (batch_size * H * W * C * dtype).
    """
    N = y.shape[0]

    # Dataset of indices 0..N-1
    idx_ds = tf.data.Dataset.range(N)
    if shuffle:
        idx_ds = idx_ds.shuffle(N, seed=seed, reshuffle_each_iteration=True)
    idx_ds = idx_ds.batch(batch_size, drop_remainder=False)

    def load_batch(indices):
        """Load batch of images from memmap and normalize on-the-fly."""
        def _np_load(idx_np):
            # idx_np is a 1-D NumPy array of indices; slicing memmap returns a view
            imgs = X_mm[idx_np]  # dtype could be uint8 / float16 / float32
            return imgs

        imgs = tf.numpy_function(_np_load, [indices], Tout=tf.uint8 if X_mm.dtype==np.uint8 else tf.float32)
        imgs.set_shape([None, *IMAGE_SHAPE])  # static shape for graph

        # Normalize to float32 [0,1]
        if X_mm.dtype == np.uint8:
            imgs = tf.cast(imgs, tf.float32) / 255.0
        else:
            # float16/float32 on disk: if they are 0..255 floats, scale; otherwise assume 0..1
            imgs = tf.cast(imgs, tf.float32)
            # Heuristic: scale if max is likely >1 — we can’t scan whole array here,
            # so prefer consistent preprocessing out-of-band if you know your range.
            # To keep it simple, assume they were uint8 originally → divide by 255.
            imgs = imgs / 255.0

        lbls = tf.gather(y, indices)
        lbls = tf.cast(lbls, tf.int32)
        return imgs, lbls

    ds = idx_ds.map(load_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(prefetch)
    return ds

def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights for binary labels {0,1}.
    Guards against zero counts.
    """
    binc = np.bincount(labels.reshape(-1), minlength=2)
    neg, pos = int(binc[0]), int(binc[1])
    total = max(1, neg + pos)
    return {0: (total / 2) / max(1, neg), 1: (total / 2) / max(1, pos)}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Train AutoKeras from merged splits (streaming + epoch metrics)")
    ap.add_argument("--merged-dir", required=True, help="Folder with train/val *_images.npy & *_labels.npy")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--model-out", default="model1.keras")
    args = ap.parse_args()

    # 1) Open arrays without loading everything into memory (tolerant to raw files)
    print("Loading image arrays (tolerant opener, no full RAM allocation)...")
    try:
        Xtr_mm, ytr, Xva_mm, yva = _open_memmaps(args.merged_dir)
    except RuntimeError as e:
        # Explicit message for object-array .npy case (pickled)
        raise SystemExit(
            f"{e}\n\nFix: convert images to numeric arrays once, e.g.\n"
            f"  python repair_to_uint8_auto.py --in-dir {args.merged_dir} --labels-dir {args.merged_dir} --out-dir {args.merged_dir}/fixed\n"
            f"and then use --merged-dir {args.merged_dir}/fixed"
        )
    print(f"Train: {Xtr_mm.shape}  Val: {Xva_mm.shape}")
    print(f"Class dist (train): {np.bincount(ytr, minlength=2)} | (val): {np.bincount(yva, minlength=2)}")

    # 2) Class weights for imbalance
    class_weight = compute_class_weights(ytr)
    print(f"Class Weights: {class_weight}")

    # 3) Build streaming tf.data pipelines
    print("Building tf.data pipelines...")
    train_ds = _make_dataset_from_memmap(Xtr_mm, ytr, args.batch_size, shuffle=True)
    val_ds   = _make_dataset_from_memmap(Xva_mm, yva, args.batch_size, shuffle=False)

    steps_per_epoch = int(np.ceil(len(ytr) / args.batch_size))
    val_steps       = int(np.ceil(len(yva) / args.batch_size))

    # 4) Define AutoKeras model
    auto_model = ak.ImageClassifier(
        max_trials=args.max_trials,
        overwrite=True,
        loss=f1_loss,  # keep your custom F1-as-loss
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="AUC"),
            F1Score(name="F1"),   # <- this prints F1 / val_F1 each epoch
        ],
    )

    # 5) Train with epoch-end metrics
    print("Starting Auto-Keras training (streaming batches)...")
    auto_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps        
    )

    # 6) Export and save best model
    best_model = auto_model.export_model()
    best_model.summary()
    best_model.save(args.model_out)
    print(f"Saved best model to {args.model_out}")

if __name__ == "__main__":
    main()
