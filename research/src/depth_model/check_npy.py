import numpy as np
import cv2
from pathlib import Path

ROOT = Path("/data/depth/pseudo_depth_dataset/")

sample = "001000"  # change this

depth_path = ROOT / "depth" / f"{sample}.npy"
image_path = ROOT / "images" / f"{sample}.png"

depth = np.load(depth_path)

print("shape:", depth.shape)
print("dtype:", depth.dtype)
print("min:", depth.min())
print("max:", depth.max())
print("mean:", depth.mean())

# normalize for viewing
dmin = depth.min()
dmax = depth.max()

depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)
depth_u8 = (depth_norm * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

cv2.imwrite("debug_depth.png", depth_color)

print("Saved debug_depth.png")