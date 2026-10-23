#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# ----- Model (same head as training) -----
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained: bool = False, out_dim: int = 8):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = ResNetRegressor(pretrained=False, out_dim=8).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    return model

def main():
    ap = argparse.ArgumentParser("Predict quad bounds once")
    ap.add_argument("--ckpt", required=True, help="Path to best_model.pt / last_model.pt")
    ap.add_argument("--image", required=True, help="Path to an image")
    ap.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    args = ap.parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Preprocess: resize to 448x448, ToTensor, ImageNet norm
    tfm = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,448,448]

    model = load_model(args.ckpt, device)
    with torch.no_grad():
        y = model(x)[0].cpu().numpy()  # 8 numbers in [~0..1]
        y = np.clip(y, 0.0, 1.0).reshape(4, 2)  # (TL,TR,BR,BL) in normalized coords

    # Denormalize to pixels of ORIGINAL image size
    y_px = y.copy()
    y_px[:, 0] *= W
    y_px[:, 1] *= H

    labels = ["TL", "TR", "BR", "BL"]
    for lbl, (px, py) in zip(labels, y_px):
        print(f"{lbl}: ({px:.1f}, {py:.1f})")

if __name__ == "__main__":
    main()
