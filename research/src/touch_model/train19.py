import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import os
import glob
from datetime import datetime

# --------------------------
# Dataset class (replace with preprocessed .npy or images)
# --------------------------
class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # shape: (N, H, W, C)
        self.labels = labels  # shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

# --------------------------
# Load preprocessed data
# --------------------------
def load_all_data(image_pattern, label_pattern):
    image_files = sorted(glob.glob(image_pattern))
    label_files = sorted(glob.glob(label_pattern))

    assert len(image_files) == len(label_files), "Mismatch between image and label files"

    all_images = []
    all_labels = []

    for img_file, lbl_file in zip(image_files, label_files):
        print(f"Loading {img_file} and {lbl_file}")
        all_images.append(np.load(img_file))
        all_labels.append(np.load(lbl_file))

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return images, labels

train_images, train_labels = load_all_data("/data/train_images*.npy", "/data/train_labels*.npy")
val_images, val_labels = load_all_data("/data/val_images*.npy", "/data/val_labels*.npy")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = MyDataset(train_images, train_labels, transform=transform)
val_dataset = MyDataset(val_images, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# --------------------------
# Model Definition
# --------------------------
class BinaryResNet(nn.Module):
    def __init__(self):
        super(BinaryResNet, self).__init__()
        self.base = models.resnet18(pretrained=True)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryResNet().to(device)

# --------------------------
# Training Setup
# --------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------
# Training Loop
# --------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # Optionally: Evaluate every epoch here

    print("Training complete.")

train_model(model, train_loader, val_loader, epochs=10)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
onnx_filename = f"resnet18_pytorch_{timestamp}.onnx"

# --------------------------
# Export to ONNX
# --------------------------
dummy_input = torch.randn(1, 3, 448, 448, device=device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
)
print("ONNX model exported as resnet18_pytorch_{timestamp}.onnx")
