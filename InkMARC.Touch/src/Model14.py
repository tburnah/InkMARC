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
train_images = np.load("/data/train_images.npy")  # shape: (N, 448, 448, 3)
train_labels = np.load("/data/train_labels.npy")  # shape: (N,)
val_images = np.load("/data/val_images.npy")
val_labels = np.load("/data/val_labels.npy")

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
        self.base = models.resnet50(pretrained=True)
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

# --------------------------
# Export to ONNX
# --------------------------
dummy_input = torch.randn(1, 3, 448, 448, device=device)
torch.onnx.export(
    model,
    dummy_input,
    "resnet50_pytorch.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
)
print("✅ ONNX model exported as resnet50_pytorch.onnx")
