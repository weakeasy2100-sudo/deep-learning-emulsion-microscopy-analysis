"""
SimpleCNN classifier for emulsion droplet size classification — Step 3.

Architecture (64×64 grayscale input):
    Conv(1→16, 3×3, pad=1) → ReLU → MaxPool(2)   # → 16 × 32 × 32
    Conv(16→32, 3×3, pad=1) → ReLU → MaxPool(2)  # → 32 × 16 × 16
    Flatten(8192) → Linear(128) → ReLU → Dropout(0.3) → Linear(3)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    """Two-layer convolutional classifier for 64×64 grayscale patches."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 64 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 32 → 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),         # 8192 → 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, device="cpu"):
    """
    Train the model and return per-epoch metrics.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'val_acc'
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        # training pass
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

    return history


def evaluate_model(model, loader, device):
    """Return (avg_loss, accuracy) on the given DataLoader."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(imgs)
    return total_loss / total, correct / total
