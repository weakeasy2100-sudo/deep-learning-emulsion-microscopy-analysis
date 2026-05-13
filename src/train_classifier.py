"""
Step 3 — Full deep learning classification pipeline.

Run from project root:
    python src/train_classifier.py

What it does
------------
1. Extract 64×64 droplet patches (skips if already done)
2. Split into train / val / test sets
3. Train SimpleCNN for 20 epochs
4. Save model weights and result figures

Outputs
-------
data/patches/<class>/   64×64 PNG patches (created if missing)
results/simple_cnn.pth
results/training_curve.png
results/confusion_matrix.png
results/prediction_examples.png
"""

import sys
from pathlib import Path

# make 'src' importable when this file is run as a script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset  import (build_patch_dataset, make_splits,
                           CLASS_ORDER, CLASS_TO_IDX, IDX_TO_CLASS, PATCHES_DIR)
from src.model    import SimpleCNN, train_model, evaluate_model
from src.classical import COLORS

# ── constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
N_EPOCHS     = 20
BATCH_SIZE   = 32
LR           = 1e-3


# ── figure helpers ────────────────────────────────────────────────────────────

def _save_training_curve(history, test_acc):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#4C9BE8")
    axes[0].plot(epochs, history["val_loss"],   label="val",   color="#E8A44C")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="#5EBD70")
    axes[1].axhline(test_acc, color="gray", linestyle="--",
                    label=f"test acc = {test_acc:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle("Step 3 — SimpleCNN Training Curve", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "training_curve.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def _save_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Step 3 — Confusion Matrix (test set)")

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

    # also print the text report to the terminal
    print("\n  Classification report:")
    report = classification_report(all_labels, all_preds, target_names=CLASS_ORDER)
    for line in report.splitlines():
        print(f"    {line}")


def _save_prediction_examples(model, test_ds, device):
    infer_tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    def _predict(path):
        img    = Image.open(path).convert("L")
        tensor = infer_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(tensor).argmax(1).item()

    model.eval()
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))

    for row_i, cls in enumerate(CLASS_ORDER):
        # pick up to 5 test patches from this class
        cls_pairs = [(p, l) for p, l in test_ds.pairs
                     if l == CLASS_TO_IDX[cls]][:5]
        for col_i, (path, true_lbl) in enumerate(cls_pairs):
            pred_lbl = _predict(path)
            correct  = pred_lbl == true_lbl

            ax = axes[row_i, col_i]
            ax.imshow(Image.open(path).convert("L"), cmap="gray")
            ax.set_title(
                f"true: {IDX_TO_CLASS[true_lbl]}\npred: {IDX_TO_CLASS[pred_lbl]}",
                fontsize=8,
                color="green" if correct else "red",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        "Step 3 — Prediction Examples  (green = correct  ·  red = wrong)",
        fontsize=11,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "prediction_examples.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: patches ──────────────────────────────────────────────────────
    print("[Step 3] Creating patches...")
    existing = list(PATCHES_DIR.glob("*/*.png"))
    if existing:
        # count per class without re-extracting
        counts = {cls: len(list((PATCHES_DIR / cls).glob("*.png")))
                  for cls in CLASS_ORDER}
        print("  (patches already exist — skipping extraction)")
    else:
        counts = build_patch_dataset()

    for cls in CLASS_ORDER:
        print(f"  {cls:8s}: {counts[cls]} patches")
    print(f"  Total   : {sum(counts.values())}  ->  {PATCHES_DIR}/")

    # ── Phase 2: splits + loaders ─────────────────────────────────────────────
    print("\n[Step 3] Building data splits...")
    train_ds, val_ds, test_ds = make_splits(val_frac=0.15, test_frac=0.15, seed=42)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ── Phase 3: train ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 3] Training SimpleCNN on {device}...")
    model   = SimpleCNN(n_classes=len(CLASS_ORDER))
    history = train_model(model, train_loader, val_loader,
                          n_epochs=N_EPOCHS, lr=LR, device=device)

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"  Test accuracy: {test_acc:.3f}  |  test loss: {test_loss:.4f}")

    # ── Phase 4: save model ───────────────────────────────────────────────────
    print("\n[Step 3] Saving model and figures...")
    model_path = RESULTS_DIR / "simple_cnn.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"  Saved -> {model_path}")

    # ── Phase 5: save figures ─────────────────────────────────────────────────
    _save_training_curve(history, test_acc)
    _save_confusion_matrix(model, test_loader, device)
    _save_prediction_examples(model, test_ds, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
