"""Train a new SimpleCNN on V1 + V2 + V2.1 patches for better robustness.
Saves results/simple_cnn_mixed.pth — does NOT overwrite results/simple_cnn.pth.
Run: python src/train_mixed_classifier.py
"""

import sys
from pathlib import Path
from collections import defaultdict

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report

from src.classical import detect_droplets, CLASS_ORDER
from src.dataset   import (make_splits, EmulsionPatchDataset,
                            PATCHES_DIR, CLASS_TO_IDX, IDX_TO_CLASS)
from src.model     import SimpleCNN, evaluate_model
from src.utils     import crop_patch

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR          = PROJECT_ROOT / "results"
V1_MODEL_PATH        = RESULTS_DIR / "simple_cnn.pth"
MIXED_MODEL_PATH     = RESULTS_DIR / "simple_cnn_mixed.pth"
REALISTIC_RAW_DIR    = PROJECT_ROOT / "data" / "realistic_raw"
REALISTIC_CSV        = PROJECT_ROOT / "data" / "realistic_metadata.csv"
HIGH_DENSITY_RAW_DIR = PROJECT_ROOT / "data" / "high_density_raw"
HIGH_DENSITY_CSV     = PROJECT_ROOT / "data" / "high_density_metadata.csv"

PATCH_SIZE = 64
BATCH_SIZE = 32
N_EPOCHS   = 30
LR         = 1e-3

# ── transforms ────────────────────────────────────────────────────────────────

# Applied only to training patches — adds diversity to match harder domains
_augment_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# Applied to validation and test patches — no stochastic changes
_eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

_PIPELINE_NOTE = "V2/V2.1 accuracy = full pipeline (detection → crop → CNN), not CNN alone."


# ── in-memory dataset ─────────────────────────────────────────────────────────

class _InMemoryPatchDataset(Dataset):
    """
    Holds (uint8 numpy patch, int label) pairs in memory.
    Accepts a transform so the same class is used for both augmented
    training and plain evaluation splits.
    """

    def __init__(self, patch_label_pairs, transform):
        self.pairs     = patch_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        arr, label = self.pairs[idx]
        return self.transform(Image.fromarray(arr, mode="L")), label


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_patches_raw(raw_dir, metadata_csv, tag):
    """
    Run classical detection on every image in raw_dir, crop 64×64 patches,
    and return a plain list of (np.ndarray, int_label) pairs.
    Nothing is written to disk.
    """
    meta  = pd.read_csv(metadata_csv)
    pairs = []
    n_det = 0

    for _, row in meta.iterrows():
        img_path = raw_dir / row["image_filename"]
        if not img_path.exists():
            continue
        img = imread(str(img_path))
        regions, _ = detect_droplets(img)
        n_det += len(regions)
        int_label = CLASS_TO_IDX[row["size_class"]]
        for reg in regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            pairs.append((patch, int_label))

    print(f"  [{tag}]  {len(meta)} images  "
          f"→  {n_det} detections  "
          f"→  {len(pairs)} valid patches")
    return pairs


def _split_pairs(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Stratified train / val / test split of (item, label) pairs.
    Uses the same fractions as the v1 pipeline (70 / 15 / 15).
    """
    rng      = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for item in pairs:
        by_label[item[1]].append(item)

    train, val, test = [], [], []
    for label_pairs in by_label.values():
        shuffled = [label_pairs[i] for i in rng.permutation(len(label_pairs))]
        n       = len(shuffled)
        n_val   = max(1, int(n * val_frac))
        n_test  = max(1, int(n * test_frac))
        n_train = n - n_val - n_test
        train  += shuffled[:n_train]
        val    += shuffled[n_train: n_train + n_val]
        test   += shuffled[n_train + n_val:]

    return train, val, test


def _collect_predictions(model, loader, device):
    """Return (all_true, all_pred) integer lists from one DataLoader pass."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.tolist())
    return all_true, all_pred


def _accuracy(all_true, all_pred):
    if not all_true:
        return 0.0
    return sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)


# ── training loop with best-checkpoint saving ─────────────────────────────────

def _train_with_checkpoint(model, train_loader, val_loader,
                            n_epochs, lr, device, save_path):
    """
    Train model and save the state dict that achieves the highest val accuracy.
    Reloads the best checkpoint before returning.

    Returns
    -------
    history      : dict  with keys train_loss, val_loss, val_acc
    best_val_acc : float
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history      = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(n_epochs):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_path))

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}"
                  + ("  ← best" if val_acc == best_val_acc else ""))

    # restore best weights
    model.load_state_dict(torch.load(str(save_path), map_location=device))
    print(f"  Best val accuracy: {best_val_acc:.3f}  →  {save_path}")
    return history, best_val_acc


# ── figures ───────────────────────────────────────────────────────────────────

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
                    label=f"V1 test acc = {test_acc:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy (mixed val set)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle("V4 — Mixed-Data SimpleCNN Training Curve", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "mixed_training_curve.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def _save_confusion_matrix(all_true, all_pred, title, filename):
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")

    plt.tight_layout()
    out = RESULTS_DIR / filename
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def _save_accuracy_comparison(v1_only_accs, mixed_accs):
    """
    Grouped bar chart: for each dataset, two bars side by side —
    V1-only model (blue) vs mixed model (orange).
    """
    ds_labels = list(v1_only_accs.keys())
    x         = np.arange(len(ds_labels))
    width     = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_v1    = ax.bar(x - width / 2,
                        [v1_only_accs[k] for k in ds_labels],
                        width, label="V1-only model", color="#4C9BE8",
                        edgecolor="white")
    bars_mixed = ax.bar(x + width / 2,
                        [mixed_accs.get(k, 0) for k in ds_labels],
                        width, label="Mixed model (V4)", color="#E8A44C",
                        edgecolor="white")

    for bar in list(bars_v1) + list(bars_mixed):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Accuracy")
    ax.set_title("V4 — V1-only Model vs Mixed-Data Model: Accuracy Comparison")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.text(
        0.5, -0.04,
        _PIPELINE_NOTE,
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    plt.tight_layout()
    out = RESULTS_DIR / "mixed_accuracy_comparison.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def _save_prediction_examples(model, example_sets, device):
    """
    One row per dataset, 5 patches each.
    Green border = correct, red = wrong.
    """
    n_rows = len(example_sets)
    fig, axes = plt.subplots(n_rows, 5, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for row_i, (ds_label, ds) in enumerate(example_sets.items()):
        n_show = min(5, len(ds))
        for col_i in range(n_show):
            patch_tensor, true_lbl = ds[col_i]
            with torch.no_grad():
                pred_lbl = model(
                    patch_tensor.unsqueeze(0).to(device)
                ).argmax(1).item()
            correct = pred_lbl == true_lbl
            display = patch_tensor.squeeze().numpy() * 0.5 + 0.5

            ax = axes[row_i, col_i]
            ax.imshow(display, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"true: {IDX_TO_CLASS[true_lbl]}\npred: {IDX_TO_CLASS[pred_lbl]}",
                fontsize=7, color="green" if correct else "red",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_i, 0].set_ylabel(ds_label, fontsize=8, labelpad=6)
        for col_i in range(n_show, 5):
            axes[row_i, col_i].axis("off")

    plt.suptitle(
        "V4 — Mixed Model Prediction Examples  "
        "(green = correct  ·  red = wrong)\n"
        "Each row = one test set  ·  "
        "Full pipeline: classical detection → patch → CNN",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "mixed_prediction_examples.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V4] Device: {device}")

    # ── Phase A: collect patches from all three sources ───────────────────────

    print("\n[V4] Phase A — Collecting patches ...")

    # V1: reuse existing patches and make_splits; re-wrap with custom transforms
    if not (PATCHES_DIR.exists() and any(PATCHES_DIR.glob("*/*.png"))):
        raise FileNotFoundError(
            f"V1 patches not found at {PATCHES_DIR}.\n"
            "Run 'python src/train_classifier.py' first."
        )
    raw_train_v1, raw_val_v1, raw_test_v1 = make_splits(
        val_frac=0.15, test_frac=0.15, seed=42
    )
    # Re-wrap with chosen transforms (augment train, plain eval for val/test)
    train_ds_v1 = EmulsionPatchDataset(raw_train_v1.pairs, transform=_augment_transform)
    val_ds_v1   = EmulsionPatchDataset(raw_val_v1.pairs,   transform=_eval_transform)
    test_ds_v1  = EmulsionPatchDataset(raw_test_v1.pairs,  transform=_eval_transform)
    print(f"  [V1 clean]  train={len(train_ds_v1)}  "
          f"val={len(val_ds_v1)}  test={len(test_ds_v1)}")

    # V2: in-memory classical detection + stratified split
    train_ds_v2 = val_ds_v2 = test_ds_v2 = None
    if not REALISTIC_RAW_DIR.exists():
        print("  [V2] data/realistic_raw/ not found — skipping.")
        print("  Run 'python src/generate_realistic_data.py' first.")
    else:
        pairs_v2 = _extract_patches_raw(
            REALISTIC_RAW_DIR, REALISTIC_CSV, "V2 realistic"
        )
        tr_v2, va_v2, te_v2 = _split_pairs(pairs_v2, seed=42)
        train_ds_v2 = _InMemoryPatchDataset(tr_v2, _augment_transform)
        val_ds_v2   = _InMemoryPatchDataset(va_v2, _eval_transform)
        test_ds_v2  = _InMemoryPatchDataset(te_v2, _eval_transform)
        print(f"  [V2]        train={len(train_ds_v2)}  "
              f"val={len(val_ds_v2)}  test={len(test_ds_v2)}")

    # V2.1: same
    train_ds_v21 = val_ds_v21 = test_ds_v21 = None
    if not HIGH_DENSITY_RAW_DIR.exists():
        print("  [V2.1] data/high_density_raw/ not found — skipping.")
        print("  Run 'python src/generate_high_density_realistic_data.py' first.")
    else:
        pairs_v21 = _extract_patches_raw(
            HIGH_DENSITY_RAW_DIR, HIGH_DENSITY_CSV, "V2.1 high-density"
        )
        tr_v21, va_v21, te_v21 = _split_pairs(pairs_v21, seed=42)
        train_ds_v21 = _InMemoryPatchDataset(tr_v21, _augment_transform)
        val_ds_v21   = _InMemoryPatchDataset(va_v21, _eval_transform)
        test_ds_v21  = _InMemoryPatchDataset(te_v21, _eval_transform)
        print(f"  [V2.1]      train={len(train_ds_v21)}  "
              f"val={len(val_ds_v21)}  test={len(test_ds_v21)}")

    # ── Phase B: build combined DataLoaders ───────────────────────────────────

    print("\n[V4] Phase B — Building combined DataLoaders ...")

    train_sources = [ds for ds in [train_ds_v1, train_ds_v2, train_ds_v21]
                     if ds is not None]
    val_sources   = [ds for ds in [val_ds_v1,   val_ds_v2,   val_ds_v21]
                     if ds is not None]

    combined_train = ConcatDataset(train_sources)
    combined_val   = ConcatDataset(val_sources)

    train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(combined_val,   batch_size=BATCH_SIZE)

    print(f"  Combined train: {len(combined_train)} patches")
    print(f"  Combined val  : {len(combined_val)} patches")

    # ── Phase C: train mixed model ────────────────────────────────────────────

    print(f"\n[V4] Phase C — Training mixed SimpleCNN "
          f"({N_EPOCHS} epochs, lr={LR}, device={device}) ...")
    mixed_model = SimpleCNN(n_classes=len(CLASS_ORDER))
    history, best_val_acc = _train_with_checkpoint(
        mixed_model, train_loader, val_loader,
        n_epochs=N_EPOCHS, lr=LR, device=device,
        save_path=MIXED_MODEL_PATH,
    )

    # ── Phase D: evaluate mixed model on each test set ────────────────────────

    print("\n[V4] Phase D — Evaluating mixed model on test sets ...")

    mixed_accs  = {}
    pred_store  = {}
    example_sets = {}

    # V1 test
    loader_te_v1 = DataLoader(test_ds_v1, batch_size=BATCH_SIZE)
    true_v1, pred_v1 = _collect_predictions(mixed_model, loader_te_v1, device)
    acc_v1 = _accuracy(true_v1, pred_v1)
    mixed_accs["V1 test (clean)"]     = acc_v1
    pred_store["V1 test (clean)"]     = (true_v1, pred_v1)
    example_sets["V1 test (clean)"]   = test_ds_v1
    print(f"  V1 test accuracy  : {acc_v1:.1%}")

    if test_ds_v2 is not None:
        loader_te_v2 = DataLoader(test_ds_v2, batch_size=BATCH_SIZE)
        true_v2, pred_v2 = _collect_predictions(mixed_model, loader_te_v2, device)
        acc_v2 = _accuracy(true_v2, pred_v2)
        mixed_accs["V2 realistic"]      = acc_v2
        pred_store["V2 realistic"]      = (true_v2, pred_v2)
        example_sets["V2 realistic"]    = test_ds_v2
        print(f"  V2 accuracy       : {acc_v2:.1%}  (full pipeline)")

    if test_ds_v21 is not None:
        loader_te_v21 = DataLoader(test_ds_v21, batch_size=BATCH_SIZE)
        true_v21, pred_v21 = _collect_predictions(mixed_model, loader_te_v21, device)
        acc_v21 = _accuracy(true_v21, pred_v21)
        mixed_accs["V2.1 high-density"]   = acc_v21
        pred_store["V2.1 high-density"]   = (true_v21, pred_v21)
        example_sets["V2.1 high-density"] = test_ds_v21
        print(f"  V2.1 accuracy     : {acc_v21:.1%}  (full pipeline)")

    # ── Phase E: evaluate OLD v1-only model on the same test sets ─────────────

    print("\n[V4] Phase E — Evaluating original V1-only model for comparison ...")
    v1_only_accs = {}

    if not V1_MODEL_PATH.exists():
        print(f"  WARNING: {V1_MODEL_PATH} not found — comparison chart skipped.")
    else:
        v1_model = SimpleCNN(n_classes=len(CLASS_ORDER))
        v1_model.load_state_dict(
            torch.load(str(V1_MODEL_PATH), map_location=device)
        )
        v1_model.to(device)

        loader_te_v1_old = DataLoader(test_ds_v1, batch_size=BATCH_SIZE)
        _, v1old_acc_v1 = evaluate_model(v1_model, loader_te_v1_old, device)
        v1_only_accs["V1 test (clean)"] = v1old_acc_v1
        print(f"  V1-only on V1 test: {v1old_acc_v1:.1%}")

        if test_ds_v2 is not None:
            loader_te_v2_old = DataLoader(test_ds_v2, batch_size=BATCH_SIZE)
            true_v2_old, pred_v2_old = _collect_predictions(
                v1_model, loader_te_v2_old, device
            )
            v1_only_accs["V2 realistic"] = _accuracy(true_v2_old, pred_v2_old)
            print(f"  V1-only on V2     : {v1_only_accs['V2 realistic']:.1%}")

        if test_ds_v21 is not None:
            loader_te_v21_old = DataLoader(test_ds_v21, batch_size=BATCH_SIZE)
            true_v21_old, pred_v21_old = _collect_predictions(
                v1_model, loader_te_v21_old, device
            )
            v1_only_accs["V2.1 high-density"] = _accuracy(
                true_v21_old, pred_v21_old
            )
            print(f"  V1-only on V2.1   : {v1_only_accs['V2.1 high-density']:.1%}")

    # ── summary ───────────────────────────────────────────────────────────────

    print(f"\n{'─' * 62}")
    print("SUMMARY")
    print(f"{'─' * 62}")
    print(f"  {'Dataset':<30s}  {'V1-only':>9s}  {'Mixed (V4)':>10s}  {'Δ':>7s}")
    print(f"  {'─'*30}  {'─'*9}  {'─'*10}  {'─'*7}")
    for ds_label in mixed_accs:
        v1_acc  = v1_only_accs.get(ds_label, float("nan"))
        mx_acc  = mixed_accs[ds_label]
        delta   = mx_acc - v1_acc if not np.isnan(v1_acc) else float("nan")
        delta_s = f"{delta:+.1%}" if not np.isnan(delta) else "  n/a"
        v1_s    = f"{v1_acc:.1%}" if not np.isnan(v1_acc) else "  n/a"
        print(f"  {ds_label:<30s}  {v1_s:>9s}  {mx_acc:>10.1%}  {delta_s:>7s}")
    print()
    print(_PIPELINE_NOTE)
    print(f"{'─' * 62}\n")

    # classification reports
    for ds_label, (all_true, all_pred) in pred_store.items():
        print(f"  Classification report — {ds_label} (mixed model):")
        for line in classification_report(
            all_true, all_pred, target_names=CLASS_ORDER
        ).splitlines():
            print(f"    {line}")
        print()

    # ── Phase F: save figures ─────────────────────────────────────────────────

    print("[V4] Phase F — Saving figures ...")

    _, test_acc_v1 = evaluate_model(
        mixed_model, DataLoader(test_ds_v1, batch_size=BATCH_SIZE), device
    )
    _save_training_curve(history, test_acc_v1)

    if "V1 test (clean)" in pred_store:
        _save_confusion_matrix(
            *pred_store["V1 test (clean)"],
            title="V4 — Mixed Model Confusion Matrix: V1 Test (clean)",
            filename="mixed_confusion_matrix_v1.png",
        )
    if "V2 realistic" in pred_store:
        _save_confusion_matrix(
            *pred_store["V2 realistic"],
            title="V4 — Mixed Model Confusion Matrix: V2 Realistic (full pipeline)",
            filename="mixed_confusion_matrix_realistic.png",
        )
    if "V2.1 high-density" in pred_store:
        _save_confusion_matrix(
            *pred_store["V2.1 high-density"],
            title="V4 — Mixed Model Confusion Matrix: V2.1 High-Density (full pipeline)",
            filename="mixed_confusion_matrix_high_density.png",
        )

    if v1_only_accs and mixed_accs:
        _save_accuracy_comparison(v1_only_accs, mixed_accs)

    if example_sets:
        _save_prediction_examples(mixed_model, example_sets, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
