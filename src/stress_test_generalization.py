"""Test whether the V1-trained SimpleCNN generalises to harder synthetic datasets.
Run: python src/stress_test_generalization.py
"""

# Note: accuracy reflects the full pipeline (detection + cropping + CNN), not CNN alone

import sys
from pathlib import Path

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report

from src.classical import detect_droplets, CLASS_ORDER
from src.dataset   import make_splits, PATCHES_DIR, CLASS_TO_IDX, IDX_TO_CLASS
from src.model     import SimpleCNN, evaluate_model
from src.utils     import crop_patch

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR          = PROJECT_ROOT / "results"
MODEL_PATH           = RESULTS_DIR / "simple_cnn.pth"
REALISTIC_RAW_DIR    = PROJECT_ROOT / "data" / "realistic_raw"
REALISTIC_CSV        = PROJECT_ROOT / "data" / "realistic_metadata.csv"
HIGH_DENSITY_RAW_DIR = PROJECT_ROOT / "data" / "high_density_raw"
HIGH_DENSITY_CSV     = PROJECT_ROOT / "data" / "high_density_metadata.csv"

PATCH_SIZE = 64
BATCH_SIZE = 32

_PIPELINE_NOTE = "Accuracy = full pipeline (detection → crop → CNN), not CNN alone."

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ── in-memory dataset ─────────────────────────────────────────────────────────

class _InMemoryPatchDataset(Dataset):
    """
    Stores (uint8 numpy patch, int label) pairs entirely in memory.
    Applies the same normalisation transform as EmulsionPatchDataset.
    Nothing is written to disk — data/patches/ is never touched.
    """

    def __init__(self, patch_label_pairs):
        self.pairs = patch_label_pairs  # list of (np.ndarray H×W, int)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        arr, label = self.pairs[idx]
        return _transform(Image.fromarray(arr, mode="L")), label


# ── patch extraction from a raw directory ─────────────────────────────────────

def _build_in_memory_patches(raw_dir, metadata_csv, tag):
    """
    Run classical detection on every image in `raw_dir` (as listed in
    `metadata_csv`), crop 64×64 patches around detected droplets, and
    return an _InMemoryPatchDataset — nothing is written to disk.

    The classical detector is the same Otsu-based pipeline used in Step 2
    (src/classical.py), so its limitations under noisy / low-contrast
    conditions are included in the accuracy figures.

    Parameters
    ----------
    raw_dir      : Path
    metadata_csv : Path
    tag          : str   label shown in progress messages

    Returns
    -------
    dataset        : _InMemoryPatchDataset
    n_images       : int  images processed
    n_detections   : int  total droplets found by classical detector
    """
    meta = pd.read_csv(metadata_csv)
    pairs        = []
    n_detections = 0

    for _, row in meta.iterrows():
        img_path = raw_dir / row["image_filename"]
        if not img_path.exists():
            continue
        img = imread(str(img_path))

        # classical detection — same pipeline as Step 2 / src/classical.py
        regions, _ = detect_droplets(img)
        n_detections += len(regions)

        int_label = CLASS_TO_IDX[row["size_class"]]
        for reg in regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            pairs.append((patch, int_label))

    print(f"  [{tag}]  {len(meta)} images  "
          f"→  {n_detections} detections  "
          f"→  {len(pairs)} valid patches")
    return _InMemoryPatchDataset(pairs), len(meta), n_detections


# ── collect predictions ───────────────────────────────────────────────────────

def _collect_predictions(model, loader, device):
    """Return (all_true, all_pred) integer lists from one full DataLoader pass."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.tolist())
    return all_true, all_pred


# ── figures ───────────────────────────────────────────────────────────────────

def _save_accuracy_comparison(acc_dict):
    """Bar chart comparing accuracy across all evaluated datasets."""
    labels = list(acc_dict.keys())
    values = [acc_dict[k] for k in labels]
    colors = ["#4C9BE8", "#E8A44C", "#E86B6B"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values,
                  color=colors[:len(labels)], edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.1%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Accuracy")
    ax.set_title("V3 — Generalization Stress Test: Pipeline Accuracy by Dataset")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.text(
        0.5, -0.05,
        "Accuracy = full pipeline: detection → crop → CNN (not CNN alone)",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    plt.tight_layout()
    out = RESULTS_DIR / "stress_test_accuracy_comparison.png"
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


def _save_prediction_examples(model, example_sets, device):
    """
    example_sets : dict of {label: Dataset}
    Shows one row per dataset (5 patches each) with green/red borders.
    Each row demonstrates the full pipeline output on that dataset.
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

            # undo normalisation for display: [-1,1] → [0,1]
            display_arr = patch_tensor.squeeze().numpy() * 0.5 + 0.5

            ax = axes[row_i, col_i]
            ax.imshow(display_arr, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"true: {IDX_TO_CLASS[true_lbl]}\npred: {IDX_TO_CLASS[pred_lbl]}",
                fontsize=7,
                color="green" if correct else "red",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

        # row label on left-most patch
        axes[row_i, 0].set_ylabel(ds_label, fontsize=8, labelpad=6)

        for col_i in range(n_show, 5):
            axes[row_i, col_i].axis("off")

    plt.suptitle(
        "V3 — Prediction Examples  (green = correct  ·  red = wrong)\n"
        "Each row = one dataset  ·  Full pipeline: classical detection → patch → CNN",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "stress_test_prediction_examples.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load model (read-only — weights file is never overwritten) ────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Run 'python src/train_classifier.py' first."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SimpleCNN(n_classes=len(CLASS_ORDER))
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.to(device)
    model.eval()
    print(f"[V3] Loaded model: {MODEL_PATH}  (device: {device})")

    print(f"\n{'─' * 62}")
    print(_PIPELINE_NOTE)
    print(f"{'─' * 62}\n")

    acc_dict     = {}   # label → accuracy float
    pred_store   = {}   # label → (all_true, all_pred)  for confusion matrices
    example_sets = {}   # label → Dataset  for prediction grid

    # ── V1 test set ───────────────────────────────────────────────────────────
    print("[V3] Evaluating on V1 clean synthetic test set ...")
    v1_available = PATCHES_DIR.exists() and any(PATCHES_DIR.glob("*/*.png"))
    if v1_available:
        _, _, test_ds_v1 = make_splits(val_frac=0.15, test_frac=0.15, seed=42)
        loader_v1 = DataLoader(test_ds_v1, batch_size=BATCH_SIZE)
        _, v1_acc = evaluate_model(model, loader_v1, device)
        acc_dict["V1 test (clean)"]     = v1_acc
        example_sets["V1 test (clean)"] = test_ds_v1
        print(f"  V1 test accuracy:  {v1_acc:.1%}  "
              f"({len(test_ds_v1)} patches from data/patches/)")
    else:
        print("  data/patches/ not found — skipping V1 evaluation.")
        print("  Run 'python src/train_classifier.py' to generate patches.")

    # ── V2 realistic ──────────────────────────────────────────────────────────
    print("\n[V3] Evaluating on V2 realistic synthetic data ...")
    if not REALISTIC_RAW_DIR.exists():
        print("  data/realistic_raw/ not found — skipping V2.")
        print("  Run 'python src/generate_realistic_data.py' first.")
    else:
        ds_v2, _, _ = _build_in_memory_patches(
            REALISTIC_RAW_DIR, REALISTIC_CSV, "V2 realistic"
        )
        if len(ds_v2) == 0:
            print("  WARNING: classical detector found 0 patches on V2 data.")
            print("  This itself indicates detector failure on harder images.")
        else:
            loader_v2         = DataLoader(ds_v2, batch_size=BATCH_SIZE)
            true_v2, pred_v2  = _collect_predictions(model, loader_v2, device)
            v2_acc = sum(t == p for t, p in zip(true_v2, pred_v2)) / len(true_v2)
            acc_dict["V2 realistic"]     = v2_acc
            pred_store["V2 realistic"]   = (true_v2, pred_v2)
            example_sets["V2 realistic"] = ds_v2
            print(f"  V2 accuracy (full pipeline):  {v2_acc:.1%}")

    # ── V2.1 high-density ─────────────────────────────────────────────────────
    print("\n[V3] Evaluating on V2.1 high-density realistic data ...")
    if not HIGH_DENSITY_RAW_DIR.exists():
        print("  data/high_density_raw/ not found — skipping V2.1.")
        print("  Run 'python src/generate_high_density_realistic_data.py' first.")
    else:
        ds_v21, _, _ = _build_in_memory_patches(
            HIGH_DENSITY_RAW_DIR, HIGH_DENSITY_CSV, "V2.1 high-density"
        )
        if len(ds_v21) == 0:
            print("  WARNING: classical detector found 0 patches on V2.1 data.")
            print("  This itself indicates detector failure on harder images.")
        else:
            loader_v21           = DataLoader(ds_v21, batch_size=BATCH_SIZE)
            true_v21, pred_v21   = _collect_predictions(model, loader_v21, device)
            v21_acc = sum(t == p for t, p in zip(true_v21, pred_v21)) / len(true_v21)
            acc_dict["V2.1 high-density"]     = v21_acc
            pred_store["V2.1 high-density"]   = (true_v21, pred_v21)
            example_sets["V2.1 high-density"] = ds_v21
            print(f"  V2.1 accuracy (full pipeline): {v21_acc:.1%}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("SUMMARY — SimpleCNN trained on V1 clean synthetic data")
    print(f"{'─' * 62}")
    if not acc_dict:
        print("  No datasets were available for evaluation.")
    for ds_label, acc in acc_dict.items():
        print(f"  {ds_label:<30s}  accuracy = {acc:.1%}")
    print()
    print(_PIPELINE_NOTE)
    print(f"{'─' * 62}\n")

    # ── classification reports ────────────────────────────────────────────────
    for ds_label, (all_true, all_pred) in pred_store.items():
        print(f"  Classification report — {ds_label}:")
        report = classification_report(all_true, all_pred, target_names=CLASS_ORDER)
        for line in report.splitlines():
            print(f"    {line}")
        print()

    # ── save figures ──────────────────────────────────────────────────────────
    print("[V3] Saving figures ...")

    if acc_dict:
        _save_accuracy_comparison(acc_dict)

    if "V2 realistic" in pred_store:
        true_v2, pred_v2 = pred_store["V2 realistic"]
        _save_confusion_matrix(
            true_v2, pred_v2,
            title="V3 — Confusion Matrix: V2 Realistic (full pipeline)",
            filename="stress_test_confusion_matrix_realistic.png",
        )

    if "V2.1 high-density" in pred_store:
        true_v21, pred_v21 = pred_store["V2.1 high-density"]
        _save_confusion_matrix(
            true_v21, pred_v21,
            title="V3 — Confusion Matrix: V2.1 High-Density (full pipeline)",
            filename="stress_test_confusion_matrix_high_density.png",
        )

    if example_sets:
        _save_prediction_examples(model, example_sets, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
