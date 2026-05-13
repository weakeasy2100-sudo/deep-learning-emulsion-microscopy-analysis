"""
Patch extraction and PyTorch Dataset — Step 3.

Crops 64×64 grayscale patches around each detected droplet and
saves them under data/patches/<size_class>/ for use by the CNN.

Public API (imported by train_classifier.py and extension scripts)
------------------------------------------------------------------
build_patch_dataset()   extract and save all patches from data/raw/
make_splits()           stratified train / val / test split (seed=42)
EmulsionPatchDataset    PyTorch Dataset: loads (patch, label) pairs
CLASS_TO_IDX            {"small": 0, "medium": 1, "large": 2}
IDX_TO_CLASS            {0: "small", 1: "medium", 2: "large"}
PATCHES_DIR             Path to data/patches/

Usage
-----
    python src/dataset.py
"""

import shutil
import sys
from pathlib import Path

# ensure project root is on sys.path whether run as script or imported
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.classical import detect_droplets, CLASS_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR  = PROJECT_ROOT / "data" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

PATCH_SIZE   = 64
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


# ── patch extraction ──────────────────────────────────────────────────────────

def _crop_patch(img, cy, cx, size):
    """Square crop of `size` centred at (cy, cx), with reflect-padding if needed."""
    half = size // 2
    r0, r1 = cy - half, cy + half
    c0, c1 = cx - half, cx + half

    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - img.shape[0])
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - img.shape[1])

    if pad_top or pad_bottom or pad_left or pad_right:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        cy += pad_top
        cx += pad_left
        r0, r1 = cy - half, cy + half
        c0, c1 = cx - half, cx + half

    return img[r0:r1, c0:c1]


def build_patch_dataset():
    """
    Extract droplet patches from all images listed in metadata.csv.
    Clears any previous patches first so re-runs are idempotent.

    Returns
    -------
    counts : dict  {size_class: n_patches_saved}
    """
    # clear previous patches to avoid duplicates on re-run
    if PATCHES_DIR.exists():
        shutil.rmtree(PATCHES_DIR)
    for cls in CLASS_ORDER:
        (PATCHES_DIR / cls).mkdir(parents=True, exist_ok=True)

    meta   = pd.read_csv(METADATA_CSV)
    counts = {cls: 0 for cls in CLASS_ORDER}

    for _, row in meta.iterrows():
        img        = imread(str(RAW_DIR / row["image_filename"]))
        regions, _ = detect_droplets(img)
        cls        = row["size_class"]
        stem       = Path(row["image_filename"]).stem

        for drop_i, reg in enumerate(regions):
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = _crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            out = PATCHES_DIR / cls / f"{stem}_drop{drop_i:02d}.png"
            Image.fromarray(patch, mode="L").save(str(out))
            counts[cls] += 1

    return counts


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

_default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),   # maps [0,1] → [-1,1]
])


class EmulsionPatchDataset(Dataset):
    """
    Loads patches from a list of (Path, int_label) pairs.
    Each patch is a 64×64 grayscale image converted to a normalised tensor.
    """

    def __init__(self, file_label_pairs, transform=None):
        self.pairs     = file_label_pairs
        self.transform = transform or _default_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


def make_splits(val_frac=0.15, test_frac=0.15, seed=42):
    """
    Collect all saved patches and split into stratified train / val / test sets.

    Returns
    -------
    train_ds, val_ds, test_ds : EmulsionPatchDataset
    """
    rng    = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for cls in CLASS_ORDER:
        files  = sorted((PATCHES_DIR / cls).glob("*.png"))
        pairs  = [(f, CLASS_TO_IDX[cls]) for f in files]
        pairs  = [pairs[i] for i in rng.permutation(len(pairs))]   # shuffle

        n       = len(pairs)
        n_val   = max(1, int(n * val_frac))
        n_test  = max(1, int(n * test_frac))
        n_train = n - n_val - n_test

        splits["train"] += pairs[:n_train]
        splits["val"]   += pairs[n_train: n_train + n_val]
        splits["test"]  += pairs[n_train + n_val:]

    return (
        EmulsionPatchDataset(splits["train"]),
        EmulsionPatchDataset(splits["val"]),
        EmulsionPatchDataset(splits["test"]),
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Extracting patches ...")
    counts = build_patch_dataset()
    for cls, n in counts.items():
        print(f"  {cls:8s}: {n} patches")
    total = sum(counts.values())
    print(f"  Total   : {total} patches  →  {PATCHES_DIR}/")
    print("Done.")
