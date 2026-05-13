"""Extract 64×64 patches around detected droplets and build a PyTorch Dataset.
Saves patches to data/patches/<class>/ for the CNN to load.
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
from src.utils     import crop_patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR  = PROJECT_ROOT / "data" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

PATCH_SIZE   = 64
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


# ── patch extraction ──────────────────────────────────────────────────────────

def build_patch_dataset():
    """Extract patches from all images in metadata.csv. Returns dict of patch counts per class."""
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
            patch  = crop_patch(img, cy, cx, PATCH_SIZE)
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
    """PyTorch Dataset — loads 64×64 grayscale patches from (path, label) pairs."""

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
    """Stratified train / val / test split of saved patches. Returns three EmulsionPatchDataset objects."""
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
