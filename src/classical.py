"""Classical droplet detection using Otsu thresholding + connected components.
Outputs a detection overlay and size distribution plots.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from skimage.io import imread
from skimage.segmentation import clear_border

# ── paths (same PROJECT_ROOT logic as generate_data.py) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── analysis parameters ───────────────────────────────────────────────────────
BLUR_SIGMA       = 1.5   # pre-threshold smoothing
MIN_AREA_PX      = 30    # discard blobs smaller than this (noise)
N_OVERLAY_IMGS   = 9     # images shown in the detection overlay grid

# post-detection filters — keep only plausible droplets
MIN_DIAMETER_PX  = 8    # smaller than this is noise (image is 256×256 px)
MAX_DIAMETER_PX  = 100  # larger than this is a background artifact
MIN_CIRCULARITY  = 0.50 # lower values let in too much noise

CLASS_ORDER = ["small", "medium", "large"]
COLORS      = {"small": "#4C9BE8", "medium": "#E8A44C", "large": "#5EBD70"}


def detect_droplets(img):
    """Run Otsu detection on one image. Returns (regions, binary_mask)."""
    blurred = filters.gaussian(img, sigma=BLUR_SIGMA, preserve_range=True)

    # Otsu finds the optimal global threshold automatically
    thresh  = filters.threshold_otsu(blurred)

    # droplets are brighter than background → take pixels above threshold
    binary  = blurred > thresh

    # clean up: remove tiny specks, fill small holes inside droplets
    binary  = morphology.remove_small_objects(binary, max_size=MIN_AREA_PX)
    binary  = morphology.remove_small_holes(binary, max_size=MIN_AREA_PX)

    # drop any blob that touches the image border (likely a partial droplet)
    binary  = clear_border(binary)

    labeled = measure.label(binary)
    all_regions = measure.regionprops(labeled)

    # keep only blobs that look like plausible droplets
    regions = [r for r in all_regions if _is_valid_droplet(r)]

    return regions, binary


def _is_valid_droplet(reg):
    """Return True if the region passes diameter and circularity checks."""
    diam = reg.equivalent_diameter_area
    if diam < MIN_DIAMETER_PX or diam > MAX_DIAMETER_PX:
        return False
    # circularity: 1.0 = perfect circle; noisy/merged blobs → much lower
    if reg.perimeter > 0:
        circularity = (4 * np.pi * reg.area) / (reg.perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            return False
    return True


def analyse_dataset(n_samples=None):
    """Detect droplets in all images from metadata.csv. Returns a DataFrame with one row per blob."""
    meta = pd.read_csv(METADATA_CSV)
    if n_samples is not None:
        meta = meta.head(n_samples)

    rows = []
    for _, row in meta.iterrows():
        img     = imread(str(RAW_DIR / row["image_filename"]))
        regions, _ = detect_droplets(img)
        for reg in regions:
            rows.append({
                "image_filename":   row["image_filename"],
                "size_class":       row["size_class"],
                "area_px":          reg.area,
                "equiv_diameter_px": round(reg.equivalent_diameter_area, 2),
                "solidity":         round(reg.solidity, 3),
            })

    return pd.DataFrame(rows)


# ── figure 1: detection overlay ───────────────────────────────────────────────

def save_detection_overlay(n_imgs=N_OVERLAY_IMGS):
    """
    Show detected blobs outlined on the original image.
    Saves results/classical_detection_overlay.png.
    """
    meta = pd.read_csv(METADATA_CSV)
    label_col = next(
        (c for c in ("size_class", "label", "class_label", "droplet_class") if c in meta.columns),
        None,
    )
    # pick 3 images per class in a fixed, readable order
    if label_col:
        classes = [c for c in CLASS_ORDER if c in meta[label_col].unique()]
        sample_rows = pd.concat([
            meta[meta[label_col] == cls].sample(3, random_state=0)
            for cls in classes
        ]).reset_index(drop=True)
    else:
        sample_rows = meta.sample(min(9, len(meta)), random_state=0).reset_index(drop=True)
        label_col = "__no_label__"
        sample_rows[label_col] = "unknown"
        classes = ["unknown"]

    n_cols = 3
    n_rows = len(sample_rows) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    for ax, (_, row) in zip(axes.flat, sample_rows.iterrows()):
        img     = imread(str(RAW_DIR / row["image_filename"]))
        regions, _ = detect_droplets(img)

        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        for reg in regions:
            cy, cx = reg.centroid
            r      = reg.equivalent_diameter_area / 2
            circle = plt.Circle((cx, cy), r,
                                 edgecolor="lime", linewidth=0.9, fill=False)
            ax.add_patch(circle)

        cls_label = row[label_col]
        ax.set_title(f"{cls_label}  ·  {len(regions)} detected",
                     fontsize=9, pad=4)
        ax.axis("off")

    plt.suptitle(
        "Step 2 — Classical Detection  (Otsu + connected components + filtering)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = RESULTS_DIR / "classical_detection_overlay.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Overlay  -> {out}")


# ── figure 2: size distribution ───────────────────────────────────────────────

def save_size_distribution(results_df):
    """
    Plot equivalent diameter distributions per size class.
    Saves results/droplet_size_distribution.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # cap x-axis at 99th percentile so extreme outliers don't dominate
    x_max = np.percentile(results_df["equiv_diameter_px"], 99) * 1.05

    # histogram — iterate in fixed class order for a natural legend
    for cls in CLASS_ORDER:
        grp = results_df[results_df["size_class"] == cls]
        if grp.empty:
            continue
        axes[0].hist(
            grp["equiv_diameter_px"], bins=25, alpha=0.65,
            label=cls, color=COLORS.get(cls, "gray"), edgecolor="white"
        )
    axes[0].set_xlim(0, x_max)
    axes[0].set_xlabel("Equivalent diameter (px)")
    axes[0].set_ylabel("Detected droplet count")
    axes[0].set_title("Detected Droplet Diameter per Class")
    axes[0].legend(title="Size class")

    # violin plot — shows distribution shape without cluttered outlier dots
    data_by_class = [
        results_df.loc[results_df["size_class"] == cls, "equiv_diameter_px"].values
        for cls in CLASS_ORDER
    ]
    parts = axes[1].violinplot(
        data_by_class, positions=range(len(CLASS_ORDER)),
        showmedians=True, showextrema=False
    )
    for body, cls in zip(parts["bodies"], CLASS_ORDER):
        body.set_facecolor(COLORS.get(cls, "gray"))
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    axes[1].set_xticks(range(len(CLASS_ORDER)))
    axes[1].set_xticklabels(CLASS_ORDER)
    axes[1].set_ylim(0, x_max)
    axes[1].set_xlabel("Size class")
    axes[1].set_ylabel("Equivalent diameter (px)")
    axes[1].set_title("Diameter Distribution (violin · median line)")

    plt.suptitle("Step 2 — Droplet Size Distribution", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "droplet_size_distribution.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Size dist -> {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Running classical analysis ...")
    save_detection_overlay()
    print("  Analysing full dataset ...")
    results_df = analyse_dataset()
    print(f"  Total blobs detected: {len(results_df)}")
    save_size_distribution(results_df)
    print("Done.")
    return results_df


if __name__ == "__main__":
    main()
