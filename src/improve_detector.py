"""
V6 — Detector Improvement Experiment.

Compares the original classical detector (src/classical.py) with an improved
detector that adds background subtraction, CLAHE, and local adaptive
thresholding — all implemented with scikit-image, no new dependencies.

IMPORTANT: src/classical.py is NOT modified.  This script is a standalone
experiment only.  The improved detector is defined here and never replaces
the original unless a future decision is made to do so.

Why the improved detector may help
-----------------------------------
The v5 patch quality diagnosis showed that on v2/v2.1 datasets:
  - detection rate drops significantly vs V1
  - many patches have low std dev (low contrast)
  - centre-surround ratio is near 1.0 (crops not centred on droplets)

Root cause: Otsu global threshold fails under strong uneven illumination and
high noise because the global histogram blurs the gap between background and
droplet brightness.

Improvements applied (in order, conservative by design):
  1. Background subtraction (Gaussian sigma=50)
     Removes low-frequency illumination gradients so the threshold step
     sees a roughly uniform background instead of a gradient.
  2. CLAHE contrast enhancement (clip_limit=0.03, conservative)
     Adaptively boosts local contrast for low-visibility droplets.
     The small clip_limit prevents noise amplification into false detections.
  3. Mild Gaussian blur (sigma=1.5, same as original)
     Noise reduction before thresholding.
  4. Local adaptive threshold (block_size=51, replaces global Otsu)
     Computes a threshold per 51×51 neighbourhood.  Robust to any
     illumination variation that survived step 1.
  5. Reduced min_diameter filter (5 px vs original 8 px)
     Catches the smaller droplets in v2.1 (radius 3-8 px) without
     relaxing the circularity filter.
  6. All other filters kept identical to original (circularity ≥ 0.50,
     max_diameter 100 px, same morphological cleanup, same clear_border).

Higher detection count alone does NOT mean a better detector.
This script reports both count AND patch quality metrics (std dev and
centre-surround ratio) so the two can be judged together.

Prerequisites
-------------
    python src/generate_data.py
    python src/generate_realistic_data.py
    python src/generate_high_density_realistic_data.py

Usage
-----
    python src/improve_detector.py

Outputs  (no existing file is overwritten)
------------------------------------------
results/detector_comparison_overlay.png
results/detector_comparison_counts.png
results/improved_patch_quality_gallery.png
"""

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
from skimage import exposure, filters, measure, morphology
from skimage.io import imread
from skimage.segmentation import clear_border

from src.classical import (detect_droplets,
                            BLUR_SIGMA, MIN_AREA_PX,
                            MAX_DIAMETER_PX, MIN_CIRCULARITY)

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR          = PROJECT_ROOT / "results"
V1_RAW_DIR           = PROJECT_ROOT / "data" / "raw"
V1_METADATA_CSV      = PROJECT_ROOT / "data" / "metadata.csv"
REALISTIC_RAW_DIR    = PROJECT_ROOT / "data" / "realistic_raw"
REALISTIC_CSV        = PROJECT_ROOT / "data" / "realistic_metadata.csv"
HIGH_DENSITY_RAW_DIR = PROJECT_ROOT / "data" / "high_density_raw"
HIGH_DENSITY_CSV     = PROJECT_ROOT / "data" / "high_density_metadata.csv"

# ── processing constants ──────────────────────────────────────────────────────
PATCH_SIZE   = 64
CENTER_HALF  = 8        # centre region = 2×CENTER_HALF square = 16×16 px
N_OVERLAY    = 2        # example images per dataset in the overlay figure
N_GALLERY    = 10       # patches per row in the gallery
OVERLAY_SEED = 0
GALLERY_SEED = 1

DS_ORDER = ["V1 clean", "V2 realistic", "V2.1 high-density"]
DS_COLORS = {
    "V1 clean":           "#4C9BE8",
    "V2 realistic":       "#E8A44C",
    "V2.1 high-density":  "#E86B6B",
}
ORIG_CIRCLE_COLOR = "lime"
IMPR_CIRCLE_COLOR = "#FF8C00"   # orange

# ── improved detector parameters (all approved in the v6 plan) ────────────────
BG_SIGMA            = 50     # sigma for background Gaussian estimation
LOCAL_BLOCK_SIZE    = 51     # local threshold neighbourhood (must be odd)
IMPROVED_MIN_DIAM   = 5      # px; original is 8 px
CLAHE_CLIP          = 0.03   # conservative; 0.01=subtle, 0.10=aggressive


# ── improved detector ─────────────────────────────────────────────────────────

def _is_valid_droplet_improved(reg):
    """Same circularity and max-size logic as original; relaxed min diameter."""
    diam = reg.equivalent_diameter_area
    if diam < IMPROVED_MIN_DIAM or diam > MAX_DIAMETER_PX:
        return False
    if reg.perimeter > 0:
        circularity = (4 * np.pi * reg.area) / (reg.perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            return False
    return True


def detect_droplets_improved(img):
    """
    Improved detector — same call signature as src.classical.detect_droplets.

    Pipeline:
      1. Background subtraction (sigma=BG_SIGMA=50) — removes illumination gradient
      2. CLAHE (clip_limit=0.03) — boosts low-contrast droplet visibility
      3. Gaussian blur (sigma=BLUR_SIGMA=1.5) — same noise reduction as original
      4. Local adaptive threshold (block_size=51) — replaces global Otsu
      5-8. Identical to original: remove_small_objects, remove_small_holes,
           clear_border, regionprops + validity filter

    Parameters
    ----------
    img : np.ndarray  uint8 or float, shape (H, W)

    Returns
    -------
    regions : list of skimage RegionProperties
    binary  : np.ndarray bool
    """
    img_f = img.astype(np.float64)

    # 1. Background subtraction
    # A large-sigma Gaussian estimates the low-frequency illumination field.
    # Subtracting it flattens the background so the threshold sees uniform grey
    # instead of a bright-to-dark gradient.
    background = filters.gaussian(img_f, sigma=BG_SIGMA, preserve_range=True)
    corrected  = img_f - background
    corrected  = corrected - corrected.mean() + 128.0   # re-centre around mid-grey
    corrected  = np.clip(corrected, 0, 255)

    # 2. CLAHE contrast enhancement
    # Conservative clip_limit keeps noise amplification low while making
    # faint droplets more distinguishable from the background.
    corrected_u8 = corrected.astype(np.uint8)
    enhanced     = exposure.equalize_adapthist(corrected_u8, clip_limit=CLAHE_CLIP)
    enhanced     = (enhanced * 255.0).astype(np.float64)  # back to [0,255]

    # 3. Mild blur — identical sigma to original detector
    blurred = filters.gaussian(enhanced, sigma=BLUR_SIGMA, preserve_range=True)

    # 4. Local adaptive threshold
    # Computes a separate threshold for each block_size × block_size neighbourhood.
    # Robust to any illumination variation that survived step 1.
    thresh_local = filters.threshold_local(blurred, block_size=LOCAL_BLOCK_SIZE)
    binary       = blurred > thresh_local

    # 5-7. Morphological cleanup — identical to original
    binary = morphology.remove_small_objects(binary, max_size=MIN_AREA_PX)
    binary = morphology.remove_small_holes(binary, max_size=MIN_AREA_PX)
    binary = clear_border(binary)

    labeled     = measure.label(binary)
    all_regions = measure.regionprops(labeled)
    regions     = [r for r in all_regions if _is_valid_droplet_improved(r)]

    return regions, binary


# ── patch crop ────────────────────────────────────────────────────────────────

def _crop_patch(img, cy, cx, size):
    """Square crop centred at (cy, cx), reflect-padded if needed."""
    half = size // 2
    r0, r1 = cy - half, cy + half
    c0, c1 = cx - half, cx + half

    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - img.shape[0])
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - img.shape[1])

    if pad_top or pad_bottom or pad_left or pad_right:
        img = np.pad(img,
                     ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="reflect")
        cy += pad_top
        cx += pad_left
        r0, r1 = cy - half, cy + half
        c0, c1 = cx - half, cx + half

    return img[r0:r1, c0:c1]


# ── patch quality statistics ──────────────────────────────────────────────────

def _compute_patch_stats(patches):
    """
    Same three quality metrics as v5 analyze_patch_quality.py.
    Returns dict of float64 arrays, one value per patch.
    """
    if not patches:
        empty = np.array([], dtype=np.float64)
        return {"mean_brightness": empty,
                "std_dev": empty,
                "center_surround_ratio": empty}

    mid = PATCH_SIZE // 2
    c0, c1 = mid - CENTER_HALF, mid + CENTER_HALF

    mean_b, std_d, cs_ratio = [], [], []
    for p in patches:
        pf = p.astype(np.float64)
        mb = pf.mean()
        sd = pf.std()
        centre = pf[c0:c1, c0:c1].mean()
        csr    = centre / mb if mb > 1e-6 else 1.0
        mean_b.append(mb)
        std_d.append(sd)
        cs_ratio.append(csr)

    return {
        "mean_brightness":       np.array(mean_b),
        "std_dev":               np.array(std_d),
        "center_surround_ratio": np.array(cs_ratio),
    }


# ── data collection ───────────────────────────────────────────────────────────

def _compare_on_dataset(raw_dir, metadata_csv, tag, overlay_rng):
    """
    Read every image once, run both detectors, collect patches and overlay samples.

    Parameters
    ----------
    raw_dir      : Path
    metadata_csv : Path
    tag          : str
    overlay_rng  : np.random.Generator  for reproducible overlay selection

    Returns
    -------
    orig_result, impr_result : dicts with patches / stats / counts
    overlay_samples : list of (img, orig_regions, impr_regions)  length N_OVERLAY
    """
    meta    = pd.read_csv(metadata_csv)
    n_total = len(meta)
    overlay_idx = set(
        overlay_rng.choice(n_total, size=min(N_OVERLAY, n_total), replace=False).tolist()
    )

    orig_patches, impr_patches = [], []
    orig_n_det,   impr_n_det   = 0, 0
    overlay_samples            = []
    processed                  = 0

    for i, (_, row) in enumerate(meta.iterrows()):
        img_path = raw_dir / row["image_filename"]
        if not img_path.exists():
            continue
        img = imread(str(img_path))
        processed += 1

        # original detector (imported from src.classical — unchanged)
        orig_regions, _ = detect_droplets(img)
        orig_n_det += len(orig_regions)
        for reg in orig_regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = _crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                orig_patches.append(patch)

        # improved detector (defined in this file)
        impr_regions, _ = detect_droplets_improved(img)
        impr_n_det += len(impr_regions)
        for reg in impr_regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = _crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                impr_patches.append(patch)

        if i in overlay_idx:
            overlay_samples.append((img.copy(), orig_regions, impr_regions))

    if processed == 0:
        return None, None, []

    def _fmt(n, ni):
        return f"{n} patches ({n/ni:.1f}/img)"

    print(f"  [{tag}]  {processed} images")
    print(f"    original : {orig_n_det} det  →  {_fmt(len(orig_patches), processed)}")
    print(f"    improved : {impr_n_det} det  →  {_fmt(len(impr_patches), processed)}")

    orig_result = dict(patches=orig_patches, n_images=processed,
                       n_valid=len(orig_patches),
                       stats=_compute_patch_stats(orig_patches))
    impr_result = dict(patches=impr_patches, n_images=processed,
                       n_valid=len(impr_patches),
                       stats=_compute_patch_stats(impr_patches))

    return orig_result, impr_result, overlay_samples


# ── figure 1: overlay comparison ──────────────────────────────────────────────

def _save_detector_comparison_overlay(results_by_ds):
    """
    3 rows (datasets) × (N_OVERLAY × 2) columns.
    Within each row, images are paired: original (left) | improved (right).
    Green circles = original detector, orange circles = improved.
    """
    ds_labels = [lb for lb in DS_ORDER if lb in results_by_ds]
    n_rows    = len(ds_labels)
    n_cols    = N_OVERLAY * 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.4, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, ds_label in enumerate(ds_labels):
        _, _, overlay_samples = results_by_ds[ds_label]

        for s_i, (img, orig_regs, impr_regs) in enumerate(overlay_samples):
            for col_offset, (regs, circle_color, det_name) in enumerate([
                (orig_regs, ORIG_CIRCLE_COLOR, "Original"),
                (impr_regs, IMPR_CIRCLE_COLOR, "Improved"),
            ]):
                ax = axes[row_i, s_i * 2 + col_offset]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                for reg in regs:
                    cy, cx = reg.centroid
                    r = reg.equivalent_diameter_area / 2
                    ax.add_patch(
                        plt.Circle((cx, cy), r,
                                   edgecolor=circle_color,
                                   linewidth=1.1, fill=False)
                    )
                ax.set_title(f"{det_name}  ·  n={len(regs)}", fontsize=8)
                ax.axis("off")

        axes[row_i, 0].set_ylabel(ds_label, fontsize=9, labelpad=5)

    # blank unused cells if any dataset had fewer than N_OVERLAY overlay samples
    for row_i, ds_label in enumerate(ds_labels):
        _, _, overlay_samples = results_by_ds[ds_label]
        for s_i in range(len(overlay_samples), N_OVERLAY):
            axes[row_i, s_i * 2].axis("off")
            axes[row_i, s_i * 2 + 1].axis("off")

    plt.suptitle(
        "V6 — Detector Comparison Overlay\n"
        "Green = original detector (Otsu)  ·  "
        "Orange = improved detector (BG-sub + CLAHE + local threshold)",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "detector_comparison_overlay.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 2: counts + quality comparison ────────────────────────────────────

def _save_detector_comparison_counts(results_by_ds):
    """
    2 × 2 grouped bar charts:
      (0,0) Valid patches per image — raw count; higher is not automatically better
      (0,1) Mean std dev (contrast proxy) — higher means patches have more signal
      (1,0) Mean centre-surround ratio — should be > 1.0 for a centred droplet crop
      (1,1) Fraction of patches with CSR > 1.05 — share that are "well-centred"

    Each chart shows original (blue) vs improved (orange) side by side per dataset.
    """
    ds_labels = [lb for lb in DS_ORDER if lb in results_by_ds]
    x         = np.arange(len(ds_labels))
    w         = 0.33
    colors    = {"original": "#4C9BE8", "improved": IMPR_CIRCLE_COLOR}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def _csr_fraction(arr):
        return float(np.mean(arr > 1.05)) if len(arr) > 0 else 0.0

    metrics = [
        # (ax, orig_fn, impr_fn, ylabel, title)
        (axes[0, 0],
         lambda r: r["n_valid"] / r["n_images"],
         lambda r: r["n_valid"] / r["n_images"],
         "Valid patches per image",
         "Detection Count per Image\n(higher ≠ automatically better — check quality too)"),
        (axes[0, 1],
         lambda r: _safe_mean(r["stats"]["std_dev"]),
         lambda r: _safe_mean(r["stats"]["std_dev"]),
         "Mean pixel std dev",
         "Patch Contrast (Std Dev)\nhigher = more signal for the CNN"),
        (axes[1, 0],
         lambda r: _safe_mean(r["stats"]["center_surround_ratio"]),
         lambda r: _safe_mean(r["stats"]["center_surround_ratio"]),
         "Mean centre-surround ratio",
         "Centre-Surround Ratio\n>1.0 = crop centred on a bright droplet"),
        (axes[1, 1],
         lambda r: _csr_fraction(r["stats"]["center_surround_ratio"]),
         lambda r: _csr_fraction(r["stats"]["center_surround_ratio"]),
         "Fraction of patches with CSR > 1.05",
         "Well-Centred Patch Fraction\nhigher = more patches actually contain a droplet"),
    ]

    for ax, orig_fn, impr_fn, ylabel, title in metrics:
        orig_vals = []
        impr_vals = []
        for lb in ds_labels:
            orig_r, impr_r, _ = results_by_ds[lb]
            orig_vals.append(orig_fn(orig_r))
            impr_vals.append(impr_fn(impr_r))

        bars_o = ax.bar(x - w / 2, orig_vals, w,
                        label="Original", color=colors["original"],
                        edgecolor="white")
        bars_i = ax.bar(x + w / 2, impr_vals, w,
                        label="Improved", color=colors["improved"],
                        edgecolor="white")

        all_vals = orig_vals + impr_vals
        top      = max(max(all_vals), 0.01)
        for bar in list(bars_o) + list(bars_i):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + top * 0.02,
                    f"{h:.2f}" if top < 2 else f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontsize=9, rotation=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, top * 1.28)
        ax.legend(fontsize=8)

    plt.suptitle(
        "V6 — Original vs Improved Detector: Count and Quality Metrics\n"
        "Evaluate all four metrics together — count alone is misleading",
        fontsize=11,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "detector_comparison_counts.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 3: improved patch gallery ─────────────────────────────────────────

def _save_improved_patch_gallery(results_by_ds):
    """
    Same format as v5 patch_quality_gallery.png, but using patches from
    the improved detector — for direct visual comparison.
    3 rows × N_GALLERY columns; each patch annotated with its std dev.
    """
    rng       = np.random.default_rng(GALLERY_SEED)
    ds_labels = [lb for lb in DS_ORDER if lb in results_by_ds]
    n_rows    = len(ds_labels)

    fig, axes = plt.subplots(n_rows, N_GALLERY,
                             figsize=(N_GALLERY * 1.6, n_rows * 1.9))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, ds_label in enumerate(ds_labels):
        _, impr_r, _ = results_by_ds[ds_label]
        patches = impr_r["patches"]
        stats   = impr_r["stats"]
        n_avail = len(patches)

        if n_avail == 0:
            for col_i in range(N_GALLERY):
                axes[row_i, col_i].axis("off")
            axes[row_i, 0].text(0.5, 0.5, "no patches\nfound",
                                ha="center", va="center",
                                fontsize=8, color="red",
                                transform=axes[row_i, 0].transAxes)
            axes[row_i, 0].set_ylabel(ds_label, fontsize=8, labelpad=6)
            continue

        sample_idx = rng.choice(n_avail,
                                size=min(N_GALLERY, n_avail),
                                replace=False)

        for col_i in range(N_GALLERY):
            ax = axes[row_i, col_i]
            if col_i < len(sample_idx):
                pi = sample_idx[col_i]
                ax.imshow(patches[pi], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"σ={stats['std_dev'][pi]:.1f}", fontsize=6.5, pad=2)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_i, 0].set_ylabel(ds_label, fontsize=8, labelpad=6)

    plt.suptitle(
        "V6 — Improved Detector: Patch Gallery\n"
        "σ = pixel std dev per patch  ·  "
        f"{N_GALLERY} random patches per dataset  ·  seed={GALLERY_SEED}\n"
        "Compare visually with v5 patch_quality_gallery.png (original detector)",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "improved_patch_quality_gallery.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── terminal comparison table ─────────────────────────────────────────────────

def _print_comparison(results_by_ds):
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0
    def _csr_frac(arr):
        return float(np.mean(arr > 1.05)) if len(arr) > 0 else 0.0

    LINE = "─" * 78
    W    = "─"
    print(f"\n{LINE}")
    print("V6 — DETECTOR COMPARISON RESULTS")
    print(LINE)
    print(f"  {'Dataset':<26}  {'Detector':<9}  "
          f"{'patch/img':>9}  {'mean_σ':>7}  "
          f"{'mean_CSR':>9}  {'CSR>1.05':>9}")
    print(f"  {W*26}  {W*9}  {W*9}  {W*7}  {W*9}  {W*9}")

    for lb in DS_ORDER:
        if lb not in results_by_ds:
            continue
        orig_r, impr_r, _ = results_by_ds[lb]
        for name, r in [("original", orig_r), ("improved", impr_r)]:
            ni   = r["n_images"]
            n    = r["n_valid"]
            rate = n / ni if ni > 0 else 0.0
            sd   = _safe_mean(r["stats"]["std_dev"])
            csr  = _safe_mean(r["stats"]["center_surround_ratio"])
            cf   = _csr_frac(r["stats"]["center_surround_ratio"])
            print(f"  {lb:<26}  {name:<9}  "
                  f"{rate:>9.2f}  {sd:>7.1f}  "
                  f"{csr:>9.3f}  {cf:>9.1%}")
        print()

    print(LINE)
    print("HOW TO READ THESE RESULTS")
    print(LINE)
    print(
        "\n  A useful improvement shows ALL of the following:\n"
        "    ✓  patch/img is equal or higher (more droplets found)\n"
        "    ✓  mean σ is equal or higher (patches have more contrast)\n"
        "    ✓  mean CSR is higher (crops are better centred on droplets)\n"
        "    ✓  CSR>1.05 fraction is higher (more patches actually contain a droplet)\n"
        "\n"
        "  If patch/img goes up but CSR and σ go down, the detector is picking up\n"
        "  more noise / background regions, not more real droplets.\n"
        "  That would be a false improvement.\n"
    )
    print(LINE + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[V6] Detector Improvement Experiment\n")
    print("  Improved detector parameters:")
    print(f"    Background subtraction: Gaussian sigma = {BG_SIGMA}")
    print(f"    CLAHE clip limit:       {CLAHE_CLIP}  (conservative)")
    print(f"    Local threshold block:  {LOCAL_BLOCK_SIZE} × {LOCAL_BLOCK_SIZE} px")
    print(f"    Min diameter filter:    {IMPROVED_MIN_DIAM} px  "
          f"(original: 8 px)\n")

    overlay_rng  = np.random.default_rng(OVERLAY_SEED)
    results_by_ds = {}

    datasets = [
        ("V1 clean",           V1_RAW_DIR,           V1_METADATA_CSV),
        ("V2 realistic",       REALISTIC_RAW_DIR,     REALISTIC_CSV),
        ("V2.1 high-density",  HIGH_DENSITY_RAW_DIR,  HIGH_DENSITY_CSV),
    ]

    for ds_label, raw_dir, csv_path in datasets:
        print(f"[V6] Processing {ds_label} ...")
        if not raw_dir.exists():
            print(f"  {raw_dir} not found — skipping.\n")
            continue
        orig_r, impr_r, overlay = _compare_on_dataset(
            raw_dir, csv_path, ds_label, overlay_rng
        )
        if orig_r is None:
            print(f"  No images processed — skipping.\n")
            continue
        results_by_ds[ds_label] = (orig_r, impr_r, overlay)
        print()

    if not results_by_ds:
        print("No datasets available. Exiting.")
        return

    print("[V6] Saving figures ...")
    _save_detector_comparison_overlay(results_by_ds)
    _save_detector_comparison_counts(results_by_ds)
    _save_improved_patch_gallery(results_by_ds)

    _print_comparison(results_by_ds)

    print("Done.")


if __name__ == "__main__":
    main()
