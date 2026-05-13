"""
Realistic synthetic emulsion-like microscopy image generator — v2 extension.

Produces images that are deliberately more challenging than the v1 generator
(src/generate_data.py) by adding:
  - stronger and more uneven background illumination
  - higher sensor noise
  - variable defocus blur
  - overlapping droplets
  - edge-cutting droplets (partially outside frame)
  - dust-like dark artefacts
  - random per-image contrast and brightness variation

v1 outputs are never touched.

Outputs
-------
data/realistic_raw/           150 PNG files  (50 per size class)
data/realistic_metadata.csv   one row per image with labels and stats
results/realistic_synthetic_examples.png   3 × 5 preview grid

Usage
-----
    python src/generate_realistic_data.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import draw, filters

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
REALISTIC_RAW_DIR = PROJECT_ROOT / "data" / "realistic_raw"
REALISTIC_CSV     = PROJECT_ROOT / "data" / "realistic_metadata.csv"
RESULTS_DIR       = PROJECT_ROOT / "results"

# ── generation parameters (shared with v1) ────────────────────────────────────
IMAGE_SIZE     = 256
N_PER_CLASS    = 50
N_DROPLETS_MIN = 4
N_DROPLETS_MAX = 9

SIZE_CLASSES = {
    "small":  (6,  14),
    "medium": (15, 25),
    "large":  (26, 40),
}

# ── realistic difficulty parameters (all higher / looser than v1) ─────────────
BG_BLOB_COUNT      = 6          # v1 uses 3
BG_BLOB_AMPLITUDE  = 40         # v1 uses 15  — stronger uneven illumination
NOISE_SIGMA_MIN    = 10         # v1 uses 4
NOISE_SIGMA_MAX    = 20         # v1 uses 9
BLUR_SIGMA_MIN     = 0.5        # v1 uses 0.8
BLUR_SIGMA_MAX     = 3.0        # v1 uses 1.6
OVERLAP_TOLERANCE  = 0.40       # v1 uses 0.75 — allows more overlap
DUST_COUNT_MIN     = 5
DUST_COUNT_MAX     = 15
DUST_RADIUS_MAX    = 3          # pixels


# ── private helpers ───────────────────────────────────────────────────────────

def _make_background(rng):
    """Uneven illumination: more blobs, larger amplitude than v1."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 190.0)
    y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    sigma_bg = (IMAGE_SIZE / 3) ** 2
    for _ in range(BG_BLOB_COUNT):
        cx   = rng.integers(0, IMAGE_SIZE)
        cy   = rng.integers(0, IMAGE_SIZE)
        blob = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma_bg))
        canvas += rng.uniform(-BG_BLOB_AMPLITUDE, BG_BLOB_AMPLITUDE) * blob
    return np.clip(canvas, 100, 240)


def _draw_droplet(canvas, cy, cx, radius, rng):
    """Bright interior + dark rim, identical approach to v1."""
    rim_w = max(1, radius // 7)
    inner = max(1, radius - rim_w)

    rr_in,  cc_in  = draw.disk((cy, cx), inner,  shape=canvas.shape)
    rr_out, cc_out = draw.disk((cy, cx), radius, shape=canvas.shape)

    canvas[rr_in, cc_in] = np.clip(
        canvas[rr_in, cc_in] + rng.uniform(25, 45), 0, 255
    )

    rim_mask = np.zeros(canvas.shape, dtype=bool)
    rim_mask[rr_out, cc_out] = True
    rim_mask[rr_in,  cc_in]  = False
    canvas[rim_mask] = np.clip(canvas[rim_mask] - 35, 0, 255)


def _place_droplets(canvas, size_class, n, rng):
    """
    Place droplets with two realistic differences vs v1:
      1. Overlap tolerance is looser  (OVERLAP_TOLERANCE = 0.40 vs 0.75)
      2. No margin guard — droplets may be partially outside the frame
         (edge-cutting artefact)
    """
    r_min, r_max = SIZE_CLASSES[size_class]
    placed   = []
    attempts = 0

    while len(placed) < n and attempts < 600:
        attempts += 1
        radius = int(rng.integers(r_min, r_max + 1))

        # allow placement close to or beyond the border (edge-cutting)
        cy = int(rng.integers(-radius // 2, IMAGE_SIZE + radius // 2))
        cx = int(rng.integers(-radius // 2, IMAGE_SIZE + radius // 2))

        too_close = any(
            np.hypot(cy - oy, cx - ox) < (radius + r) * OVERLAP_TOLERANCE
            for oy, ox, r in placed
        )
        if too_close:
            continue

        _draw_droplet(canvas, cy, cx, radius, rng)
        placed.append((cy, cx, radius))

    return placed


def _add_dust(canvas, rng):
    """Scatter small dark dust-like artefacts across the image."""
    n_dust = int(rng.integers(DUST_COUNT_MIN, DUST_COUNT_MAX + 1))
    for _ in range(n_dust):
        cy     = int(rng.integers(0, IMAGE_SIZE))
        cx     = int(rng.integers(0, IMAGE_SIZE))
        radius = int(rng.integers(1, DUST_RADIUS_MAX + 1))
        rr, cc = draw.disk((cy, cx), radius, shape=canvas.shape)
        canvas[rr, cc] = np.clip(canvas[rr, cc] - rng.uniform(30, 60), 0, 255)


def _apply_contrast_brightness(canvas, rng):
    """Random per-image contrast scaling and brightness offset."""
    contrast   = rng.uniform(0.7, 1.3)
    brightness = rng.uniform(-20, 20)
    mean       = canvas.mean()
    canvas     = (canvas - mean) * contrast + mean + brightness
    return np.clip(canvas, 0, 255)


# ── public API ────────────────────────────────────────────────────────────────

def generate_realistic_image(size_class, seed):
    """
    Generate one realistic synthetic emulsion image.

    Parameters
    ----------
    size_class : str   one of 'small', 'medium', 'large'
    seed       : int   random seed for full reproducibility

    Returns
    -------
    img     : np.ndarray  uint8, shape (256, 256)
    droplets: list of (cy, cx, radius)
    """
    rng    = np.random.default_rng(seed)
    canvas = _make_background(rng)
    n      = int(rng.integers(N_DROPLETS_MIN, N_DROPLETS_MAX + 1))
    drops  = _place_droplets(canvas, size_class, n, rng)

    _add_dust(canvas, rng)

    # variable defocus blur (wider range than v1)
    blur_sigma = rng.uniform(BLUR_SIGMA_MIN, BLUR_SIGMA_MAX)
    canvas     = filters.gaussian(canvas, sigma=blur_sigma, preserve_range=True)

    # stronger sensor noise
    noise_sigma = rng.uniform(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX)
    canvas     += rng.normal(0, noise_sigma, canvas.shape)

    canvas = _apply_contrast_brightness(canvas, rng)

    return np.clip(canvas, 0, 255).astype(np.uint8), drops


def _save_preview(images, labels):
    """Save a 3 × 5 grid (one row per class, five examples each)."""
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    for row_i, cls in enumerate(SIZE_CLASSES):
        cls_imgs = [im for im, lb in zip(images, labels) if lb == cls][:5]
        for col_i, im in enumerate(cls_imgs):
            axes[row_i, col_i].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[row_i, col_i].set_title(cls, fontsize=9)
            axes[row_i, col_i].axis("off")
    plt.suptitle("Realistic Synthetic Emulsion-like Images  (v2)", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "realistic_synthetic_examples.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  Preview  -> {out}")


def main():
    REALISTIC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows   = []
    images = []
    labels = []

    print("Generating realistic images ...")
    idx = 0
    for size_class in SIZE_CLASSES:
        for i in range(N_PER_CLASS):
            # offset seed space from v1 (v1 uses idx*137+42) to avoid collision
            seed = idx * 137 + 99
            img, drops = generate_realistic_image(size_class, seed)

            fname = f"{size_class}_{i:03d}.png"
            Image.fromarray(img, mode="L").save(str(REALISTIC_RAW_DIR / fname))

            mean_r = float(np.mean([r for _, _, r in drops])) if drops else 0.0
            rows.append({
                "image_filename": fname,
                "droplet_count":  len(drops),
                "mean_radius":    round(mean_r, 2),
                "size_class":     size_class,
                "random_seed":    seed,
            })
            images.append(img)
            labels.append(size_class)
            idx += 1

    with open(REALISTIC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Images   -> {REALISTIC_RAW_DIR}/  ({len(rows)} files)")
    print(f"  Metadata -> {REALISTIC_CSV}")
    _save_preview(images, labels)
    print("Done.")


if __name__ == "__main__":
    main()
