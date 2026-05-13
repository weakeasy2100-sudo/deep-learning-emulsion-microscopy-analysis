"""
High-density small-droplet realistic synthetic image generator — v2.1 extension.

Produces images that are more challenging than both v1 (generate_data.py) and
v2 (generate_realistic_data.py) by simulating real emulsion microscopy conditions:
  - many more droplets per image (10–25)
  - smaller droplet radius ranges across all classes
  - very loose overlap tolerance — touching and partially overlapping droplets
  - higher proportion of edge-cutting droplets
  - weaker droplet-to-background contrast
  - stronger and more uneven background illumination
  - heavier sensor noise
  - more and larger dust-like artefacts
  - wider contrast and brightness variation per image

v1 and v2 outputs are never touched.

Outputs
-------
data/high_density_raw/           150 PNG files  (50 per size class)
data/high_density_metadata.csv   one row per image with labels and stats
results/high_density_synthetic_examples.png   3 × 5 preview grid

Usage
-----
    python src/generate_high_density_realistic_data.py
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
PROJECT_ROOT         = Path(__file__).resolve().parent.parent
HIGH_DENSITY_RAW_DIR = PROJECT_ROOT / "data" / "high_density_raw"
HIGH_DENSITY_CSV     = PROJECT_ROOT / "data" / "high_density_metadata.csv"
RESULTS_DIR          = PROJECT_ROOT / "results"

# ── generation parameters ─────────────────────────────────────────────────────
IMAGE_SIZE     = 256
N_PER_CLASS    = 50
N_DROPLETS_MIN = 10     # v1: 4,  v2: 4
N_DROPLETS_MAX = 25     # v1: 9,  v2: 9

# Smaller radius ranges across all classes vs v1/v2
SIZE_CLASSES = {
    "small":  (3,   8),   # v1/v2: (6,  14)
    "medium": (6,  14),   # v1/v2: (15, 25)
    "large":  (10, 20),   # v1/v2: (26, 40)
}

# ── difficulty parameters ─────────────────────────────────────────────────────
BG_BLOB_COUNT      = 8          # v1: 3,  v2: 6
BG_BLOB_AMPLITUDE  = 55         # v1: 15, v2: 40
NOISE_SIGMA_MIN    = 15         # v1: 4,  v2: 10
NOISE_SIGMA_MAX    = 30         # v1: 9,  v2: 20
BLUR_SIGMA_MIN     = 1.0        # v1: 0.8, v2: 0.5
BLUR_SIGMA_MAX     = 3.5        # v1: 1.6, v2: 3.0
OVERLAP_TOLERANCE  = 0.20       # v1: 0.75, v2: 0.40 — allows touching/overlap
BRIGHTNESS_BOOST_MIN = 10       # v1/v2: 25 — weaker droplet contrast
BRIGHTNESS_BOOST_MAX = 25       # v1/v2: 45
DUST_COUNT_MIN     = 15         # v1: 0,  v2: 5
DUST_COUNT_MAX     = 30         # v1: 0,  v2: 15
DUST_RADIUS_MAX    = 4          # v1: 0,  v2: 3
CONTRAST_MIN       = 0.6        # v1: 1.0, v2: 0.7
CONTRAST_MAX       = 1.4        # v1: 1.0, v2: 1.3
BRIGHTNESS_OFFSET  = 30         # v1: 0,  v2: 20


# ── private helpers ───────────────────────────────────────────────────────────

def _make_background(rng):
    """Strong uneven illumination with more and larger blobs than v1/v2."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 190.0)
    y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    sigma_bg = (IMAGE_SIZE / 3) ** 2
    for _ in range(BG_BLOB_COUNT):
        cx   = rng.integers(0, IMAGE_SIZE)
        cy   = rng.integers(0, IMAGE_SIZE)
        blob = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma_bg))
        canvas += rng.uniform(-BG_BLOB_AMPLITUDE, BG_BLOB_AMPLITUDE) * blob
    return np.clip(canvas, 80, 245)


def _draw_droplet(canvas, cy, cx, radius, rng):
    """
    Bright interior + dark rim with weaker contrast than v1/v2.
    brightness boost is lower (BRIGHTNESS_BOOST_MIN/MAX) to simulate
    weak droplet-to-background separation.
    """
    rim_w = max(1, radius // 7)
    inner = max(1, radius - rim_w)

    rr_in,  cc_in  = draw.disk((cy, cx), inner,  shape=canvas.shape)
    rr_out, cc_out = draw.disk((cy, cx), radius, shape=canvas.shape)

    canvas[rr_in, cc_in] = np.clip(
        canvas[rr_in, cc_in] + rng.uniform(BRIGHTNESS_BOOST_MIN, BRIGHTNESS_BOOST_MAX),
        0, 255
    )

    rim_mask = np.zeros(canvas.shape, dtype=bool)
    rim_mask[rr_out, cc_out] = True
    rim_mask[rr_in,  cc_in]  = False
    canvas[rim_mask] = np.clip(canvas[rim_mask] - 25, 0, 255)


def _place_droplets(canvas, size_class, n, rng):
    """
    Place up to n droplets with high-density settings:
      - very loose overlap tolerance (OVERLAP_TOLERANCE = 0.20)
      - larger edge offset allows more edge-cutting than v2
    """
    r_min, r_max = SIZE_CLASSES[size_class]
    placed   = []
    attempts = 0

    while len(placed) < n and attempts < 1000:
        attempts += 1
        radius = int(rng.integers(r_min, r_max + 1))

        # allow up to 60% of radius outside the frame (more edge-cutting than v2)
        edge_offset = int(radius * 0.6)
        cy = int(rng.integers(-edge_offset, IMAGE_SIZE + edge_offset))
        cx = int(rng.integers(-edge_offset, IMAGE_SIZE + edge_offset))

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
    """More and slightly larger dust-like dark artefacts than v2."""
    n_dust = int(rng.integers(DUST_COUNT_MIN, DUST_COUNT_MAX + 1))
    for _ in range(n_dust):
        cy     = int(rng.integers(0, IMAGE_SIZE))
        cx     = int(rng.integers(0, IMAGE_SIZE))
        radius = int(rng.integers(1, DUST_RADIUS_MAX + 1))
        rr, cc = draw.disk((cy, cx), radius, shape=canvas.shape)
        canvas[rr, cc] = np.clip(canvas[rr, cc] - rng.uniform(30, 65), 0, 255)


def _apply_contrast_brightness(canvas, rng):
    """Wider contrast and brightness variation per image than v2."""
    contrast   = rng.uniform(CONTRAST_MIN, CONTRAST_MAX)
    brightness = rng.uniform(-BRIGHTNESS_OFFSET, BRIGHTNESS_OFFSET)
    mean       = canvas.mean()
    canvas     = (canvas - mean) * contrast + mean + brightness
    return np.clip(canvas, 0, 255)


# ── public API ────────────────────────────────────────────────────────────────

def generate_high_density_image(size_class, seed):
    """
    Generate one high-density realistic synthetic emulsion image.

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

    blur_sigma  = rng.uniform(BLUR_SIGMA_MIN, BLUR_SIGMA_MAX)
    canvas      = filters.gaussian(canvas, sigma=blur_sigma, preserve_range=True)

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
    plt.suptitle("High-Density Realistic Synthetic Emulsion Images  (v2.1)", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "high_density_synthetic_examples.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  Preview  -> {out}")


def main():
    HIGH_DENSITY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows   = []
    images = []
    labels = []

    print("Generating high-density realistic images ...")
    idx = 0
    for size_class in SIZE_CLASSES:
        for i in range(N_PER_CLASS):
            # seed space offset from v1 (idx*137+42) and v2 (idx*137+99)
            seed = idx * 137 + 199
            img, drops = generate_high_density_image(size_class, seed)

            fname = f"{size_class}_{i:03d}.png"
            Image.fromarray(img, mode="L").save(str(HIGH_DENSITY_RAW_DIR / fname))

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

    with open(HIGH_DENSITY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Images   -> {HIGH_DENSITY_RAW_DIR}/  ({len(rows)} files)")
    print(f"  Metadata -> {HIGH_DENSITY_CSV}")
    _save_preview(images, labels)
    print("Done.")


if __name__ == "__main__":
    main()
