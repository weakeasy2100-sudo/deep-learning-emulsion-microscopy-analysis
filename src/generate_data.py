"""Generate synthetic emulsion-like microscopy images.
150 images total (50 per size class: small / medium / large).
Run: python src/generate_data.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
from PIL import Image
from skimage import draw, filters

# ── paths (always relative to project root, regardless of working directory) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── generation parameters ─────────────────────────────────────────────────────
IMAGE_SIZE       = 256
N_PER_CLASS      = 50          # 50 x 3 classes = 150 images total
N_DROPLETS_MIN   = 4
N_DROPLETS_MAX   = 9

# radius range (pixels) for each size class
SIZE_CLASSES = {
    "small":  (6,  14),
    "medium": (15, 25),
    "large":  (26, 40),
}


# ── private helpers ───────────────────────────────────────────────────────────

def _make_background(rng):
    """Gray canvas with mild low-frequency illumination variation."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 190.0)
    y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    sigma_bg = (IMAGE_SIZE / 3) ** 2
    for _ in range(3):
        cx   = rng.integers(0, IMAGE_SIZE)
        cy   = rng.integers(0, IMAGE_SIZE)
        blob = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma_bg))
        canvas += rng.uniform(-15, 15) * blob
    return np.clip(canvas, 150, 230)


def _draw_droplet(canvas, cy, cx, radius, rng):
    """In-place: bright interior + dark rim, mimicking bright-field emulsion."""
    rim_w  = max(1, radius // 7)
    inner  = max(1, radius - rim_w)

    rr_in,  cc_in  = draw.disk((cy, cx), inner,  shape=canvas.shape)
    rr_out, cc_out = draw.disk((cy, cx), radius, shape=canvas.shape)

    # brighten interior
    canvas[rr_in, cc_in] = np.clip(
        canvas[rr_in, cc_in] + rng.uniform(25, 45), 0, 255
    )

    # darken rim
    rim_mask = np.zeros(canvas.shape, dtype=bool)
    rim_mask[rr_out, cc_out] = True
    rim_mask[rr_in,  cc_in]  = False
    canvas[rim_mask] = np.clip(canvas[rim_mask] - 35, 0, 255)


def _place_droplets(canvas, size_class, n, rng):
    """
    Try to place n droplets of size_class on canvas.
    Returns list of (cy, cx, radius) for each placed droplet.
    Mild overlap (< 25% of combined radius) is allowed.
    """
    r_min, r_max = SIZE_CLASSES[size_class]
    placed = []   # (cy, cx, radius)
    attempts = 0

    while len(placed) < n and attempts < 600:
        attempts += 1
        radius = int(rng.integers(r_min, r_max + 1))
        margin = radius + 3
        if 2 * margin >= IMAGE_SIZE:
            continue
        cy = int(rng.integers(margin, IMAGE_SIZE - margin))
        cx = int(rng.integers(margin, IMAGE_SIZE - margin))

        too_close = any(
            np.hypot(cy - oy, cx - ox) < (radius + r) * 0.75
            for oy, ox, r in placed
        )
        if too_close:
            continue

        _draw_droplet(canvas, cy, cx, radius, rng)
        placed.append((cy, cx, radius))

    return placed


# ── public API ────────────────────────────────────────────────────────────────

def generate_image(size_class, seed):
    """Generate one synthetic image. Returns (img, droplets) where img is uint8 256×256."""
    rng    = np.random.default_rng(seed)
    canvas = _make_background(rng)
    n      = int(rng.integers(N_DROPLETS_MIN, N_DROPLETS_MAX + 1))
    drops  = _place_droplets(canvas, size_class, n, rng)

    # slight defocus blur
    canvas = filters.gaussian(canvas, sigma=rng.uniform(0.8, 1.6), preserve_range=True)
    # sensor noise
    canvas += rng.normal(0, rng.uniform(4, 9), canvas.shape)

    return np.clip(canvas, 0, 255).astype(np.uint8), drops


def _save_preview(images, labels):
    """Save a 3 x 5 grid (one row per class, five examples each)."""
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    for row_i, cls in enumerate(SIZE_CLASSES):
        cls_imgs = [im for im, lb in zip(images, labels) if lb == cls][:5]
        for col_i, im in enumerate(cls_imgs):
            axes[row_i, col_i].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[row_i, col_i].set_title(cls, fontsize=9)
            axes[row_i, col_i].axis("off")
    plt.suptitle("Synthetic Emulsion-like Images  (Step 1)", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "synthetic_examples.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  Preview  -> {out}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows   = []
    images = []
    labels = []

    print("Generating images ...")
    idx = 0
    for size_class in SIZE_CLASSES:
        for i in range(N_PER_CLASS):
            seed = idx * 137 + 42          # deterministic, unique per image
            img, drops = generate_image(size_class, seed)

            fname = f"{size_class}_{i:03d}.png"
            Image.fromarray(img, mode="L").save(str(RAW_DIR / fname))

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

    # write metadata CSV
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Images   -> {RAW_DIR}/  ({len(rows)} files)")
    print(f"  Metadata -> {METADATA_CSV}")
    _save_preview(images, labels)
    print("Done.")


if __name__ == "__main__":
    main()
