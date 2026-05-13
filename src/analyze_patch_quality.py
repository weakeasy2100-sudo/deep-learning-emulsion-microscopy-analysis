"""V5 — Diagnosing why accuracy drops on harder datasets.

V3 and V4 showed a big accuracy gap on v2/v2.1 data, but I couldn't tell
if that was the CNN's fault or the classical detector's. This script checks
the quality of patches the detector actually hands to the CNN — brightness,
contrast (std dev), and whether the crop is centred on a real droplet
(centre-surround ratio > 1.0 means the centre is brighter than the edges,
which is what you'd expect for a correctly-detected droplet). Also reports
how many patches per image the detector finds on each dataset. No model is
loaded or retrained here.
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
from PIL import Image
from skimage.io import imread

from src.classical import detect_droplets
from src.dataset   import PATCHES_DIR
from src.utils     import crop_patch

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR          = PROJECT_ROOT / "results"
V1_METADATA_CSV      = PROJECT_ROOT / "data" / "metadata.csv"
REALISTIC_RAW_DIR    = PROJECT_ROOT / "data" / "realistic_raw"
REALISTIC_CSV        = PROJECT_ROOT / "data" / "realistic_metadata.csv"
HIGH_DENSITY_RAW_DIR = PROJECT_ROOT / "data" / "high_density_raw"
HIGH_DENSITY_CSV     = PROJECT_ROOT / "data" / "high_density_metadata.csv"

PATCH_SIZE   = 64
CENTER_HALF  = 8      # centre region = 2×CENTER_HALF px square = 16×16
N_GALLERY    = 10     # patches shown per row in the gallery
GALLERY_SEED = 0

DS_ORDER  = ["V1 clean", "V2 realistic", "V2.1 high-density"]
DS_COLORS = {
    "V1 clean":           "#4C9BE8",
    "V2 realistic":       "#E8A44C",
    "V2.1 high-density":  "#E86B6B",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_v1_patches():
    """
    Load V1 patches directly from data/patches/ — no re-detection needed
    because train_classifier.py already ran the classical detector and saved
    the results.  This is the asymmetric design agreed in the plan.

    Returns
    -------
    patches  : list of np.ndarray uint8 (64×64)
    n_images : int  total V1 source images (from metadata.csv)
    """
    if not PATCHES_DIR.exists():
        return [], 0

    all_files = sorted(PATCHES_DIR.glob("*/*.png"))
    patches   = [np.array(Image.open(f).convert("L")) for f in all_files]

    n_images = len(patches)          # safe fallback
    if V1_METADATA_CSV.exists():
        try:
            n_images = len(pd.read_csv(V1_METADATA_CSV))
        except Exception:
            pass

    return patches, n_images


def _extract_patches(raw_dir, metadata_csv, tag):
    """
    Run classical detection on every image listed in metadata_csv, crop
    64×64 patches, and return them as a list.  Nothing is written to disk.

    Returns
    -------
    patches      : list of np.ndarray uint8 (64×64)
    n_images     : int
    n_detections : int   raw detector output count (before shape filter)
    """
    meta = pd.read_csv(metadata_csv)
    patches      = []
    n_detections = 0

    for _, row in meta.iterrows():
        img_path = raw_dir / row["image_filename"]
        if not img_path.exists():
            continue
        img = imread(str(img_path))
        regions, _ = detect_droplets(img)
        n_detections += len(regions)
        for reg in regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                patches.append(patch)

    print(f"  [{tag}]  {len(meta)} images  "
          f"→  {n_detections} raw detections  "
          f"→  {len(patches)} valid patches")
    return patches, len(meta), n_detections


# ── per-patch statistics ──────────────────────────────────────────────────────

def _compute_patch_stats(patches):
    """
    Compute quality metrics for every patch in the list.

    Returns
    -------
    dict with keys:
        mean_brightness       : np.ndarray  float64, one value per patch
        std_dev               : np.ndarray  float64
        center_surround_ratio : np.ndarray  float64
            Ratio of the central 16×16 mean to the full-patch mean.
            >1.0 means the centre is brighter than the surroundings —
            expected for a well-centred droplet.  Near 1.0 means the crop
            did not capture a droplet in its centre.
    """
    if not patches:
        return {k: np.array([]) for k in
                ("mean_brightness", "std_dev", "center_surround_ratio")}

    mid = PATCH_SIZE // 2
    c0, c1 = mid - CENTER_HALF, mid + CENTER_HALF  # 24:40 for 64-px patch

    mean_b, std_d, cs_ratio = [], [], []
    for p in patches:
        pf = p.astype(np.float64)
        mb = pf.mean()
        sd = pf.std()
        centre_mean = pf[c0:c1, c0:c1].mean()
        csr = centre_mean / mb if mb > 1e-6 else 1.0
        mean_b.append(mb)
        std_d.append(sd)
        cs_ratio.append(csr)

    return {
        "mean_brightness":       np.array(mean_b),
        "std_dev":               np.array(std_d),
        "center_surround_ratio": np.array(cs_ratio),
    }


# ── figure 1: gallery ─────────────────────────────────────────────────────────

def _save_gallery(data_by_ds):
    """
    3 rows × N_GALLERY columns.
    Each patch shows its pixel std dev (σ) as a subtitle:
    low σ = low contrast = CNN has little to classify.
    """
    rng    = np.random.default_rng(GALLERY_SEED)
    labels = [lb for lb in DS_ORDER if lb in data_by_ds]
    n_rows = len(labels)

    fig, axes = plt.subplots(n_rows, N_GALLERY,
                             figsize=(N_GALLERY * 1.6, n_rows * 1.9))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, lb in enumerate(labels):
        info    = data_by_ds[lb]
        patches = info["patches"]
        stats   = info["stats"]
        n_avail = len(patches)

        if n_avail == 0:
            for col_i in range(N_GALLERY):
                axes[row_i, col_i].axis("off")
            axes[row_i, 0].text(0.5, 0.5, "no patches\nfound",
                                ha="center", va="center",
                                fontsize=8, color="red",
                                transform=axes[row_i, 0].transAxes)
            axes[row_i, 0].set_ylabel(lb, fontsize=9, labelpad=6)
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

        # row label on the leftmost cell
        axes[row_i, 0].set_ylabel(lb, fontsize=8, labelpad=6)

    plt.suptitle(
        "V5 — Patch Quality Gallery\n"
        "σ = pixel std dev  ·  low σ = low contrast  ·  "
        f"{N_GALLERY} random patches per dataset  ·  seed={GALLERY_SEED}",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "patch_quality_gallery.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 2: statistics ──────────────────────────────────────────────────────

def _save_statistics(data_by_ds):
    """
    2 × 2 grid:
      (0,0)  Mean brightness violin
      (0,1)  Std dev violin             — low std = low-contrast patch
      (1,0)  Centre-surround ratio      — near 1.0 = crop not centred
      (1,1)  Detection rate bar chart   — patches per source image
    """
    labels = [lb for lb in DS_ORDER if lb in data_by_ds]
    colors = [DS_COLORS[lb] for lb in labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── shared violin helper ──────────────────────────────────────────────────
    def _violin_ax(ax, metric_key, ylabel, title, ref_line=None):
        arrays    = [data_by_ds[lb]["stats"][metric_key] for lb in labels]
        non_empty = [(i, a) for i, a in enumerate(arrays) if len(a) > 0]

        if not non_empty:
            ax.set_title(title + "\n(no data)", fontsize=10)
            return

        pos  = [i + 1 for i, _ in non_empty]
        arrs = [a for _, a in non_empty]
        lbls = [labels[i] for i, _ in non_empty]
        cols = [colors[i] for i, _ in non_empty]

        parts = ax.violinplot(arrs, positions=pos,
                              showmedians=True, showextrema=True)
        for body, col in zip(parts["bodies"], cols):
            body.set_facecolor(col)
            body.set_alpha(0.75)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.2)

        if ref_line is not None:
            ax.axhline(ref_line, color="red", linestyle="--",
                       linewidth=1.1, alpha=0.8, label=f"ref = {ref_line}")
            ax.legend(fontsize=8)

        # give headroom for annotations
        all_max = max(float(a.max()) for a in arrs)
        all_min = min(float(a.min()) for a in arrs)
        span    = max(all_max - all_min, 1.0)
        ax.set_ylim(all_min - span * 0.08,
                    all_max + span * 0.32)

        for p, a in zip(pos, arrs):
            ax.text(p, float(a.max()) + span * 0.04,
                    f"n={len(a)}\nmed={np.median(a):.1f}",
                    ha="center", va="bottom", fontsize=7, color="#333333")

        ax.set_xticks(pos)
        ax.set_xticklabels(lbls, fontsize=9, rotation=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)

    # ── (0,0) mean brightness ─────────────────────────────────────────────────
    _violin_ax(
        axes[0, 0],
        "mean_brightness",
        "Mean pixel value (0–255)",
        "Mean Brightness per Patch",
    )

    # ── (0,1) std dev ─────────────────────────────────────────────────────────
    _violin_ax(
        axes[0, 1],
        "std_dev",
        "Pixel std dev",
        "Contrast (Std Dev per Patch)\nlow σ → CNN sees mostly noise",
    )

    # ── (1,0) centre-surround ratio ───────────────────────────────────────────
    _violin_ax(
        axes[1, 0],
        "center_surround_ratio",
        "Centre 16×16 mean / full 64×64 mean",
        "Centre-Surround Ratio\n≈1.0 → crop not centred on droplet",
        ref_line=1.0,
    )

    # ── (1,1) detection rate bar ──────────────────────────────────────────────
    ax = axes[1, 1]
    rates  = []
    bar_lb = []
    bar_cl = []
    for lb in labels:
        info = data_by_ds[lb]
        ni   = info["n_images"]
        rate = info["n_patches"] / ni if ni > 0 else 0.0
        rates.append(rate)
        bar_lb.append(lb)
        bar_cl.append(DS_COLORS[lb])

    bars = ax.bar(range(len(bar_lb)), rates,
                  color=bar_cl, edgecolor="white", width=0.55)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rates) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(bar_lb)))
    ax.set_xticklabels(bar_lb, fontsize=9, rotation=12)
    ax.set_ylabel("Valid patches per source image", fontsize=9)
    ax.set_title(
        "Detection Rate\n(classical detector + shape filter output)", fontsize=10
    )
    ax.set_ylim(0, max(rates) * 1.25 if rates else 1.0)

    plt.suptitle("V5 — Patch Quality Statistics Across Datasets", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "patch_quality_statistics.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── terminal diagnosis ────────────────────────────────────────────────────────

def _print_diagnosis(data_by_ds):
    """
    Print a summary table and a written diagnosis with quantitative thresholds.
    Concludes whether the accuracy gap is driven by pipeline failure or CNN
    domain shift — or both.
    """
    LINE = "─" * 72

    print(f"\n{LINE}")
    print("V5 — PATCH QUALITY DIAGNOSIS")
    print(LINE)
    print(f"  {'Dataset':<26}  {'n_patches':>9}  {'patch/img':>9}  "
          f"{'mean_bright':>11}  {'mean_σ':>7}  {'mean_CSR':>9}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*7}  {'─'*9}")

    summary = {}
    for lb in DS_ORDER:
        if lb not in data_by_ds:
            continue
        info = data_by_ds[lb]
        n    = info["n_patches"]
        ni   = info["n_images"]
        rate = n / ni if ni > 0 else 0.0
        st   = info["stats"]
        mb   = float(np.mean(st["mean_brightness"]))        if n > 0 else 0.0
        sd   = float(np.mean(st["std_dev"]))                if n > 0 else 0.0
        csr  = float(np.mean(st["center_surround_ratio"])) if n > 0 else 1.0
        summary[lb] = dict(n=n, n_images=ni, rate=rate, mb=mb, sd=sd, csr=csr)
        print(f"  {lb:<26}  {n:>9d}  {rate:>9.2f}  "
              f"{mb:>11.1f}  {sd:>7.1f}  {csr:>9.3f}")

    # reference values from V1
    v1      = summary.get("V1 clean", {})
    v1_rate = v1.get("rate", 1.0)
    v1_csr  = v1.get("csr",  1.10)
    v1_sd   = v1.get("sd",   30.0)

    print(f"\n{LINE}")
    print("PER-DATASET FINDINGS")
    print(LINE)
    print(
        "  NOTE: all metrics reflect the FULL pipeline —\n"
        "  classical detection → patch cropping → (then the CNN would classify).\n"
        "  Low scores below indicate pipeline failures upstream of the CNN.\n"
    )

    pipeline_issue_count = {}

    for lb in ["V2 realistic", "V2.1 high-density"]:
        s = summary.get(lb)
        if s is None:
            print(f"\n  {lb}: dataset not available — skipped.")
            continue

        issues = 0
        print(f"\n  {lb}:")

        # 1. detection rate
        if v1_rate > 0:
            drop = (v1_rate - s["rate"]) / v1_rate
            if drop > 0.50:
                print(f"    ⚠  Detection rate:  {s['rate']:.2f} patches/img  "
                      f"(V1: {v1_rate:.2f},  drop: {drop:.0%})")
                print( "       Classical detector misses or rejects most droplets.")
                print( "       Fewer inputs reach the CNN → accuracy is limited by coverage.")
                issues += 1
            elif drop > 0.25:
                print(f"    △  Detection rate:  {s['rate']:.2f} patches/img  "
                      f"(V1: {v1_rate:.2f},  drop: {drop:.0%})")
                print( "       Detector output is noticeably lower — some droplets missed.")
            else:
                print(f"    ✓  Detection rate:  {s['rate']:.2f} patches/img  "
                      f"(V1: {v1_rate:.2f},  drop: {drop:.0%})")

        # 2. centre-surround ratio
        if s["csr"] < 1.04:
            print(f"    ⚠  Centre-surround ratio: {s['csr']:.3f}  (near 1.0)")
            print( "       Crops are NOT centred on droplets.")
            print( "       The 64×64 patch the CNN receives does not contain a "
                   "droplet in its centre.")
            print( "       This is a detection / cropping failure — the CNN "
                   "cannot fix this.")
            issues += 1
        elif s["csr"] < v1_csr * 0.90:
            print(f"    △  Centre-surround ratio: {s['csr']:.3f}  "
                  f"(V1: {v1_csr:.3f})")
            print( "       Crops are partially off-centre — some patches are "
                   "still useful but quality is degraded.")
        else:
            print(f"    ✓  Centre-surround ratio: {s['csr']:.3f}  "
                  f"(V1: {v1_csr:.3f}) — crops appear well-centred.")

        # 3. contrast
        if s["sd"] < 12.0:
            print(f"    ⚠  Mean contrast (σ): {s['sd']:.1f}  (very low)")
            print( "       Patches are nearly uniform noise.")
            print( "       The CNN receives almost no structural signal to classify.")
            issues += 1
        elif s["sd"] < v1_sd * 0.55:
            print(f"    △  Mean contrast (σ): {s['sd']:.1f}  (V1: {v1_sd:.1f})")
            print( "       Contrast is substantially lower than V1.")
            print( "       CNN domain shift is expected even if crops are centred.")
        else:
            print(f"    ✓  Mean contrast (σ): {s['sd']:.1f}  (V1: {v1_sd:.1f})")

        pipeline_issue_count[lb] = issues

    # overall conclusion
    print(f"\n{LINE}")
    print("OVERALL CONCLUSION")
    print(LINE)

    has_any_issue = any(v > 0 for v in pipeline_issue_count.values())

    if has_any_issue:
        print(
            "\n  The accuracy gap observed in v3 / v4 is at least partly caused by\n"
            "  classical detection and patch-cropping failures — NOT only by CNN\n"
            "  domain shift.\n"
            "\n"
            "  Recommended next steps (in order of expected impact):\n"
            "    1. Replace or improve the classical detector so it finds droplets\n"
            "       reliably under noise and low contrast (e.g. learned detector,\n"
            "       adaptive threshold, or multi-scale approach).\n"
            "    2. Only after improving detection quality does retraining the CNN\n"
            "       on mixed / augmented data become the dominant bottleneck.\n"
        )
    else:
        print(
            "\n  Detection and cropping quality appears comparable to V1.\n"
            "  The accuracy gap is more likely driven by CNN domain shift:\n"
            "  the model was trained on clean data and has not seen noisy /\n"
            "  low-contrast patches.\n"
            "\n"
            "  Recommended next step:\n"
            "    Retrain or fine-tune the CNN on mixed / augmented data\n"
            "    (v4 train_mixed_classifier.py already does this).\n"
        )

    print(LINE + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[V5] Patch Quality Diagnosis\n")

    data_by_ds = {}

    # ── V1: read from existing data/patches/ (no re-detection) ───────────────
    print("[V5] Loading V1 patches from data/patches/ ...")
    v1_patches, v1_n_images = _load_v1_patches()
    if not v1_patches:
        print("  data/patches/ not found or empty.")
        print("  Run 'python src/train_classifier.py' first.")
    else:
        print(f"  [V1 clean]  {len(v1_patches)} patches from {v1_n_images} images")
        data_by_ds["V1 clean"] = {
            "patches":   v1_patches,
            "n_patches": len(v1_patches),
            "n_images":  v1_n_images,
            "stats":     _compute_patch_stats(v1_patches),
        }

    # ── V2: classical detection in-memory ────────────────────────────────────
    print("\n[V5] Extracting V2 patches (classical detection, in-memory) ...")
    if not REALISTIC_RAW_DIR.exists():
        print("  data/realistic_raw/ not found — skipping V2.")
        print("  Run 'python src/generate_realistic_data.py' first.")
    else:
        v2_patches, v2_n_imgs, _ = _extract_patches(
            REALISTIC_RAW_DIR, REALISTIC_CSV, "V2 realistic"
        )
        data_by_ds["V2 realistic"] = {
            "patches":   v2_patches,
            "n_patches": len(v2_patches),
            "n_images":  v2_n_imgs,
            "stats":     _compute_patch_stats(v2_patches),
        }

    # ── V2.1: classical detection in-memory ───────────────────────────────────
    print("\n[V5] Extracting V2.1 patches (classical detection, in-memory) ...")
    if not HIGH_DENSITY_RAW_DIR.exists():
        print("  data/high_density_raw/ not found — skipping V2.1.")
        print("  Run 'python src/generate_high_density_realistic_data.py' first.")
    else:
        v21_patches, v21_n_imgs, _ = _extract_patches(
            HIGH_DENSITY_RAW_DIR, HIGH_DENSITY_CSV, "V2.1 high-density"
        )
        data_by_ds["V2.1 high-density"] = {
            "patches":   v21_patches,
            "n_patches": len(v21_patches),
            "n_images":  v21_n_imgs,
            "stats":     _compute_patch_stats(v21_patches),
        }

    if not data_by_ds:
        print("No datasets available. Exiting.")
        return

    # ── figures ───────────────────────────────────────────────────────────────
    print("\n[V5] Saving figures ...")
    _save_gallery(data_by_ds)
    _save_statistics(data_by_ds)

    # ── diagnosis ─────────────────────────────────────────────────────────────
    _print_diagnosis(data_by_ds)

    print("Done.")


if __name__ == "__main__":
    main()
