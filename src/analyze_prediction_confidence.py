"""V7 — Looking at how confident the model actually is on each dataset.

Instead of just reporting one accuracy number, this script splits predictions
into high-confidence (softmax ≥ 0.80) and uncertain (< 0.80) and checks
whether confidence tracks patch quality at all. The important caveat is that
high confidence just means the model is certain about its output — it doesn't
mean the prediction is correct, since there's no human-labelled ground truth
to compare against. Both the V1-only model and the mixed model (if available)
are evaluated across all three datasets.
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
import torch
import torchvision.transforms as T

from src.classical import detect_droplets
from src.dataset   import PATCHES_DIR, CLASS_ORDER, IDX_TO_CLASS
from src.model     import SimpleCNN
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

# ── constants ─────────────────────────────────────────────────────────────────
PATCH_SIZE          = 64
CENTER_HALF         = 8      # centre region = 2×8 = 16 px
HIGH_CONF_THRESHOLD = 0.80   # approved in v7 plan
BATCH_SIZE          = 64
N_EXAMPLES          = 5      # high / low conf patches per dataset in Figure 2
N_GALLERY           = 10     # uncertain patches per row in Figure 3
GALLERY_SEED        = 42

DS_ORDER  = ["V1 clean", "V2 realistic", "V2.1 high-density"]
DS_COLORS = {
    "V1 clean":           "#4C9BE8",
    "V2 realistic":       "#E8A44C",
    "V2.1 high-density":  "#E86B6B",
}
# colors by predicted class index (same palette as project-wide)
CLASS_COLORS = {0: "#4C9BE8", 1: "#E8A44C", 2: "#5EBD70"}

_CAVEAT = "High confidence = model certainty, not validated accuracy."

_infer_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ── patch collection ──────────────────────────────────────────────────────────

def _load_v1_patches():
    """Read existing V1 patches from data/patches/ as uint8 arrays."""
    if not PATCHES_DIR.exists():
        return []
    pairs = []
    for class_dir in sorted(PATCHES_DIR.iterdir()):
        if class_dir.is_dir():
            for f in sorted(class_dir.glob("*.png")):
                arr = np.array(Image.open(f).convert("L"))
                pairs.append((arr, "V1 clean"))
    return pairs


def _extract_patches_in_memory(raw_dir, metadata_csv, ds_label, tag):
    """Classical detection → in-memory 64×64 crops. Returns list of (arr, ds_label)."""
    meta  = pd.read_csv(metadata_csv)
    pairs = []
    for _, row in meta.iterrows():
        img_path = raw_dir / row["image_filename"]
        if not img_path.exists():
            continue
        img     = imread(str(img_path))
        regions, _ = detect_droplets(img)
        for reg in regions:
            cy, cx = int(reg.centroid[0]), int(reg.centroid[1])
            patch  = crop_patch(img, cy, cx, PATCH_SIZE)
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                pairs.append((patch, ds_label))
    print(f"  [{tag}]  {len(meta)} images → {len(pairs)} patches")
    return pairs


def _collect_all_patches():
    """
    Gather (patch_arr uint8, ds_label) from all available sources.
    V1 is loaded from saved files; V2/V2.1 use in-memory detection.
    """
    all_pairs = []

    print("[V7] Collecting patches ...")

    v1 = _load_v1_patches()
    if v1:
        print(f"  [V1 clean]  loaded {len(v1)} patches from {PATCHES_DIR}")
    else:
        print("  [V1 clean]  data/patches/ not found — skipping.")
        print("              Run 'python src/train_classifier.py' first.")
    all_pairs.extend(v1)

    if REALISTIC_RAW_DIR.exists():
        all_pairs.extend(
            _extract_patches_in_memory(
                REALISTIC_RAW_DIR, REALISTIC_CSV, "V2 realistic", "V2 realistic"
            )
        )
    else:
        print("  [V2]  data/realistic_raw/ not found — skipping.")

    if HIGH_DENSITY_RAW_DIR.exists():
        all_pairs.extend(
            _extract_patches_in_memory(
                HIGH_DENSITY_RAW_DIR, HIGH_DENSITY_CSV,
                "V2.1 high-density", "V2.1 high-density"
            )
        )
    else:
        print("  [V2.1]  data/high_density_raw/ not found — skipping.")

    print(f"  Total: {len(all_pairs)} patches across all datasets.\n")
    return all_pairs


# ── inference ─────────────────────────────────────────────────────────────────

def _patch_quality(arr):
    """Return (std_dev, centre_surround_ratio) for a uint8 patch array."""
    pf  = arr.astype(np.float64)
    mid = PATCH_SIZE // 2
    c0, c1 = mid - CENTER_HALF, mid + CENTER_HALF
    mb  = pf.mean()
    csr = pf[c0:c1, c0:c1].mean() / mb if mb > 1e-6 else 1.0
    return float(pf.std()), float(csr)


def _run_inference(model, model_label, patches_with_ds, device):
    """
    Batch inference over all patches. Returns list of record dicts.

    Each record holds:
      dataset, model, pred_label, max_conf, probs (array),
      patch_std, patch_csr, is_high_conf, patch (uint8 array)

    patch arrays are shared references (not copied) to keep memory low.
    """
    records = []
    total   = len(patches_with_ds)

    for start in range(0, total, BATCH_SIZE):
        batch = patches_with_ds[start: start + BATCH_SIZE]

        tensors = [
            _infer_transform(Image.fromarray(arr, mode="L"))
            for arr, _ in batch
        ]
        batch_t = torch.stack(tensors).to(device)

        with torch.no_grad():
            probs_batch = torch.softmax(model(batch_t), dim=1).cpu().numpy()

        for (patch_arr, ds_label), probs in zip(batch, probs_batch):
            max_conf   = float(probs.max())
            pred_class = int(probs.argmax())
            std_dev, csr = _patch_quality(patch_arr)

            records.append({
                "dataset":      ds_label,
                "model":        model_label,
                "pred_class":   pred_class,
                "pred_label":   IDX_TO_CLASS[pred_class],
                "max_conf":     max_conf,
                "probs":        probs,
                "patch_std":    std_dev,
                "patch_csr":    csr,
                "is_high_conf": max_conf >= HIGH_CONF_THRESHOLD,
                "patch":        patch_arr,     # shared reference, no copy
            })

    return records


# ── figure 1: confidence distribution ────────────────────────────────────────

def _save_confidence_distribution(all_records, model_labels):
    """
    Histogram of max softmax confidence per (model × dataset).
    Red dashed line = HIGH_CONF_THRESHOLD.
    Fraction of high-conf patches annotated as subtitle.
    """
    n_models = len(model_labels)
    n_cols   = 3

    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(n_cols * 4.2, n_models * 3.5),
                             sharey=False)
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row_i, ml in enumerate(model_labels):
        for col_i, ds in enumerate(DS_ORDER):
            ax   = axes[row_i, col_i]
            recs = [r for r in all_records
                    if r["model"] == ml and r["dataset"] == ds]

            if not recs:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        fontsize=10, transform=ax.transAxes, color="gray")
                ax.set_title(f"{ds}", fontsize=9)
                ax.axis("off")
                continue

            confs      = [r["max_conf"] for r in recs]
            high_frac  = sum(r["is_high_conf"] for r in recs) / len(recs)
            uncert_n   = sum(not r["is_high_conf"] for r in recs)

            ax.hist(confs, bins=20, range=(0, 1),
                    color=DS_COLORS.get(ds, "gray"),
                    alpha=0.75, edgecolor="white")
            ax.axvline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                       linewidth=1.4, alpha=0.9,
                       label=f"threshold {HIGH_CONF_THRESHOLD:.0%}")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Max softmax confidence", fontsize=8)
            ax.set_ylabel("Patch count", fontsize=8)
            ax.set_title(
                f"{ds}\n"
                f"high-conf: {high_frac:.1%}  ·  uncertain: {uncert_n}  "
                f"(n={len(recs)})",
                fontsize=8.5,
            )
            ax.legend(fontsize=7)

        axes[row_i, 0].set_ylabel(f"{ml}\nPatch count", fontsize=8)

    plt.suptitle(
        "V7 — Softmax Confidence Distribution per Dataset\n"
        f"Threshold = {HIGH_CONF_THRESHOLD:.0%}  ·  "
        + _CAVEAT,
        fontsize=9.5,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "confidence_distribution.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 2: high vs low confidence examples ─────────────────────────────────

def _save_high_vs_low_examples(all_records, model_labels):
    """
    For each dataset: 5 highest-confidence patches (green border) beside
    5 lowest-confidence patches (orange border).
    Uses the first available model only.
    """
    ml       = model_labels[0]
    ds_avail = [ds for ds in DS_ORDER
                if any(r["dataset"] == ds and r["model"] == ml
                       for r in all_records)]
    if not ds_avail:
        print("  [Figure 2] no records available — skipping.")
        return

    n_rows = len(ds_avail)
    n_cols = N_EXAMPLES * 2   # high | low
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.55, n_rows * 2.1))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, ds in enumerate(ds_avail):
        recs   = sorted(
            [r for r in all_records if r["dataset"] == ds and r["model"] == ml],
            key=lambda r: r["max_conf"],
            reverse=True,
        )
        high_recs = recs[:N_EXAMPLES]
        low_recs  = recs[-N_EXAMPLES:]

        for side_i, (side_recs, edge_col, side_label) in enumerate([
            (high_recs, "#22AA44", "HIGH"),
            (low_recs,  "#FF6633", "LOW"),
        ]):
            for j in range(N_EXAMPLES):
                ax = axes[row_i, side_i * N_EXAMPLES + j]
                if j < len(side_recs):
                    rec = side_recs[j]
                    ax.imshow(rec["patch"], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(
                        f"{rec['max_conf']:.0%}\n{rec['pred_label']}",
                        fontsize=6.5, color=edge_col,
                    )
                    for spine in ax.spines.values():
                        spine.set_edgecolor(edge_col)
                        spine.set_linewidth(2.5)
                else:
                    ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])

        axes[row_i, 0].set_ylabel(ds, fontsize=8, labelpad=5)

    # column group labels at the top
    mid_high = (N_EXAMPLES - 1) / 2
    mid_low  = N_EXAMPLES + (N_EXAMPLES - 1) / 2
    axes[0, int(mid_high)].set_title(
        f"← {N_EXAMPLES} highest-confidence →\n"
        + axes[0, int(mid_high)].get_title(),
        fontsize=7.5,
    )
    axes[0, int(mid_low)].set_title(
        f"← {N_EXAMPLES} lowest-confidence →\n"
        + axes[0, int(mid_low)].get_title(),
        fontsize=7.5,
    )

    plt.suptitle(
        f"V7 — High vs Low Confidence Examples  (model: {ml})\n"
        "Green border = high-conf  ·  Orange border = low/uncertain  ·  "
        + _CAVEAT,
        fontsize=9,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "high_vs_low_confidence_examples.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 3: uncertain patch gallery ────────────────────────────────────────

def _save_uncertain_patch_gallery(all_records, model_labels):
    """
    Gallery of patches where max_conf < HIGH_CONF_THRESHOLD.
    Shows what the model is uncertain about — often noisy or
    poorly-cropped patches.
    Uses the first available model only.
    """
    rng = np.random.default_rng(GALLERY_SEED)
    ml  = model_labels[0]

    uncertain_by_ds = {
        ds: [r for r in all_records
             if r["dataset"] == ds and r["model"] == ml
             and not r["is_high_conf"]]
        for ds in DS_ORDER
    }
    # only include datasets that actually have uncertain patches
    ds_with_uncert = [ds for ds in DS_ORDER if uncertain_by_ds[ds]]

    if not ds_with_uncert:
        print("  [Figure 3] no uncertain patches found — skipping.")
        return

    n_rows = len(ds_with_uncert)
    fig, axes = plt.subplots(n_rows, N_GALLERY,
                             figsize=(N_GALLERY * 1.6, n_rows * 1.95))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, ds in enumerate(ds_with_uncert):
        recs   = uncertain_by_ds[ds]
        n_avail = len(recs)
        sample_idx = rng.choice(n_avail,
                                size=min(N_GALLERY, n_avail),
                                replace=False)

        for col_i in range(N_GALLERY):
            ax = axes[row_i, col_i]
            if col_i < len(sample_idx):
                rec = recs[sample_idx[col_i]]
                ax.imshow(rec["patch"], cmap="gray", vmin=0, vmax=255)
                ax.set_title(
                    f"conf={rec['max_conf']:.0%}\n{rec['pred_label']}",
                    fontsize=6.5,
                    color="#FF6633",
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor("#FF6633")
                    spine.set_linewidth(1.5)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_i, 0].set_ylabel(
            f"{ds}\n(n={n_avail} uncertain)", fontsize=8, labelpad=5
        )

    plt.suptitle(
        f"V7 — Uncertain Patch Gallery  (conf < {HIGH_CONF_THRESHOLD:.0%},  "
        f"model: {ml})\n"
        "These patches may be noisy, low-contrast, or poorly centred on a droplet.",
        fontsize=10,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "uncertain_patch_gallery.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── figure 4: patch quality vs confidence ────────────────────────────────────

def _save_quality_confidence_relationship(all_records, model_labels):
    """
    Scatter plot: patch std dev (quality proxy) vs max confidence.
    One column per dataset, one row per model.
    Colour = predicted class.
    Horizontal line at HIGH_CONF_THRESHOLD.
    Trend line shows whether higher-quality patches get higher confidence.
    """
    n_models = len(model_labels)
    n_cols   = 3

    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(n_cols * 4.2, n_models * 3.8))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row_i, ml in enumerate(model_labels):
        for col_i, ds in enumerate(DS_ORDER):
            ax   = axes[row_i, col_i]
            recs = [r for r in all_records
                    if r["model"] == ml and r["dataset"] == ds]

            if not recs:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        fontsize=10, transform=ax.transAxes, color="gray")
                ax.set_title(f"{ds}", fontsize=9)
                ax.axis("off")
                continue

            stds  = np.array([r["patch_std"]  for r in recs])
            confs = np.array([r["max_conf"]   for r in recs])
            cls   = np.array([r["pred_class"] for r in recs])

            # scatter, coloured by predicted class
            for c_idx, c_label in enumerate(CLASS_ORDER):
                mask = cls == c_idx
                if mask.sum() == 0:
                    continue
                ax.scatter(stds[mask], confs[mask],
                           c=CLASS_COLORS[c_idx], alpha=0.45,
                           s=12, label=c_label, linewidths=0)

            # confidence threshold line
            ax.axhline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.8,
                       label=f"threshold {HIGH_CONF_THRESHOLD:.0%}")

            # linear trend line (std → confidence)
            if len(stds) > 2 and stds.std() > 0:
                z    = np.polyfit(stds, confs, 1)
                x_ln = np.linspace(stds.min(), stds.max(), 100)
                ax.plot(x_ln, np.polyval(z, x_ln),
                        color="black", linewidth=1.2, alpha=0.7,
                        linestyle="-", label=f"trend (slope={z[0]:+.4f})")

            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Patch std dev (contrast proxy)", fontsize=8)
            ax.set_ylabel("Max softmax confidence", fontsize=8)
            ax.set_title(f"{ds}  (n={len(recs)})", fontsize=9)
            ax.legend(fontsize=6.5, markerscale=1.5)

        axes[row_i, 0].set_ylabel(f"{ml}\nMax confidence", fontsize=8)

    plt.suptitle(
        "V7 — Patch Quality vs Model Confidence\n"
        "Positive trend = higher-contrast patches get higher CNN confidence  ·  "
        + _CAVEAT,
        fontsize=9.5,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "quality_confidence_relationship.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── terminal summary ──────────────────────────────────────────────────────────

def _print_summary(all_records, model_labels):
    LINE = "─" * 70

    print(f"\n{LINE}")
    print("V7 — CONFIDENCE-AWARE PREDICTION SUMMARY")
    print(LINE)

    for ml in model_labels:
        print(f"\n  Model: {ml}\n")
        print(f"  {'Dataset':<26}  {'total':>6}  "
              f"{'high-conf':>9}  {'uncertain':>9}  {'high-conf%':>10}")
        print(f"  {'─'*26}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*10}")

        for ds in DS_ORDER:
            recs = [r for r in all_records
                    if r["model"] == ml and r["dataset"] == ds]
            if not recs:
                print(f"  {ds:<26}  {'(no data)':>36}")
                continue
            total     = len(recs)
            high_n    = sum(r["is_high_conf"] for r in recs)
            uncert_n  = total - high_n
            high_pct  = high_n / total
            print(f"  {ds:<26}  {total:>6}  "
                  f"{high_n:>9}  {uncert_n:>9}  {high_pct:>10.1%}")

    print(f"\n{LINE}")
    print("INTERPRETATION")
    print(LINE)

    # check if high-density is more uncertain
    first_ml = model_labels[0]
    fracs    = {}
    for ds in DS_ORDER:
        recs = [r for r in all_records
                if r["model"] == first_ml and r["dataset"] == ds]
        if recs:
            fracs[ds] = sum(r["is_high_conf"] for r in recs) / len(recs)

    if len(fracs) >= 2:
        ordered = sorted(fracs.items(), key=lambda x: x[1], reverse=True)
        print("\n  Confidence ranking (model: {})".format(first_ml))
        for ds, frac in ordered:
            bar = "█" * int(frac * 20)
            print(f"    {ds:<26}  {bar:<20}  {frac:.1%}")

    v21_frac = fracs.get("V2.1 high-density", None)
    v1_frac  = fracs.get("V1 clean", None)
    if v21_frac is not None and v1_frac is not None:
        if v21_frac < v1_frac * 0.6:
            print(
                "\n  ✓  V2.1 high-density data produces substantially more uncertain\n"
                "     predictions than V1 clean data, consistent with the known\n"
                "     difficulty of that dataset (v5 diagnosis: low contrast,\n"
                "     poor centre-surround ratio, lower detection rate)."
            )
        else:
            print(
                "\n  △  V2.1 high-density confidence is not dramatically lower than V1.\n"
                "     The mixed-data training (v4) may have improved robustness."
            )

    print(f"\n{LINE}")
    print(f"Note: high confidence (≥{HIGH_CONF_THRESHOLD:.0%}) = model certainty, not validated accuracy.")
    print("Confident-but-wrong predictions are possible. Flag uncertain patches for human review.")
    print(LINE + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load available models ─────────────────────────────────────────────────
    print("[V7] Loading models ...")
    available_models = []  # list of (SimpleCNN, label_str)

    if V1_MODEL_PATH.exists():
        m = SimpleCNN(n_classes=len(CLASS_ORDER))
        m.load_state_dict(torch.load(str(V1_MODEL_PATH), map_location=device))
        m.to(device).eval()
        available_models.append((m, "v1-only"))
        print(f"  Loaded: {V1_MODEL_PATH}")
    else:
        print(f"  WARNING: {V1_MODEL_PATH} not found.")
        print("  Run 'python src/train_classifier.py' first.")

    if MIXED_MODEL_PATH.exists():
        m = SimpleCNN(n_classes=len(CLASS_ORDER))
        m.load_state_dict(torch.load(str(MIXED_MODEL_PATH), map_location=device))
        m.to(device).eval()
        available_models.append((m, "mixed"))
        print(f"  Loaded: {MIXED_MODEL_PATH}")
    else:
        print(f"  (optional) {MIXED_MODEL_PATH} not found — skipping mixed model.")

    if not available_models:
        raise RuntimeError("No trained model found. Cannot proceed.")

    model_labels = [label for _, label in available_models]
    print(f"  Running with: {model_labels}\n")

    # ── collect patches ───────────────────────────────────────────────────────
    patches_with_ds = _collect_all_patches()
    if not patches_with_ds:
        raise RuntimeError(
            "No patches available. "
            "Run the data-generation and training scripts first."
        )

    # ── run inference for each model ──────────────────────────────────────────
    print("[V7] Running inference ...")
    all_records = []
    for model, label in available_models:
        print(f"  [{label}] inferring on {len(patches_with_ds)} patches ...")
        recs = _run_inference(model, label, patches_with_ds, device)
        all_records.extend(recs)
        n_high = sum(r["is_high_conf"] for r in recs)
        print(f"    high-conf: {n_high}/{len(recs)} "
              f"({n_high/len(recs):.1%})\n")

    # ── save figures ──────────────────────────────────────────────────────────
    print("[V7] Saving figures ...")
    _save_confidence_distribution(all_records, model_labels)
    _save_high_vs_low_examples(all_records, model_labels)
    _save_uncertain_patch_gallery(all_records, model_labels)
    _save_quality_confidence_relationship(all_records, model_labels)

    # ── terminal summary ──────────────────────────────────────────────────────
    _print_summary(all_records, model_labels)

    print("Done.")


if __name__ == "__main__":
    main()
