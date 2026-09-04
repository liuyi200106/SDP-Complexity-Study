"""Regenerate every figure in the paper.

  figure1_framework.png     Schematic of the pipeline under study and the
                            evaluation protocol (Section 3.1). Drawn here; it
                            depends on no experimental output.
  figure2_cd_diagram.png    Critical-difference diagrams (Nemenyi, alpha = 0.05)
                            over the imbalanced-ensemble comparison, for F1 and
                            G-mean (Section 4.8.1). Component panels are also
                            written to figures/components/.
  figure3_distributions.png Per-fold score distributions for the same
                            configurations, shown for description only; no test is
                            computed on per-fold values (Section 3.7).

Figures 2 and 3 read results/summary/imbalanced_ensemble_*.csv, so run
scripts/generate_tables.py first if those are missing.

Usage
-----
    python scripts/generate_figures.py                # all three
    python scripts/generate_figures.py --figure 1
    python scripts/generate_figures.py --dpi 600      # override output DPI
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "summary"
FIGURES = PROJECT_ROOT / "figures"
CONFIGS = ["C_Reference", "BalancedRF", "EasyEnsemble", "ABF", "ABF_Hybrid"]
DEFAULT_DPI = 300


FIG1_SCALE = 1.3


def figure1(dpi: int) -> None:
    s = FIG1_SCALE
    font = FontProperties(family=["DejaVu Sans", "sans-serif"])
    fig, ax = plt.subplots(figsize=(9.5 * s, 4.6 * s))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef1f6", ec="#333333", fontsize=9.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=3",
                                    linewidth=1.1 * s, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize * s,
                fontproperties=font, weight=weight, zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=14 * s, linewidth=1.2 * s,
                                     color="#333333", zorder=1))

    y0, h0 = 30, 10
    box(1, y0, 13, h0, "14 datasets\nPROMISE/NASA MDP\n+ Java CK", fontsize=8.5)
    arrow(14, y0 + h0 / 2, 16.6, y0 + h0 / 2)
    box(16.8, y0, 19, h0, "Stage 1\nSHAP-guided filter\n+ RFE", fc="#dbe7f5", weight="bold")
    arrow(35.8, y0 + h0 / 2, 38.4, y0 + h0 / 2)
    box(38.6, y0, 19, h0, "Stage 2\nFIB-SMOTE\n(Adaptive SMOTE)", fc="#dbe7f5", weight="bold")
    arrow(57.6, y0 + h0 / 2, 60.2, y0 + h0 / 2)
    box(60.4, y0, 19, h0, "Stage 3\nAdaptive Balanced\nForest (ABF)", fc="#dbe7f5", weight="bold")
    arrow(79.4, y0 + h0 / 2, 82, y0 + h0 / 2)
    box(82.2, y0, 16.8, h0, "Defect-proneness\nprediction", fontsize=8.5)

    box(20, 16, 27, 7, "Feature importances\nreused for weighting",
        fc="#fdf3d8", ec="#8a6d00", fontsize=8)
    arrow(26.3, y0, 26.3, 23)
    arrow(33, 19.5, 38.6, y0 + 1.5)

    box(4, 2, 92, 9.5,
        "Evaluation: stratified 5-fold cross-validation\n(training-fold preprocessing only)\n"
        "F1 / AUC / MCC / G-mean  →  dataset-level statistical testing",
        fc="#eef7ea", ec="#4a7a3a", fontsize=8.6)
    arrow(48, 29.6, 48, 11.5)

    plt.tight_layout()
    out = FIGURES / "figure1_framework.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {out.relative_to(PROJECT_ROOT)}")


def _cd_panel(avg_ranks: pd.Series, cd: float, metric: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    order = avg_ranks.sort_values()
    ax.hlines(1, order.min() - 0.5, order.max() + 0.5, color="black", linewidth=1)

    span = max(order.max() - order.min(), 1e-6)
    min_sep = span * 0.18
    last_x = {1: None, -1: None}
    row_of = {1: 0, -1: 0}
    seen: list[float] = []
    for i, (name, rank) in enumerate(order.items()):
        n_tied = sum(1 for r in seen if abs(r - rank) < 1e-6)
        seen.append(rank)
        y = 1 + 0.06 * n_tied
        ax.plot(rank, y, "o", color="black")

        side = 1 if i % 2 == 0 else -1
        prev = last_x[side]
        row_of[side] = row_of[side] + 1 if (prev is not None and abs(rank - prev) < min_sep) else 0
        last_x[side] = rank
        base = 16 if side == 1 else -34
        step = 26 if side == 1 else -26
        ax.annotate(f"{name}\n({rank:.2f})", (rank, y), textcoords="offset points",
                    xytext=(0, base + step * row_of[side]), ha="center", fontsize=8)

    cd_x0 = order.min()
    ax.plot([cd_x0, cd_x0 + cd], [2.30, 2.30], color="red", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", (cd_x0 + cd / 2, 2.36), ha="center", fontsize=8, color="red")
    ax.set_ylim(0.30, 2.60)
    ax.set_yticks([])
    ax.set_xlabel(f"Average rank ({metric}, 1 = best)")
    ax.set_title(f"Critical Difference diagram — {metric} (Nemenyi, α=0.05)")
    plt.tight_layout()

    out = FIGURES / "components" / f"cd_diagram_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    return out


def figure2(dpi: int) -> None:
    path = SUMMARY / "imbalanced_ensemble_dataset_level.csv"
    if not path.exists():
        print(f"  [skip] figure 2: {path.name} missing -- run scripts/generate_tables.py")
        return
    dl = pd.read_csv(path)
    n_datasets = dl["dataset"].nunique()
    k = len(CONFIGS)
    cd = 2.728 * np.sqrt(k * (k + 1) / (6 * n_datasets))

    panels = []
    for metric in ("F1", "Gmean"):
        pivot = dl.pivot(index="dataset", columns="config", values=metric)[CONFIGS]
        ranks = pivot.rank(axis=1, ascending=False).mean().reindex(CONFIGS)
        panels.append(_cd_panel(ranks, cd, metric, dpi))

    top, bottom = (Image.open(p) for p in panels)
    combined = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), "white")
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height))
    out = FIGURES / "figure2_cd_diagram.png"
    combined.save(out, dpi=(dpi, dpi))
    print(f"  {out.relative_to(PROJECT_ROOT)}  (CD = {cd:.3f}, n = {n_datasets})")


def figure3(dpi: int) -> None:
    path = SUMMARY / "imbalanced_ensemble_fold_level.csv"
    if not path.exists():
        print(f"  [skip] figure 3: {path.name} missing -- run scripts/generate_tables.py")
        return
    fold = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("F1", "Gmean")):
        data = [fold.loc[fold["config"] == c, metric].values for c in CONFIGS]
        ax.boxplot(data, tick_labels=CONFIGS, showmeans=True)
        ax.set_title(metric)
        ax.set_xticklabels(CONFIGS, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
    plt.tight_layout()
    out = FIGURES / "figure3_distributions.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    n_obs = len(fold[fold["config"] == CONFIGS[0]])
    print(f"  {out.relative_to(PROJECT_ROOT)}  (n = {n_obs} folds per configuration)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", choices=["1", "2", "3"], default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"output resolution (default {DEFAULT_DPI})")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures at {args.dpi} dpi:")
    if args.figure in (None, "1"):
        figure1(args.dpi)
    if args.figure in (None, "2"):
        figure2(args.dpi)
    if args.figure in (None, "3"):
        figure3(args.dpi)


if __name__ == "__main__":
    main()
