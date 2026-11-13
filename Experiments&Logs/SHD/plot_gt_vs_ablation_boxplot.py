#!/usr/bin/env python3
"""Boxplot of per-seed test Accuracy (%) for GT Ablation vs GT (GT_NMC) on SHD,
with an independent-samples t-test significance annotation.

Figure contract
---------------
- Core conclusion: on SHD, GT achieves higher test accuracy than GT Ablation,
  and the difference is statistically significant.
- Evidence: per-seed Accuracy (%) from 4 random seeds (seed42-seed45) per
  algorithm, parsed from the experiment logs; independent t-test on the two
  groups, significance annotated on the figure.
- Archetype: quantitative grid (single comparison panel).
- Export: single-column width, editable-text SVG/PDF + 600-dpi TIFF.

Usage: python3 plot_gt_vs_ablation_boxplot.py
"""

import os
import re
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ["seed42", "seed43", "seed44", "seed45"]
GROUPS = [("GT_Ablation", "GT Ablation"), ("GT_NMC", "GT")]
COLORS = ["#1964af", "#7daffa"]

LINE_RE = re.compile(
    r"Best-step test metrics: accuracy=([0-9.]+), f1=([0-9.]+), "
    r"precision=([0-9.]+), recall=([0-9.]+)"
)

mpl.rcParams.update({
    # family selector is "sans-serif"; the resolved list points to Times New
    # Roman (user requirement), which is present on this system.
    "font.family": "sans-serif",
    "font.sans-serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                        "DejaVu Serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # editable TrueType text in PDF
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def parse_accuracy(log_path):
    """Test accuracy (%) from the first matching line in the log."""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                return float(m.group(1)) * 100.0
    raise ValueError(f"No 'Best-step test metrics' line found in {log_path}")


def find_log(folder, seed):
    """Top-level .log in the seed dir (excludes LSM_lib subdirectories)."""
    seed_dir = os.path.join(BASE, folder, seed)
    logs = [
        os.path.join(seed_dir, name)
        for name in os.listdir(seed_dir)
        if name.endswith(".log") and os.path.isfile(os.path.join(seed_dir, name))
    ]
    if len(logs) != 1:
        raise AssertionError(
            f"Expected exactly 1 top-level .log in {seed_dir}, got {logs}"
        )
    return logs[0]


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    # --- data ---
    data = {folder: [parse_accuracy(find_log(folder, s)) for s in SEEDS]
            for folder, _ in GROUPS}
    vals = [data[folder] for folder, _ in GROUPS]

    # --- independent t-test (Student's, pooled variance) ---
    # Welch variant also reported in the console for transparency; the effect
    # size here is large enough that both give the same significance level.
    t_stud, p_stud = stats.ttest_ind(vals[0], vals[1], equal_var=True)
    t_welch, p_welch = stats.ttest_ind(vals[0], vals[1], equal_var=False)

    print("== per-seed Accuracy (%) ==")
    for (folder, label), v in zip(GROUPS, vals):
        print(f"  {label}: " + ", ".join(f"{x:.2f}" for x in v)
              + f"  (mean {mean(v):.2f})")
    print(f"Independent t-test (Student, pooled): t = {t_stud:.4f}, "
          f"p = {p_stud:.4e} (df = {2 * len(SEEDS) - 2})")
    print(f"Independent t-test (Welch)          : t = {t_welch:.4f}, "
          f"p = {p_welch:.4e}")
    print(f"Annotation: {significance_stars(p_stud)}")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(3.4, 3.4))  # square canvas (width = height)
    ax.set_axisbelow(True)  # keep the horizontal grid behind the boxes
    ax.grid(axis="y", color="0.85", linewidth=0.6, linestyle="-")
    bp = ax.boxplot(
        vals,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="black"),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # overlay the four seed values (fixed deterministic jitter) for data
    # integrity: n = 4 is too small to hide the raw points behind a box
    jitter = [-0.06, 0.06, -0.02, 0.02]
    for i, v in enumerate(vals):
        ax.scatter(i + 1 + np.array(jitter), v, s=14, color="black", zorder=3,
                   linewidths=0)

    # significance bracket + stars
    ymax = max(max(v) for v in vals)
    ymin = min(min(v) for v in vals)
    span = ymax - ymin
    top = ymax + 0.08 * span
    ax.plot([1, 1, 2, 2], [top, top + 0.05 * span, top + 0.05 * span, top],
            color="black", linewidth=0.8)
    ax.text(1.5, top + 0.09 * span, significance_stars(p_stud),
            ha="center", va="bottom", fontsize=9)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label for _, label in GROUPS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(ymin - 0.15 * span, top + 0.30 * span)

    # fixed margins on the square canvas; exported as-is (no tight crop) so
    # the output image is exactly square
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.10)

    # --- export ---
    stem = os.path.join(BASE, "boxplot_gt_vs_gt_ablation")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.tiff", dpi=600)
    fig.savefig(f"{stem}.png", dpi=300)  # QA preview
    print(f"Wrote {stem}.svg/.pdf/.tiff/.png")


if __name__ == "__main__":
    main()
