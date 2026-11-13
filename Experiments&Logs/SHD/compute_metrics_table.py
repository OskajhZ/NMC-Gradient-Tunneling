#!/usr/bin/env python3
"""Compute mean +/- standard error (STE, not SD) of "Best-step test metrics"
from the five SHD experiment folders (e-prop / FPTT / pp-prop / GT_Ablation /
GT_NMC), each with seeds 42-45, and emit a LaTeX three-line table.

The table has 6 rows x 5 columns:
  header row: Algorithm | Accuracy | Precision | Recall | F1
  data rows : e-prop, FPTT, pp-prop, GT Ablation, GT

Usage: python3 compute_metrics_table.py
Outputs: metrics_table.tex  (LaTeX table, three-line style)
"""

import os
import re
from statistics import mean, stdev

BASE = os.path.dirname(os.path.abspath(__file__))

# folder name -> row label in the table
ALGORITHMS = [
    ("e-prop", "e-prop"),
    ("FPTT", "FPTT"),
    ("pp-prop", "pp-prop"),
    ("GT_Ablation", "GT Ablation"),
    ("GT_NMC", "GT"),
]
SEEDS = ["seed42", "seed43", "seed44", "seed45"]

# column order in the table (header -> metric key)
COLUMNS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
]

LINE_RE = re.compile(
    r"Best-step test metrics: accuracy=([0-9.]+), f1=([0-9.]+), "
    r"precision=([0-9.]+), recall=([0-9.]+)"
)


def parse_metrics(log_path):
    """Return (accuracy, f1, precision, recall) from the first matching line."""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                acc, f1, prec, rec = (float(x) for x in m.groups())
                return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}
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


def ste(values):
    """Standard error of the mean: sample std / sqrt(n)."""
    n = len(values)
    return stdev(values) / (n ** 0.5)


def fmt(mean_val, ste_val):
    """Format as percentage (x100) with two decimals, e.g. 67.53$\\pm$0.36."""
    return f"{mean_val * 100:.2f}$\\pm${ste_val * 100:.2f}"


def main():
    rows = {}      # folder -> {metric: (mean, ste)}
    raw = {}       # folder -> {seed: {metric: value}}
    for folder, _label in ALGORITHMS:
        per_seed = {seed: parse_metrics(find_log(folder, seed)) for seed in SEEDS}
        raw[folder] = per_seed
        rows[folder] = {
            key: (mean([per_seed[s][key] for s in SEEDS]),
                  ste([per_seed[s][key] for s in SEEDS]))
            for _header, key in COLUMNS
        }

    # --- console report (for verification) ---
    for folder, label in ALGORITHMS:
        print(f"== {label} ==")
        for seed in SEEDS:
            v = raw[folder][seed]
            print(f"  {seed}: acc={v['accuracy']:.6f} prec={v['precision']:.6f} "
                  f"rec={v['recall']:.6f} f1={v['f1']:.6f}")
        for header, key in COLUMNS:
            print(f"  {header}: mean={rows[folder][key][0]:.6f} "
                  f"ste={rows[folder][key][1]:.6f}")

    # --- LaTeX three-line table ---
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Test metrics of learning algorithms on SHD in "
                 "percentage (mean$\\pm$standard error over four seeds).}")
    lines.append("\\label{tab:shd-metrics}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append(" & ".join(["\\textbf{Algorithm}"]
                            + [f"\\textbf{{{h}}}" for h, _ in COLUMNS]) + " \\\\")
    lines.append("\\midrule")
    for folder, label in ALGORITHMS:
        cells = [label] + [fmt(*rows[folder][key]) for _h, key in COLUMNS]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = os.path.join(BASE, "metrics_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()
