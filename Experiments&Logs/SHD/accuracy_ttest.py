#!/usr/bin/env python3
"""Student's t-tests on test-set Accuracy between GT (folder GT_NMC) and the
e-prop / FPTT baselines, using the four random seeds (seed42-seed45).

For each baseline-vs-GT comparison, two variants are reported:
  - Paired t-test (scipy.stats.ttest_rel): per-seed pairing (seed42 vs seed42,
    ...), df = n - 1. Seeds are matched across algorithms, so this removes
    seed-level variance and is the more powerful choice.
  - Independent-sample Student's t-test (scipy.stats.ttest_ind,
    equal_var=True): classical two-sample t-test with pooled variance,
    df = 2n - 2.

Both p-values are two-tailed. n = 4 seeds per algorithm.

Usage: python3 accuracy_ttest.py
"""

import os
import re
from statistics import mean
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ["seed42", "seed43", "seed44", "seed45"]

# matches the "Best-step test metrics" line in each experiment log
LINE_RE = re.compile(
    r"Best-step test metrics: accuracy=([0-9.]+), f1=([0-9.]+), "
    r"precision=([0-9.]+), recall=([0-9.]+)"
)


def parse_accuracy(log_path):
    """Accuracy from the first matching line in the log."""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                return float(m.group(1))
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


def accuracy_per_seed(folder):
    """{seed: accuracy} for the four seeds of one algorithm folder."""
    return {seed: parse_accuracy(find_log(folder, seed)) for seed in SEEDS}


def fmt_p(p):
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def report(baseline, gt_vals):
    """Paired + independent t-test of baseline accuracy vs GT accuracy."""
    base_vals = [accuracy_per_seed(baseline)[s] for s in SEEDS]

    t_paired, p_paired = stats.ttest_rel(base_vals, gt_vals)
    t_ind, p_ind = stats.ttest_ind(base_vals, gt_vals, equal_var=True)

    print(f"== {baseline} vs GT (n = {len(SEEDS)}) ==")
    print(f"  {baseline} seed accuracies: "
          + ", ".join(f"{v:.4f}" for v in base_vals))
    print(f"  GT      seed accuracies: "
          + ", ".join(f"{v:.4f}" for v in gt_vals))
    print(f"  mean acc  {baseline}: {mean(base_vals):.4f}   GT: {mean(gt_vals):.4f}"
          f"   diff (GT - {baseline}): {mean(gt_vals) - mean(base_vals):+.4f}")
    print(f"  Paired t-test        : t = {t_paired:.4f}, "
          f"p = {fmt_p(p_paired)} (df = {len(SEEDS) - 1})")
    print(f"  Independent t-test   : t = {t_ind:.4f}, "
          f"p = {fmt_p(p_ind)} (df = {2 * len(SEEDS) - 2})")
    print()


def main():
    gt_vals = [accuracy_per_seed("GT_NMC")[s] for s in SEEDS]
    for baseline in ("e-prop", "FPTT"):
        report(baseline, gt_vals)


if __name__ == "__main__":
    main()
