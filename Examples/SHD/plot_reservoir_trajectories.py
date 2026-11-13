"""Render reservoir-trajectory heatmaps from data saved by Examples/SHD/shd.py.

During training, ``SHDTrainer`` records one ``.npz`` per epoch containing the
reservoir output over the training-visible 50-step window (see
``_record_reservoir_trajectory``). This script reads a trajectory data folder
and writes a heatmap PNG per ``.npz``, with the same appearance the training
code used to produce inline.

Usage:
    python Examples/SHD/plot_reservoir_trajectories.py --trace-dir logs/shd/reservoir_trajectories
    python Examples/SHD/plot_reservoir_trajectories.py --trace-dir logs/shd/reservoir_trajectories --output-dir logs/shd/trajectory_figures
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_one_npz(npz_path: str, output_dir: str) -> None:
    with np.load(npz_path) as data:
        trajectory = data["trajectory"]
        step = int(data["step"])
        sample_idx = int(data["sample_idx"])
        label = int(data["label"])

    os.makedirs(output_dir, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(npz_path))[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(
        trajectory,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reservoir neuron index")
    ax.set_title(
        f"Reservoir trajectory - epoch {step}, sample {sample_idx}, label {label}"
    )
    fig.colorbar(heatmap, ax=ax, label="Reservoir output")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{file_stem}.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render reservoir-trajectory heatmaps from recorded .npz data."
    )
    parser.add_argument(
        "--trace-dir",
        default="logs/shd/reservoir_trajectories",
        help="Directory containing the epoch_*.npz trajectory files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the rendered PNGs (default: same as --trace-dir).",
    )
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.trace_dir, "*.npz")))
    if not npz_files:
        raise SystemExit(f"No .npz trajectory files found in {args.trace_dir}")
    output_dir = args.output_dir or args.trace_dir
    for npz_path in npz_files:
        plot_one_npz(npz_path, output_dir)
        print(f"Saved {os.path.basename(npz_path)} -> {output_dir}")
    print(f"Rendered {len(npz_files)} trajectory figure(s).")


if __name__ == "__main__":
    main()
