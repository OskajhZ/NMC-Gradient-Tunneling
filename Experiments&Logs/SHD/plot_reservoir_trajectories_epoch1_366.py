#!/usr/bin/env python3
"""Two-panel reservoir-trajectory heatmaps for GT_NMC seed42: epoch 1 vs
epoch 366, sharing one colorbar. Appearance follows the reference
Examples/SHD/plot_reservoir_trajectories.py (imshow, viridis, vmin=0, vmax=1,
origin="lower"), combined onto a single 2:1 canvas.

Figure contract
---------------
- Core conclusion: reservoir output evolves from a compressed, near-uniform
  state in epoch 1 to a differentiated trajectory by epoch 366.
- Evidence: epoch_0001 / epoch_0366 trajectory recordings of the same sample.
- Export: 2:1 canvas exported as-is (no tight crop) so the output aspect is
  exactly 2:1; editable-text SVG/PDF + 600-dpi TIFF.

Usage: python3 plot_reservoir_trajectories_epoch1_366.py
"""

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(
    "/home/xiangnan/E/EDoc/Research/神经形态芯片/GradientTunneling",
    "Logs/正式实验/SHD/GT_NMC/seed42/logs/shd/log_0/reservoir_trajectories",
)
EPOCHS = [(1, "Firing Rate Trajectory in SHD Epoch 1"),
          (366, "Firing Rate Trajectory in SHD Epoch 366")]

mpl.rcParams.update({
    # family selector is "sans-serif"; the resolved list points to Times New
    # Roman, which is present on this system.
    "font.family": "sans-serif",
    "font.sans-serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                        "DejaVu Serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # editable TrueType text in PDF
    "font.size": 8,
})


def load_trajectory(epoch):
    """(trajectory, recorded_epoch) for the single .npz of the given epoch."""
    npz_files = glob.glob(os.path.join(TRACE_DIR, f"epoch_{epoch:04d}_*.npz"))
    if len(npz_files) != 1:
        raise AssertionError(
            f"Expected exactly 1 .npz for epoch {epoch}, got {npz_files}"
        )
    with np.load(npz_files[0]) as data:
        return data["trajectory"], int(data["step"])


def main():
    fig = plt.figure(figsize=(7.2, 3.6))  # 2:1 canvas
    # explicit axes positions: wide gap between the two heatmaps, colorbar
    # close to the right panel (its tick labels extend into the right margin)
    axes = [
        fig.add_axes([0.07, 0.13, 0.35, 0.74]),
        fig.add_axes([0.53, 0.13, 0.35, 0.74]),
    ]
    cax = fig.add_axes([0.925, 0.13, 0.02, 0.74])

    # shared color scale tightened to the observed data range (both epochs),
    # so per-epoch color differences are more visible than the fixed 0-1 range
    data = {epoch: load_trajectory(epoch)[0] for epoch, _ in EPOCHS}
    vmin = min(t.min() for t in data.values())
    vmax = max(t.max() for t in data.values())
    print(f"Data range {vmin:.4f}-{vmax:.4f}; "
          f"colorbar range tightened to [{vmin:.4f}, {vmax:.4f}]")

    ims = []
    for ax, (epoch, title) in zip(axes, EPOCHS):
        trajectory = data[epoch]
        im = ax.imshow(
            trajectory,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron Index")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ims.append(im)

    fig.colorbar(ims[0], cax=cax)

    # export the full 2:1 canvas as-is (no tight crop) so the output aspect
    # is exactly 2:1
    stem = os.path.join(BASE, "trajectory_epoch_1_vs_366")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.tiff", dpi=600)
    fig.savefig(f"{stem}.png", dpi=300)  # QA preview
    print(f"Wrote {stem}.svg/.pdf/.tiff/.png")


if __name__ == "__main__":
    main()
