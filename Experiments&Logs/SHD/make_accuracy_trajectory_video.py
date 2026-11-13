#!/usr/bin/env python3
"""Animate GT_NMC seed42 SHD training into a 1:1 MP4 video.

Top panel: cumulative TensorBoard scalar curve (default tag test/accuracy)
read from the training event file, up to the current epoch.
Bottom panel: the per-epoch reservoir trajectory heatmap, drawn with the same
appearance as Examples/SHD/plot_reservoir_trajectories.py (imshow, viridis,
vmin=0, vmax=1, origin="lower") plus the labels used in the SHD figure script.

The two panels are composited into square 1000x1000 frames and encoded with
ffmpeg (H.264 / yuv420p), following
downloads/log_0_o/generate_accuracy_trajectory_gif.py but outputting MP4.

Usage: python3 make_accuracy_trajectory_video.py [--duration 15] [--tag test/accuracy]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from fractions import Fraction
from io import BytesIO
from pathlib import Path

# Avoid Matplotlib cache warnings on read-only home directories.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = BASE / "GT_NMC" / "seed42" / "logs" / "shd" / "log_0"
TRACE_DIR = LOG_DIR / "reservoir_trajectories"

WIDTH = 1000
HEIGHT = 1000          # 1:1 video
CURVE_H = 333          # top panel (1/3)
TRAJ_H = HEIGHT - CURVE_H   # bottom panel (2/3), ratio 1:2
TAG_DISPLAY = "Test Accuracy"   # display form of the TB tag (title-cased)

mpl.rcParams.update({
    # family selector is "sans-serif"; the resolved list points to Times New
    # Roman, which is present on this system.
    "font.family": "sans-serif",
    "font.sans-serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                        "DejaVu Serif"],
    "svg.fonttype": "none",     # editable text if SVG is ever exported
    "pdf.fonttype": 42,         # editable text if PDF is ever exported
    "font.size": 12,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-glob", default="events.out.tfevents.*",
                        help="Glob for TensorBoard event files under LOG_DIR.")
    parser.add_argument("--tag", default="test/accuracy",
                        help="TensorBoard scalar tag for the top curve.")
    parser.add_argument("--output", default=BASE / "accuracy_trajectory_gt_nmc_seed42.mp4",
                        help="Output MP4 path.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Total video duration in seconds.")
    parser.add_argument("--first-duration", type=float, default=1.0,
                        help="Hold time of the first frame in seconds.")
    return parser.parse_args()


def load_scalar(event_glob: str, tag: str) -> dict[int, float]:
    event_files = sorted(LOG_DIR.glob(event_glob))
    if not event_files:
        raise FileNotFoundError(f"No event files matched {event_glob!r} under {LOG_DIR}")
    values: dict[int, float] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file),
                                       size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            values[int(event.step)] = float(event.value)
    if not values:
        raise RuntimeError(f"TensorBoard scalar tag {tag!r} was not found")
    return values


def load_trajectory(epoch: int) -> np.ndarray:
    npz_files = sorted(TRACE_DIR.glob(f"epoch_{epoch:04d}_*.npz"))
    if len(npz_files) != 1:
        raise AssertionError(
            f"Expected exactly 1 .npz for epoch {epoch}, got {npz_files}"
        )
    with np.load(npz_files[0]) as data:
        return data["trajectory"]


def make_curve_panel(epochs: list[int], values: dict[int, float],
                     current_epoch: int, y_lo: float, y_hi: float) -> Image.Image:
    visible_epochs = [e for e in epochs if e <= current_epoch]
    visible_values = [values[e] for e in visible_epochs]

    fig = plt.figure(figsize=(WIDTH / 100, CURVE_H / 100), dpi=100,
                     facecolor="white")
    # compact margins: bottom fits the xlabel + tick labels, top fits the title
    ax = fig.add_axes((0.08, 0.15, 0.88, 0.70))
    ax.plot(visible_epochs, visible_values, color="#1677ff", linewidth=2.4)
    ax.scatter([visible_epochs[-1]], [visible_values[-1]],
               color="#e84a5f", edgecolors="white", linewidths=1.0, s=38,
               zorder=3)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(y_lo, y_hi)   # tightened to the observed min/max (see main)
    ax.set_xlabel("Epoch", fontsize=12, labelpad=2)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, labelpad=2)
    ax.set_title(
        f"{TAG_DISPLAY} Through SHD Epoch {current_epoch}   |   "
        f"Current: {visible_values[-1] * 100:.2f}%",
        fontsize=18, pad=4,
    )
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=10.5, length=3)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _pos: f"{v * 100:.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    return _fig_to_pil(fig)


def make_traj_panel(trajectory: np.ndarray, epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(WIDTH / 100, TRAJ_H / 100), dpi=100,
                     facecolor="white")
    # heatmap with colorbar on the right; compact top margin (title only)
    ax = fig.add_axes((0.07, 0.08, 0.80, 0.87))
    im = ax.imshow(trajectory, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Time Step", fontsize=12, labelpad=2)
    ax.set_ylabel("Neuron Index", fontsize=12, labelpad=2)
    ax.set_title(f"Firing Rate Trajectory in SHD Epoch {epoch}",
                 fontsize=18, pad=4)
    ax.tick_params(labelsize=10.5, length=3)

    cax = fig.add_axes((0.90, 0.08, 0.02, 0.87))
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10.5)

    return _fig_to_pil(fig)


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as panel:
        return panel.convert("RGB").copy()


def encode_mp4(frame_dir: Path, output: Path, fps: Fraction) -> None:
    """CFR H.264 from consecutive frame_%04d.png images at a fractional fps.

    The first frame is written multiple times (``n_first`` copies) so it holds
    for ~1 s; every remaining epoch is exactly one frame, so all epochs get
    identical screen time. fps = (n_epochs - 1) / (total - first) keeps the
    remaining frames at exactly 9/365 s each.
    """
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", f"{fps.numerator}/{fps.denominator}",
         "-i", str(frame_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart",
         str(output)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = BASE / output

    values = load_scalar(args.event_glob, args.tag)
    epochs = sorted(values)

    # tightened y range from the observed accuracy min/max (5% padding)
    acc_vals = [values[e] for e in epochs]
    acc_lo, acc_hi = min(acc_vals), max(acc_vals)
    pad = 0.05 * (acc_hi - acc_lo)
    y_lo, y_hi = acc_lo - pad, acc_hi + pad
    print(f"Accuracy range {acc_lo:.4f}-{acc_hi:.4f}; "
          f"tightened ylim {y_lo:.4f}-{y_hi:.4f} "
          f"({y_lo*100:.1f}%-{y_hi*100:.1f}%)")

    output.parent.mkdir(parents=True, exist_ok=True)

    # one frame per remaining epoch; first frame duplicated to hold ~1 s.
    # fps = 365/9 -> each remaining epoch is exactly 9/365 s (sum 9 s).
    fps = Fraction(len(epochs) - 1) / Fraction(
        str(args.duration - args.first_duration))
    n_first = int(round(args.first_duration * float(fps)))

    with tempfile.TemporaryDirectory(prefix="training_video_") as temp_name:
        temp_dir = Path(temp_name)
        frame_number = 0
        for epoch in epochs:
            curve = make_curve_panel(epochs, values, epoch, y_lo, y_hi)
            traj = make_traj_panel(load_trajectory(epoch), epoch)
            frame = Image.new("RGB", (WIDTH, HEIGHT), "white")
            frame.paste(curve, (0, 0))
            frame.paste(traj, (0, CURVE_H))
            copies = n_first if epoch == epochs[0] else 1
            for _ in range(copies):
                frame.save(temp_dir / f"frame_{frame_number:04d}.png",
                           optimize=True)
                frame_number += 1

        encode_mp4(temp_dir, output, fps)

    total_dur = frame_number / float(fps)
    first_dur = n_first / float(fps)
    rest_dur = (len(epochs) - 1) / float(fps)
    print(f"Created {output}: {frame_number} frames at {fps} fps "
          f"({float(fps):.3f}): first frame x{n_first} = {first_dur:.3f}s, "
          f"remaining {len(epochs) - 1} x1 = {rest_dur:.3f}s, "
          f"total {total_dur:.3f}s, {WIDTH}x{HEIGHT}.")


if __name__ == "__main__":
    main()
