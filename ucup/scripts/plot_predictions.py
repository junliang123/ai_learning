#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucup_rsrp.data import discover_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predicted RSRP maps on local coordinates.")
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--predictions", type=Path, default=Path("submissions/path_baseline_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/path_baseline_v1_maps"))
    parser.add_argument("--bins", type=int, default=256)
    return parser.parse_args()


def bin_predictions(frame: pd.DataFrame, bins: int) -> np.ndarray:
    minimum, maximum = -1024.0, 1024.0
    x_index = np.floor((frame["x"].to_numpy() - minimum) / (maximum - minimum) * bins)
    y_index = np.floor((frame["y"].to_numpy() - minimum) / (maximum - minimum) * bins)
    x_index = np.clip(x_index.astype(int), 0, bins - 1)
    y_index = np.clip(y_index.astype(int), 0, bins - 1)
    flat = y_index * bins + x_index
    count = np.bincount(flat, minlength=bins * bins)
    total = np.bincount(
        flat, weights=frame["rsrp_pred"].to_numpy(), minlength=bins * bins
    )
    grid = np.full(bins * bins, np.nan, dtype=np.float32)
    occupied = count > 0
    grid[occupied] = total[occupied] / count[occupied]
    return grid.reshape(bins, bins)


def add_direction(ax: plt.Axes, azimuth: float) -> None:
    radians = math.radians(azimuth)
    ax.scatter([0], [0], marker="*", s=32, color="white", edgecolor="black", linewidth=0.5)
    ax.arrow(
        0,
        0,
        180 * math.sin(radians),
        180 * math.cos(radians),
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        head_width=35,
        length_includes_head=True,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = [cell for cell in discover_cells(args.data) if cell.split == "test"]
    grids: dict[str, np.ndarray] = {}
    for cell in cells:
        points = cell.read_points()
        prediction = pd.read_csv(
            args.predictions / f"{cell.numeric_id}.csv", dtype={"point_id": "string"}
        )
        frame = points.merge(prediction, on="point_id", validate="one_to_one")
        grid = bin_predictions(frame, args.bins)
        grids[cell.cell_id] = grid

        figure, ax = plt.subplots(figsize=(6, 5.4), constrained_layout=True)
        image = ax.imshow(
            grid,
            origin="lower",
            extent=(-1024, 1024, -1024, 1024),
            cmap="viridis",
            vmin=-120,
            vmax=-70,
            interpolation="nearest",
        )
        add_direction(ax, cell.azimuth)
        ax.set(title=f"{cell.cell_id}  {cell.band}", xlabel="Local x (m)", ylabel="Local y (m)")
        figure.colorbar(image, ax=ax, label="Predicted RSRP (dBm)")
        figure.savefig(args.output / f"{cell.cell_id}.png", dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(5, 4, figsize=(13, 15.5), constrained_layout=True)
    last_image = None
    for ax, cell in zip(axes.flat, cells, strict=False):
        last_image = ax.imshow(
            grids[cell.cell_id],
            origin="lower",
            extent=(-1024, 1024, -1024, 1024),
            cmap="viridis",
            vmin=-120,
            vmax=-70,
            interpolation="nearest",
        )
        add_direction(ax, cell.azimuth)
        ax.set_title(f"{cell.cell_id} · {cell.band}")
        ax.set_xticks([-1000, 0, 1000])
        ax.set_yticks([-1000, 0, 1000])
    for ax in axes.flat[len(cells) :]:
        ax.axis("off")
    if last_image is not None:
        figure.colorbar(last_image, ax=axes, label="Predicted RSRP (dBm)", shrink=0.65)
    figure.savefig(args.output / "all_test_cells.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
