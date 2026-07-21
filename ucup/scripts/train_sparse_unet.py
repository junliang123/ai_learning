#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from ucup_rsrp.data import Cell, discover_cells
from ucup_rsrp.geometry import BAND_MHZ, add_geometry_features
from ucup_rsrp.modeling import GEOMETRY_FEATURES, PATH_MODEL_FEATURES, add_path_model_features
from ucup_rsrp.raster import PointCloudRaster
from ucup_rsrp.unet import TinyUNet


GRID_SIZE = 512
MINIMUM_XY = -1024.0
RESOLUTION_M = 4.0
RESIDUAL_SCALE_DB = 20.0
INPUT_CHANNELS = 14


@dataclass
class CellArrays:
    inputs: np.ndarray
    target: np.ndarray | None
    mask: np.ndarray | None
    label_pixels: np.ndarray | None


@dataclass
class ValidationRecord:
    cell: Cell
    points: pd.DataFrame
    truth: np.ndarray
    geometry_prediction: np.ndarray
    path_prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse-label U-Net RSRP experiment.")
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument(
        "--geometry-artifact", type=Path, default=Path("artifacts/geometry_baseline")
    )
    parser.add_argument("--path-artifact", type=Path, default=Path("artifacts/path_baseline"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sparse_unet_v1"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def geometry_grid(cell: Cell) -> pd.DataFrame:
    centers = MINIMUM_XY + (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) * RESOLUTION_M
    x = np.tile(centers, GRID_SIZE)
    y = np.repeat(centers, GRID_SIZE)
    return add_geometry_features(
        pd.DataFrame({"x": x, "y": y}),
        base_x=cell.x,
        base_y=cell.y,
        height=cell.height,
        azimuth=cell.azimuth,
        downtilt=cell.downtilt,
        band=cell.band,
    )


def aggregate_target(points: pd.DataFrame, baseline_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_index = np.floor((points["x"].to_numpy() - MINIMUM_XY) / RESOLUTION_M).astype(int)
    y_index = np.floor((points["y"].to_numpy() - MINIMUM_XY) / RESOLUTION_M).astype(int)
    valid = (
        (x_index >= 0)
        & (x_index < GRID_SIZE)
        & (y_index >= 0)
        & (y_index < GRID_SIZE)
    )
    flat_index = y_index[valid] * GRID_SIZE + x_index[valid]
    count = np.bincount(flat_index, minlength=GRID_SIZE * GRID_SIZE)
    total = np.bincount(
        flat_index,
        weights=points["rsrp"].to_numpy()[valid],
        minlength=GRID_SIZE * GRID_SIZE,
    )
    mask = count.reshape(GRID_SIZE, GRID_SIZE) > 0
    label_mean = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    occupied = count > 0
    label_mean[occupied] = (total[occupied] / count[occupied]).astype(np.float32)
    residual = label_mean.reshape(GRID_SIZE, GRID_SIZE) - baseline_map
    return (residual / RESIDUAL_SCALE_DB).astype(np.float16), mask


def build_cell_arrays(
    cell: Cell,
    raster_path: Path,
    geometry_model: object,
    *,
    include_target: bool,
) -> CellArrays:
    raster = PointCloudRaster.load(raster_path)
    geometry = geometry_grid(cell)
    baseline_map = np.asarray(
        geometry_model.predict(geometry[GEOMETRY_FEATURES]), dtype=np.float32
    ).reshape(GRID_SIZE, GRID_SIZE)
    occupancy = np.isfinite(raster.max_z)
    band_normalized = np.log10(BAND_MHZ[cell.band] / 800.0) / np.log10(3500.0 / 800.0)

    channels = np.stack(
        [
            np.nan_to_num(raster.max_z / 60.0),
            np.nan_to_num(raster.min_z / 60.0),
            np.nan_to_num(raster.mean_z / 60.0),
            np.nan_to_num(raster.std_z / 20.0),
            raster.log_density / 8.0,
            occupancy.astype(np.float32),
            geometry["log10_distance"].to_numpy().reshape(GRID_SIZE, GRID_SIZE) / 3.2,
            geometry["azimuth_delta_sin"].to_numpy().reshape(GRID_SIZE, GRID_SIZE),
            geometry["azimuth_delta_cos"].to_numpy().reshape(GRID_SIZE, GRID_SIZE),
            np.clip(
                geometry["vertical_delta_deg"].to_numpy().reshape(GRID_SIZE, GRID_SIZE)
                / 30.0,
                -2.0,
                2.0,
            ),
            np.full((GRID_SIZE, GRID_SIZE), band_normalized, dtype=np.float32),
            np.full((GRID_SIZE, GRID_SIZE), cell.height / 60.0, dtype=np.float32),
            np.full((GRID_SIZE, GRID_SIZE), cell.downtilt / 15.0, dtype=np.float32),
            (baseline_map + 100.0) / 30.0,
        ]
    ).astype(np.float16)
    if not include_target:
        return CellArrays(channels, None, None, None)

    points = cell.read_points()
    target, mask = aggregate_target(points, baseline_map)
    label_pixels = np.argwhere(mask)
    return CellArrays(channels, target, mask, label_pixels)


def sample_batch(
    arrays: dict[str, CellArrays],
    cell_ids: list[str],
    *,
    batch_size: int,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for cell_id in rng.choice(cell_ids, size=batch_size, replace=True):
        cell_arrays = arrays[str(cell_id)]
        assert cell_arrays.label_pixels is not None
        assert cell_arrays.target is not None
        assert cell_arrays.mask is not None
        row, column = cell_arrays.label_pixels[rng.integers(len(cell_arrays.label_pixels))]
        row_offset = rng.integers(patch_size // 4, 3 * patch_size // 4)
        column_offset = rng.integers(patch_size // 4, 3 * patch_size // 4)
        top = int(np.clip(row - row_offset, 0, GRID_SIZE - patch_size))
        left = int(np.clip(column - column_offset, 0, GRID_SIZE - patch_size))
        image = cell_arrays.inputs[:, top : top + patch_size, left : left + patch_size]
        target = cell_arrays.target[top : top + patch_size, left : left + patch_size]
        mask = cell_arrays.mask[top : top + patch_size, left : left + patch_size]

        rotations = int(rng.integers(4))
        image = np.rot90(image, rotations, axes=(1, 2))
        target = np.rot90(target, rotations)
        mask = np.rot90(mask, rotations)
        if rng.random() < 0.5:
            image = image[:, :, ::-1]
            target = target[:, ::-1]
            mask = mask[:, ::-1]
        inputs.append(np.ascontiguousarray(image, dtype=np.float32))
        targets.append(np.ascontiguousarray(target, dtype=np.float32))
        masks.append(np.ascontiguousarray(mask, dtype=np.float32))
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(masks)),
    )


def bilinear_sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    column = np.clip((x - MINIMUM_XY) / RESOLUTION_M - 0.5, 0.0, GRID_SIZE - 1.0)
    row = np.clip((y - MINIMUM_XY) / RESOLUTION_M - 0.5, 0.0, GRID_SIZE - 1.0)
    left = np.floor(column).astype(int)
    top = np.floor(row).astype(int)
    right = np.minimum(left + 1, GRID_SIZE - 1)
    bottom = np.minimum(top + 1, GRID_SIZE - 1)
    horizontal = column - left
    vertical = row - top
    return (
        grid[top, left] * (1 - horizontal) * (1 - vertical)
        + grid[top, right] * horizontal * (1 - vertical)
        + grid[bottom, left] * (1 - horizontal) * vertical
        + grid[bottom, right] * horizontal * vertical
    )


def prepare_validation_records(
    valid_cells: list[Cell],
    *,
    raster_dir: Path,
    raster_mapping: dict[str, str],
    geometry_model: object,
    path_model: object,
) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    for cell in valid_cells:
        points = cell.read_points()
        geometry = add_geometry_features(
            points,
            base_x=cell.x,
            base_y=cell.y,
            height=cell.height,
            azimuth=cell.azimuth,
            downtilt=cell.downtilt,
            band=cell.band,
        )
        geometry_prediction = geometry_model.predict(geometry[GEOMETRY_FEATURES])
        path_features = add_path_model_features(
            cell, points, raster_dir / raster_mapping[cell.cell_id]
        )
        path_prediction = path_model.predict(path_features[PATH_MODEL_FEATURES])
        records.append(
            ValidationRecord(
                cell=cell,
                points=points,
                truth=points["rsrp"].to_numpy(),
                geometry_prediction=np.asarray(geometry_prediction),
                path_prediction=np.asarray(path_prediction),
            )
        )
    return records


def evaluate(
    model: nn.Module,
    arrays: dict[str, CellArrays],
    records: list[ValidationRecord],
) -> dict[str, object]:
    model.eval()
    truth_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    path_parts: list[np.ndarray] = []
    unet_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for record in records:
            inputs = torch.from_numpy(arrays[record.cell.cell_id].inputs)
            residual = model(inputs.float().unsqueeze(0))[0, 0].cpu().numpy()
            residual_at_points = bilinear_sample(
                residual,
                record.points["x"].to_numpy(),
                record.points["y"].to_numpy(),
            )
            unet_prediction = record.geometry_prediction + np.clip(
                residual_at_points * RESIDUAL_SCALE_DB, -40.0, 40.0
            )
            truth_parts.append(record.truth)
            geometry_parts.append(record.geometry_prediction)
            path_parts.append(record.path_prediction)
            unet_parts.append(unet_prediction)
            cell_parts.append(np.full(len(record.truth), record.cell.cell_id, dtype=object))
    truth = np.concatenate(truth_parts)
    geometry_prediction = np.concatenate(geometry_parts)
    path_prediction = np.concatenate(path_parts)
    unet_prediction = np.concatenate(unet_parts)
    cell_ids = np.concatenate(cell_parts)

    best_alpha, best_blend_mae = 0.0, float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        blend = path_prediction + alpha * (unet_prediction - path_prediction)
        mae = float(np.mean(np.abs(truth - blend)))
        if mae < best_blend_mae:
            best_alpha, best_blend_mae = float(alpha), mae
    per_cell: dict[str, dict[str, float]] = {}
    for cell_id in np.unique(cell_ids):
        selected = cell_ids == cell_id
        per_cell[str(cell_id)] = {
            "geometry_mae": float(
                np.mean(np.abs(truth[selected] - geometry_prediction[selected]))
            ),
            "path_mae": float(np.mean(np.abs(truth[selected] - path_prediction[selected]))),
            "unet_mae": float(np.mean(np.abs(truth[selected] - unet_prediction[selected]))),
        }
    return {
        "points": len(truth),
        "geometry_mae": float(np.mean(np.abs(truth - geometry_prediction))),
        "path_mae": float(np.mean(np.abs(truth - path_prediction))),
        "unet_mae": float(np.mean(np.abs(truth - unet_prediction))),
        "best_path_unet_blend_mae": best_blend_mae,
        "best_path_unet_blend_alpha": best_alpha,
        "per_cell": per_cell,
    }


def main() -> None:
    args = parse_args()
    if args.patch_size % 8:
        raise ValueError("Patch size must be divisible by 8.")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)

    split_metrics = json.loads(
        (args.geometry_artifact / "metrics.json").read_text(encoding="utf-8")
    )
    train_ids = split_metrics["train_cells"]
    valid_ids = split_metrics["valid_cells"]
    cells = discover_cells(args.data)
    cells_by_id = {cell.cell_id: cell for cell in cells}
    train_cells = [cells_by_id[cell_id] for cell_id in train_ids]
    valid_cells = [cells_by_id[cell_id] for cell_id in valid_ids]
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )
    geometry_model = joblib.load(args.geometry_artifact / "model.joblib")
    path_model = joblib.load(args.path_artifact / "model.joblib")

    arrays: dict[str, CellArrays] = {}
    selected_cells = train_cells + valid_cells
    for index, cell in enumerate(selected_cells, start=1):
        arrays[cell.cell_id] = build_cell_arrays(
            cell,
            args.raster_dir / raster_mapping[cell.cell_id],
            geometry_model,
            include_target=cell.cell_id in train_ids,
        )
        print(f"2D arrays [{index:02d}/{len(selected_cells)}] {cell.cell_id}")

    validation_records = prepare_validation_records(
        valid_cells,
        raster_dir=args.raster_dir,
        raster_mapping=raster_mapping,
        geometry_model=geometry_model,
        path_model=path_model,
    )
    model = TinyUNet(INPUT_CHANNELS, base_channels=args.base_channels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, float | int]] = []
    best_unet_mae = float("inf")
    epochs_without_improvement = 0
    checkpoint = args.output / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            inputs, target, mask = sample_batch(
                arrays,
                train_ids,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, 0]
            loss = (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        metrics = evaluate(model, arrays, validation_records)
        epoch_result = {
            "epoch": epoch,
            "train_masked_mae_scaled": float(np.mean(losses)),
            "geometry_mae": float(metrics["geometry_mae"]),
            "path_mae": float(metrics["path_mae"]),
            "unet_mae": float(metrics["unet_mae"]),
            "blend_mae": float(metrics["best_path_unet_blend_mae"]),
            "blend_alpha": float(metrics["best_path_unet_blend_alpha"]),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))
        if epoch_result["unet_mae"] < best_unet_mae:
            best_unet_mae = epoch_result["unet_mae"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    final_metrics = evaluate(model, arrays, validation_records)
    result = {
        "config": vars(args),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_cells": train_ids,
        "valid_cells": valid_ids,
        "history": history,
        "best_validation": final_metrics,
    }
    result["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
