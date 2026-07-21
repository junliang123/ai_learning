#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ucup_rsrp.data import Cell, discover_cells
from ucup_rsrp.geometry import add_geometry_features
from ucup_rsrp.path_features import PATH_FEATURES, extract_path_features
from ucup_rsrp.raster import PointCloudRaster
from ucup_rsrp.sites import assign_site_ids


FEATURES = [
    "distance_m",
    "log10_distance",
    "azimuth_delta_deg",
    "azimuth_delta_sin",
    "azimuth_delta_cos",
    "downward_angle_deg",
    "vertical_delta_deg",
    "frequency_mhz",
    "log10_frequency",
    "free_space_loss_db",
    "base_height_m",
    "downtilt_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a point-cloud-free RSRP baseline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("TrainingData.26UCupSummer"),
        help="Dataset root.",
    )
    parser.add_argument("--max-points-per-cell", type=int, default=5_000)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, default=Path("artifacts/geometry_baseline"))
    parser.add_argument("--use-path-features", action="store_true")
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    return parser.parse_args()


def load_training_cell(
    cell: Cell,
    max_points: int,
    seed: int,
    *,
    raster_path: Path | None = None,
) -> pd.DataFrame:
    frame = cell.read_points()
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed + cell.numeric_id)
    frame = add_geometry_features(
        frame,
        base_x=cell.x,
        base_y=cell.y,
        height=cell.height,
        azimuth=cell.azimuth,
        downtilt=cell.downtilt,
        band=cell.band,
    )
    if raster_path is not None:
        raster = PointCloudRaster.load(raster_path)
        path_frame = extract_path_features(
            frame,
            raster,
            antenna_height_m=cell.height,
            band=cell.band,
        )
        frame = frame.join(path_frame)
    frame["cell_id"] = cell.cell_id
    return frame


def mae_by_cell(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    errors = np.abs(frame["rsrp"].to_numpy() - prediction)
    working = pd.DataFrame({"cell_id": frame["cell_id"].to_numpy(), "error": errors})
    return working.groupby("cell_id")["error"].mean().sort_index().to_dict()


def main() -> None:
    args = parse_args()
    cells = [cell for cell in discover_cells(args.data) if cell.split == "train"]
    site_ids = assign_site_ids(cells)
    raster_mapping: dict[str, str] = {}
    if args.use_path_features:
        raster_mapping = json.loads(
            (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
        )
    frames = [
        load_training_cell(
            cell,
            args.max_points_per_cell,
            args.seed,
            raster_path=(
                args.raster_dir / raster_mapping[cell.cell_id]
                if args.use_path_features
                else None
            ),
        )
        for cell in cells
    ]
    data = pd.concat(frames, ignore_index=True)
    features = FEATURES + (PATH_FEATURES if args.use_path_features else [])
    groups = data["cell_id"].map(site_ids).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.valid_fraction, random_state=args.seed
    )
    train_indices, valid_indices = next(splitter.split(data, groups=groups))
    train = data.iloc[train_indices]
    valid = data.iloc[valid_indices]

    train_cell_counts = train.groupby("cell_id").size()
    train_weights = train["cell_id"].map(1.0 / train_cell_counts).to_numpy()
    train_weights *= len(train_weights) / train_weights.sum()

    model = lgb.LGBMRegressor(
        objective="mae",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=300,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=8,
        verbosity=-1,
    )
    model.fit(
        train[features],
        train["rsrp"],
        sample_weight=train_weights,
        eval_set=[(valid[features], valid["rsrp"])],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)],
    )
    prediction = model.predict(valid[features])
    cell_metrics = mae_by_cell(valid, prediction)
    metrics = {
        "best_iteration": int(model.best_iteration_),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_cells": sorted(train["cell_id"].unique().tolist()),
        "valid_cells": sorted(valid["cell_id"].unique().tolist()),
        "valid_sites": sorted(np.unique(groups[valid_indices]).astype(int).tolist()),
        "point_mae": float(np.mean(np.abs(valid["rsrp"].to_numpy() - prediction))),
        "cell_macro_mae": float(np.mean(list(cell_metrics.values()))),
        "cell_mae": cell_metrics,
        "feature_importance_gain": {
            feature: float(importance)
            for feature, importance in sorted(
                zip(
                    features,
                    model.booster_.feature_importance(importance_type="gain"),
                    strict=True,
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
