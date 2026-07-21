#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ucup_rsrp.data import Cell, discover_cells
from ucup_rsrp.modeling import PATH_MODEL_FEATURES, add_path_model_features, make_path_model
from ucup_rsrp.transfer import add_knn_transfer_features, find_same_sector_matches


ALLOWED_BAND_PAIRS = {frozenset(("800M", "2.1G"))}
TRANSFER_FEATURES = [
    "base_pred",
    "source_rsrp_nearest",
    "source_rsrp_weighted",
    "source_rsrp_median",
    "source_rsrp_std",
    "source_distance_1_m",
    "source_distance_k_m",
    "source_neighbors_5m",
    "source_neighbors_10m",
    "source_neighbors_20m",
    "source_frequency_mhz",
    "target_frequency_mhz",
    "log_frequency_ratio",
    "same_sector_site_distance_m",
    "same_sector_azimuth_difference_deg",
    "distance_m",
    "log10_distance",
    "azimuth_delta_cos",
    "vertical_delta_deg",
    "path_mean_positive_clearance_m",
    "path_blocked_fraction",
    "path_fresnel_blocked_fraction",
    "corridor_max_clearance_m",
    "corridor_blocked_fraction",
    "target_ground_z",
    "target_surface_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and apply co-site, same-sector cross-band label transfer."
    )
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument(
        "--base-submission", type=Path, default=Path("submissions/path_baseline_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("submissions/same_sector_transfer_v1")
    )
    parser.add_argument("--max-points-per-cell", type=int, default=5_000)
    parser.add_argument("--path-trees", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def pair_id(left: Cell, right: Cell) -> str:
    return "|".join(sorted((left.cell_id, right.cell_id)))


def equal_cell_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("cell_id").size()
    weights = frame["cell_id"].map(1.0 / counts).to_numpy()
    return weights * len(weights) / weights.sum()


def make_transfer_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="mae",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=200,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )


def add_transfer_features(
    source: Cell,
    target: Cell,
    target_points: pd.DataFrame,
    *,
    raster_path: Path,
) -> pd.DataFrame:
    featured = add_path_model_features(target, target_points, raster_path)
    source_points = source.read_points()
    knn = add_knn_transfer_features(
        source_points,
        target_points,
        source=source,
        target=target,
    )
    return featured.join(knn)


def macro_cell_mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    errors = pd.DataFrame(
        {
            "cell_id": frame["cell_id"].to_numpy(),
            "error": np.abs(frame["rsrp"].to_numpy() - prediction),
        }
    )
    return float(errors.groupby("cell_id")["error"].mean().mean())


def tune_blend(
    truth: np.ndarray,
    base_prediction: np.ndarray,
    transfer_prediction: np.ndarray,
    nearest_distance: np.ndarray,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for scale in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, math.inf):
        spatial_weight = (
            np.ones_like(nearest_distance)
            if math.isinf(scale)
            else np.exp(-np.square(nearest_distance / scale))
        )
        for alpha in np.linspace(0.0, 1.0, 11):
            weight = alpha * spatial_weight
            blended = base_prediction + weight * (transfer_prediction - base_prediction)
            mae = float(np.mean(np.abs(truth - blended)))
            if best is None or mae < best["mae"]:
                best = {"mae": mae, "scale_m": scale, "alpha": float(alpha)}
    assert best is not None
    return best


def blend_predictions(
    base_prediction: np.ndarray,
    transfer_prediction: np.ndarray,
    nearest_distance: np.ndarray,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    scale = parameters["scale_m"]
    spatial_weight = (
        np.ones_like(nearest_distance)
        if math.isinf(scale)
        else np.exp(-np.square(nearest_distance / scale))
    )
    weight = parameters["alpha"] * spatial_weight
    return base_prediction + weight * (transfer_prediction - base_prediction), weight


def direction_key(source_frequency_mhz: float, target_frequency_mhz: float) -> str:
    return f"{int(source_frequency_mhz)}->{int(target_frequency_mhz)}"


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )
    cells = discover_cells(args.data)
    train_cells = [cell for cell in cells if cell.split == "train"]
    test_cells = [cell for cell in cells if cell.split == "test"]
    directed_train_matches = find_same_sector_matches(
        train_cells,
        train_cells,
        allowed_band_pairs=ALLOWED_BAND_PAIRS,
    )
    train_pair_ids = sorted(
        {pair_id(match.source, match.target) for match in directed_train_matches}
    )
    print("training transfer pairs:", train_pair_ids)
    if len(train_pair_ids) < 3:
        raise RuntimeError("At least three independent cross-band pairs are required.")

    sampled_frames: list[pd.DataFrame] = []
    for index, cell in enumerate(train_cells, start=1):
        frame = cell.read_points()
        if len(frame) > args.max_points_per_cell:
            frame = frame.sample(
                args.max_points_per_cell,
                random_state=args.seed + cell.numeric_id,
            )
        frame = add_path_model_features(
            cell, frame, args.raster_dir / raster_mapping[cell.cell_id]
        )
        frame["cell_id"] = cell.cell_id
        sampled_frames.append(frame)
        print(f"base features [{index:02d}/{len(train_cells)}] {cell.cell_id}: {len(frame)}")
    sampled_train = pd.concat(sampled_frames, ignore_index=True)

    transfer_examples: dict[str, pd.DataFrame] = {}
    for match in directed_train_matches:
        target_points = match.target.read_points()
        featured = add_transfer_features(
            match.source,
            match.target,
            target_points,
            raster_path=args.raster_dir / raster_mapping[match.target.cell_id],
        )
        featured["cell_id"] = match.target.cell_id
        featured["pair_id"] = pair_id(match.source, match.target)
        featured["source_cell_id"] = match.source.cell_id
        transfer_examples[match.target.cell_id] = featured

    for fold_index, held_out_pair in enumerate(train_pair_ids, start=1):
        excluded_cells = held_out_pair.split("|")
        fold_train = sampled_train[~sampled_train["cell_id"].isin(excluded_cells)]
        path_model = make_path_model(
            trees=args.path_trees,
            seed=args.seed + fold_index,
        )
        path_model.fit(
            fold_train[PATH_MODEL_FEATURES],
            fold_train["rsrp"],
            sample_weight=equal_cell_weights(fold_train),
        )
        for cell_id in excluded_cells:
            examples = transfer_examples[cell_id]
            examples["base_pred"] = path_model.predict(examples[PATH_MODEL_FEATURES])
        print(f"OOF path predictions: {held_out_pair}")

    all_examples = pd.concat(transfer_examples.values(), ignore_index=True)
    all_examples["transfer_pred"] = np.nan
    for fold_index, held_out_pair in enumerate(train_pair_ids, start=1):
        calibration_train = all_examples[all_examples["pair_id"] != held_out_pair]
        calibration_valid = all_examples[all_examples["pair_id"] == held_out_pair]
        model = make_transfer_model(args.seed + fold_index)
        model.fit(
            calibration_train[TRANSFER_FEATURES],
            calibration_train["rsrp"],
            sample_weight=equal_cell_weights(calibration_train),
        )
        all_examples.loc[calibration_valid.index, "transfer_pred"] = model.predict(
            calibration_valid[TRANSFER_FEATURES]
        )
        print(f"OOF transfer predictions: {held_out_pair}")

    truth = all_examples["rsrp"].to_numpy()
    base_prediction = all_examples["base_pred"].to_numpy()
    transfer_prediction = all_examples["transfer_pred"].to_numpy()
    nearest_distance = all_examples["source_distance_1_m"].to_numpy()
    base_mae = float(np.mean(np.abs(truth - base_prediction)))
    all_examples["direction"] = [
        direction_key(source, target)
        for source, target in zip(
            all_examples["source_frequency_mhz"],
            all_examples["target_frequency_mhz"],
            strict=True,
        )
    ]
    blend_parameters: dict[str, dict[str, float]] = {}
    best_prediction = base_prediction.copy()
    for direction, group in all_examples.groupby("direction"):
        index = group.index.to_numpy()
        parameters = tune_blend(
            truth[index],
            base_prediction[index],
            transfer_prediction[index],
            nearest_distance[index],
        )
        best_prediction[index], _ = blend_predictions(
            base_prediction[index],
            transfer_prediction[index],
            nearest_distance[index],
            parameters,
        )
        blend_parameters[direction] = parameters

    per_cell: dict[str, dict[str, float | int]] = {}
    for cell_id, group in all_examples.groupby("cell_id"):
        index = group.index.to_numpy()
        per_cell[cell_id] = {
            "points": len(group),
            "base_mae": float(np.mean(np.abs(truth[index] - base_prediction[index]))),
            "transfer_mae": float(np.mean(np.abs(truth[index] - best_prediction[index]))),
            "nearest_source_median_m": float(np.median(nearest_distance[index])),
        }

    transfer_mae = float(np.mean(np.abs(truth - best_prediction)))
    cv_metrics = {
        "training_pairs": train_pair_ids,
        "points": len(all_examples),
        "base_point_mae": base_mae,
        "transfer_point_mae": transfer_mae,
        "improvement_db": base_mae - transfer_mae,
        "base_cell_macro_mae": macro_cell_mae(all_examples, base_prediction),
        "transfer_cell_macro_mae": macro_cell_mae(all_examples, best_prediction),
        "blend_parameters": blend_parameters,
        "per_cell": per_cell,
    }
    print(json.dumps(cv_metrics, indent=2))

    final_transfer_model = make_transfer_model(args.seed)
    final_transfer_model.fit(
        all_examples[TRANSFER_FEATURES],
        all_examples["rsrp"],
        sample_weight=equal_cell_weights(all_examples),
    )
    joblib.dump(final_transfer_model, args.output / "transfer_model.joblib")

    test_matches = find_same_sector_matches(
        train_cells,
        test_cells,
        allowed_band_pairs=ALLOWED_BAND_PAIRS,
    )
    test_match_by_target = {match.target.cell_id: match for match in test_matches}
    applied: dict[str, dict[str, float | str]] = {}
    csv_paths: list[Path] = []
    for cell in test_cells:
        base_csv = pd.read_csv(
            args.base_submission / f"{cell.numeric_id}.csv",
            dtype={"point_id": "string"},
        )
        match = test_match_by_target.get(cell.cell_id)
        if match is not None:
            target_points = cell.read_points()
            featured = add_transfer_features(
                match.source,
                cell,
                target_points,
                raster_path=args.raster_dir / raster_mapping[cell.cell_id],
            )
            featured["base_pred"] = base_csv["rsrp_pred"].to_numpy()
            candidate = final_transfer_model.predict(featured[TRANSFER_FEATURES])
            distance = featured["source_distance_1_m"].to_numpy()
            direction = direction_key(
                featured["source_frequency_mhz"].iloc[0],
                featured["target_frequency_mhz"].iloc[0],
            )
            parameters = blend_parameters[direction]
            transferred, weight = blend_predictions(
                base_csv["rsrp_pred"].to_numpy(),
                candidate,
                distance,
                parameters,
            )
            base_csv["rsrp_pred"] = np.clip(
                transferred,
                -140.0,
                -30.0,
            )
            applied[cell.cell_id] = {
                "source": match.source.cell_id,
                "source_band": match.source.band,
                "target_band": cell.band,
                "mean_blend_weight": float(np.mean(weight)),
                "median_source_distance_m": float(np.median(distance)),
                "blend_alpha": parameters["alpha"],
                "blend_scale_m": parameters["scale_m"],
            }
        destination = args.output / f"{cell.numeric_id}.csv"
        base_csv.to_csv(destination, index=False)
        csv_paths.append(destination)

    archive_path = args.output / "output.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in csv_paths:
            archive.write(path, arcname=path.name)
    result = {"cross_validation": cv_metrics, "applied_test_cells": applied}
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("test matches:", json.dumps(applied, indent=2))
    print("wrote", archive_path)


if __name__ == "__main__":
    main()
