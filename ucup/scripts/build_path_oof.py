#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ucup_rsrp.data import Cell, discover_cells
from ucup_rsrp.modeling import PATH_MODEL_FEATURES, add_path_model_features, make_path_model
from ucup_rsrp.oof import cell_balanced_weights, make_site_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build physical-site OOF LightGBM predictions for path-field residuals."
    )
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument(
        "--split-metrics", type=Path, default=Path("artifacts/path_baseline/metrics.json")
    )
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/path_oof_strict_v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-train-points-per-cell", type=int, default=5_000)
    parser.add_argument(
        "--max-prediction-points-per-cell",
        type=int,
        default=0,
        help="Zero predicts every point; a positive value is intended only for smoke tests.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=100_000)
    parser.add_argument("--trees", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--skip-full-model", action="store_true")
    return parser.parse_args()


def point_id_digest(point_ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in point_ids.astype("string"):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cache_cell_features(
    cell: Cell,
    *,
    data_limit: int,
    raster_path: Path,
    feature_dir: Path,
    target_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    feature_path = feature_dir / f"{cell.cell_id}.npy"
    target_path = target_dir / f"{cell.cell_id}.npy"
    read_kwargs = {"nrows": data_limit} if data_limit > 0 else {}
    points = cell.read_points(**read_kwargs)
    expected_shape = (len(points), len(PATH_MODEL_FEATURES))
    cache_is_valid = False
    if not force and feature_path.is_file() and target_path.is_file():
        cached_features = np.load(feature_path, mmap_mode="r")
        cached_target = np.load(target_path, mmap_mode="r")
        cache_is_valid = (
            cached_features.shape == expected_shape
            and cached_target.shape == (len(points),)
        )
    if not cache_is_valid:
        featured = add_path_model_features(cell, points, raster_path)
        features = featured[PATH_MODEL_FEATURES].to_numpy(dtype=np.float32)
        target = points["rsrp"].to_numpy(dtype=np.float32)
        np.save(feature_path, features)
        np.save(target_path, target)
    return {
        "rows": len(points),
        "point_id_sha256": point_id_digest(points["point_id"]),
        "feature_file": str(feature_path),
        "target_file": str(target_path),
    }


def sampled_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=maximum, replace=False))


def training_arrays(
    cells: list[Cell],
    *,
    feature_dir: Path,
    target_dir: Path,
    maximum_per_cell: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    for cell in cells:
        features = np.load(feature_dir / f"{cell.cell_id}.npy", mmap_mode="r")
        target = np.load(target_dir / f"{cell.cell_id}.npy", mmap_mode="r")
        indices = sampled_indices(
            len(target), maximum_per_cell, seed + cell.numeric_id
        )
        feature_parts.append(np.asarray(features[indices], dtype=np.float32))
        target_parts.append(np.asarray(target[indices], dtype=np.float32))
        cell_parts.append(np.full(len(indices), cell.cell_id, dtype=object))
    return (
        np.concatenate(feature_parts),
        np.concatenate(target_parts),
        np.concatenate(cell_parts),
    )


def predict_feature_file(
    model: object,
    feature_path: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    features = np.load(feature_path, mmap_mode="r")
    prediction = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        frame = pd.DataFrame(
            features[start:stop], columns=PATH_MODEL_FEATURES, copy=False
        )
        prediction[start:stop] = model.predict(frame)
    return prediction


def fit_path_model(
    cells: list[Cell],
    *,
    feature_dir: Path,
    target_dir: Path,
    maximum_per_cell: int,
    trees: int,
    seed: int,
) -> tuple[object, int]:
    features, target, cell_ids = training_arrays(
        cells,
        feature_dir=feature_dir,
        target_dir=target_dir,
        maximum_per_cell=maximum_per_cell,
        seed=seed,
    )
    weights = cell_balanced_weights(cell_ids)
    model = make_path_model(trees=trees, seed=seed)
    model.fit(
        pd.DataFrame(features, columns=PATH_MODEL_FEATURES, copy=False),
        target,
        sample_weight=weights,
    )
    return model, len(target)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "features"
    target_dir = args.output / "targets"
    prediction_dir = args.output / "predictions"
    model_dir = args.output / "models"
    for directory in (feature_dir, target_dir, prediction_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_train_cells = [cell for cell in discover_cells(args.data) if cell.split == "train"]
    split_metrics = json.loads(args.split_metrics.read_text(encoding="utf-8"))
    selected_ids = (
        {cell.cell_id for cell in all_train_cells}
        if args.all_cells
        else set(split_metrics["train_cells"])
    )
    cells = [cell for cell in all_train_cells if cell.cell_id in selected_ids]
    trees = args.trees or int(split_metrics["best_iteration"])
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )

    manifest: dict[str, dict[str, object]] = {}
    for index, cell in enumerate(cells, start=1):
        manifest[cell.cell_id] = cache_cell_features(
            cell,
            data_limit=args.max_prediction_points_per_cell,
            raster_path=args.raster_dir / raster_mapping[cell.cell_id],
            feature_dir=feature_dir,
            target_dir=target_dir,
        )
        print(f"feature cache [{index:02d}/{len(cells)}] {cell.cell_id}")

    fold_by_cell = make_site_folds(cells, n_splits=args.folds, seed=args.seed)
    fold_metrics: dict[str, dict[str, object]] = {}
    all_errors: list[np.ndarray] = []
    cell_mae: dict[str, float] = {}
    for fold in range(args.folds):
        train_cells = [cell for cell in cells if fold_by_cell[cell.cell_id] != fold]
        valid_cells = [cell for cell in cells if fold_by_cell[cell.cell_id] == fold]
        model, training_rows = fit_path_model(
            train_cells,
            feature_dir=feature_dir,
            target_dir=target_dir,
            maximum_per_cell=args.max_train_points_per_cell,
            trees=trees,
            seed=args.seed + fold,
        )
        joblib.dump(model, model_dir / f"fold_{fold}.joblib")
        fold_errors: list[np.ndarray] = []
        for cell in valid_cells:
            prediction = predict_feature_file(
                model,
                feature_dir / f"{cell.cell_id}.npy",
                batch_size=args.prediction_batch_size,
            )
            np.save(prediction_dir / f"{cell.cell_id}.npy", prediction)
            target = np.load(target_dir / f"{cell.cell_id}.npy", mmap_mode="r")
            error = np.abs(np.asarray(target) - prediction)
            fold_errors.append(error)
            all_errors.append(error)
            cell_mae[cell.cell_id] = float(np.mean(error))
        fold_error = np.concatenate(fold_errors)
        fold_metrics[str(fold)] = {
            "training_rows": training_rows,
            "train_cells": [cell.cell_id for cell in train_cells],
            "valid_cells": [cell.cell_id for cell in valid_cells],
            "valid_rows": len(fold_error),
            "mae": float(np.mean(fold_error)),
        }
        print(
            f"OOF fold {fold}: train={training_rows}, valid={len(fold_error)}, "
            f"mae={np.mean(fold_error):.6f}"
        )

    full_model_rows = 0
    if not args.skip_full_model:
        full_model, full_model_rows = fit_path_model(
            cells,
            feature_dir=feature_dir,
            target_dir=target_dir,
            maximum_per_cell=args.max_train_points_per_cell,
            trees=trees,
            seed=args.seed,
        )
        joblib.dump(full_model, args.output / "full_model.joblib")

    combined_error = np.concatenate(all_errors)
    metadata = {
        "mode": "all_cells" if args.all_cells else "strict_train_cells_only",
        "features": PATH_MODEL_FEATURES,
        "trees": trees,
        "folds": args.folds,
        "max_train_points_per_cell": args.max_train_points_per_cell,
        "max_prediction_points_per_cell": args.max_prediction_points_per_cell,
        "selected_cells": [cell.cell_id for cell in cells],
        "fold_by_cell": fold_by_cell,
        "fold_metrics": fold_metrics,
        "full_model_rows": full_model_rows,
        "oof_rows": len(combined_error),
        "oof_mae": float(np.mean(combined_error)),
        "oof_macro_mae": float(np.mean(list(cell_mae.values()))),
        "cell_mae": cell_mae,
        "cache_manifest": manifest,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "oof_rows": metadata["oof_rows"],
                "oof_mae": metadata["oof_mae"],
                "oof_macro_mae": metadata["oof_macro_mae"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
