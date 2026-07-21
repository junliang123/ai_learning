#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ucup_rsrp.data import discover_cells
from ucup_rsrp.modeling import PATH_MODEL_FEATURES, add_path_model_features, make_path_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the path baseline and write a submission.")
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument("--output", type=Path, default=Path("submissions/path_baseline_v1"))
    parser.add_argument("--max-points-per-cell", type=int, default=10_000)
    parser.add_argument("--trees", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )
    cells = discover_cells(args.data)
    train_cells = [cell for cell in cells if cell.split == "train"]
    test_cells = [cell for cell in cells if cell.split == "test"]

    train_frames: list[pd.DataFrame] = []
    for index, cell in enumerate(train_cells, start=1):
        frame = cell.read_points()
        if len(frame) > args.max_points_per_cell:
            frame = frame.sample(
                args.max_points_per_cell, random_state=args.seed + cell.numeric_id
            )
        frame = add_path_model_features(
            cell, frame, args.raster_dir / raster_mapping[cell.cell_id]
        )
        frame["cell_id"] = cell.cell_id
        train_frames.append(frame)
        print(f"train features [{index:02d}/{len(train_cells)}] {cell.cell_id}: {len(frame)}")
    train = pd.concat(train_frames, ignore_index=True)
    cell_counts = train.groupby("cell_id").size()
    weights = train["cell_id"].map(1.0 / cell_counts).to_numpy()
    weights *= len(weights) / weights.sum()

    model = make_path_model(trees=args.trees, seed=args.seed)
    model.fit(train[PATH_MODEL_FEATURES], train["rsrp"], sample_weight=weights)
    joblib.dump(model, args.output / "model.joblib")

    csv_paths: list[Path] = []
    prediction_summary: dict[str, dict[str, float | int]] = {}
    for index, cell in enumerate(test_cells, start=1):
        frame = cell.read_points()
        featured = add_path_model_features(
            cell, frame, args.raster_dir / raster_mapping[cell.cell_id]
        )
        prediction = np.clip(model.predict(featured[PATH_MODEL_FEATURES]), -140.0, -30.0)
        submission = pd.DataFrame(
            {"point_id": frame["point_id"].astype("string"), "rsrp_pred": prediction}
        )
        destination = args.output / f"{cell.numeric_id}.csv"
        submission.to_csv(destination, index=False)
        csv_paths.append(destination)
        prediction_summary[cell.cell_id] = {
            "points": len(frame),
            "minimum": float(np.min(prediction)),
            "mean": float(np.mean(prediction)),
            "maximum": float(np.max(prediction)),
        }
        print(f"test prediction [{index:02d}/{len(test_cells)}] {cell.cell_id}: {len(frame)}")

    archive_path = args.output / "output.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in csv_paths:
            archive.write(path, arcname=path.name)
    metadata = {
        "training_rows": len(train),
        "training_cells": len(train_cells),
        "test_cells": len(test_cells),
        "trees": args.trees,
        "features": PATH_MODEL_FEATURES,
        "predictions": prediction_summary,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {archive_path}")


if __name__ == "__main__":
    main()
