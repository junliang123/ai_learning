#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ucup_rsrp.data import discover_cells
from ucup_rsrp.modeling import PATH_MODEL_FEATURES, add_path_model_features
from ucup_rsrp.path_field import (
    RESIDUAL_SCALE_DB,
    ConditionalPathField,
    PathFieldConfig,
    build_bev_input,
    extract_path_sequence,
    make_scalar_features,
)
from ucup_rsrp.raster import PointCloudRaster


COORDINATE_SCALE_M = 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a path-field checkpoint on every point of held-out sites."
    )
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument("--path-artifact", type=Path, default=Path("artifacts/path_baseline"))
    parser.add_argument(
        "--model-artifact", type=Path, default=Path("artifacts/path_field_ablation_v1/gradient")
    )
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model_metrics = json.loads(
        (args.model_artifact / "metrics.json").read_text(encoding="utf-8")
    )
    config = PathFieldConfig(**model_metrics["config"])
    model = ConditionalPathField(config)
    model.load_state_dict(
        torch.load(args.model_artifact / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    path_metrics = json.loads(
        (args.path_artifact / "metrics.json").read_text(encoding="utf-8")
    )
    valid_ids = path_metrics["valid_cells"]
    path_model = joblib.load(args.path_artifact / "model.joblib")
    cells = {cell.cell_id: cell for cell in discover_cells(args.data)}
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )

    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    per_cell: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for index, cell_id in enumerate(valid_ids, start=1):
            cell = cells[cell_id]
            points = cell.read_points()
            raster = PointCloudRaster.load(args.raster_dir / raster_mapping[cell_id])
            featured = add_path_model_features(
                cell, points, args.raster_dir / raster_mapping[cell_id]
            )
            baseline = np.asarray(
                path_model.predict(featured[PATH_MODEL_FEATURES]), dtype=np.float32
            )
            scalars = make_scalar_features(
                featured,
                baseline,
                band=cell.band,
                base_height_m=cell.height,
                downtilt_deg=cell.downtilt,
            )
            scene_features = None
            if config.use_bev:
                bev = torch.from_numpy(build_bev_input(raster)).unsqueeze(0)
                scene_features = model.encode_scene(bev)
            residual_parts: list[np.ndarray] = []
            for start in range(0, len(points), args.batch_size):
                stop = min(start + args.batch_size, len(points))
                point_batch = points.iloc[start:stop]
                sequence = extract_path_sequence(
                    point_batch,
                    raster,
                    antenna_height_m=cell.height,
                    band=cell.band,
                    samples=config.samples,
                )
                coordinates = point_batch[["x", "y"]].to_numpy(dtype=np.float32)
                coordinates = np.clip(
                    coordinates / COORDINATE_SCALE_M, -1.0, 1.0
                ).astype(np.float32)
                residual = model.forward_queries(
                    torch.from_numpy(sequence),
                    torch.from_numpy(scalars[start:stop]),
                    torch.from_numpy(coordinates),
                    scene_features=scene_features,
                )
                residual_parts.append(residual.numpy() * RESIDUAL_SCALE_DB)
            correction = np.concatenate(residual_parts)
            truth = points["rsrp"].to_numpy(dtype=np.float32)
            baseline_mae = float(np.mean(np.abs(truth - baseline)))
            path_field_mae = float(
                np.mean(
                    np.abs(truth - np.clip(baseline + correction, -140.0, -30.0))
                )
            )
            per_cell[cell_id] = {
                "points": len(points),
                "baseline_mae": baseline_mae,
                "path_field_mae": path_field_mae,
                "improvement_db": baseline_mae - path_field_mae,
            }
            truth_parts.append(truth)
            baseline_parts.append(baseline)
            correction_parts.append(correction)
            print(
                f"full validation [{index:02d}/{len(valid_ids)}] {cell_id}: "
                f"{baseline_mae:.4f} -> {path_field_mae:.4f}"
            )

    truth = np.concatenate(truth_parts)
    baseline = np.concatenate(baseline_parts)
    correction = np.concatenate(correction_parts)
    scale_mae: dict[str, float] = {}
    for scale in np.linspace(0.0, 1.0, 21):
        prediction = np.clip(baseline + scale * correction, -140.0, -30.0)
        scale_mae[f"{scale:.2f}"] = float(np.mean(np.abs(truth - prediction)))
    best_scale = min(scale_mae, key=scale_mae.get)
    result = {
        "model_artifact": str(args.model_artifact),
        "points": len(truth),
        "baseline_mae": scale_mae["0.00"],
        "path_field_mae": scale_mae["1.00"],
        "improvement_db": scale_mae["0.00"] - scale_mae["1.00"],
        "best_residual_scale": float(best_scale),
        "best_scaled_mae": scale_mae[best_scale],
        "best_scaled_improvement_db": scale_mae["0.00"] - scale_mae[best_scale],
        "baseline_macro_mae": float(
            np.mean([metrics["baseline_mae"] for metrics in per_cell.values()])
        ),
        "path_field_macro_mae": float(
            np.mean([metrics["path_field_mae"] for metrics in per_cell.values()])
        ),
        "residual_scale_mae": scale_mae,
        "per_cell": per_cell,
    }
    output = args.output or (args.model_artifact / "full_validation.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
