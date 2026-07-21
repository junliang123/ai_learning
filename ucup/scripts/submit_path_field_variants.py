#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ucup_rsrp.data import Cell, discover_cells
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
DEFAULT_VARIANTS = ("sequence", "bev", "fourier", "moe", "gradient")


@dataclass
class LoadedVariant:
    name: str
    artifact: Path
    config: PathFieldConfig
    model: ConditionalPathField
    metrics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate submission ZIPs for every conditional path-field ablation."
    )
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("submissions/path_baseline_v1/model.joblib"),
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/path_field_ablation_v1")
    )
    parser.add_argument("--output-root", type=Path, default=Path("submissions"))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--residual-scales", default="0.30,1.00")
    parser.add_argument("--batch-size", type=int, default=2_048)
    return parser.parse_args()


def load_variants(root: Path, names: list[str]) -> list[LoadedVariant]:
    result: list[LoadedVariant] = []
    for name in names:
        artifact = root / name
        metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
        config = PathFieldConfig(**metrics["config"])
        model = ConditionalPathField(config)
        model.load_state_dict(
            torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        result.append(
            LoadedVariant(
                name=name,
                artifact=artifact,
                config=config,
                model=model,
                metrics=metrics,
            )
        )
    sample_counts = {variant.config.samples for variant in result}
    if len(sample_counts) != 1:
        raise ValueError(f"All variants must use the same path sample count: {sample_counts}")
    return result


def scale_tag(scale: float) -> str:
    return f"scale{round(scale * 100):03d}"


def submission_directory(root: Path, variant: str, scale: float) -> Path:
    return root / f"path_field_{variant}_{scale_tag(scale)}_v1"


def predict_cell(
    cell: Cell,
    points: pd.DataFrame,
    raster: PointCloudRaster,
    baseline: np.ndarray,
    scalars: np.ndarray,
    variants: list[LoadedVariant],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    bev = torch.from_numpy(build_bev_input(raster)).unsqueeze(0)
    scene_features: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
    with torch.inference_mode():
        for variant in variants:
            scene_features[variant.name] = (
                variant.model.encode_scene(bev) if variant.config.use_bev else None
            )

        correction_parts: dict[str, list[np.ndarray]] = {
            variant.name: [] for variant in variants
        }
        samples = variants[0].config.samples
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            point_batch = points.iloc[start:stop]
            sequence = extract_path_sequence(
                point_batch,
                raster,
                antenna_height_m=cell.height,
                band=cell.band,
                samples=samples,
            )
            coordinates = point_batch[["x", "y"]].to_numpy(dtype=np.float32)
            coordinates = np.clip(
                coordinates / COORDINATE_SCALE_M, -1.0, 1.0
            ).astype(np.float32)
            sequence_tensor = torch.from_numpy(sequence)
            scalar_tensor = torch.from_numpy(scalars[start:stop])
            coordinate_tensor = torch.from_numpy(coordinates)
            for variant in variants:
                residual = variant.model.forward_queries(
                    sequence_tensor,
                    scalar_tensor,
                    coordinate_tensor,
                    scene_features=scene_features[variant.name],
                )
                correction_parts[variant.name].append(
                    residual.numpy() * RESIDUAL_SCALE_DB
                )
    return {
        name: np.concatenate(parts) for name, parts in correction_parts.items()
    }


def validate_archive(
    archive_path: Path,
    test_cells: list[Cell],
    expected_ids: dict[str, np.ndarray],
) -> None:
    expected_names = {f"{cell.numeric_id}.csv" for cell in test_cells}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if names != expected_names:
            raise ValueError(
                f"Unexpected files in {archive_path}: {sorted(names ^ expected_names)}"
            )
        for cell in test_cells:
            name = f"{cell.numeric_id}.csv"
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, dtype={"point_id": "string"})
            if frame.columns.tolist() != ["point_id", "rsrp_pred"]:
                raise ValueError(f"Unexpected columns in {archive_path}/{name}")
            if not np.array_equal(
                frame["point_id"].to_numpy(dtype=str), expected_ids[cell.cell_id]
            ):
                raise ValueError(f"point_id mismatch in {archive_path}/{name}")
            if not np.isfinite(frame["rsrp_pred"].to_numpy()).all():
                raise ValueError(f"Non-finite prediction in {archive_path}/{name}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    variant_names = [item.strip() for item in args.variants.split(",") if item.strip()]
    scales = [float(item.strip()) for item in args.residual_scales.split(",")]
    if not scales or any(scale < 0.0 or scale > 1.0 for scale in scales):
        raise ValueError("Residual scales must lie in [0, 1].")
    variants = load_variants(args.artifact_root, variant_names)
    base_model = joblib.load(args.base_model)
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )
    test_cells = [cell for cell in discover_cells(args.data) if cell.split == "test"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    output_directories: dict[tuple[str, float], Path] = {}
    summaries: dict[tuple[str, float], dict[str, dict[str, float | int]]] = {}
    for variant in variants:
        for scale in scales:
            key = (variant.name, scale)
            output_directories[key] = submission_directory(
                args.output_root, variant.name, scale
            )
            output_directories[key].mkdir(parents=True, exist_ok=True)
            summaries[key] = {}

    expected_ids: dict[str, np.ndarray] = {}
    for cell_index, cell in enumerate(test_cells, start=1):
        points = cell.read_points()
        expected_ids[cell.cell_id] = points["point_id"].astype("string").to_numpy(dtype=str)
        raster_path = args.raster_dir / raster_mapping[cell.cell_id]
        raster = PointCloudRaster.load(raster_path)
        featured = add_path_model_features(cell, points, raster_path)
        baseline = np.asarray(
            base_model.predict(featured[PATH_MODEL_FEATURES]), dtype=np.float32
        )
        scalars = make_scalar_features(
            featured,
            baseline,
            band=cell.band,
            base_height_m=cell.height,
            downtilt_deg=cell.downtilt,
        )
        corrections = predict_cell(
            cell,
            points,
            raster,
            baseline,
            scalars,
            variants,
            batch_size=args.batch_size,
        )
        for variant in variants:
            correction = corrections[variant.name]
            for scale in scales:
                prediction = np.clip(baseline + scale * correction, -140.0, -30.0)
                key = (variant.name, scale)
                destination = output_directories[key] / f"{cell.numeric_id}.csv"
                pd.DataFrame(
                    {
                        "point_id": points["point_id"].astype("string"),
                        "rsrp_pred": prediction,
                    }
                ).to_csv(destination, index=False)
                summaries[key][cell.cell_id] = {
                    "points": len(points),
                    "minimum": float(np.min(prediction)),
                    "mean": float(np.mean(prediction)),
                    "maximum": float(np.max(prediction)),
                    "mean_absolute_correction": float(
                        np.mean(np.abs(scale * correction))
                    ),
                }
        print(
            f"test predictions [{cell_index:02d}/{len(test_cells)}] "
            f"{cell.cell_id}: {len(points)}"
        )

    for variant in variants:
        for scale in scales:
            key = (variant.name, scale)
            output = output_directories[key]
            archive_path = output / "output.zip"
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for cell in test_cells:
                    csv_path = output / f"{cell.numeric_id}.csv"
                    archive.write(csv_path, arcname=csv_path.name)
            metadata = {
                "variant": variant.name,
                "residual_scale": scale,
                "source_checkpoint": str(variant.artifact / "model.pt"),
                "training_scope": "strict-site-split training cells",
                "base_model": str(args.base_model),
                "model_config": variant.metrics["config"],
                "validation": variant.metrics["best_validation"],
                "predictions": summaries[key],
            }
            (output / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            validate_archive(archive_path, test_cells, expected_ids)
            print(f"validated {archive_path}")


if __name__ == "__main__":
    main()
