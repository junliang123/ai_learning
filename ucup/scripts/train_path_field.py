#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ucup_rsrp.data import Cell, discover_cells
from ucup_rsrp.geometry import add_geometry_features
from ucup_rsrp.modeling import PATH_MODEL_FEATURES, add_path_model_features
from ucup_rsrp.path_field import (
    RESIDUAL_SCALE_DB,
    ConditionalPathField,
    PathFieldConfig,
    build_bev_input,
    environment_aware_gradient_loss,
    extract_path_sequence,
    make_scalar_features,
)
from ucup_rsrp.raster import PointCloudRaster


COORDINATE_SCALE_M = 1024.0
VALID_VARIANTS = ("sequence", "bev", "fourier", "moe", "gradient")


@dataclass
class CellSamples:
    cell: Cell
    x: np.ndarray
    y: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    scalars: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class RasterBundle:
    raster: PointCloudRaster
    bev: np.ndarray


class RasterStore:
    def __init__(
        self,
        raster_dir: Path,
        mapping: dict[str, str],
        *,
        capacity: int = 4,
    ) -> None:
        self.raster_dir = raster_dir
        self.mapping = mapping
        self.capacity = capacity
        self.cache: OrderedDict[str, RasterBundle] = OrderedDict()

    def get(self, cell_id: str) -> RasterBundle:
        if cell_id in self.cache:
            self.cache.move_to_end(cell_id)
            return self.cache[cell_id]
        raster = PointCloudRaster.load(self.raster_dir / self.mapping[cell_id])
        bundle = RasterBundle(raster=raster, bev=build_bev_input(raster))
        self.cache[cell_id] = bundle
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conditional path-field ablations on a strict site split."
    )
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--raster-dir", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument("--path-artifact", type=Path, default=Path("artifacts/path_baseline"))
    parser.add_argument(
        "--oof-dir",
        type=Path,
        help="Optional OOF artifact. OOF predictions are used only for training cells.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/path_field_v1"))
    parser.add_argument(
        "--variants",
        default=",".join(VALID_VARIANTS),
        help=f"Comma-separated subset of: {','.join(VALID_VARIANTS)}",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batches-per-cell", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--max-train-points-per-cell", type=int, default=20_000)
    parser.add_argument("--max-valid-points-per-cell", type=int, default=20_000)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--path-width", type=int, default=32)
    parser.add_argument("--condition-width", type=int, default=64)
    parser.add_argument("--bev-base-channels", type=int, default=8)
    parser.add_argument("--fourier-levels", type=int, default=5)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def select_points(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum > 0 and len(frame) > maximum:
        return frame.sample(maximum, random_state=seed)
    return frame


def load_cell_samples(
    cell: Cell,
    *,
    maximum_points: int,
    seed: int,
    raster_path: Path,
    path_model: object,
    oof_prediction_path: Path | None = None,
) -> CellSamples:
    full_oof: np.ndarray | None = None
    all_points = cell.read_points()
    if oof_prediction_path is not None:
        full_oof = np.load(oof_prediction_path, mmap_mode="r")
        if len(full_oof) < len(all_points):
            all_points = all_points.iloc[: len(full_oof)]
    points = select_points(all_points, maximum_points, seed + cell.numeric_id)
    if oof_prediction_path is None:
        featured = add_path_model_features(cell, points, raster_path)
        baseline = np.asarray(
            path_model.predict(featured[PATH_MODEL_FEATURES]), dtype=np.float32
        )
    else:
        assert full_oof is not None
        row_indices = points.index.to_numpy(dtype=np.int64)
        if len(row_indices) and int(np.max(row_indices)) >= len(full_oof):
            raise ValueError(
                f"OOF predictions for {cell.cell_id} have {len(full_oof)} rows, "
                f"but row {int(np.max(row_indices))} was requested."
            )
        baseline = np.asarray(full_oof[row_indices], dtype=np.float32)
        featured = add_geometry_features(
            points,
            base_x=cell.x,
            base_y=cell.y,
            height=cell.height,
            azimuth=cell.azimuth,
            downtilt=cell.downtilt,
            band=cell.band,
        )
    scalars = make_scalar_features(
        featured,
        baseline,
        band=cell.band,
        base_height_m=cell.height,
        downtilt_deg=cell.downtilt,
    )
    return CellSamples(
        cell=cell,
        x=points["x"].to_numpy(dtype=np.float32),
        y=points["y"].to_numpy(dtype=np.float32),
        target=points["rsrp"].to_numpy(dtype=np.float32),
        baseline=baseline,
        scalars=scalars,
    )


def variant_config(name: str, args: argparse.Namespace) -> tuple[PathFieldConfig, float]:
    position = VALID_VARIANTS.index(name)
    config = PathFieldConfig(
        samples=args.samples,
        path_width=args.path_width,
        condition_width=args.condition_width,
        bev_base_channels=args.bev_base_channels,
        use_bev=position >= VALID_VARIANTS.index("bev"),
        fourier_levels=args.fourier_levels if position >= VALID_VARIANTS.index("fourier") else 0,
        experts=args.experts if position >= VALID_VARIANTS.index("moe") else 1,
    )
    gradient_weight = args.gradient_weight if name == "gradient" else 0.0
    return config, gradient_weight


def numpy_batch(
    samples: CellSamples,
    bundle: RasterBundle,
    indices: np.ndarray,
    *,
    path_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_frame = pd.DataFrame({"x": samples.x[indices], "y": samples.y[indices]})
    sequence = extract_path_sequence(
        point_frame,
        bundle.raster,
        antenna_height_m=samples.cell.height,
        band=samples.cell.band,
        samples=path_samples,
    )
    coordinates = np.column_stack([samples.x[indices], samples.y[indices]])
    coordinates = np.clip(coordinates / COORDINATE_SCALE_M, -1.0, 1.0).astype(np.float32)
    target_residual = (samples.target[indices] - samples.baseline[indices]) / RESIDUAL_SCALE_DB
    return sequence, samples.scalars[indices], coordinates, target_residual.astype(np.float32)


def evaluate(
    model: ConditionalPathField,
    datasets: list[CellSamples],
    raster_store: RasterStore,
    *,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    baseline_errors: list[np.ndarray] = []
    model_errors: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    per_cell: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for dataset in datasets:
            bundle = raster_store.get(dataset.cell.cell_id)
            scene_features = None
            if model.config.use_bev:
                bev = torch.from_numpy(bundle.bev).unsqueeze(0)
                scene_features = model.encode_scene(bev)
            prediction_parts: list[np.ndarray] = []
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                indices = np.arange(start, stop)
                sequence, scalars, coordinates, _ = numpy_batch(
                    dataset, bundle, indices, path_samples=model.config.samples
                )
                residual, gate = model.forward_queries(
                    torch.from_numpy(sequence),
                    torch.from_numpy(scalars),
                    torch.from_numpy(coordinates),
                    scene_features=scene_features,
                    return_gate=True,
                )
                prediction_parts.append(residual.numpy())
                gate_parts.append(gate.numpy())
            residual = np.concatenate(prediction_parts) * RESIDUAL_SCALE_DB
            prediction = np.clip(dataset.baseline + residual, -140.0, -30.0)
            baseline_error = np.abs(dataset.target - dataset.baseline)
            model_error = np.abs(dataset.target - prediction)
            baseline_errors.append(baseline_error)
            model_errors.append(model_error)
            truth_parts.append(dataset.target)
            baseline_parts.append(dataset.baseline)
            correction_parts.append(residual)
            per_cell[dataset.cell.cell_id] = {
                "points": len(dataset),
                "baseline_mae": float(np.mean(baseline_error)),
                "path_field_mae": float(np.mean(model_error)),
                "improvement_db": float(np.mean(baseline_error) - np.mean(model_error)),
            }
    all_baseline_errors = np.concatenate(baseline_errors)
    all_model_errors = np.concatenate(model_errors)
    truth = np.concatenate(truth_parts)
    baseline = np.concatenate(baseline_parts)
    correction = np.concatenate(correction_parts)
    residual_scales = np.linspace(0.0, 1.0, 21)
    scaled_mae = {
        f"{scale:.2f}": float(
            np.mean(
                np.abs(
                    truth - np.clip(baseline + scale * correction, -140.0, -30.0)
                )
            )
        )
        for scale in residual_scales
    }
    best_scale = min(scaled_mae, key=scaled_mae.get)
    gate_mean = np.concatenate(gate_parts).mean(axis=0).tolist()
    return {
        "points": int(len(all_model_errors)),
        "baseline_mae": float(np.mean(all_baseline_errors)),
        "path_field_mae": float(np.mean(all_model_errors)),
        "improvement_db": float(np.mean(all_baseline_errors) - np.mean(all_model_errors)),
        "baseline_macro_mae": float(
            np.mean([cell["baseline_mae"] for cell in per_cell.values()])
        ),
        "path_field_macro_mae": float(
            np.mean([cell["path_field_mae"] for cell in per_cell.values()])
        ),
        "best_residual_scale": float(best_scale),
        "best_scaled_mae": scaled_mae[best_scale],
        "residual_scale_mae": scaled_mae,
        "gate_mean": gate_mean,
        "per_cell": per_cell,
    }


def train_variant(
    name: str,
    config: PathFieldConfig,
    gradient_weight: float,
    train_datasets: list[CellSamples],
    valid_datasets: list[CellSamples],
    raster_store: RasterStore,
    args: argparse.Namespace,
) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ConditionalPathField(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    history: list[dict[str, object]] = []
    best_mae = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_datasets))
        losses: list[float] = []
        gradient_losses: list[float] = []
        for step in range(args.steps_per_epoch):
            order_index = (step // args.batches_per_cell) % len(order)
            dataset = train_datasets[int(order[order_index])]
            bundle = raster_store.get(dataset.cell.cell_id)
            indices = rng.integers(0, len(dataset), size=args.batch_size)
            sequence, scalars, coordinates, target = numpy_batch(
                dataset, bundle, indices, path_samples=config.samples
            )
            coordinate_tensor = torch.from_numpy(coordinates)
            coordinate_tensor.requires_grad_(gradient_weight > 0.0)
            bev = torch.from_numpy(bundle.bev).unsqueeze(0) if config.use_bev else None
            optimizer.zero_grad(set_to_none=True)
            prediction, gate = model(
                torch.from_numpy(sequence),
                torch.from_numpy(scalars),
                coordinate_tensor,
                bev=bev,
                return_gate=True,
            )
            data_loss = F.smooth_l1_loss(
                prediction, torch.from_numpy(np.clip(target, -2.0, 2.0)), beta=0.1
            )
            gradient_loss = prediction.new_zeros(())
            if gradient_weight > 0.0:
                gradient_loss = environment_aware_gradient_loss(
                    prediction, coordinate_tensor, torch.from_numpy(sequence)
                )
            mean_gate = gate.mean(dim=0)
            balance_loss = torch.sum(
                mean_gate * torch.log(mean_gate * config.experts + 1e-8)
            )
            loss = data_loss + gradient_weight * gradient_loss + 0.005 * balance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(data_loss.detach()))
            gradient_losses.append(float(gradient_loss.detach()))

        validation = evaluate(
            model, valid_datasets, raster_store, batch_size=args.eval_batch_size
        )
        epoch_result: dict[str, object] = {
            "epoch": epoch,
            "train_residual_loss": float(np.mean(losses)),
            "train_gradient_loss": float(np.mean(gradient_losses)),
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps({"variant": name, **epoch_result}))
        validation_mae = float(validation["path_field_mae"])
        if validation_mae < best_mae:
            best_mae = validation_mae
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    best_validation = evaluate(
        model, valid_datasets, raster_store, batch_size=args.eval_batch_size
    )
    result = {
        "variant": name,
        "config": asdict(config),
        "training_baseline": (
            {"type": "site_oof", "artifact": str(args.oof_dir)}
            if args.oof_dir is not None
            else {"type": "in_sample_model", "artifact": str(args.path_artifact)}
        ),
        "gradient_weight": gradient_weight,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_cells": [dataset.cell.cell_id for dataset in train_datasets],
        "valid_cells": [dataset.cell.cell_id for dataset in valid_datasets],
        "history": history,
        "best_validation": best_validation,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VALID_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if args.samples < 8:
        raise ValueError("At least eight path samples are required.")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((args.path_artifact / "metrics.json").read_text(encoding="utf-8"))
    train_ids = metrics["train_cells"]
    valid_ids = metrics["valid_cells"]
    cells_by_id = {cell.cell_id: cell for cell in discover_cells(args.data)}
    raster_mapping: dict[str, str] = json.loads(
        (args.raster_dir / "cell_to_raster.json").read_text(encoding="utf-8")
    )
    path_model = joblib.load(args.path_artifact / "model.joblib")
    oof_metadata: dict[str, object] | None = None
    if args.oof_dir is not None:
        oof_metadata = json.loads(
            (args.oof_dir / "metadata.json").read_text(encoding="utf-8")
        )
        missing_oof = sorted(set(train_ids) - set(oof_metadata["selected_cells"]))
        if missing_oof:
            raise ValueError(f"OOF artifact is missing training cells: {missing_oof}")

    train_datasets: list[CellSamples] = []
    valid_datasets: list[CellSamples] = []
    selected = [(cell_id, "train") for cell_id in train_ids]
    selected += [(cell_id, "valid") for cell_id in valid_ids]
    for index, (cell_id, split) in enumerate(selected, start=1):
        cell = cells_by_id[cell_id]
        maximum = (
            args.max_train_points_per_cell
            if split == "train"
            else args.max_valid_points_per_cell
        )
        dataset = load_cell_samples(
            cell,
            maximum_points=maximum,
            seed=args.seed,
            raster_path=args.raster_dir / raster_mapping[cell_id],
            path_model=path_model,
            oof_prediction_path=(
                args.oof_dir / "predictions" / f"{cell_id}.npy"
                if args.oof_dir is not None and split == "train"
                else None
            ),
        )
        (train_datasets if split == "train" else valid_datasets).append(dataset)
        print(f"points [{index:02d}/{len(selected)}] {cell_id}: {len(dataset)}")

    raster_store = RasterStore(args.raster_dir, raster_mapping)
    results: dict[str, object] = {}
    for variant in variants:
        config, gradient_weight = variant_config(variant, args)
        print(f"training {variant}: {json.dumps(asdict(config))}")
        results[variant] = train_variant(
            variant,
            config,
            gradient_weight,
            train_datasets,
            valid_datasets,
            raster_store,
            args,
        )
    summary = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": {
            name: result["best_validation"] for name, result in results.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
