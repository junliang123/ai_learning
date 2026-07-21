from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .data import Cell
from .geometry import add_geometry_features
from .path_features import PATH_FEATURES, extract_path_features
from .raster import PointCloudRaster


GEOMETRY_FEATURES = [
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
PATH_MODEL_FEATURES = GEOMETRY_FEATURES + PATH_FEATURES


def add_path_model_features(
    cell: Cell,
    frame: pd.DataFrame,
    raster_path: str | Path,
) -> pd.DataFrame:
    frame = add_geometry_features(
        frame,
        base_x=cell.x,
        base_y=cell.y,
        height=cell.height,
        azimuth=cell.azimuth,
        downtilt=cell.downtilt,
        band=cell.band,
    )
    raster = PointCloudRaster.load(raster_path)
    return frame.join(
        extract_path_features(
            frame,
            raster,
            antenna_height_m=cell.height,
            band=cell.band,
        )
    )


def make_path_model(*, trees: int, seed: int, jobs: int = 8) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="mae",
        n_estimators=trees,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=300,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=jobs,
        verbosity=-1,
    )

