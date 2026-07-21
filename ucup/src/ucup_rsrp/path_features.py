from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from .geometry import BAND_MHZ
from .raster import PointCloudRaster


PATH_FEATURES = [
    "path_valid_fraction",
    "path_max_clearance_m",
    "path_mean_positive_clearance_m",
    "path_blocked_fraction",
    "path_first_blockage_fraction",
    "path_last_blockage_fraction",
    "path_max_fresnel_intrusion_m",
    "path_fresnel_blocked_fraction",
    "path_mean_max_z",
    "path_max_max_z",
    "path_mean_log_density",
    "corridor_max_clearance_m",
    "corridor_blocked_fraction",
    "target_ground_z",
    "target_surface_z",
    "target_object_height_m",
    "target_log_density",
]


def _nanmax_rows(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.full(values.shape[0], fallback, dtype=np.float32)
    rows = valid.any(axis=1)
    result[rows] = np.nanmax(values[rows], axis=1)
    return result


def extract_path_features(
    points: pd.DataFrame,
    raster: PointCloudRaster,
    *,
    antenna_height_m: float,
    band: str,
    receiver_height_m: float = 1.5,
    samples: int = 96,
    corridor_radius_pixels: int = 2,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Extract center-ray and narrow-corridor obstruction statistics for every query point."""
    x_all = points["x"].to_numpy(dtype=np.float64, copy=False)
    y_all = points["y"].to_numpy(dtype=np.float64, copy=False)
    output = np.empty((len(points), len(PATH_FEATURES)), dtype=np.float32)
    fractions = np.linspace(0.02, 0.98, samples, dtype=np.float32)
    wavelength_m = 299_792_458.0 / (BAND_MHZ[band] * 1_000_000.0)
    corridor_max = maximum_filter(
        np.nan_to_num(raster.max_z, nan=-1_000.0),
        size=2 * corridor_radius_pixels + 1,
        mode="nearest",
    )

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        x = x_all[start:stop]
        y = y_all[start:stop]
        distance = np.maximum(np.hypot(x, y), 1.0)
        sample_x = x[:, None] * fractions[None, :]
        sample_y = y[:, None] * fractions[None, :]
        x_index = np.floor(
            (sample_x - raster.minimum_xy) / raster.resolution_m
        ).astype(np.int32)
        y_index = np.floor(
            (sample_y - raster.minimum_xy) / raster.resolution_m
        ).astype(np.int32)
        inside = (
            (x_index >= 0)
            & (x_index < raster.size)
            & (y_index >= 0)
            & (y_index < raster.size)
        )
        safe_x = np.clip(x_index, 0, raster.size - 1)
        safe_y = np.clip(y_index, 0, raster.size - 1)
        center_z = raster.max_z[safe_y, safe_x]
        density = raster.log_density[safe_y, safe_x]
        corridor_z = corridor_max[safe_y, safe_x]
        valid = inside & np.isfinite(center_z)
        center_z = np.where(valid, center_z, np.nan)
        density = np.where(valid, density, np.nan)
        corridor_z = np.where(inside & (corridor_z > -999.0), corridor_z, np.nan)

        target_x = np.floor((x - raster.minimum_xy) / raster.resolution_m).astype(int)
        target_y = np.floor((y - raster.minimum_xy) / raster.resolution_m).astype(int)
        target_inside = (
            (target_x >= 0)
            & (target_x < raster.size)
            & (target_y >= 0)
            & (target_y < raster.size)
        )
        safe_target_x = np.clip(target_x, 0, raster.size - 1)
        safe_target_y = np.clip(target_y, 0, raster.size - 1)
        target_ground = raster.min_z[safe_target_y, safe_target_x].astype(np.float32)
        target_surface = raster.max_z[safe_target_y, safe_target_x].astype(np.float32)
        fallback_ground = np.float32(-antenna_height_m)
        target_ground = np.where(
            target_inside & np.isfinite(target_ground), target_ground, fallback_ground
        )
        target_surface = np.where(
            target_inside & np.isfinite(target_surface), target_surface, target_ground
        )
        receiver_z = target_ground + receiver_height_m
        line_z = receiver_z[:, None] * fractions[None, :]
        clearance = center_z - line_z
        corridor_clearance = corridor_z - line_z
        positive_clearance = np.maximum(clearance, 0.0)
        blocked = clearance > 0.0
        corridor_blocked = corridor_clearance > 0.0
        valid_count = valid.sum(axis=1)
        denominator = np.maximum(valid_count, 1)

        first_blockage = np.where(
            blocked.any(axis=1),
            np.argmax(blocked, axis=1) / max(samples - 1, 1),
            1.0,
        )
        last_blockage = np.where(
            blocked.any(axis=1),
            (samples - 1 - np.argmax(blocked[:, ::-1], axis=1)) / max(samples - 1, 1),
            0.0,
        )
        fresnel_radius = np.sqrt(
            wavelength_m
            * distance[:, None]
            * fractions[None, :]
            * (1.0 - fractions[None, :])
        )
        fresnel_intrusion = clearance + 0.6 * fresnel_radius
        fresnel_blocked = fresnel_intrusion > 0.0

        features = np.column_stack(
            [
                valid_count / samples,
                _nanmax_rows(clearance),
                np.nansum(positive_clearance, axis=1) / denominator,
                np.sum(blocked & valid, axis=1) / denominator,
                first_blockage,
                last_blockage,
                _nanmax_rows(fresnel_intrusion),
                np.sum(fresnel_blocked & valid, axis=1) / denominator,
                np.nansum(center_z, axis=1) / denominator,
                _nanmax_rows(center_z),
                np.nansum(density, axis=1) / denominator,
                _nanmax_rows(corridor_clearance),
                np.sum(corridor_blocked & np.isfinite(corridor_z), axis=1)
                / np.maximum(np.isfinite(corridor_z).sum(axis=1), 1),
                target_ground,
                target_surface,
                target_surface - target_ground,
                raster.log_density[safe_target_y, safe_target_x],
            ]
        )
        output[start:stop] = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(output, columns=PATH_FEATURES, index=points.index)

