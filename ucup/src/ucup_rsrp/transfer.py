from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import Cell
from .geometry import BAND_MHZ


KNN_TRANSFER_FEATURES = [
    "source_rsrp_nearest",
    "source_rsrp_weighted",
    "source_rsrp_median",
    "source_rsrp_std",
    "source_distance_1_m",
    "source_distance_k_m",
    "source_neighbors_5m",
    "source_neighbors_10m",
    "source_neighbors_20m",
]


@dataclass(frozen=True)
class SameSectorMatch:
    source: Cell
    target: Cell
    site_distance_m: float
    azimuth_difference_deg: float


def angular_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def find_same_sector_matches(
    sources: list[Cell],
    targets: list[Cell],
    *,
    position_tolerance_m: float = 5.0,
    azimuth_tolerance_deg: float = 5.0,
    allowed_band_pairs: set[frozenset[str]] | None = None,
) -> list[SameSectorMatch]:
    matches: list[SameSectorMatch] = []
    for target in targets:
        candidates: list[SameSectorMatch] = []
        for source in sources:
            if source.cell_id == target.cell_id or source.band == target.band:
                continue
            band_pair = frozenset((source.band, target.band))
            if allowed_band_pairs is not None and band_pair not in allowed_band_pairs:
                continue
            site_distance = math.hypot(source.x - target.x, source.y - target.y)
            azimuth_difference = angular_difference(source.azimuth, target.azimuth)
            if (
                site_distance <= position_tolerance_m
                and azimuth_difference <= azimuth_tolerance_deg
            ):
                candidates.append(
                    SameSectorMatch(
                        source=source,
                        target=target,
                        site_distance_m=site_distance,
                        azimuth_difference_deg=azimuth_difference,
                    )
                )
        if candidates:
            matches.append(
                min(
                    candidates,
                    key=lambda match: (
                        match.azimuth_difference_deg,
                        match.site_distance_m,
                    ),
                )
            )
    return matches


def add_knn_transfer_features(
    source_points: pd.DataFrame,
    target_points: pd.DataFrame,
    *,
    source: Cell,
    target: Cell,
    neighbors: int = 16,
) -> pd.DataFrame:
    if "rsrp" not in source_points:
        raise ValueError("The source points must contain RSRP labels.")
    neighbor_count = min(neighbors, len(source_points))
    source_xy = np.column_stack(
        [
            source_points["x"].to_numpy(dtype=np.float64) + source.x,
            source_points["y"].to_numpy(dtype=np.float64) + source.y,
        ]
    )
    target_xy = np.column_stack(
        [
            target_points["x"].to_numpy(dtype=np.float64) + target.x,
            target_points["y"].to_numpy(dtype=np.float64) + target.y,
        ]
    )
    tree = cKDTree(source_xy)
    distances, indices = tree.query(target_xy, k=neighbor_count, workers=-1)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    source_rsrp = source_points["rsrp"].to_numpy(dtype=np.float64)[indices]
    weights = 1.0 / np.square(distances + 0.5)
    weighted = np.sum(weights * source_rsrp, axis=1) / np.sum(weights, axis=1)

    result = pd.DataFrame(index=target_points.index)
    result["source_rsrp_nearest"] = source_rsrp[:, 0]
    result["source_rsrp_weighted"] = weighted
    result["source_rsrp_median"] = np.median(source_rsrp, axis=1)
    result["source_rsrp_std"] = np.std(source_rsrp, axis=1)
    result["source_distance_1_m"] = distances[:, 0]
    result["source_distance_k_m"] = distances[:, -1]
    result["source_neighbors_5m"] = np.sum(distances <= 5.0, axis=1)
    result["source_neighbors_10m"] = np.sum(distances <= 10.0, axis=1)
    result["source_neighbors_20m"] = np.sum(distances <= 20.0, axis=1)
    result["source_frequency_mhz"] = BAND_MHZ[source.band]
    result["target_frequency_mhz"] = BAND_MHZ[target.band]
    result["log_frequency_ratio"] = math.log10(BAND_MHZ[target.band] / BAND_MHZ[source.band])
    result["same_sector_site_distance_m"] = math.hypot(source.x - target.x, source.y - target.y)
    result["same_sector_azimuth_difference_deg"] = angular_difference(
        source.azimuth, target.azimuth
    )
    return result

