from __future__ import annotations

import numpy as np
import pandas as pd


BAND_MHZ = {"800M": 800.0, "2.1G": 2100.0, "3.5G": 3500.0}


def wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def add_geometry_features(
    points: pd.DataFrame,
    *,
    base_x: float,
    base_y: float,
    height: float,
    azimuth: float,
    downtilt: float,
    band: str,
    receiver_height: float = 1.5,
) -> pd.DataFrame:
    """Add propagation geometry for local coordinates whose +y axis is true north."""
    result = points.copy()
    x = result["x"].to_numpy(dtype=np.float64, copy=False)
    y = result["y"].to_numpy(dtype=np.float64, copy=False)
    distance = np.hypot(x, y)
    safe_distance = np.maximum(distance, 1.0)
    bearing = np.degrees(np.arctan2(x, y)) % 360.0
    azimuth_delta = wrap_degrees(bearing - azimuth)
    downward_angle = np.degrees(np.arctan2(height - receiver_height, safe_distance))
    vertical_delta = downward_angle - downtilt
    frequency_mhz = BAND_MHZ[band]

    result["distance_m"] = distance
    result["log10_distance"] = np.log10(safe_distance)
    result["bearing_deg"] = bearing
    result["azimuth_delta_deg"] = azimuth_delta
    result["azimuth_delta_sin"] = np.sin(np.radians(azimuth_delta))
    result["azimuth_delta_cos"] = np.cos(np.radians(azimuth_delta))
    result["downward_angle_deg"] = downward_angle
    result["vertical_delta_deg"] = vertical_delta
    result["frequency_mhz"] = frequency_mhz
    result["log10_frequency"] = np.log10(frequency_mhz)
    result["free_space_loss_db"] = (
        32.44 + 20.0 * np.log10(safe_distance / 1000.0) + 20.0 * np.log10(frequency_mhz)
    )
    result["base_height_m"] = height
    result["downtilt_deg"] = downtilt
    result["global_x"] = base_x + x
    result["global_y"] = base_y + y
    return result

