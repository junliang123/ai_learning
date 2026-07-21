from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import read_ply_vertices


@dataclass(frozen=True)
class PointCloudRaster:
    resolution_m: float
    minimum_xy: float
    maximum_xy: float
    max_z: np.ndarray
    min_z: np.ndarray
    mean_z: np.ndarray
    std_z: np.ndarray
    log_density: np.ndarray
    mean_red: np.ndarray
    mean_green: np.ndarray
    mean_blue: np.ndarray

    @property
    def size(self) -> int:
        return self.max_z.shape[0]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            resolution_m=np.float32(self.resolution_m),
            minimum_xy=np.float32(self.minimum_xy),
            maximum_xy=np.float32(self.maximum_xy),
            max_z=self.max_z,
            min_z=self.min_z,
            mean_z=self.mean_z,
            std_z=self.std_z,
            log_density=self.log_density,
            mean_red=self.mean_red,
            mean_green=self.mean_green,
            mean_blue=self.mean_blue,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PointCloudRaster":
        with np.load(path) as archive:
            return cls(
                resolution_m=float(archive["resolution_m"]),
                minimum_xy=float(archive["minimum_xy"]),
                maximum_xy=float(archive["maximum_xy"]),
                max_z=archive["max_z"],
                min_z=archive["min_z"],
                mean_z=archive["mean_z"],
                std_z=archive["std_z"],
                log_density=archive["log_density"],
                mean_red=archive["mean_red"],
                mean_green=archive["mean_green"],
                mean_blue=archive["mean_blue"],
            )


def point_cloud_fingerprint(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    """Fingerprint large PLYs from separated samples so exact duplicates can share a raster."""
    source = Path(path)
    size = source.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(size).encode("ascii"))
    with source.open("rb") as handle:
        for offset in (0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes)):
            handle.seek(offset)
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def rasterize_point_cloud(
    path: str | Path,
    *,
    resolution_m: float = 4.0,
    minimum_xy: float = -1024.0,
    maximum_xy: float = 1024.0,
    chunk_size: int = 1_000_000,
) -> PointCloudRaster:
    span = maximum_xy - minimum_xy
    size = int(round(span / resolution_m))
    if not np.isclose(size * resolution_m, span):
        raise ValueError("The raster extent must be divisible by the resolution.")

    vertices = read_ply_vertices(path)
    pixel_count = size * size
    count = np.zeros(pixel_count, dtype=np.uint32)
    sum_z = np.zeros(pixel_count, dtype=np.float64)
    sum_z2 = np.zeros(pixel_count, dtype=np.float64)
    sum_red = np.zeros(pixel_count, dtype=np.float64)
    sum_green = np.zeros(pixel_count, dtype=np.float64)
    sum_blue = np.zeros(pixel_count, dtype=np.float64)
    max_z = np.full(pixel_count, -np.inf, dtype=np.float32)
    min_z = np.full(pixel_count, np.inf, dtype=np.float32)

    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start : start + chunk_size]
        x_index = np.floor((chunk["x"] - minimum_xy) / resolution_m).astype(np.int32)
        y_index = np.floor((chunk["y"] - minimum_xy) / resolution_m).astype(np.int32)
        valid = (x_index >= 0) & (x_index < size) & (y_index >= 0) & (y_index < size)
        flat_index = y_index[valid].astype(np.int64) * size + x_index[valid]
        z = chunk["z"][valid]

        count += np.bincount(flat_index, minlength=pixel_count).astype(np.uint32)
        sum_z += np.bincount(flat_index, weights=z, minlength=pixel_count)
        sum_z2 += np.bincount(flat_index, weights=z * z, minlength=pixel_count)
        sum_red += np.bincount(
            flat_index, weights=chunk["red"][valid], minlength=pixel_count
        )
        sum_green += np.bincount(
            flat_index, weights=chunk["green"][valid], minlength=pixel_count
        )
        sum_blue += np.bincount(
            flat_index, weights=chunk["blue"][valid], minlength=pixel_count
        )
        np.maximum.at(max_z, flat_index, z)
        np.minimum.at(min_z, flat_index, z)

    occupied = count > 0
    mean_z = np.full(pixel_count, np.nan, dtype=np.float32)
    std_z = np.full(pixel_count, np.nan, dtype=np.float32)
    colors = [np.full(pixel_count, np.nan, dtype=np.float32) for _ in range(3)]
    mean_z[occupied] = (sum_z[occupied] / count[occupied]).astype(np.float32)
    variance = sum_z2[occupied] / count[occupied] - mean_z[occupied].astype(np.float64) ** 2
    std_z[occupied] = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    for destination, total in zip(colors, (sum_red, sum_green, sum_blue), strict=True):
        destination[occupied] = (total[occupied] / count[occupied]).astype(np.float32)
    max_z[~occupied] = np.nan
    min_z[~occupied] = np.nan

    reshape = lambda array: array.reshape(size, size)  # noqa: E731
    return PointCloudRaster(
        resolution_m=resolution_m,
        minimum_xy=minimum_xy,
        maximum_xy=maximum_xy,
        max_z=reshape(max_z),
        min_z=reshape(min_z),
        mean_z=reshape(mean_z),
        std_z=reshape(std_z),
        log_density=reshape(np.log1p(count).astype(np.float32)),
        mean_red=reshape(colors[0]),
        mean_green=reshape(colors[1]),
        mean_blue=reshape(colors[2]),
    )

