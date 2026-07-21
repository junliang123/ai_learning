from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CELL_PATTERN = re.compile(r"cell_(\d{4})$")
PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


@dataclass(frozen=True)
class Cell:
    path: Path
    cell_id: str
    numeric_id: int
    split: str
    x: float
    y: float
    height: float
    azimuth: float
    downtilt: float
    band: str

    @property
    def point_cloud_path(self) -> Path:
        return self.path / "local_points_3d_origin_aerial.ply"

    @property
    def points_path(self) -> Path:
        filename = "train_signal.csv" if self.split == "train" else "test_points.csv"
        return self.path / filename

    def read_points(self, **kwargs: object) -> pd.DataFrame:
        return pd.read_csv(self.points_path, dtype={"point_id": "string"}, **kwargs)


def discover_cells(dataset_root: str | Path) -> list[Cell]:
    root = Path(dataset_root)
    cells: list[Cell] = []
    for path in sorted(root.glob("cell_[01][0-9][0-9][0-9]")):
        match = CELL_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        numeric_id = int(match.group(1))
        split = "train" if numeric_id < 1000 else "test"
        expected = "train_signal.csv" if split == "train" else "test_points.csv"
        if not (path / expected).is_file():
            continue
        with (path / "ep.json").open(encoding="utf-8") as handle:
            ep = json.load(handle)
        cells.append(
            Cell(
                path=path,
                cell_id=ep["cell_id"],
                numeric_id=numeric_id,
                split=split,
                x=float(ep["x"]),
                y=float(ep["y"]),
                height=float(ep["height"]),
                azimuth=float(ep["azimuth"]),
                downtilt=float(ep["downtilt"]),
                band=str(ep["band"]),
            )
        )
    return cells


def _ply_data_offset(path: str | Path) -> tuple[int, int]:
    vertex_count: int | None = None
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            stripped = line.decode("ascii").strip()
            if stripped == "format binary_little_endian 1.0":
                pass
            elif stripped.startswith("format "):
                raise ValueError(f"Unsupported PLY format {stripped!r}: {path}")
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.rsplit(" ", 1)[1])
            elif stripped == "end_header":
                if vertex_count is None:
                    raise ValueError(f"PLY header has no vertex count: {path}")
                return handle.tell(), vertex_count


def read_ply_vertices(path: str | Path) -> np.memmap:
    """Memory-map the challenge's binary PLY vertices without loading the file into RAM."""
    offset, vertex_count = _ply_data_offset(path)
    expected_size = offset + vertex_count * PLY_VERTEX_DTYPE.itemsize
    actual_size = Path(path).stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected PLY size for {path}: expected {expected_size}, got {actual_size}"
        )
    return np.memmap(
        path,
        mode="r",
        dtype=PLY_VERTEX_DTYPE,
        offset=offset,
        shape=(vertex_count,),
    )

