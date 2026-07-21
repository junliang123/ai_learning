from pathlib import Path

import numpy as np

from ucup_rsrp.data import PLY_VERTEX_DTYPE, read_ply_vertices
from ucup_rsrp.raster import rasterize_point_cloud


def write_test_ply(path: Path, values: list[tuple[float, float, float, int, int, int]]) -> None:
    array = np.array(values, dtype=PLY_VERTEX_DTYPE)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(array)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def test_read_and_rasterize_binary_ply(tmp_path: Path) -> None:
    path = tmp_path / "tiny.ply"
    write_test_ply(path, [(0.1, 0.1, 2.0, 10, 20, 30), (0.2, 0.2, 4.0, 30, 40, 50)])
    vertices = read_ply_vertices(path)
    assert len(vertices) == 2
    raster = rasterize_point_cloud(
        path, resolution_m=1.0, minimum_xy=0.0, maximum_xy=2.0, chunk_size=1
    )
    assert raster.max_z[0, 0] == 4.0
    assert raster.min_z[0, 0] == 2.0
    assert raster.mean_z[0, 0] == 3.0
    assert raster.mean_red[0, 0] == 20.0
    assert np.isnan(raster.max_z[1, 1])

