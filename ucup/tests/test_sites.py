from pathlib import Path

from ucup_rsrp.data import Cell
from ucup_rsrp.sites import assign_site_ids


def make_cell(cell_id: str, x: float, y: float) -> Cell:
    return Cell(
        path=Path(cell_id),
        cell_id=cell_id,
        numeric_id=int(cell_id[-4:]),
        split="train",
        x=x,
        y=y,
        height=30.0,
        azimuth=0.0,
        downtilt=5.0,
        band="800M",
    )


def test_site_clustering_uses_transitive_proximity() -> None:
    cells = [
        make_cell("cell_0000", 0.0, 0.0),
        make_cell("cell_0001", 6.0, 0.0),
        make_cell("cell_0002", 12.0, 0.0),
        make_cell("cell_0003", 100.0, 0.0),
    ]
    sites = assign_site_ids(cells, radius_m=10.0)
    assert sites["cell_0000"] == sites["cell_0002"]
    assert sites["cell_0000"] != sites["cell_0003"]

