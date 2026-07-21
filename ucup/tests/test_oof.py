from pathlib import Path

import numpy as np

from ucup_rsrp.data import Cell
from ucup_rsrp.oof import cell_balanced_weights, make_site_folds


def make_cell(cell_id: str, numeric_id: int, x: float, y: float) -> Cell:
    return Cell(
        path=Path(cell_id),
        cell_id=cell_id,
        numeric_id=numeric_id,
        split="train",
        x=x,
        y=y,
        height=30.0,
        azimuth=0.0,
        downtilt=5.0,
        band="800M",
    )


def test_make_site_folds_keeps_co_sited_cells_together() -> None:
    cells = [
        make_cell("cell_0000", 0, 0.0, 0.0),
        make_cell("cell_0001", 1, 2.0, 1.0),
        make_cell("cell_0002", 2, 100.0, 0.0),
        make_cell("cell_0003", 3, 200.0, 0.0),
        make_cell("cell_0004", 4, 300.0, 0.0),
    ]
    folds = make_site_folds(cells, n_splits=3, seed=7)
    assert folds["cell_0000"] == folds["cell_0001"]
    assert set(folds) == {cell.cell_id for cell in cells}
    assert set(folds.values()) == {0, 1, 2}


def test_cell_balanced_weights_equalize_cell_totals() -> None:
    cell_ids = np.array(["a", "a", "a", "b"], dtype=object)
    weights = cell_balanced_weights(cell_ids)
    assert np.isclose(weights[cell_ids == "a"].sum(), weights[cell_ids == "b"].sum())
    assert np.isclose(weights.mean(), 1.0)
