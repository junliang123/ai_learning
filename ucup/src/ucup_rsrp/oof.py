from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import Cell
from .sites import assign_site_ids


def make_site_folds(
    cells: list[Cell],
    *,
    n_splits: int = 5,
    seed: int = 20260720,
) -> dict[str, int]:
    """Assign every physical site, and therefore all of its cells, to one OOF fold."""
    if n_splits < 2:
        raise ValueError("OOF requires at least two folds.")
    site_ids = assign_site_ids(cells)
    unique_sites = sorted(set(site_ids.values()))
    if n_splits > len(unique_sites):
        raise ValueError(
            f"Cannot create {n_splits} folds from only {len(unique_sites)} physical sites."
        )
    groups = np.array([site_ids[cell.cell_id] for cell in cells], dtype=np.int32)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_cell: dict[str, int] = {}
    placeholders = np.zeros((len(cells), 1), dtype=np.float32)
    for fold, (_, valid_indices) in enumerate(splitter.split(placeholders, groups=groups)):
        for index in valid_indices:
            fold_by_cell[cells[int(index)].cell_id] = fold
    if len(fold_by_cell) != len(cells):
        raise RuntimeError("OOF assignment did not cover every cell.")
    return fold_by_cell


def cell_balanced_weights(cell_ids: np.ndarray) -> np.ndarray:
    """Return row weights that give every cell the same total weight."""
    unique, inverse, counts = np.unique(cell_ids, return_inverse=True, return_counts=True)
    if len(unique) == 0:
        return np.empty(0, dtype=np.float32)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)
