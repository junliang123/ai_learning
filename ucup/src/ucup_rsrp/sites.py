from __future__ import annotations

import math

from .data import Cell


def assign_site_ids(cells: list[Cell], radius_m: float = 10.0) -> dict[str, int]:
    """Cluster the small set of cells by engineering-coordinate proximity."""
    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(cells):
        for j in range(i):
            right = cells[j]
            if math.hypot(left.x - right.x, left.y - right.y) <= radius_m:
                union(i, j)

    roots: dict[int, int] = {}
    result: dict[str, int] = {}
    for index, cell in enumerate(cells):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        result[cell.cell_id] = roots[root]
    return result

