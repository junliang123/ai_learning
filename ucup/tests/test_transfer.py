from pathlib import Path

import pandas as pd

from ucup_rsrp.data import Cell
from ucup_rsrp.transfer import add_knn_transfer_features, find_same_sector_matches


def make_cell(cell_id: str, band: str, x: float = 0.0, azimuth: float = 0.0) -> Cell:
    return Cell(
        path=Path(cell_id),
        cell_id=cell_id,
        numeric_id=int(cell_id[-4:]),
        split="train",
        x=x,
        y=0.0,
        height=30.0,
        azimuth=azimuth,
        downtilt=5.0,
        band=band,
    )


def test_find_same_sector_match_filters_direction_and_band() -> None:
    source = make_cell("cell_0000", "800M", x=1.0, azimuth=2.0)
    target = make_cell("cell_1000", "2.1G")
    wrong_direction = make_cell("cell_0001", "800M", azimuth=90.0)
    matches = find_same_sector_matches([wrong_direction, source], [target])
    assert len(matches) == 1
    assert matches[0].source == source


def test_knn_features_use_global_coordinates() -> None:
    source = make_cell("cell_0000", "800M", x=10.0)
    target = make_cell("cell_1000", "2.1G", x=11.0)
    source_points = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 0.0], "rsrp": [-80, -100]})
    target_points = pd.DataFrame({"x": [-1.0], "y": [0.0]})
    features = add_knn_transfer_features(
        source_points, target_points, source=source, target=target, neighbors=2
    )
    assert features.loc[0, "source_distance_1_m"] == 0.0
    assert features.loc[0, "source_rsrp_nearest"] == -80.0

