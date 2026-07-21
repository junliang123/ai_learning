import numpy as np
import pandas as pd

from ucup_rsrp.geometry import add_geometry_features, wrap_degrees


def test_wrap_degrees() -> None:
    values = np.array([0.0, 180.0, 181.0, 359.0, 360.0])
    np.testing.assert_allclose(wrap_degrees(values), [0.0, -180.0, -179.0, -1.0, 0.0])


def test_bearing_uses_clockwise_angle_from_north() -> None:
    points = pd.DataFrame({"x": [0.0, 1.0, 0.0, -1.0], "y": [1.0, 0.0, -1.0, 0.0]})
    result = add_geometry_features(
        points,
        base_x=0.0,
        base_y=0.0,
        height=30.0,
        azimuth=0.0,
        downtilt=5.0,
        band="800M",
    )
    np.testing.assert_allclose(result["bearing_deg"], [0.0, 90.0, 180.0, 270.0])

