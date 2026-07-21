import numpy as np
import pandas as pd
import torch

from ucup_rsrp.path_field import (
    BEV_INPUT_CHANNELS,
    PATH_SEQUENCE_CHANNELS,
    RESIDUAL_SCALE_DB,
    SCALAR_FEATURES,
    ConditionalPathField,
    PathFieldConfig,
    build_bev_input,
    environment_aware_gradient_loss,
    extract_path_sequence,
)
from ucup_rsrp.raster import PointCloudRaster


def make_raster() -> PointCloudRaster:
    shape = (8, 8)
    ground = np.full(shape, -20.0, dtype=np.float32)
    surface = ground.copy()
    surface[3:5, 3:5] = 5.0
    density = np.ones(shape, dtype=np.float32)
    color = np.full(shape, 100.0, dtype=np.float32)
    return PointCloudRaster(
        resolution_m=1.0,
        minimum_xy=-4.0,
        maximum_xy=4.0,
        max_z=surface,
        min_z=ground,
        mean_z=(surface + ground) / 2.0,
        std_z=np.abs(surface - ground) / 2.0,
        log_density=density,
        mean_red=color,
        mean_green=color,
        mean_blue=color,
    )


def test_extract_path_sequence_preserves_ordered_profile() -> None:
    raster = make_raster()
    points = pd.DataFrame({"x": [3.0, -3.0], "y": [3.0, -3.0]})
    sequence = extract_path_sequence(
        points, raster, antenna_height_m=20.0, band="800M", samples=16
    )
    assert sequence.shape == (2, len(PATH_SEQUENCE_CHANNELS), 16)
    assert np.all(np.diff(sequence[0, 0]) > 0)
    assert np.max(sequence[0, 6]) > 0.0
    assert np.isfinite(sequence).all()


def test_build_bev_input_has_expected_channels() -> None:
    bev = build_bev_input(make_raster())
    assert bev.shape == (len(BEV_INPUT_CHANNELS), 8, 8)
    assert np.isfinite(bev).all()


def test_path_field_supports_all_components_and_coordinate_gradients() -> None:
    config = PathFieldConfig(
        samples=16,
        path_width=8,
        condition_width=16,
        bev_base_channels=4,
        fourier_levels=3,
        experts=3,
        use_bev=True,
    )
    model = ConditionalPathField(config)
    batch = 5
    sequence = torch.zeros(batch, len(PATH_SEQUENCE_CHANNELS), config.samples)
    scalars = torch.zeros(batch, len(SCALAR_FEATURES))
    coordinates = torch.linspace(-0.7, 0.7, batch * 2).reshape(batch, 2)
    coordinates.requires_grad_(True)
    bev = torch.from_numpy(build_bev_input(make_raster())).unsqueeze(0)
    prediction, gate = model(
        sequence, scalars, coordinates, bev=bev, return_gate=True
    )
    assert prediction.shape == (batch,)
    assert gate.shape == (batch, config.experts)
    assert torch.allclose(gate.sum(dim=1), torch.ones(batch), atol=1e-5)
    assert torch.max(torch.abs(prediction * RESIDUAL_SCALE_DB)) <= 40.0
    regularizer = environment_aware_gradient_loss(
        prediction, coordinates, sequence
    )
    assert regularizer.ndim == 0
    assert torch.isfinite(regularizer)
    (prediction.abs().mean() + regularizer).backward()


def test_sequence_only_path_field() -> None:
    config = PathFieldConfig(
        samples=12, path_width=8, condition_width=16, use_bev=False
    )
    model = ConditionalPathField(config)
    prediction = model(
        torch.zeros(4, len(PATH_SEQUENCE_CHANNELS), config.samples),
        torch.zeros(4, len(SCALAR_FEATURES)),
        torch.zeros(4, 2),
    )
    assert prediction.shape == (4,)
