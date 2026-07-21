import torch

from ucup_rsrp.unet import TinyUNet


def test_tiny_unet_preserves_spatial_shape() -> None:
    model = TinyUNet(input_channels=14, base_channels=4)
    inputs = torch.zeros((2, 14, 64, 64))
    assert model(inputs).shape == (2, 1, 64, 64)

