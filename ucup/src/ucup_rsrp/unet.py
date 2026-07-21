from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(4, output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, 2, stride=2)
        self.conv = DoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        return self.conv(torch.cat((inputs, skip), dim=1))


class TinyUNet(nn.Module):
    """A small U-Net intended for sparse CPU experiments on 2D point-cloud rasters."""

    def __init__(self, input_channels: int, base_channels: int = 8) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(input_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))
        decoder3 = self.decoder3(bottleneck, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        return self.output(decoder1)

