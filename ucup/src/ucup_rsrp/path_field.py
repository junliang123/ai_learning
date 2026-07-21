from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import maximum_filter
from torch import nn
from torch.nn import functional as F

from .geometry import BAND_MHZ
from .raster import PointCloudRaster


PATH_SEQUENCE_CHANNELS = [
    "path_fraction",
    "surface_z",
    "ground_z",
    "object_height",
    "los_z",
    "clearance",
    "positive_clearance",
    "fresnel_intrusion",
    "log_density",
    "corridor_clearance",
    "valid",
]

BEV_INPUT_CHANNELS = [
    "surface_z",
    "ground_z",
    "mean_z",
    "std_z",
    "log_density",
    "occupancy",
    "red",
    "green",
    "blue",
]

SCALAR_FEATURES = [
    "baseline_rsrp",
    "log10_distance",
    "azimuth_delta_sin",
    "azimuth_delta_cos",
    "vertical_delta",
    "band",
    "base_height",
    "downtilt",
    "free_space_loss",
]

RESIDUAL_SCALE_DB = 20.0


@dataclass(frozen=True)
class PathFieldConfig:
    samples: int = 96
    path_width: int = 32
    condition_width: int = 64
    bev_base_channels: int = 8
    fourier_levels: int = 0
    experts: int = 1
    use_bev: bool = False


def _safe_raster_lookup(
    array: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
    inside: np.ndarray,
    *,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    safe_x = np.clip(x_index, 0, array.shape[1] - 1)
    safe_y = np.clip(y_index, 0, array.shape[0] - 1)
    values = array[safe_y, safe_x]
    valid = inside & np.isfinite(values)
    return np.where(valid, values, fill).astype(np.float32), valid


def extract_path_sequence(
    points: pd.DataFrame,
    raster: PointCloudRaster,
    *,
    antenna_height_m: float,
    band: str,
    receiver_height_m: float = 1.5,
    samples: int = 96,
    corridor_radius_pixels: int = 2,
) -> np.ndarray:
    """Return normalized, ordered BS-to-receiver profiles with shape (N, C, S)."""
    x = points["x"].to_numpy(dtype=np.float64, copy=False)
    y = points["y"].to_numpy(dtype=np.float64, copy=False)
    fractions = np.linspace(0.02, 0.98, samples, dtype=np.float32)
    sample_x = x[:, None] * fractions[None, :]
    sample_y = y[:, None] * fractions[None, :]
    x_index = np.floor((sample_x - raster.minimum_xy) / raster.resolution_m).astype(
        np.int32
    )
    y_index = np.floor((sample_y - raster.minimum_xy) / raster.resolution_m).astype(
        np.int32
    )
    inside = (
        (x_index >= 0)
        & (x_index < raster.size)
        & (y_index >= 0)
        & (y_index < raster.size)
    )

    surface_z, surface_valid = _safe_raster_lookup(
        raster.max_z, x_index, y_index, inside
    )
    ground_z, ground_valid = _safe_raster_lookup(
        raster.min_z, x_index, y_index, inside
    )
    log_density, density_valid = _safe_raster_lookup(
        raster.log_density, x_index, y_index, inside
    )
    corridor = maximum_filter(
        np.nan_to_num(raster.max_z, nan=-1_000.0),
        size=2 * corridor_radius_pixels + 1,
        mode="nearest",
    )
    corridor_z, corridor_valid = _safe_raster_lookup(
        corridor, x_index, y_index, inside, fill=-1_000.0
    )
    corridor_valid &= corridor_z > -999.0

    target_x = np.floor((x - raster.minimum_xy) / raster.resolution_m).astype(np.int32)
    target_y = np.floor((y - raster.minimum_xy) / raster.resolution_m).astype(np.int32)
    target_inside = (
        (target_x >= 0)
        & (target_x < raster.size)
        & (target_y >= 0)
        & (target_y < raster.size)
    )
    target_ground, target_valid = _safe_raster_lookup(
        raster.min_z, target_x, target_y, target_inside, fill=-antenna_height_m
    )
    target_ground = np.where(target_valid, target_ground, -antenna_height_m)
    receiver_z = target_ground + receiver_height_m
    los_z = receiver_z[:, None] * fractions[None, :]

    clearance = surface_z - los_z
    positive_clearance = np.maximum(clearance, 0.0)
    distance = np.maximum(np.hypot(x, y), 1.0)
    wavelength_m = 299_792_458.0 / (BAND_MHZ[band] * 1_000_000.0)
    fresnel_radius = np.sqrt(
        wavelength_m
        * distance[:, None]
        * fractions[None, :]
        * (1.0 - fractions[None, :])
    )
    fresnel_intrusion = clearance + 0.6 * fresnel_radius
    corridor_clearance = corridor_z - los_z
    valid = surface_valid & ground_valid & density_valid

    sequence = np.stack(
        [
            np.broadcast_to(fractions[None, :], surface_z.shape),
            surface_z / 60.0,
            ground_z / 60.0,
            np.maximum(surface_z - ground_z, 0.0) / 40.0,
            los_z / 60.0,
            clearance / 30.0,
            positive_clearance / 30.0,
            fresnel_intrusion / 30.0,
            log_density / 8.0,
            np.where(corridor_valid, corridor_clearance, 0.0) / 30.0,
            valid.astype(np.float32),
        ],
        axis=1,
    )
    sequence[:, 1:9] *= valid[:, None, :]
    return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def build_bev_input(raster: PointCloudRaster) -> np.ndarray:
    """Build a compact normalized BEV tensor without changing the cached raster format."""
    occupancy = np.isfinite(raster.max_z)
    channels = np.stack(
        [
            np.nan_to_num(raster.max_z / 60.0),
            np.nan_to_num(raster.min_z / 60.0),
            np.nan_to_num(raster.mean_z / 60.0),
            np.nan_to_num(raster.std_z / 20.0),
            raster.log_density / 8.0,
            occupancy.astype(np.float32),
            np.nan_to_num(raster.mean_red / 255.0),
            np.nan_to_num(raster.mean_green / 255.0),
            np.nan_to_num(raster.mean_blue / 255.0),
        ]
    )
    return channels.astype(np.float32)


def make_scalar_features(
    featured: pd.DataFrame,
    baseline_prediction: np.ndarray,
    *,
    band: str,
    base_height_m: float,
    downtilt_deg: float,
) -> np.ndarray:
    band_value = np.log10(BAND_MHZ[band] / 800.0) / np.log10(3500.0 / 800.0)
    features = np.column_stack(
        [
            (np.asarray(baseline_prediction) + 100.0) / 30.0,
            featured["log10_distance"].to_numpy() / 3.2,
            featured["azimuth_delta_sin"].to_numpy(),
            featured["azimuth_delta_cos"].to_numpy(),
            np.clip(featured["vertical_delta_deg"].to_numpy() / 30.0, -2.0, 2.0),
            np.full(len(featured), band_value),
            np.full(len(featured), base_height_m / 60.0),
            np.full(len(featured), downtilt_deg / 15.0),
            (featured["free_space_loss_db"].to_numpy() - 100.0) / 30.0,
        ]
    )
    return features.astype(np.float32)


def fourier_encode(coordinates: torch.Tensor, levels: int) -> torch.Tensor:
    if levels <= 0:
        return coordinates.new_zeros((len(coordinates), 0))
    frequencies = 2.0 ** torch.arange(
        levels, dtype=coordinates.dtype, device=coordinates.device
    )
    angles = torch.pi * coordinates.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs))


class PathSequenceEncoder(nn.Module):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        groups = min(8, width)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=5, padding=2),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(ResidualBlock1D(width), ResidualBlock1D(width))
        self.output_dim = width * 4

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(sequence))
        return torch.cat(
            [
                hidden.mean(dim=-1),
                hidden.amax(dim=-1),
                hidden[:, :, 0],
                hidden[:, :, -1],
            ],
            dim=1,
        )


class BEVEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        self.level1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(min(4, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.GroupNorm(min(8, base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output_channels = base_channels * 3

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        level1 = self.level1(inputs)
        return level1, self.level2(level1)


def _query_plane(plane: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    sampled = F.grid_sample(
        plane,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0].permute(1, 0, 2)


class ConditionalPathField(nn.Module):
    """Shared conditional continuous field that predicts a path-baseline residual."""

    def __init__(self, config: PathFieldConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "fractions", torch.linspace(0.02, 0.98, config.samples), persistent=False
        )
        self.bev_encoder: BEVEncoder | None = None
        bev_channels = 0
        if config.use_bev:
            self.bev_encoder = BEVEncoder(len(BEV_INPUT_CHANNELS), config.bev_base_channels)
            bev_channels = self.bev_encoder.output_channels
        self.path_encoder = PathSequenceEncoder(
            len(PATH_SEQUENCE_CHANNELS) + bev_channels, config.path_width
        )
        coordinate_dim = 4 * config.fourier_levels
        condition_input_dim = len(SCALAR_FEATURES) + coordinate_dim + bev_channels
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, config.condition_width),
            nn.SiLU(),
            nn.Linear(config.condition_width, config.condition_width),
            nn.SiLU(),
        )
        fused_dim = self.path_encoder.output_dim + config.condition_width
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, config.condition_width),
                    nn.SiLU(),
                    nn.Linear(config.condition_width, 1),
                )
                for _ in range(config.experts)
            ]
        )
        self.gate: nn.Module | None = None
        if config.experts > 1:
            self.gate = nn.Sequential(
                nn.Linear(condition_input_dim, config.condition_width // 2),
                nn.SiLU(),
                nn.Linear(config.condition_width // 2, config.experts),
            )

    def encode_scene(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bev_encoder is None:
            raise RuntimeError("This model was configured without a BEV encoder.")
        return self.bev_encoder(bev)

    def _query_scene(
        self,
        coordinates: torch.Tensor,
        scene_features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path_grid = coordinates[:, None, :] * self.fractions[None, :, None]
        endpoint_grid = coordinates[:, None, :]
        path_parts = [_query_plane(plane, path_grid) for plane in scene_features]
        endpoint_parts = [
            _query_plane(plane, endpoint_grid).squeeze(-1) for plane in scene_features
        ]
        return torch.cat(path_parts, dim=1), torch.cat(endpoint_parts, dim=1)

    def forward_queries(
        self,
        raw_sequence: torch.Tensor,
        scalar_features: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        scene_features: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sequence = raw_sequence
        endpoint_features = scalar_features.new_zeros((len(scalar_features), 0))
        if self.config.use_bev:
            if scene_features is None:
                raise ValueError("scene_features are required when use_bev=True")
            path_features, endpoint_features = self._query_scene(
                coordinates, scene_features
            )
            sequence = torch.cat([sequence, path_features], dim=1)
        coordinate_features = fourier_encode(coordinates, self.config.fourier_levels)
        condition_input = torch.cat(
            [scalar_features, coordinate_features, endpoint_features], dim=1
        )
        condition = self.condition_encoder(condition_input)
        fused = torch.cat([self.path_encoder(sequence), condition], dim=1)
        expert_predictions = torch.cat([expert(fused) for expert in self.experts], dim=1)
        if self.gate is None:
            gate = expert_predictions.new_ones(expert_predictions.shape)
            prediction = expert_predictions[:, 0]
        else:
            gate = torch.softmax(self.gate(condition_input), dim=1)
            prediction = torch.sum(gate * expert_predictions, dim=1)
        prediction = 2.0 * torch.tanh(prediction)
        if return_gate:
            return prediction, gate
        return prediction

    def forward(
        self,
        raw_sequence: torch.Tensor,
        scalar_features: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        bev: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scene_features = None
        if self.config.use_bev:
            if bev is None:
                raise ValueError("bev is required when use_bev=True")
            scene_features = self.encode_scene(bev)
        return self.forward_queries(
            raw_sequence,
            scalar_features,
            coordinates,
            scene_features=scene_features,
            return_gate=return_gate,
        )


def environment_aware_gradient_loss(
    prediction: torch.Tensor,
    coordinates: torch.Tensor,
    sequence: torch.Tensor,
    *,
    environment_alpha: float = 3.0,
    distance_scale: float = 0.08,
) -> torch.Tensor:
    """Local Taylor-consistency penalty, downweighted across environmental boundaries."""
    if len(prediction) < 2 or not coordinates.requires_grad:
        return prediction.new_zeros(())
    gradient = torch.autograd.grad(
        prediction.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    with torch.no_grad():
        distances = torch.cdist(coordinates, coordinates)
        distances.fill_diagonal_(float("inf"))
        neighbor = distances.argmin(dim=1)
        nearest_distance = distances.gather(1, neighbor[:, None]).squeeze(1)
        environment_difference = torch.mean(
            torch.abs(sequence - sequence[neighbor]), dim=(1, 2)
        )
        weight = torch.exp(-environment_alpha * environment_difference)
        weight *= torch.exp(-nearest_distance / distance_scale)
    delta = coordinates[neighbor] - coordinates
    predicted_difference = prediction[neighbor] - prediction
    taylor_difference = torch.sum(gradient * delta, dim=1)
    error = F.smooth_l1_loss(
        predicted_difference, taylor_difference, reduction="none", beta=0.05
    )
    return torch.sum(weight * error) / weight.sum().clamp_min(1e-6)
