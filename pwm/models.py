"""
Predictive World Model (PWM)
First executable vision prototype.

Pipeline
--------
Frame_t
    -> coarse latent map
    -> world transformer
    -> latent observation query

Frame_(t+1)
    -> coarse latent map / keys

query x keys
    -> soft spatial fixation
    -> differentiable HD foveal crop
    -> target retinal latent

world representation
    -> predictor
    -> predicted retinal latent

Training minimizes latent prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PWMOutput:
    predicted_latent: Tensor
    target_latent: Tensor

    attention_logits: Tensor
    attention_weights: Tensor

    fixation_xy: Tensor
    foveal_crop: Tensor


# ============================================================================
# Coarse Encoder
# ============================================================================

class CoarseEncoder(nn.Module):
    """
    Converts a full frame into a low-resolution spatial latent map.

    Input:
        [B, 3, H, W]

    Output:
        [B, D, Hc, Wc]

    Spatial organization is preserved so every latent key still corresponds
    to a location in the original frame.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, frame: Tensor) -> Tensor:
        return self.encoder(frame)


# ============================================================================
# 2D Positional Encoding
# ============================================================================

class LearnedPosition2D(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        max_height: int = 64,
        max_width: int = 64,
    ) -> None:
        super().__init__()

        self.row_embedding = nn.Embedding(max_height, latent_dim)
        self.col_embedding = nn.Embedding(max_width, latent_dim)

    def forward(self, feature_map: Tensor) -> Tensor:

        batch, channels, height, width = feature_map.shape

        if height > self.row_embedding.num_embeddings:
            raise ValueError("Coarse feature-map height exceeds positional capacity.")

        if width > self.col_embedding.num_embeddings:
            raise ValueError("Coarse feature-map width exceeds positional capacity.")

        rows = torch.arange(height, device=feature_map.device)
        cols = torch.arange(width, device=feature_map.device)

        position = (
            self.row_embedding(rows)[:, None, :]
            +
            self.col_embedding(cols)[None, :, :]
        )

        position = position.reshape(1, height * width, channels)

        return position.expand(batch, -1, -1)


# ============================================================================
# World Transformer
# ============================================================================

class WorldTransformer(nn.Module):
    """
    Processes the coarse latent scene.

    Produces:

        world_latent
        query

    query describes what visual information should be searched for next.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()

        self.position = LearnedPosition2D(latent_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

        self.query_token = nn.Parameter(
            torch.randn(1, 1, latent_dim) * 0.02
        )

        self.query_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

        self.world_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        coarse_map: Tensor,
    ) -> tuple[Tensor, Tensor]:

        batch, channels, height, width = coarse_map.shape

        tokens = coarse_map.flatten(2).transpose(1, 2)

        tokens = tokens + self.position(coarse_map)

        query_token = self.query_token.expand(batch, -1, -1)

        sequence = torch.cat(
            [query_token, tokens],
            dim=1,
        )

        sequence = self.transformer(sequence)

        query_state = sequence[:, 0]

        scene_states = sequence[:, 1:]

        world_latent = self.world_head(
            scene_states.mean(dim=1)
        )

        query = self.query_head(query_state)

        return world_latent, query


# ============================================================================
# Foveal Search
# ============================================================================

class FovealSearch(nn.Module):
    """
    Reverse-searches a coarse spatial latent map using the query.

    Q dot K produces a spatial attention distribution.

    During training, soft attention is used instead of argmax so gradients
    can influence the query and therefore the learned fixation policy.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.temperature = temperature

        self.key_projection = nn.Conv2d(
            latent_dim,
            latent_dim,
            kernel_size=1,
        )

    def forward(
        self,
        query: Tensor,
        coarse_map: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        batch, channels, height, width = coarse_map.shape

        keys = self.key_projection(coarse_map)

        keys = keys.flatten(2).transpose(1, 2)

        query = F.normalize(query, dim=-1)
        keys = F.normalize(keys, dim=-1)

        logits = torch.einsum(
            "bd,bnd->bn",
            query,
            keys,
        )

        logits = logits / self.temperature

        weights = F.softmax(logits, dim=-1)

        ys = torch.linspace(
            -1.0,
            1.0,
            height,
            device=coarse_map.device,
            dtype=coarse_map.dtype,
        )

        xs = torch.linspace(
            -1.0,
            1.0,
            width,
            device=coarse_map.device,
            dtype=coarse_map.dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        coordinates = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)],
            dim=-1,
        )

        fixation_xy = torch.einsum(
            "bn,nc->bc",
            weights,
            coordinates,
        )

        return fixation_xy, logits, weights


# ============================================================================
# Differentiable Fovea
# ============================================================================

class DifferentiableFovea(nn.Module):
    """
    Extracts a differentiable high-resolution crop centered on fixation_xy.

    fixation_xy uses normalized coordinates:

        x, y in [-1, 1]

    grid_sample allows prediction gradients to flow back into the fixation.
    """

    def __init__(
        self,
        crop_size: int = 128,
        radius: float = 0.25,
    ) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.radius = radius

    def forward(
        self,
        frame: Tensor,
        fixation_xy: Tensor,
    ) -> Tensor:

        batch = frame.shape[0]

        axis = torch.linspace(
            -self.radius,
            self.radius,
            self.crop_size,
            device=frame.device,
            dtype=frame.dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            axis,
            axis,
            indexing="ij",
        )

        local_grid = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        )

        local_grid = local_grid.unsqueeze(0).expand(
            batch,
            -1,
            -1,
            -1,
        )

        grid = (
            local_grid
            + fixation_xy[:, None, None, :]
        )

        return F.grid_sample(
            frame,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


# ============================================================================
# Retinal Encoder
# ============================================================================

class RetinalEncoder(nn.Module):
    """
    Converts the high-resolution foveal observation into a latent target.
    """

    def __init__(
        self,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),

            nn.Conv2d(128, latent_dim, 3, 2, 1),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, crop: Tensor) -> Tensor:

        features = self.encoder(crop)

        return self.projector(features)


# ============================================================================
# Predictor
# ============================================================================

class LatentPredictor(nn.Module):

    def __init__(
        self,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.GELU(),

            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(
        self,
        world_latent: Tensor,
        query: Tensor,
    ) -> Tensor:

        x = torch.cat(
            [world_latent, query],
            dim=-1,
        )

        return self.network(x)


# ============================================================================
# PWM Model
# ============================================================================

class PWMModel(nn.Module):

    def __init__(
        self,
        latent_dim: int = 256,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        crop_size: int = 128,
        fovea_radius: float = 0.25,
        search_temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.coarse_encoder = CoarseEncoder(
            latent_dim=latent_dim,
        )

        self.world_transformer = WorldTransformer(
            latent_dim=latent_dim,
            depth=transformer_depth,
            heads=transformer_heads,
        )

        self.foveal_search = FovealSearch(
            latent_dim=latent_dim,
            temperature=search_temperature,
        )

        self.fovea = DifferentiableFovea(
            crop_size=crop_size,
            radius=fovea_radius,
        )

        self.retinal_encoder = RetinalEncoder(
            latent_dim=latent_dim,
        )

        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
        )

    def forward(
        self,
        frame_t: Tensor,
        frame_next: Tensor,
    ) -> PWMOutput:

        # ---------------------------------------------------------------
        # 1. Encode current coarse scene
        # ---------------------------------------------------------------

        coarse_t = self.coarse_encoder(frame_t)

        # ---------------------------------------------------------------
        # 2. Generate world representation and observation query
        # ---------------------------------------------------------------

        world_latent, query = self.world_transformer(
            coarse_t
        )

        # ---------------------------------------------------------------
        # 3. Encode next frame as searchable coarse keys
        # ---------------------------------------------------------------

        coarse_next = self.coarse_encoder(frame_next)

        # ---------------------------------------------------------------
        # 4. Reverse-search query against next-frame keys
        # ---------------------------------------------------------------

        fixation_xy, logits, weights = self.foveal_search(
            query,
            coarse_next,
        )

        # ---------------------------------------------------------------
        # 5. Extract HD foveal observation
        # ---------------------------------------------------------------

        crop = self.fovea(
            frame_next,
            fixation_xy,
        )

        # ---------------------------------------------------------------
        # 6. Encode observed target
        # ---------------------------------------------------------------

        target_latent = self.retinal_encoder(crop)

        # ---------------------------------------------------------------
        # 7. Predict latent observation from current world + query
        # ---------------------------------------------------------------

        predicted_latent = self.predictor(
            world_latent,
            query,
        )

        return PWMOutput(
            predicted_latent=predicted_latent,
            target_latent=target_latent,
            attention_logits=logits,
            attention_weights=weights,
            fixation_xy=fixation_xy,
            foveal_crop=crop,
        )