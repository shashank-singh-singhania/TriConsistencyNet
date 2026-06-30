"""
TriConsistencyNet

Cross-Consistency Attention (CCA)

Author: Shashank Singh
"""

import torch
import torch.nn as nn


class CrossConsistencyAttention(nn.Module):
    """
    Cross-Consistency Attention

    Inputs:
        Spatial Features   : (B,1280,7,7)
        Frequency Features : (B,256,7,7)

    Outputs:
        Refined Features : (B,1280,7,7)
        Attention Map    : (B,1280,7,7)
    """

    def __init__(
        self,
        spatial_channels=1280,
        frequency_channels=256,
    ):

        super().__init__()

        # ----------------------------------------
        # Frequency Projection
        # 256 -> 1280
        # ----------------------------------------

        self.frequency_projection = nn.Sequential(

            nn.Conv2d(
                frequency_channels,
                spatial_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(spatial_channels),

            nn.SiLU(inplace=True),
        )

        # ----------------------------------------
        # Spatial Projection
        # 1280 -> 1280
        # ----------------------------------------

        self.spatial_projection = nn.Sequential(

            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(spatial_channels),

            nn.SiLU(inplace=True),
        )

        # ----------------------------------------
        # Attention Generator
        # ----------------------------------------

        self.attention = nn.Sequential(

            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(spatial_channels),

            nn.SiLU(inplace=True),

            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        spatial_features,
        frequency_features,
    ):

        spatial = self.spatial_projection(
            spatial_features
        )

        frequency = self.frequency_projection(
            frequency_features
        )

        consistency = spatial * frequency

        attention_map = self.attention(
            consistency
        )

        refined = spatial_features * (
            1.0 + attention_map
        )

        return refined, attention_map
