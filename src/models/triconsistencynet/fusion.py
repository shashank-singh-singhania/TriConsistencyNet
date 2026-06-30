"""
TriConsistencyNet

Adaptive Feature Fusion (AFF)

Author: Shashank Singh
"""

import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion

    Input:
        (B,1280,7,7)

    Output:
        (B,1280)
    """

    def __init__(
        self,
        channels=1280,
        reduction=4,
    ):

        super().__init__()

        self.pre_pool = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = channels // reduction

        self.channel_gate = nn.Sequential(

            nn.Linear(
                channels,
                hidden,
            ),

            nn.SiLU(inplace=True),

            nn.Linear(
                hidden,
                channels,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.pre_pool(x)

        pooled = self.pool(x)

        pooled = pooled.flatten(1)

        gate = self.channel_gate(pooled)

        fused = pooled * gate

        return fused
