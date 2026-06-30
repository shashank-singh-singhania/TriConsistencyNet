"""
TriConsistencyNet

Frequency Guidance Encoder (FGE)

Author: Shashank Singh
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.SiLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)

        out += identity

        out = self.activation(out)

        return out


class FrequencyGuidanceEncoder(nn.Module):

    """
    RGB
      ↓
    FFT
      ↓
    Log Magnitude
      ↓
    Normalize
      ↓
    CNN
      ↓
    (B,256,7,7)
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.SiLU(inplace=True),

            nn.MaxPool2d(2),

            ResidualBlock(64),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),

            nn.SiLU(inplace=True),

            ResidualBlock(128),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),

            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d((7, 7)),
        )

    @staticmethod
    def fft_preprocess(x):

        # Force float32 for FFT to avoid extremely slow float16 complex operations under autocast
        x_f32 = x.float()

        fft = torch.fft.fft2(
            x_f32,
            norm="ortho",
        )

        magnitude = torch.abs(fft)

        magnitude = torch.log1p(magnitude)

        min_val = magnitude.amin(
            dim=(2, 3),
            keepdim=True,
        )

        max_val = magnitude.amax(
            dim=(2, 3),
            keepdim=True,
        )

        magnitude = (
            magnitude - min_val
        ) / (
            max_val - min_val + 1e-8
        )

        return magnitude.to(x.dtype)

    def forward(self, x):

        frequency = self.fft_preprocess(x)

        frequency = self.encoder(frequency)

        return frequency
