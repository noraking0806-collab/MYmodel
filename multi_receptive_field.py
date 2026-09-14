"""Lightweight multi-receptive-field adapter for frozen DINO features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class MultiReceptiveFieldAdapter(nn.Module):
    """Add inexpensive local-to-context reasoning to a frozen feature grid.

    The frozen DINO map is reduced once, processed by parallel depthwise
    convolutions with different dilation rates, fused, and added back through
    a small learnable residual gate.  GroupNorm is used because the retained
    HP profiles have batches as small as two images.
    """

    def __init__(
        self,
        channels: int = 384,
        bottleneck: int = 64,
        dilations: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("channels and bottleneck must be positive")
        dilation_values = tuple(int(value) for value in dilations)
        if not dilation_values or any(value <= 0 for value in dilation_values):
            raise ValueError("dilations must contain positive integers")

        self.channels = int(channels)
        self.bottleneck = int(bottleneck)
        self.dilations = dilation_values
        bottleneck_groups = _group_count(self.bottleneck)
        output_groups = _group_count(self.channels)

        self.reduce = nn.Sequential(
            nn.Conv2d(self.channels, self.bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(bottleneck_groups, self.bottleneck),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.bottleneck,
                        self.bottleneck,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=self.bottleneck,
                        bias=False,
                    ),
                    nn.GroupNorm(bottleneck_groups, self.bottleneck),
                    nn.GELU(),
                )
                for dilation in self.dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.bottleneck * len(self.dilations),
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(output_groups, self.channels),
        )
        # Near-identity initialization retains the pretrained DINO geometry,
        # while a nonzero value lets branch weights receive gradients at step 1.
        self.residual_gate = nn.Parameter(torch.tensor(1.0e-3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW features with {self.channels} channels, "
                f"got {tuple(features.shape)}"
            )
        reduced = self.reduce(features)
        context = torch.cat(
            [branch(reduced) for branch in self.branches], dim=1
        )
        delta = self.fuse(context)
        return features + torch.tanh(self.residual_gate) * delta


__all__ = ["MultiReceptiveFieldAdapter"]
