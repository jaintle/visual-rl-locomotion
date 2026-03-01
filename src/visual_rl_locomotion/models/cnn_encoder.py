"""
CNN encoder for pixel-based observations.

Converts a batch of CHW float32 frames in [0, 1] to a fixed-size
latent vector suitable for feeding into a policy or value head.

Architecture:
    Conv(3  -> 32, kernel=8, stride=4)  ReLU
    Conv(32 -> 64, kernel=4, stride=2)  ReLU
    Conv(64 -> 64, kernel=3, stride=1)  ReLU
    Flatten
    Linear(computed_flat_dim -> latent_dim)  ReLU

This is the standard "Nature DQN" convolutional stack, scaled down to
work comfortably on 64×64 frames.  The flattened size after the conv
layers is computed from a dummy forward pass, so the encoder works
with any square input resolution (≥ 16px recommended).

No attention, no transformers, no skip connections.  Clarity first.
"""

from typing import Tuple

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Three-layer convolutional encoder.

    Args:
        obs_shape:  Shape of a single observation in CHW format,
                    e.g. (3, 64, 64).  The batch dimension is NOT included.
        latent_dim: Dimensionality of the output feature vector.  Default 256.
    """

    def __init__(self, obs_shape: Tuple[int, int, int], latent_dim: int = 256) -> None:
        super().__init__()

        C, H, W = obs_shape

        # Three conv layers — standard DQN-style stack.
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Compute flattened size via a dummy forward pass.
        # This is safe (no side effects) and more robust than algebraic formulae.
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            flat_size = int(self.conv(dummy).numel())  # == 64 * h' * w'

        self.linear = nn.Sequential(
            nn.Linear(flat_size, latent_dim),
            nn.ReLU(),
        )

        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of pixel observations.

        Args:
            obs: Float32 tensor of shape (B, 3, H, W) with values in [0, 1].

        Returns:
            Float32 tensor of shape (B, latent_dim).
        """
        # obs: (B, C, H, W)
        x = self.conv(obs)          # (B, 64, h', w')
        x = x.reshape(x.size(0), -1)  # (B, flat_size)  — reshape avoids contiguous issues
        return self.linear(x)       # (B, latent_dim)
