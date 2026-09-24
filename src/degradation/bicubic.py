"""Bicubic downsample I use for the synthetic SR experiment. It can backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bicubic_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample a BCHW tensor. DIP fit this same operator, so train and test match."""
    if image.ndim != 4:
        raise ValueError("image must have shape (batch, channels, height, width).")
    if scale < 1:
        raise ValueError("scale must be at least 1.")

    height, width = image.shape[-2:]
    if height % scale != 0 or width % scale != 0:
        raise ValueError("image height and width must be divisible by scale.")

    return F.interpolate(
        image,
        size=(height // scale, width // scale),
        mode="bicubic",
        align_corners=False,
        # antialias=False, because MPS cannot backward through antialiased bicubic.
        antialias=False,
    )
