"""Shared differentiable bicubic degradation for synthetic SR experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bicubic_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample a BCHW image tensor using the operator fitted by DIP."""
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
        # MPS does not implement the antialiased bicubic backward operation.
        antialias=False,
    )
