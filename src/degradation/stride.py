"""Downsample by keeping every Nth row and column.

x2 is `::2`, x4 is `::4`.

If I only do stride, high frequency will alias and look like fake low
frequency pattern. Supervisor said I should blur first, then stride.
`filtered_stride_downsample` use torchvision `gaussian_blur`, I don't
write my own kernel.

Bicubic in PyTorch/PIL is interpolation, and I also set `antialias=False`
because MPS cannot backward it. So I don't use bicubic to *make* the LR.
"""

from __future__ import annotations

import math

import torch
from torchvision.transforms.functional import gaussian_blur


def _require_bchw_and_scale(image: torch.Tensor, scale: int) -> None:
    if image.ndim != 4:
        raise ValueError("image must have shape (batch, channels, height, width).")
    if scale < 1:
        raise ValueError("scale must be at least 1.")
    height, width = image.shape[-2:]
    if height % scale != 0 or width % scale != 0:
        raise ValueError("image height and width must be divisible by scale.")


def stride_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Keep every `scale` pixel. No blur, so it will alias."""
    _require_bchw_and_scale(image, scale)
    return image[..., ::scale, ::scale]


def filtered_stride_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Gaussian blur first, then keep every `scale` pixel.

    I use torchvision.transforms.functional.gaussian_blur. Sigma is
    `scale / 2`, this is a common choice so the cutoff is near the new
    Nyquist frequency. Kernel width is about 3 sigma, this is just the
    parameter of that function, not a filter I design myself.
    """
    _require_bchw_and_scale(image, scale)
    if scale == 1:
        return image

    sigma = 0.5 * float(scale)
    kernel = int(2 * math.ceil(3 * sigma) + 1)
    if kernel % 2 == 0:
        kernel += 1
    blurred = gaussian_blur(
        image,
        kernel_size=[kernel, kernel],
        sigma=[sigma, sigma],
    )
    return blurred[..., ::scale, ::scale]
