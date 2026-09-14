"""Regular-grid downsampling: keep every Nth row and column.

×2 keeps every other pixel (`::2`). ×4 keeps every fourth (`::4`).

Raw stride aliases: high-frequency detail folds into fake low-frequency
patterns. The supervisor's full recipe is therefore a low-pass filter first,
then this stride. `filtered_stride_downsample` does that with torchvision's
existing `gaussian_blur`, not a hand-written kernel.

PyTorch/PIL bicubic is an interpolator. Our previous bicubic downsample also
sets `antialias=False`, because MPS cannot back-propagate through antialiased
bicubic. That is why bicubic is not used to *create* the LR image.
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
    """Keep every `scale`-th pixel. No blur — this aliases on its own."""
    _require_bchw_and_scale(image, scale)
    return image[..., ::scale, ::scale]


def filtered_stride_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Gaussian low-pass, then keep every `scale`-th pixel.

    The blur uses torchvision.transforms.functional.gaussian_blur. Sigma is
    `scale / 2`, a standard choice so the cutoff sits near the new Nyquist
    frequency. The kernel width is the usual 3-sigma support; that is a
    parameter of the existing function, not a custom filter.
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
