from collections.abc import Callable

import torch

from .bicubic import bicubic_downsample
from .stride import filtered_stride_downsample, stride_downsample

DownsampleFn = Callable[[torch.Tensor, int], torch.Tensor]


def get_downsample(name: str) -> DownsampleFn:
    """Return the HR→LR operator used both to make the LR image and inside DIP."""
    if name == "bicubic":
        return bicubic_downsample
    if name == "stride":
        return stride_downsample
    if name == "filtered_stride":
        return filtered_stride_downsample
    raise ValueError(
        f"Unknown downsample mode {name!r}. "
        "Use 'bicubic', 'stride', or 'filtered_stride'."
    )


__all__ = [
    "bicubic_downsample",
    "filtered_stride_downsample",
    "get_downsample",
    "stride_downsample",
]
