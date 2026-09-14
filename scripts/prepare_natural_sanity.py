"""Prepare one full DIV2K photograph for the natural-image sanity check.

Default: no crop, scale ×4 (every fourth row and column) after a Gaussian
low-pass. This is not bicubic shrinking.

This script only makes the HR/LR pair and a degradation figure. It does not
run DIP or BATDiff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import get_downsample, stride_downsample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DIV2K image with filtered stride downsampling."
    )
    parser.add_argument("--image-name", default="0801.png")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Centre crop in pixels. 0 means use the original image (default).",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--downsample",
        choices=("filtered_stride", "stride", "bicubic"),
        default="filtered_stride",
        help="How to make the LR image. Default is blur, then every Nth pixel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sanity" / "div2k_filtered_stride_x4",
    )
    return parser.parse_args()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def upsample(image: Image.Image, size: tuple[int, int], resample: int) -> Image.Image:
    return image.resize(size, resample)


def centre_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image {image.size} is smaller than the requested crop {crop_size}."
        )
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def trim_to_multiple(image: Image.Image, multiple: int) -> Image.Image:
    """Drop the right/bottom edge so width and height divide `multiple`.

    DIP's U-Net pools twice, so the HR size must also be divisible by 4.
    """
    width, height = image.size
    step = multiple
    new_w = width - (width % step)
    new_h = height - (height % step)
    if new_w < step or new_h < step:
        raise ValueError(f"Image {image.size} is too small for multiple={step}.")
    if (new_w, new_h) != (width, height):
        image = image.crop((0, 0, new_w, new_h))
        print(f"Trimmed HR from {(width, height)} to {image.size} so it divides {step}.")
    return image


def save_degradation_figure(
    hr: Image.Image,
    stride_lr: Image.Image,
    filtered_lr: Image.Image,
    scale: int,
    path: Path,
) -> None:
    """HR vs skip-only vs official blur-then-stride. No bicubic downsample."""
    hr_size = hr.size
    panels = [
        (hr, "HR (original)"),
        (
            upsample(stride_lr, hr_size, Image.Resampling.NEAREST),
            f"Stride ×{scale} only (::{scale})\nno low-pass; aliases",
        ),
        (
            upsample(filtered_lr, hr_size, Image.Resampling.NEAREST),
            f"Official LR ×{scale}:\nGaussian blur, then ::{scale}",
        ),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 5))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.crop_size < 0:
        raise ValueError("crop-size must be 0 (full image) or a positive pixel size.")
    if args.crop_size and args.crop_size % args.scale != 0:
        raise ValueError("crop-size must be divisible by scale.")

    hr_path = PROJECT_ROOT / "data" / "DIV2K_valid_HR" / args.image_name
    if not hr_path.exists():
        raise FileNotFoundError(f"Missing DIV2K image: {hr_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hr = Image.open(hr_path).convert("RGB")
    print(f"Loaded: {hr_path.name} {hr.size}")
    if args.crop_size:
        hr = centre_crop(hr, args.crop_size)
        print(f"Centre crop: {hr.size}")
    # Divisible by scale, and by 4 so DIP's U-Net can pool twice.
    hr = trim_to_multiple(hr, multiple=max(args.scale, 4))

    hr_tensor = image_to_tensor(hr)
    downsample_fn = get_downsample(args.downsample)
    lr_tensor = downsample_fn(hr_tensor, args.scale)
    lr = tensor_to_image(lr_tensor)

    stride_only = tensor_to_image(stride_downsample(hr_tensor, args.scale))
    filtered = tensor_to_image(get_downsample("filtered_stride")(hr_tensor, args.scale))

    hr.save(output_dir / "hr.png")
    hr.save(output_dir / "hr_crop.png")
    lr.save(output_dir / "lr.png")
    stride_only.save(output_dir / "lr_stride_only.png")
    save_degradation_figure(
        hr,
        stride_only,
        filtered,
        args.scale,
        output_dir / "degradation_comparison.png",
    )

    print(f"HR: {hr.size}")
    print(f"LR ({args.downsample} ×{args.scale}): {lr.size}")
    print(f"Saved: {output_dir / 'hr.png'}")
    print(f"Saved: {output_dir / 'lr.png'}  (official LR: blur then ::{args.scale})")
    print(f"Saved: {output_dir / 'lr_stride_only.png'}  (no blur, for the figure only)")
    print(f"Saved: {output_dir / 'degradation_comparison.png'}")
    print(
        "Next: python scripts/run_dip.py "
        f"--output-dir {output_dir} --scale {args.scale} "
        f"--downsample {args.downsample}"
    )


if __name__ == "__main__":
    main()
