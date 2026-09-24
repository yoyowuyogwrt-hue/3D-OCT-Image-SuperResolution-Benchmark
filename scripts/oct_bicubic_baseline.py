"""Do one OCT B-scan and score the bicubic x8 baseline.

I take the original B-scan as HR. I shrink it x8 to make a fake LR, then
bicubic upsample back, and score with PSNR, SSIM and LPIPS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import bicubic_downsample
from src.metrics.evaluate import evaluate

SCALE = 8
SLICE_INDEX = 256
VOLUME = "TX12_D0_A1L"
TIFF_PATH = (
    PROJECT_ROOT
    / "data"
    / "oct"
    / "TX12D0"
    / VOLUME
    / f"{VOLUME}_{SLICE_INDEX}.tif"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "oct"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def uint16_to_uint8(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Change 16-bit OCT slice to 8-bit, so I can score it like DIV2K.

    Most pixel is much smaller than 65535, so if I just /65535 the image
    is almost black. I clip to 0.5% ~ 99.5% of this slice, then scale to
    0-255. I only do this one time, and after that this 8-bit image is my HR.
    """
    low, high = np.percentile(raw, (0.5, 99.5))
    if high <= low:
        high = float(raw.max())
        low = float(raw.min())
    stretched = np.clip(raw, low, high)
    scaled = (stretched - low) / (high - low)
    return np.rint(scaled * 255.0).astype(np.uint8), float(low), float(high)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Copy the grey channel 3 times. LPIPS is trained on RGB so it need 3 channel."""
    return np.stack([gray, gray, gray], axis=2)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def bicubic_upsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    return torch.nn.functional.interpolate(
        image,
        size=(height * scale, width * scale),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )


def save_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def save_comparison(
    hr: np.ndarray,
    lr: np.ndarray,
    bicubic: np.ndarray,
    path: Path,
) -> None:
    lr_shown = np.array(
        Image.fromarray(lr, mode="RGB").resize(
            (hr.shape[1], hr.shape[0]),
            resample=Image.NEAREST,
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 8))
    titles = (
        "HR B-scan (8-bit reference)",
        f"LR ×{SCALE} (nearest, just to show pixels)",
        "Bicubic upsample",
    )
    for axis, image, title in zip(axes, (hr, lr_shown, bicubic), titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Step 1. Load one B-scan from the volume I downloaded.")
    print(f"  path: {TIFF_PATH}")
    if not TIFF_PATH.exists():
        raise FileNotFoundError(f"Missing {TIFF_PATH}. Download the volume first.")

    raw = np.array(Image.open(TIFF_PATH))
    print(f"  numpy shape (height, width): {raw.shape}")
    print(f"  dtype: {raw.dtype}  min={int(raw.min())}  max={int(raw.max())}")
    print("  This is one 2D slice through a 3D OCT volume, not a colour photo.")

    print("\nStep 2. Convert 16-bit values to an 8-bit RGB image.")
    gray8, low, high = uint16_to_uint8(raw)
    hr = gray_to_rgb(gray8)
    print(f"  display window: {low:.0f} to {high:.0f} (0.5th–99.5th percentile)")
    print(f"  HR RGB shape: {hr.shape}")
    hr_path = OUT_DIR / "hr_bscan.png"
    save_png(hr, hr_path)
    print(f"  saved: {hr_path}")

    print(f"\nStep 3. Shrink HR by ×{SCALE} with the same bicubic operator DIP uses.")
    height, width = hr.shape[:2]
    print(f"  {width}×{height} must divide by {SCALE}: {width % SCALE == 0 and height % SCALE == 0}")
    lr_tensor = bicubic_downsample(to_tensor(hr), SCALE)
    lr = from_tensor(lr_tensor)
    print(f"  LR RGB shape: {lr.shape}  (width={lr.shape[1]}, height={lr.shape[0]})")
    lr_path = OUT_DIR / "lr_bscan.png"
    save_png(lr, lr_path)
    print(f"  saved: {lr_path}")

    print("\nStep 4. Enlarge LR back to HR size with bicubic interpolation (the baseline).")
    restored_tensor = bicubic_upsample(lr_tensor, SCALE)
    restored = from_tensor(restored_tensor)
    print(f"  restored shape: {restored.shape}")
    restored_path = OUT_DIR / "bicubic_bscan.png"
    save_png(restored, restored_path)
    print(f"  saved: {restored_path}")

    print("\nStep 5. Score restored vs HR. PSNR/SSIM higher better; LPIPS lower better.")
    scores = evaluate(hr, restored)
    print(f"  PSNR:  {scores['PSNR']:.4f}")
    print(f"  SSIM:  {scores['SSIM']:.4f}")
    print(f"  LPIPS: {scores['LPIPS']:.4f}")

    comparison_path = OUT_DIR / "comparison.png"
    save_comparison(hr, lr, restored, comparison_path)
    print(f"\nStep 6. Saved side-by-side figure: {comparison_path}")

    scores_path = OUT_DIR / "scores.txt"
    scores_path.write_text(
        "OCT bicubic ×8 baseline on "
        f"{VOLUME} slice {SLICE_INDEX}\n"
        f"HR TIFF: {TIFF_PATH.name}\n"
        f"HR size (W×H): {width}×{height}\n"
        f"LR size (W×H): {lr.shape[1]}×{lr.shape[0]}\n"
        f"16-bit window: {low:.1f} to {high:.1f}\n"
        f"PSNR:  {scores['PSNR']:.4f}\n"
        f"SSIM:  {scores['SSIM']:.4f}\n"
        f"LPIPS: {scores['LPIPS']:.4f}\n"
    )
    print(f"  scores: {scores_path}")
    print("Done.")


if __name__ == "__main__":
    main()
