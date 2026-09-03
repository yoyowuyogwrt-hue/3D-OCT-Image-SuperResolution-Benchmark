"""Run DIP on one crop of the OCT B-scan and compare it with bicubic.

I reuse the 8-bit HR B-scan from scripts/oct_bicubic_baseline.py so the
intensity window stays the same. I crop 256x256, shrink by x8, fit DIP for a
fixed number of iterations, and score DIP against bicubic on that same crop.
"""

from __future__ import annotations

import argparse
import csv
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
from src.models.dip import train_dip
from src.models.dip.train import get_device

SCALE = 8
CROP_SIZE = 256
HR_BSCAN_PATH = PROJECT_ROOT / "outputs" / "oct" / "hr_bscan.png"
OUT_DIR = PROJECT_ROOT / "outputs" / "oct" / "dip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIP on one OCT B-scan crop.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000],
    )
    parser.add_argument("--ema-decay", type=float, default=0.9)
    return parser.parse_args()


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
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


def centre_crop(rgb: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    height, width = rgb.shape[:2]
    if height < size or width < size:
        raise ValueError(f"B-scan {width}x{height} is smaller than crop {size}.")
    top = (height - size) // 2
    left = (width - size) // 2
    return rgb[top : top + size, left : left + size], top, left


def save_loss_curve(losses: list[float], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("LR reconstruction MSE")
    plt.title("DIP training loss (OCT crop)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison(
    hr: np.ndarray,
    bicubic: np.ndarray,
    dip: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes,
        (hr, bicubic, dip),
        ("HR crop", "Bicubic", "DIP"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_curves(
    checkpoint_rows: list[dict[str, float]],
    bicubic_scores: dict[str, float],
    path: Path,
) -> None:
    iterations = [int(row["iteration"]) for row in checkpoint_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, metric, higher_is_better in zip(
        axes,
        ("PSNR", "SSIM", "LPIPS"),
        (True, True, False),
    ):
        axis.plot(iterations, [row[metric] for row in checkpoint_rows], marker="o")
        axis.axhline(
            bicubic_scores[metric],
            color="tab:orange",
            linestyle="--",
            label="Bicubic",
        )
        arrow = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{metric} ({arrow})")
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not HR_BSCAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HR_BSCAN_PATH}. Run `python scripts/oct_bicubic_baseline.py` first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1. Load the 8-bit HR B-scan from the bicubic experiment.")
    hr_full = np.asarray(Image.open(HR_BSCAN_PATH).convert("RGB"))
    print(f"  {HR_BSCAN_PATH.name}: shape {hr_full.shape}")

    print(f"\nStep 2. Take a centre {CROP_SIZE}x{CROP_SIZE} crop so DIP can run on this Mac.")
    hr_crop, top, left = centre_crop(hr_full, CROP_SIZE)
    print(f"  crop top={top}, left={left}, shape={hr_crop.shape}")
    Image.fromarray(hr_crop).save(OUT_DIR / "hr_crop.png")

    print(f"\nStep 3. Shrink the crop by x{SCALE} with the DIP bicubic operator.")
    hr_tensor = to_tensor(hr_crop)
    lr_tensor = bicubic_downsample(hr_tensor, SCALE)
    lr = from_tensor(lr_tensor)
    print(f"  LR shape: {lr.shape}")
    Image.fromarray(lr).save(OUT_DIR / "lr.png")

    print("\nStep 4. Bicubic baseline: enlarge LR back to 256x256.")
    bicubic = from_tensor(bicubic_upsample(lr_tensor, SCALE))
    Image.fromarray(bicubic).save(OUT_DIR / "bicubic.png")

    print("\nStep 5. Score bicubic vs the HR crop (this is the number DIP must beat).")
    bicubic_scores = evaluate(hr_crop, bicubic)
    print(
        f"  Bicubic: PSNR={bicubic_scores['PSNR']:.4f}, "
        f"SSIM={bicubic_scores['SSIM']:.4f}, "
        f"LPIPS={bicubic_scores['LPIPS']:.4f}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    checkpoint_iterations = sorted(
        {iteration for iteration in args.checkpoints if 0 < iteration <= args.iterations}
        | {args.iterations}
    )
    checkpoint_tensors: dict[int, torch.Tensor] = {}

    def keep_checkpoint(iteration: int, output: torch.Tensor) -> None:
        checkpoint_tensors[iteration] = output

    print(f"\nStep 6. Train DIP for {args.iterations} iterations on {device}.")
    print("  I do not show the HR crop to the network. Stopping rule is fixed iterations.")
    dip_tensor, losses = train_dip(
        lr_tensor,
        scale=SCALE,
        num_iter=args.iterations,
        ema_decay=args.ema_decay,
        checkpoint_iterations=tuple(checkpoint_iterations),
        checkpoint_callback=keep_checkpoint,
        device=device,
    )
    dip = from_tensor(dip_tensor)
    Image.fromarray(dip).save(OUT_DIR / "dip.png")

    print("\nStep 7. Score DIP vs the same HR crop.")
    dip_scores = evaluate(hr_crop, dip)
    print(
        f"  DIP:     PSNR={dip_scores['PSNR']:.4f}, "
        f"SSIM={dip_scores['SSIM']:.4f}, "
        f"LPIPS={dip_scores['LPIPS']:.4f}"
    )

    checkpoint_rows: list[dict[str, float]] = []
    for iteration in checkpoint_iterations:
        checkpoint_image = tensor_to_image(checkpoint_tensors[iteration])
        checkpoint_image.save(checkpoint_dir / f"iter_{iteration:04d}.png")
        checkpoint_scores = evaluate(hr_crop, np.asarray(checkpoint_image))
        checkpoint_rows.append({"iteration": iteration, **checkpoint_scores})

    with (OUT_DIR / "checkpoint_scores.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["iteration", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    save_metric_curves(checkpoint_rows, bicubic_scores, OUT_DIR / "checkpoint_metrics.png")
    save_loss_curve(losses, OUT_DIR / "loss_curve.png")
    save_comparison(hr_crop, bicubic, dip, OUT_DIR / "comparison.png")

    improved = (
        dip_scores["PSNR"] > bicubic_scores["PSNR"]
        and dip_scores["SSIM"] > bicubic_scores["SSIM"]
        and dip_scores["LPIPS"] < bicubic_scores["LPIPS"]
    )
    verdict = (
        "Verdict: DIP improves all three metrics over bicubic."
        if improved
        else "Verdict: DIP does not improve all three metrics over bicubic."
    )
    lines = [
        "OCT DIP vs bicubic, TX12_D0_A1L slice 256, centre 256x256 crop, x8",
        f"Stopping rule: fixed {args.iterations} iterations (not chosen from HR scores).",
        (
            f"Bicubic: PSNR={bicubic_scores['PSNR']:.4f}, "
            f"SSIM={bicubic_scores['SSIM']:.4f}, "
            f"LPIPS={bicubic_scores['LPIPS']:.4f}"
        ),
        (
            f"DIP:     PSNR={dip_scores['PSNR']:.4f}, "
            f"SSIM={dip_scores['SSIM']:.4f}, "
            f"LPIPS={dip_scores['LPIPS']:.4f}"
        ),
        verdict,
        (
            f"Delta (DIP - bicubic): PSNR={dip_scores['PSNR'] - bicubic_scores['PSNR']:+.4f}, "
            f"SSIM={dip_scores['SSIM'] - bicubic_scores['SSIM']:+.4f}, "
            f"LPIPS={dip_scores['LPIPS'] - bicubic_scores['LPIPS']:+.4f}"
        ),
        "Note: checkpoint HR metrics are diagnostic only; they must not be used to select a test-image checkpoint.",
        "Note: these scores are for the 256 crop, not the full 1024x512 B-scan bicubic run.",
    ]
    (OUT_DIR / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved files in {OUT_DIR}")


if __name__ == "__main__":
    main()
