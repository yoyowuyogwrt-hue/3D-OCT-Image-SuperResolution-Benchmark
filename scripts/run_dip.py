"""Run my 2D DIV2K Deep Image Prior experiment.

I load the LR image I prepared, train DIP, save the result, and compare
with bicubic upsample. The score is against the HR image.
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

from src.metrics.evaluate import evaluate
from src.models.dip import train_dip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIP on one DIV2K crop.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dip",
        help="Folder that already contains lr.png and hr_crop.png.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of DIP training iterations (default: 2000).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Super-resolution scale factor (default: 8).",
    )
    parser.add_argument(
        "--downsample",
        choices=("bicubic", "stride", "filtered_stride"),
        default="bicubic",
        help="Must match how lr.png was made (default: bicubic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for a reproducible run (default: 0).",
    )
    parser.add_argument(
        "--input-depth",
        type=int,
        default=32,
        help="Channels in DIP's random-noise input (default: 32).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision to reduce GPU memory.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000],
        help="Iterations whose reconstructions and metrics I save.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="Output-averaging strength used to suppress unstable noise (default: 0.9).",
    )
    return parser.parse_args()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor (1, 3, H, W)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Float tensor (1, 3, H, W) -> PIL RGB image."""
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_loss_curve(losses: list[float], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("LR reconstruction MSE")
    plt.title("DIP training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison(
    hr: Image.Image,
    bicubic: Image.Image,
    dip: Image.Image,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes,
        (hr, bicubic, dip),
        ("HR reference", "Bicubic upsample", "DIP"),
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
    """Plot the checkpoint scores so I can see the training. I don't use this to pick the best model."""
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
    output_dir = args.output_dir
    lr_path = output_dir / "lr.png"
    hr_path = output_dir / "hr_crop.png"

    if not lr_path.exists() or not hr_path.exists():
        raise FileNotFoundError(
            "lr.png or hr_crop.png is missing. "
            "For the old bicubic ×8 pair run `python scripts/make_lr.py`. "
            "For the natural-image pair run `python scripts/prepare_natural_sanity.py`."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_iterations = sorted(
        {iteration for iteration in args.checkpoints if 0 < iteration <= args.iterations}
        | {args.iterations}
    )
    checkpoint_tensors: dict[int, torch.Tensor] = {}

    def keep_checkpoint(iteration: int, output: torch.Tensor) -> None:
        checkpoint_tensors[iteration] = output

    lr_image = Image.open(lr_path).convert("RGB")
    hr_image = Image.open(hr_path).convert("RGB")
    expected_size = (lr_image.width * args.scale, lr_image.height * args.scale)
    if hr_image.size != expected_size:
        raise ValueError(
            f"HR size {hr_image.size} does not match LR size {lr_image.size} "
            f"at scale x{args.scale}; expected {expected_size}."
        )

    print(f"Loaded LR image: {lr_path} ({lr_image.size})")
    print(
        f"Training DIP for {args.iterations} iterations at x{args.scale} "
        f"with {args.downsample} downsampling..."
    )
    dip_tensor, losses = train_dip(
        image_to_tensor(lr_image),
        scale=args.scale,
        num_iter=args.iterations,
        input_depth=args.input_depth,
        ema_decay=args.ema_decay,
        downsample=args.downsample,
        use_amp=args.amp,
        checkpoint_iterations=tuple(checkpoint_iterations),
        checkpoint_callback=keep_checkpoint,
    )

    dip_image = tensor_to_image(dip_tensor)
    bicubic_image = lr_image.resize(hr_image.size, Image.Resampling.BICUBIC)
    dip_path = output_dir / "dip.png"
    bicubic_path = output_dir / "bicubic.png"
    dip_image.save(dip_path)
    bicubic_image.save(bicubic_path)

    hr_array = np.asarray(hr_image)
    bicubic_scores = evaluate(hr_array, np.asarray(bicubic_image))
    dip_scores = evaluate(hr_array, np.asarray(dip_image))

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows: list[dict[str, float]] = []
    for iteration in checkpoint_iterations:
        checkpoint_image = tensor_to_image(checkpoint_tensors[iteration])
        checkpoint_image.save(checkpoint_dir / f"iter_{iteration:04d}.png")
        checkpoint_scores = evaluate(hr_array, np.asarray(checkpoint_image))
        checkpoint_rows.append({"iteration": iteration, **checkpoint_scores})

    checkpoint_scores_path = output_dir / "checkpoint_scores.csv"
    with checkpoint_scores_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["iteration", "PSNR", "SSIM", "LPIPS"],
        )
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    save_metric_curves(
        checkpoint_rows,
        bicubic_scores,
        output_dir / "checkpoint_metrics.png",
    )

    score_lines = [
        f"Degradation: {args.downsample} ×{args.scale} (DIP forward operator matches this).",
        (
            f"Bicubic upsample: PSNR={bicubic_scores['PSNR']:.4f}, "
            f"SSIM={bicubic_scores['SSIM']:.4f}, "
            f"LPIPS={bicubic_scores['LPIPS']:.4f}"
        ),
        (
            f"DIP:              PSNR={dip_scores['PSNR']:.4f}, "
            f"SSIM={dip_scores['SSIM']:.4f}, "
            f"LPIPS={dip_scores['LPIPS']:.4f}"
        ),
    ]

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
    score_lines.append(verdict)
    score_lines.extend(
        [
            (
                f"Delta (DIP - bicubic): PSNR={dip_scores['PSNR'] - bicubic_scores['PSNR']:+.4f}, "
                f"SSIM={dip_scores['SSIM'] - bicubic_scores['SSIM']:+.4f}, "
                f"LPIPS={dip_scores['LPIPS'] - bicubic_scores['LPIPS']:+.4f}"
            ),
            (
                "Note: checkpoint HR metrics are diagnostic only; they must not be "
                "used to select a test-image checkpoint."
            ),
        ]
    )

    scores_path = output_dir / "scores.txt"
    scores_path.write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    save_loss_curve(losses, output_dir / "loss_curve.png")
    save_comparison(hr_image, bicubic_image, dip_image, output_dir / "comparison.png")

    print("\n".join(score_lines))
    print(f"Saved DIP reconstruction: {dip_path}")
    print(f"Saved metrics: {scores_path}")
    print(f"Saved visual comparison: {output_dir / 'comparison.png'}")
    print(f"Saved loss curve: {output_dir / 'loss_curve.png'}")
    print(f"Saved checkpoints: {checkpoint_dir}")
    print(f"Saved checkpoint metrics: {checkpoint_scores_path}")


if __name__ == "__main__":
    main()
