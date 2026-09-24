"""
Here I train my Deep Image Prior network.

Every step I put noise into the U-Net and it give me a HR guess. Then I
downsample this guess and compare with my LR image by MSE. After that I
update the weights to make the loss smaller. I never show the real HR
image to the network when I train, because DIP should only see the LR.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.degradation import get_downsample

from .unet import SkipUNet


def get_device() -> torch.device:
    """I use GPU if the computer has one. MPS first, then CUDA, or just CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_dip(
    lr: torch.Tensor,
    scale: int = 8,
    num_iter: int = 2000,
    learning_rate: float = 0.01,
    input_depth: int = 32,
    reg_noise_std: float = 0.03,
    ema_decay: float = 0.9,
    downsample: str = "bicubic",
    checkpoint_iterations: tuple[int, ...] = (),
    checkpoint_callback: Callable[[int, torch.Tensor], None] | None = None,
    device: torch.device | None = None,
    show_progress: bool = True,
    use_amp: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """
    I fit a random U-Net so after downsample, the output is close to `lr`.

    Args:
        lr: My LR image, shape (1, 3, h, w), pixel value from 0 to 1.
        scale: How many times bigger the output is (8 means 32 become 256).
        num_iter: How many step I train.
        learning_rate: How big each weight update is.
        input_depth: Channel number of the fixed noise input.
        reg_noise_std: Small extra noise, so the network not overfit too fast.
        ema_decay: How much I average the outputs. 0 means I don't average.
        downsample: Must be the same way I made the LR image
            (`bicubic`, `stride`, or `filtered_stride`).
        checkpoint_iterations: Which step (start from 1) I want to save.
        checkpoint_callback: A function I call when I save a checkpoint.
        device: CPU or GPU. If None I pick it myself.
        show_progress: If True I show the progress bar.
        use_amp: On CUDA I use mixed precision, so it use less GPU memory.

    Returns:
        hr_estimate: The averaged HR image after the last step.
        losses: MSE of every step, I can plot this later.
    """
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be between 0 (inclusive) and 1 (exclusive).")

    if device is None:
        device = get_device()
    if use_amp and device.type != "cuda":
        raise ValueError("Mixed precision is only supported on CUDA in this experiment.")

    lr = lr.to(device)
    downsample_fn = get_downsample(downsample)
    _, _, lr_h, lr_w = lr.shape
    hr_h, hr_w = lr_h * scale, lr_w * scale

    net = SkipUNet(input_depth=input_depth, output_depth=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # The noise is fixed and the size is the HR size I want. DIP paper do it like this.
    net_input = torch.randn(1, input_depth, hr_h, hr_w, device=device)
    net_input_saved = net_input.detach().clone()

    losses: list[float] = []
    output_average: torch.Tensor | None = None
    checkpoint_set = set(checkpoint_iterations)

    iterator = range(num_iter)
    if show_progress:
        iterator = tqdm(iterator, desc="DIP training", leave=True)

    for iteration_index in iterator:
        optimizer.zero_grad(set_to_none=True)

        # I change the noise a little every step, otherwise it overfit very quick.
        if reg_noise_std > 0:
            noise = torch.randn_like(net_input_saved) * reg_noise_std
            net_input = net_input_saved + noise
        else:
            net_input = net_input_saved

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hr_guess = net(net_input)
            lr_guess = downsample_fn(hr_guess, scale)
            loss = F.mse_loss(lr_guess, lr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.item())
        losses.append(loss_value)

        # One step output is quite noisy, so I average it with the previous ones.
        current_out = hr_guess.detach()
        if output_average is None:
            output_average = current_out.clone()
        else:
            output_average.mul_(ema_decay).add_(current_out, alpha=1 - ema_decay)

        iteration = iteration_index + 1
        if iteration in checkpoint_set and checkpoint_callback is not None:
            checkpoint_callback(iteration, output_average.clamp(0, 1).cpu())

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss_value:.6f}")

    if output_average is None:
        raise ValueError("num_iter must be at least 1.")
    return output_average.clamp(0, 1).cpu(), losses
