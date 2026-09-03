"""
Here I train my Deep Image Prior network.

At each iteration, I pass noise through the U-Net to create a
high-resolution estimate. I shrink this estimate and compare it with my
low-resolution image using MSE loss. I then update the network to reduce
the error. I do not give the original high-resolution image to the network
during training.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.degradation import bicubic_downsample

from .unet import SkipUNet


def get_device() -> torch.device:
    """I use an available GPU when possible and otherwise use the CPU."""
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
    checkpoint_iterations: tuple[int, ...] = (),
    checkpoint_callback: Callable[[int, torch.Tensor], None] | None = None,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[float]]:
    """
    I fit a randomly initialised U-Net so its downscaled output matches `lr`.

    Args:
        lr: My blurry image, with shape (1, 3, h, w) and values from 0 to 1.
        scale: How much larger I make the output (8 changes 32 into 256).
        num_iter: The number of training iterations I use.
        learning_rate: The size of each update to the network weights.
        input_depth: The number of channels in my fixed random-noise input.
        reg_noise_std: Extra noise I add to reduce overfitting.
        ema_decay: How strongly I smooth consecutive outputs (0 disables averaging).
        checkpoint_iterations: One-based iterations whose averaged outputs I save.
        checkpoint_callback: A function that receives each checkpoint image.
        device: The CPU or GPU used for training. I detect it if this is None.
        show_progress: Whether I display a progress bar.

    Returns:
        hr_estimate: My averaged high-resolution estimate after the final iteration.
        losses: The MSE loss from each iteration, which I can plot later.
    """
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be between 0 (inclusive) and 1 (exclusive).")

    if device is None:
        device = get_device()

    lr = lr.to(device)
    _, _, lr_h, lr_w = lr.shape
    hr_h, hr_w = lr_h * scale, lr_w * scale

    net = SkipUNet(input_depth=input_depth, output_depth=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # I create a fixed random-noise input with my target image size.
    net_input = torch.randn(1, input_depth, hr_h, hr_w, device=device)
    net_input_saved = net_input.detach().clone()

    losses: list[float] = []
    output_average: torch.Tensor | None = None
    checkpoint_set = set(checkpoint_iterations)

    iterator = range(num_iter)
    if show_progress:
        iterator = tqdm(iterator, desc="DIP training", leave=True)

    for iteration_index in iterator:
        optimizer.zero_grad()

        # I slightly change the noise at each step to reduce overfitting.
        if reg_noise_std > 0:
            noise = torch.randn_like(net_input_saved) * reg_noise_std
            net_input = net_input_saved + noise
        else:
            net_input = net_input_saved

        hr_guess = net(net_input)
        lr_guess = bicubic_downsample(hr_guess, scale)
        loss = F.mse_loss(lr_guess, lr)

        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)

        # Averaging consecutive outputs suppresses unstable high-frequency noise.
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
