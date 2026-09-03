"""
Here I define the small U-Net used in my Deep Image Prior experiment.

I give the network random noise as its input, and it produces a colour
image with the same size as my high-resolution crop. The encoder reduces
the image size, the decoder restores it, and skip connections pass details
between the two sides of the network.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """I use two convolution layers to make one reusable U-Net block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SkipUNet(nn.Module):
    """
    This is my small U-Net with skip connections.

    Height and width must be divisible by 4 because I pool twice.
    The same network works on a 256 crop or a full 1024x512 B-scan:
      noise in  -> (batch, input_depth, height, width)
      image out -> (batch, 3, height, width)   # 3 = R,G,B colour channels
    """

    def __init__(self, input_depth: int = 32, output_depth: int = 3):
        super().__init__()

        # I reduce the image size on the encoder side.
        self.down1 = _conv_block(input_depth, 64)
        self.pool1 = nn.MaxPool2d(2)  # I half the height and width.

        self.down2 = _conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # I half the height and width again.

        # This is the middle of my U-Net.
        self.middle = _conv_block(128, 256)

        # I restore the image size on the decoder side.
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = _conv_block(256 + 128, 128)  # I add 128 channels from the skip connection.

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = _conv_block(128 + 64, 64)  # I add 64 channels from the skip connection.

        # I convert the result into a three-channel RGB image.
        self.final = nn.Conv2d(64, output_depth, kernel_size=1)
        self.out_act = nn.Sigmoid()  # I keep all pixel values between 0 and 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I pass the input down the encoder.
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.middle(self.pool2(d2))

        # I pass it up the decoder and join the skip connections.
        u2 = self.up_conv2(torch.cat([self.up2(m), d2], dim=1))
        u1 = self.up_conv1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out_act(self.final(u1))
