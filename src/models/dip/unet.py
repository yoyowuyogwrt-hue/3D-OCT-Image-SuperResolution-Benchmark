"""
Here I write the small U-Net for my Deep Image Prior experiment.

Input is random noise, output is a colour image, same size as my HR crop.
Encoder make the feature map smaller, decoder make it big again, and skip
connection pass the detail from encoder to decoder.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two conv layers. I reuse this block in encoder, middle and decoder."""
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
    My small U-Net with skip connection.

    H and W must can divide by 4, because I pool two times.
    Same network can do a 256 crop or the full 1024x512 B-scan:
      noise in  -> (batch, input_depth, height, width)
      image out -> (batch, 3, height, width)   # 3 = R,G,B
    """

    def __init__(self, input_depth: int = 32, output_depth: int = 3):
        super().__init__()

        # Encoder side, make the feature map smaller.
        self.down1 = _conv_block(input_depth, 64)
        self.pool1 = nn.MaxPool2d(2)  # half the H and W

        self.down2 = _conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # half again

        # Middle of the U-Net (bottleneck).
        self.middle = _conv_block(128, 256)

        # Decoder side, upsample back.
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = _conv_block(256 + 128, 128)  # 256+128 because I concat the skip

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = _conv_block(128 + 64, 64)  # same, concat 64 channel from encoder

        # Last conv, change to 3 channel (RGB).
        self.final = nn.Conv2d(64, output_depth, kernel_size=1)
        self.out_act = nn.Sigmoid()  # keep pixel in 0~1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Go down first, and I keep d1, d2 for the skip.
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.middle(self.pool2(d2))

        # Then go up, and concat with the encoder feature.
        u2 = self.up_conv2(torch.cat([self.up2(m), d2], dim=1))
        u1 = self.up_conv1(torch.cat([self.up1(u2), d1], dim=1))

        return self.out_act(self.final(u1))
