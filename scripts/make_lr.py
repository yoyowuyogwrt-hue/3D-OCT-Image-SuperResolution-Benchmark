"""
Here I prepare one image for my DIP experiment.

I load a high-resolution image, crop a small section from its centre,
shrink the crop by a factor of 8, and save both versions for later.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# I use these settings for this experiment.
SCALE = 8
CROP_SIZE = 256  # I use a small patch so training is faster on my CPU.
IMAGE_NAME = "0801.png"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import bicubic_downsample

HR_PATH = PROJECT_ROOT / "data" / "DIV2K_valid_HR" / IMAGE_NAME
OUT_DIR = PROJECT_ROOT / "outputs" / "dip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# I load the original high-resolution image.
hr_full = Image.open(HR_PATH).convert("RGB")
print(f"Loaded: {HR_PATH.name}")
print(f"Full HR size (width, height): {hr_full.size}")

# I crop the centre because training on the full image would be slow on my CPU.
w, h = hr_full.size
left = (w - CROP_SIZE) // 2
top = (h - CROP_SIZE) // 2
hr_crop = hr_full.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
print(f"Cropped HR size: {hr_crop.size}")

# I create the low-resolution image using bicubic downscaling.
lr_w, lr_h = CROP_SIZE // SCALE, CROP_SIZE // SCALE  # My 256-pixel crop becomes 32 pixels.
hr_array = np.asarray(hr_crop, dtype=np.float32) / 255.0
hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1).unsqueeze(0)
lr_tensor = bicubic_downsample(hr_tensor, SCALE)
lr_array = lr_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
lr = Image.fromarray(np.rint(lr_array * 255.0).astype(np.uint8), mode="RGB")
print(f"LR size after ×{SCALE} downscale: {lr.size}")

# I save both images so I can use them in the next DIP steps.
hr_crop.save(OUT_DIR / "hr_crop.png")
lr.save(OUT_DIR / "lr.png")
print(f"Saved: {OUT_DIR / 'hr_crop.png'}")
print(f"Saved: {OUT_DIR / 'lr.png'}")
print("Step 4.1 done.")
