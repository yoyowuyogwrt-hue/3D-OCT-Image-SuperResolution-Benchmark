import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import numpy as np
from src.metrics.evaluate import evaluate

image_path = Path("data/DIV2K_valid_HR/0801.png")
hr = Image.open(image_path).convert("RGB")

w, h = hr.size
lr = hr.resize((w // 8, h // 8), Image.BICUBIC)
restored = lr.resize((w, h), Image.BICUBIC)

hr_array = np.array(hr)
restored_array = np.array(restored)

scores = evaluate(hr_array, restored_array)

print(f"Testing on: {image_path.name}")
print(f"Original size: {hr.size}")
print("---")
print(f"PSNR:  {scores['PSNR']:.2f}  (higher = better)")
print(f"SSIM:  {scores['SSIM']:.4f}  (higher = better, max 1.0)")
print(f"LPIPS: {scores['LPIPS']:.4f}  (lower = better)")
print("---")
print("Metrics test passed!")
