from pathlib import Path

from PIL import Image

data_dir = Path("data/DIV2K_valid_HR")
images = sorted(data_dir.glob("*.png"))

print(f"Found {len(images)} images")

if not images:
    print("No images found. Check that data/DIV2K_valid_HR exists.")
else:
    first = images[0]
    img = Image.open(first)
    print(f"First image: {first.name}")
    print(f"Size (width x height): {img.size}")
    print("Data check passed!")
