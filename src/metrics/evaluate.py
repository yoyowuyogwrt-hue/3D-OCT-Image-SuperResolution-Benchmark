import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image_as_array(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def compute_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def compute_lpips(img1, img2):
    loss_fn = lpips.LPIPS(net="alex")
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0)
    with torch.no_grad():
        return loss_fn(to_tensor(img1), to_tensor(img2)).item()


def evaluate(img1, img2):
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    lpips_score = compute_lpips(img1, img2)
    return {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips_score}
