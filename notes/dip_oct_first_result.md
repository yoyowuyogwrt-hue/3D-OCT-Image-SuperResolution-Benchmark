# First DIP experiment on OCT (my notes)

I am writing this in the first person so I can paste it into the dissertation later. These results are from one 2D B-scan crop, not from a full 3D volume.

## What I ran

- Data: murine OCT volume `TX12_D0_A1L`, B-scan 256, converted to 8-bit with the same 0.5th–99.5th percentile window as my bicubic baseline, then a centre 256×256 crop.
- Degradation: bicubic downscaling by ×8 (256 → 32).
- DIP: 2000 iterations, Adam, learning rate 0.01, extra input noise, EMA decay 0.9. Same settings as my DIV2K run.
- Stopping rule, decided in advance: always 2000 iterations. I do not pick a checkpoint from HR PSNR.

## Numbers (official result = iteration 2000)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic | 12.87 | 0.063 | 1.042 |
| DIP | 12.24 | 0.047 | 0.670 |

DIP does **not** beat bicubic on all three metrics. PSNR is about 0.64 dB worse and SSIM is slightly worse. LPIPS is better (lower) by about 0.37.

Diagnostic HR scores at checkpoints (not used to pick a model):

| Iteration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 100 | 11.89 | 0.044 | 0.684 |
| 250 | 11.89 | 0.043 | 0.644 |
| 500 | 12.00 | 0.045 | 0.634 |
| 1000 | 12.12 | 0.046 | 0.643 |
| 2000 | 12.24 | 0.047 | 0.670 |

No checkpoint reached bicubic PSNR (12.87). LPIPS was best around iteration 500, but I must not treat that as the official result.

## What I see

The LR reconstruction MSE falls to nearly zero by about iteration 50, then stays flat to 2000. DIP quickly matches the **low-resolution** crop. That is not the same as recovering the **high-resolution** speckle.

In `comparison.png`, the HR crop is grainy (OCT speckle). Bicubic is a blurry, smoother version of the same brightness pattern. DIP is less blurry in a perceptual sense (better LPIPS) but adds **horizontal banding** and does not restore the original grain. Visually I would not call it a successful super-resolution of the retina.

These scores are for the **256 crop**, not the earlier full B-scan bicubic run (PSNR 13.50). I must not mix the two tables.

## How this sits next to DIV2K

On DIV2K ×8, DIP also lost on all three metrics. On this OCT crop it still loses PSNR and SSIM, but LPIPS improves. I will treat that as a mixed finding, not a win: the project rule I am using is that a method should improve **all three** agreed metrics, or I explain why it does not.

## What this does not mean

I have not run 3D DIP, CycleGAN, or BATDiff. One centre crop can miss the layered retina if those layers sit away from the middle of the B-scan. A later experiment can use a layer-centred crop, still with a stop rule fixed in advance.
