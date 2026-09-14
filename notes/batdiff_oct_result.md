# BATDiff on the full OCT B-scan (my notes)

I am writing this in the first person so I can paste it into the dissertation later. These results are from one 2D B-scan, not from a 3D volume. Organised files live in `outputs/oct/batdiff/`.

## What I am testing

BATDiff trains a diffusion model on à-trous wavelet planes of a single reference image `x_ref`. In the published code, `x_ref` is the low-resolution input stretched back up with bicubic interpolation. That reference has almost no high-frequency content, so the finest wavelet plane is nearly empty.

My contribution is only this substitution: I give BATDiff my DIP reconstruction as `x_ref` instead of the bicubic upsample. The wavelet decomposition then runs on the DIP image, so every scale the diffusion model is trained to reproduce changes. I do not average DIP and BATDiff outputs, and I do not train on a new LR made from the DIP image.

So the controlled comparison is:

| Run | `x_ref` | Role |
|---|---|---|
| A | bicubic upsample of LR | published BATDiff |
| B | DIP output (`outputs/oct/dip_full/`) | this project |

Everything else is meant to stay the same.

## What I ran

- Data: murine OCT volume `TX12_D0_A1L`, B-scan 256, full 512×1024 slice, same 8-bit window as my DIP full-slice run. Degradation: bicubic ×8 (512×1024 → 64×128).
- Code: `notebooks/batdiff_oct.ipynb` on Colab GPU, after `scripts/batdiff_dip_patch.py` on a BATDiff checkout.
- Shared settings (full run, not the smoke test): `dim=64`, `train_num_steps=4000`, `timesteps=100`, `atrous_level=6`, `sr_factor=8`, à-trous wavelet `b3`. Upstream default `dim` is 200; I used 64 because 512×1024 with six full-resolution scales does not fit a T4 at the default size.
- Run A finished around 14:06 on 8 September 2026; run B around 15:16. I score with the same `src/metrics/evaluate.py` as DIP.
- I copied the Colab dump from `batdiff_oct_results (2)/` into `outputs/oct/batdiff/` without the `.pt` weights.

## Numbers (official table)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×8 | 13.50 | 0.075 | 1.064 |
| DIP alone | 13.11 | 0.064 | 0.949 |
| BATDiff (bicubic ref) | 11.36 | 0.024 | 0.980 |
| BATDiff (DIP ref) | 10.95 | 0.018 | 0.883 |

Contribution (DIP-ref minus bicubic-ref BATDiff):

- PSNR −0.41 dB (worse)
- SSIM −0.006 (worse)
- LPIPS −0.097 (better; lower is better)

BATDiff with a DIP reference does **not** beat upstream BATDiff on PSNR or SSIM. It is also worse than both bicubic and DIP-alone on those two metrics. LPIPS is the only number that moves in the direction I hoped.

The bicubic and DIP rows match `outputs/oct/dip_full/scores.txt`. I must not mix this table with the old 256-crop DIP numbers.

## What I see

In `comparison.png`, HR is a grainy layered B-scan. Bicubic is a smooth, washed-out version of the brightness pattern. DIP is still slightly blurrier than HR on PSNR/SSIM, but the bright bands look a bit crisper than bicubic, which is why LPIPS preferred it.

Both BATDiff panels look like **high-frequency noise**, not like a retina. The layered structure is almost gone. `sr_finest_scale.png` for each run is essentially salt-and-pepper texture. That matches the collapse in SSIM (0.075 → 0.02). I should describe this as a failed reconstruction of structure, not as “sharper speckle.”

The training-loss plots (`running_loss.png`) have about 40 points on the x-axis after a huge spike at the first step. If the notebook logged every 100 of 4000 steps, that would be expected. After the spike, run A sits around 0.02–0.03; run B stays noisier and higher, around 0.06–0.10. Loss going down is not the same as recovering the HR B-scan.

## Why I think this happened

Three things can be true at once, and I should not pick only the story that sounds good.

1. **The substitution did change what was learned.** DIP-ref LPIPS is better than bicubic-ref BATDiff, and the loss curve is different. So `x_ref` is not a no-op. The finest-scale DIP wavelet energy argument in the notebook still explains *why* I made the change. It does not prove that the change produces a usable image.

2. **This BATDiff run is under-powered relative to the paper.** I used `dim=64` instead of 200, on one slice, at ×8. If the finest-scale samples are noise, the diffusion model did not synthesise retinal structure at this capacity and step count. I cannot claim I reproduced BATDiff at the authors’ operating point.

3. **LPIPS on noisy OCT is easy to over-interpret.** A noisy image can be closer to grainy HR speckle in a natural-image perceptual network than a blur is, while still being useless. The comparison figure is the check: I would not call either BATDiff panel a successful super-resolution.

The perception–distortion tradeoff still applies. PSNR/SSIM punish invented texture; LPIPS can reward it. That is a reason to keep all three metrics, not a reason to declare a win on LPIPS alone.

## What this does and does not mean

This does **not** mean the contribution idea is disproven forever. It means that on this one B-scan, with this reduced `dim`, DIP-as-`x_ref` did not improve PSNR or SSIM over standard BATDiff, and neither BATDiff output beat bicubic on those metrics.

This does **not** mean DIP was wasted. DIP still beats bicubic on LPIPS on the same slice, and it is the reference I actually fed into run B.

I have not run BATDiff on any other slice, at ×16, or in 3D. I have not run CycleGAN as super-resolution.

## What I do next

1. Put this table and `comparison.png` in the thesis results chapter, with the honest visual description (BATDiff looks like noise).
2. If there is GPU time before 25 September, a second pair of runs at higher `dim` (or fewer wavelet scales) is the only way to test whether this is a capacity problem. I should not keep changing `x_ref` until a number looks good.
3. If there is no time, the dissertation result is: DIP reference changes BATDiff, LPIPS moves a little, PSNR/SSIM and the pictures do not support a claim that DIP+BATDiff is better super-resolution.
