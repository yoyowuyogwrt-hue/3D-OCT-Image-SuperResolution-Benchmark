# DIP on the full OCT B-scan (my notes)

I am writing this in the first person so I can paste it into the dissertation later. This replaces my 256 crop run as the DIP result I will report for OCT.

## Why I re-ran this on the full slice

My supervisor said the crop result was not convincing, and I now agree the crop was the wrong experimental unit. A 256×256 centre crop holds only a small part of the retina. DIP's whole argument is that the network structure prefers coherent, multi-scale image content, so if I cut away the layer structure and the surrounding context I am testing the prior on the material it is least suited to. I also ran my bicubic baseline on the full B-scan, so a full-slice DIP run is the only version that compares against it directly.

## What I ran

- Data: murine OCT volume `TX12_D0_A1L`, B-scan 256, converted to 8-bit with the same 0.5th–99.5th percentile window as my bicubic baseline. **No crop** — the whole 512×1024 slice.
- Degradation: bicubic downscaling by ×8 (512×1024 → 64×128).
- DIP: 3000 iterations, Adam, learning rate 0.01, input depth 32, extra input noise (std 0.03), EMA decay 0.9, seed 0.
- Stopping rule, decided in advance: always 3000 iterations. I do not pick a checkpoint from HR scores.
- Diagnostic checkpoints at 100, 250, 500, 1000, 2000 and 3000.
- Runtime: about 1 hour 27 minutes on the Mac GPU (MPS), roughly 1.05 iterations per second.

I chose 3000 rather than the 2000 I used before because the full slice has eight times as many pixels as the crop, and I wanted to give the network more steps before I judged it. I fixed that number before I looked at any score.

## Numbers (official result = iteration 3000)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic | 13.50 | 0.075 | 1.064 |
| DIP | 13.11 | 0.064 | 0.949 |

DIP does **not** beat bicubic on all three metrics. PSNR is about 0.39 dB worse, SSIM is about 0.011 worse, LPIPS is better (lower) by about 0.115.

Diagnostic HR scores at checkpoints (not used to pick a model):

| Iteration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 100 | 13.18 | 0.069 | 0.885 |
| 250 | 13.22 | 0.068 | 0.889 |
| 500 | 13.27 | 0.068 | 0.916 |
| 1000 | 13.28 | 0.066 | 0.976 |
| 2000 | 13.08 | 0.065 | 0.959 |
| 3000 | 13.11 | 0.064 | 0.949 |

No checkpoint reached bicubic PSNR (13.50) or bicubic SSIM (0.075). Every checkpoint beat bicubic LPIPS.

## What I see

`loss_curve.png` behaves like my earlier runs: the LR reconstruction MSE starts at about 0.13 and collapses to near zero by roughly iteration 100, then stays flat. Once again, matching the **low-resolution** image is easy and happens early, and it is not the same as recovering the high-resolution one. Everything interesting happens after the loss has already flattened, which the loss curve cannot show me.

There is one thing I did not expect: a sharp spike near iteration 1500, where the loss jumps to about 0.043 and then returns to near zero within a few tens of iterations. I take this as an optimisation instability rather than something meaningful. I should not read anything into the fact that LPIPS is worse at 1000 and better at 2000 and 3000, because I have only one run with one seed and cannot separate that from ordinary variation.

In `comparison.png`, the difference from the crop run is obvious. Bicubic is a smooth, washed-out version of the brightness pattern; the bright reflective bands are there but their edges are smeared. DIP gives those bands noticeably crisper edges and keeps the vessel shadows more distinct. It also produces a grainy texture where bicubic produces smooth mush. That grain is what LPIPS is rewarding.

But the grain is not the real speckle. The HR B-scan has very fine, high-frequency speckle; DIP's grain is coarser and blobbier. So DIP is not recovering OCT speckle, it is inventing a texture that is perceptually closer to speckle than blur is. I should say exactly that in the dissertation and not claim more.

One improvement worth recording: the horizontal banding artefact that ruined my crop run is gone. I think the crop was too small for the network's two pooling stages to see any large-scale structure, so it fell back on a repeating pattern. On the full slice there is enough context that this does not happen.

## Does DIP overfit here?

The checkpoint table suggests it does. LPIPS is best at iteration 100 (0.885) and is worse at every later checkpoint, ending at 0.949. SSIM falls steadily from 0.069 to 0.064. PSNR rises a little to iteration 1000 and then drops.

This matches what Ulyanov et al. describe: the network fits coherent structure first, then starts fitting noise in the input. OCT is full of speckle, so there is a lot of noise for it to fit, and I would expect the effect to be stronger here than on a natural photograph.

I want to be careful about what I am allowed to do with this. I **cannot** report iteration 100 as my result, because I only know it is best by comparing checkpoints against the HR image, and the HR image is my test data. That is oracle stopping and it would invalidate the number. What I can do is state that a smaller fixed iteration count is a defensible hyperparameter choice, set it in advance, and re-run. If I do that, the honest description is that I chose it from this run's behaviour, so it is a tuned hyperparameter and not an independent result.

## How this sits next to the 256 crop

| Run | Bicubic PSNR | DIP PSNR | Difference |
|---|---|---|---|
| 256 centre crop, 2000 iterations | 12.87 | 12.24 | −0.64 |
| Full B-scan, 3000 iterations | 13.50 | 13.11 | −0.39 |

DIP is closer to bicubic on the full slice than on the crop, which supports the decision to change the experimental unit. I need to be careful with this comparison though: the crop and the full slice are different images with different content, and the iteration counts differ too, so this is suggestive rather than a controlled result. I must not put the two tables side by side as if one experiment turned into the other.

For the same reason I am not going to claim the LPIPS improvement got smaller (0.37 on the crop, 0.115 on the full slice). LPIPS values are not comparable across different images.

## Why DIP still loses PSNR and SSIM

I think the honest explanation is that PSNR and SSIM are the wrong instruments for what DIP does, rather than that DIP failed.

PSNR is mean squared error in disguise, and MSE is minimised by predicting the average of all plausible answers. At ×8 there are enormously many HR images consistent with my LR input, and their average is blurry. Bicubic is close to that blurry average, so it scores well by construction. DIP commits to one specific sharp answer. Every piece of texture it puts in the slightly wrong place is punished twice, once for missing the real detail and once for adding detail that is not there. SSIM behaves similarly on this data because the local statistics it compares are dominated by speckle.

This is the perception–distortion tradeoff described by Blau and Michaeli: past a certain point, a method cannot improve perceptual quality and distortion at the same time. My three metrics are showing me both ends of it, which is a reason to keep reporting all three rather than a reason to drop one.

## What this does not mean

It does not mean DIP is the wrong method for this project. For my purposes DIP does not have to beat bicubic on its own — its job is to give BATDiff a better starting reference than bicubic does, and the property that matters for that is perceptual structure, which is the one metric DIP wins on.

It also does not mean I have benchmarked DIP on OCT. This is one B-scan from one volume at one downscaling factor, in 2D. I have not run any slice other than 256, I have not run ×16, and I have not run anything in 3D.

## What I do next

1. Use this DIP output as the wavelet reference inside BATDiff, in place of its bicubic upsample, and compare against BATDiff run normally. That is my contribution.
2. If there is time, repeat the DIP run with a smaller fixed iteration count and report it as a tuned hyperparameter, to test whether an early-stopped reference works better than a fully trained one.
3. If there is time after that, run more than one B-scan so I can report a mean rather than a single number.
