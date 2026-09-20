# DIP on the full DIV2K photograph (Phase A sanity check)

I am writing this in the first person so I can paste it into the dissertation later. This is the natural-image check requested after the failed OCT BATDiff run: DIP alone, full photograph, blur-then-stride ×4. It is not an OCT result.

## What this run is for

The ×8 bicubic OCT protocol produced BATDiff outputs that looked like noise. My supervisor treated that as a failed reconstruction, not as evidence about the DIP reference. Before I go back to OCT I need to know that DIP itself can reconstruct a photograph under the corrected degradation. The pass/fail rule is visual: the output must look like a photo, not like snow. Beating bicubic on the metrics is useful but is not the pass criterion.

## What I ran

- Data: complete DIV2K validation image `0801.png` (a rockhopper penguin). Original size 2040 × 1356. No spatial crop.
- Degradation: Gaussian blur, then keep every fourth row and column (`filtered_stride`, ×4). LR size 510 × 339. Bicubic was **not** used to create the LR image.
- DIP forward operator: the same blur-then-stride, so training matches how LR was made.
- DIP: 2000 iterations, Adam, learning rate 0.01, extra input noise, EMA decay 0.9, seed 0.
- Memory adjustment, fixed before the scored run: 16 random-input channels and CUDA mixed precision, because 32 channels exceeded the 14.6 GB Tesla T4. U-Net widths and image size were not reduced.
- Stopping rule, decided in advance: always 2000 iterations. I do not pick a checkpoint from HR scores.
- Diagnostic checkpoints at 5, 20, 100, 250, 500, 1000 and 2000.
- Files: `outputs/sanity/div2k_filtered_stride_x4/` (unzipped from `dip_natural_full.zip`).

Bicubic interpolation is only the **upsample** baseline: I enlarge `lr.png` back to 2040 × 1356 and score it against the original photograph.

## Numbers (official result = iteration 2000)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample | 24.24 | 0.657 | 0.460 |
| DIP | 27.90 | 0.807 | 0.293 |

DIP beats bicubic on all three metrics: PSNR +3.66 dB, SSIM +0.150, LPIPS −0.167.

Diagnostic HR scores at checkpoints (not used to pick a model):

| Iteration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 100 | 22.69 | 0.514 | 0.646 |
| 250 | 24.34 | 0.620 | 0.538 |
| 500 | 26.01 | 0.703 | 0.433 |
| 1000 | 27.23 | 0.768 | 0.339 |
| 2000 | 27.90 | 0.807 | 0.293 |

All three curves improve through the whole run. DIP crosses bicubic PSNR between iterations 100 and 250, and crosses bicubic SSIM and LPIPS between 250 and 500. There is no OCT-style late collapse: LPIPS keeps falling to 2000. I still report iteration 2000 only, because that was the pre-set stopping rule.

## What I see

`comparison.png` is the decision image. The HR panel is a sharp photograph: yellow crest, pink bill, water on the rocks, feather edges. Bicubic is recognisably the same scene but softer; rock texture and the crest are smeared. DIP is also recognisably a photograph of the same penguin. It is sharper than bicubic and closer to the original. It is not noise, not snow, and not the failed OCT BATDiff texture.

At iteration 20 the output is still grainy and almost grey, with only a coarse penguin-shaped blob. That is expected for a smoke test. By 2000 the colour and structure have returned. The remaining error versus HR is mild blur and some lost fine rock grain, not a different image.

`loss_curve.png` matches every previous DIP run: LR reconstruction MSE starts near 0.08, collapses close to zero by about iteration 100, then stays flat. Matching the small image is easy and happens early. The HR metrics keep improving long after that, which I could not have seen from the loss curve alone.

## How this compares with my earlier DIP runs

| Run | DIP vs bicubic | Looks like |
|---|---|---|
| DIV2K 256 crop, bicubic ×8 | loses all three | photograph, but noisier than bicubic |
| DIV2K full, stride-only ×4 (no blur) | PSNR up, SSIM and LPIPS down | photograph |
| OCT full B-scan, bicubic ×8 | LPIPS only | retina, grainy |
| **This run: full DIV2K, blur+stride ×4** | **wins all three** | **photograph** |

Two things I will write into the methods chapter:

1. The implementation can reconstruct a natural image. The OCT BATDiff snow was not “DIP/BATDiff can never make a picture”.
2. The corrected degradation matters. On the same photograph, stride without blur did not win all metrics; blur-then-stride did. I should not treat those two operators as interchangeable.

This still does not prove DIP will win on OCT. A penguin photo has no speckle. Phase A only tells me DIP is working on the easy case.

## What I do next

Phase B is now done: BATDiff **alone** on the same pair is still noise.
Write-up: `notes/batdiff_natural_full_result.md`. I do not go back to OCT yet.
