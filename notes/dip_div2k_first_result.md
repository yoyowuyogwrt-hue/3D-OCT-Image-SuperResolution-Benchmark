# First DIP experiment on DIV2K (my notes)

I am writing this in the first person so I can paste it into the dissertation later. These results are from one 2D natural-image crop, not from 3D OCT yet.

## What I am trying to do

My project compares methods that try to recover a high-resolution (HR) image from a low-resolution (LR) image. I do not have a paired “same scan, two machines” HR OCT for super-resolution, so I create a fair test: I start from a sharp image, shrink it, and ask each method to enlarge it again. I then score the reconstruction against the original using PSNR and SSIM (higher is better) and LPIPS (lower is better).

Bicubic interpolation is my baseline. It is a standard, cheap way to enlarge an image. A learned or prior-based method is only useful here if it beats this baseline on the agreed metrics, or if I can explain clearly why it does not.

## What Deep Image Prior is (in my own words)

Deep Image Prior (DIP) does not use a large training dataset. I randomly initialise a U-Net and keep a fixed random-noise tensor as input. At every iteration the network outputs an HR estimate. I downsample that estimate to the LR size and measure mean squared error against my LR image. I update the network to reduce that error. I never show the original HR image to the network during training.

The idea, from Ulyanov et al., is that the structure of the network itself prefers images that look coherent rather than pure noise, so early in optimisation the output can look like a plausible photograph. If I train for too long, the network can start fitting noise in the LR image instead of a clean HR image. That is why early stopping matters, and why I must not cheat when I choose when to stop.

## What I ran

- Data: one DIV2K validation crop (a close-up of an animal, with a sharp black stripe and textured fur).
- Degradation: bicubic downscaling by ×8.
- DIP: 2000 iterations, Adam, learning rate 0.01, extra input noise, and exponential moving average of outputs (decay 0.9) to reduce flicker.
- I saved diagnostic checkpoints at iterations 100, 250, 500, 1000 and 2000.
- I compared the final DIP output with bicubic enlargement of the same LR image.

## Numbers (final iteration 2000)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic | 25.04 | 0.70 | 0.55 |
| DIP | 19.50 | 0.36 | 0.70 |

DIP is worse on all three metrics. The differences are large: PSNR about 5.5 dB lower, SSIM about 0.33 lower, LPIPS about 0.15 worse.

Diagnostic HR scores at checkpoints (not used to pick a model):

| Iteration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 100 | 18.81 | 0.33 | 0.66 |
| 250 | 18.55 | 0.32 | 0.69 |
| 500 | 18.64 | 0.32 | 0.70 |
| 1000 | 18.96 | 0.33 | 0.70 |
| 2000 | 19.50 | 0.36 | 0.70 |

HR PSNR never approached bicubic. It drifted only slightly upward at the end, still far below 25 dB.

## What I see in the figures

In `comparison.png`, the HR reference has a clean white region, a smooth curved black stripe, and fine texture. Bicubic is blurry and blocky: the stripe edge is jagged and the texture is gone. DIP is less blocky than bicubic, but it adds grain, colour shift (the brown region looks muddy), wavy edges, and repeating artefacts. Visually, DIP does not recover the missing detail; it invents a different, noisier image.

In `loss_curve.png`, the LR reconstruction MSE starts high (about 0.08) and falls close to zero by roughly iteration 100, then stays flat to 2000. That means DIP quickly learned to match the **low-resolution** image. Matching LR is not the same as recovering HR. After the LR loss has collapsed, extra iterations can still change the HR image without improving the scores that matter.

## Why I think DIP lost on ×8

×8 throws away most of the pixels. Many different HR images can shrink to the same LR image, so the problem is badly underdetermined. Bicubic answers with a smooth guess. DIP answers by fitting a deep network to one LR image. On this crop, that guess was noisier and less accurate than interpolation.

I also think 2000 iterations is late relative to the loss curve: the LR error had already reached near zero around iteration 100. I did not stop there. I am **not** allowed to look at the HR PSNR table and pick the “best” checkpoint as my official result, because that would use the test image’s ground truth to choose the model. The checkpoint table is only to help me understand the run.

If I do another DIP experiment later, I should decide the stopping rule in advance (for example a fixed iteration, or a rule that only looks at LR loss) and keep ×8 as the hard setting from the project brief. A ×4 run would be easier and might show whether DIP can beat bicubic at all; it would not replace the ×8/×16 OCT experiments.

## What this does and does not mean for the dissertation

This does **not** mean DIP is useless, and it does **not** mean the project has failed. It means that on this first 2D ×8 test, a structural prior without extra training data did not beat interpolation. I will treat that as a finding: a benchmark must include negative results.

I have not yet run DIP, CycleGAN, or BATDiff on the lab 3D OCT volumes. The OCT data I will use for super-resolution are 3D mouse OCT TIFFs (Nicholson and Ward, DOI 10.5523/bris.ypfrg4sz8jwi2ehjqjubbq526). The OCT2Confocal set (DOI 10.5523/bris.1hvd8aw0g6l6g28fnub18hgse4) pairs OCT with confocal microscopy and is for cross-modal translation, not for treating confocal as HR OCT.

Next I will keep this evaluation pipeline and transfer it to OCT B-scans under the same metrics, still against bicubic, before I add a second method.
