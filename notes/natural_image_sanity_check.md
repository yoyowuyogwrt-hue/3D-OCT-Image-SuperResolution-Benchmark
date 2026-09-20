# Natural-image sanity check after the failed OCT protocol

## Rationale

The first BATDiff experiment on the full OCT B-scan used bicubic downsampling at
×8. Both BATDiff outputs resembled high-frequency noise rather than retinal
structure. At the supervisory meeting on 9 September 2026, this was treated as a
failed reconstruction and a protocol problem rather than evidence that the
proposed DIP reference was successful.

The revised experiment therefore checks DIP and BATDiff separately on an easier
natural photograph before either method is applied to OCT again. This separates
an implementation failure from difficulty caused by OCT speckle and retinal
structure. A method passes this sanity check if its output remains recognisably
photographic rather than collapsing to noise. DIP is tested first; BATDiff is
tested independently afterward using its published bicubic reference. The DIP
output is not supplied to BATDiff at this stage.

## Corrected degradation

The test image is the complete DIV2K validation photograph `0801.png`, without a
spatial crop. The original RGB image is 2040 × 1356 pixels. To create the
synthetic low-resolution observation, a Gaussian low-pass filter is applied
before retaining every fourth row and column. This produces a 510 × 339 LR
image at scale ×4. Bicubic interpolation is not used to create the LR image.

The same blur-then-stride operator is used in the DIP forward model. At every
iteration, DIP produces an HR-sized estimate, degrades that estimate with the
known operator, and minimizes mean squared error against the observed LR image.
The original HR photograph is not shown to DIP during optimization; it is used
only afterward for visual comparison and calculation of PSNR, SSIM, and LPIPS.

Bicubic interpolation is retained only as an upsampling baseline: the LR image
is enlarged back to 2040 × 1356 and compared with the same HR reference. This
does not conflict with the decision to exclude bicubic downsampling, because
the two operations have different roles.

## DIP configuration and computational adjustment

The full photograph was substantially more memory-intensive than the earlier
256 × 256 crop. With the original 32-channel random input, one full-resolution
iteration exceeded the 14.6 GB memory of a Tesla T4. The experiment therefore
uses 16 random-input channels and CUDA mixed-precision training. The U-Net
feature widths, complete image dimensions, ×4 degradation, optimizer, and
2000-iteration stopping rule remain unchanged. The memory adjustment is fixed
before examining the final reconstruction.

A 20-iteration smoke test completed successfully and showed the coarse penguin
structure, confirming that the full-image pipeline runs without immediate
collapse. Its quantitative scores are not reported as experimental results
because 20 iterations are insufficient for convergence. The formal DIP run
uses 2000 iterations.

## Phase A results (DIP, 2000 iterations)

Official scores from `outputs/sanity/div2k_filtered_stride_x4/scores.txt`.
Write-up: `notes/dip_natural_full_result.md`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample | 24.24 | 0.657 | 0.460 |
| DIP | 27.90 | 0.807 | 0.293 |

DIP improves all three metrics (PSNR +3.66 dB, SSIM +0.150, LPIPS −0.167).
In `comparison.png` the DIP panel is a photograph of the same penguin, sharper
than bicubic and not noise. Phase A **passes**. Diagnostic checkpoints improve
through 2000; there is no late HR collapse on this image. Iteration 2000 remains
the official result because that stopping rule was fixed in advance.

## Phase B results (BATDiff alone, published bicubic `x_ref`)

Official scores from `outputs/sanity/div2k_filtered_stride_x4/batdiff/scores.txt`.
Write-up: `notes/batdiff_natural_full_result.md`. Decision image:
`outputs/sanity/div2k_filtered_stride_x4/batdiff/comparison.png`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample | 24.24 | 0.657 | 0.460 |
| DIP alone | 27.90 | 0.807 | 0.293 |
| BATDiff alone | 10.74 | 0.060 | 0.727 |

BATDiff is worse than bicubic on all three metrics and the panel is noise, not a
penguin. Phase B **fails** the visual rule. Do not return to OCT, and do not
pass `dip.png` into BATDiff, until the supervisor agrees the next step.
