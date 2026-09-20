# BATDiff on the full DIV2K photograph (Phase B sanity check)

I am writing this in the first person so I can paste it into the dissertation later. This is BATDiff **alone** on the same penguin photograph as Phase A. It is not an OCT result, and it does not use a DIP reference.

## What this run is for

Phase A showed that DIP can reconstruct this photograph under blur-then-stride ×4. The remaining question was whether BATDiff, with its published bicubic `x_ref`, can also produce a picture rather than snow. The pass/fail rule is the same as Phase A: the output must look like a photo, not like noise. Beating bicubic on the metrics is useful but is not the pass criterion.

I did **not** pass `dip.png` into BATDiff. This is the published method, not the contribution.

## What I ran

- Data: the same pair as Phase A. DIV2K `0801.png`, full 2040 × 1356 photograph. LR is `filtered_stride` ×4 (510 × 339). Bicubic was not used to create the LR image.
- BATDiff `x_ref`: bicubic *enlargement* of `lr.png` (upstream `create_img_scales()`). No `--xref_image`.
- Code: `notebooks/batdiff_natural.ipynb` settings, run on Colab T4 after a runtime reset. Three new cells at the bottom of the notebook: clone and patch, upload `lr.png`/`hr.png` only, train then download the zip immediately.
- Settings: `dim=64`, `train_num_steps=4000`, `timesteps=100`, `atrous_level=6`, `sr_factor=4`, `--ts 1`, à-trous wavelet `b3`. Upstream default `dim` is 200; 64 is the same reduced width I used on the OCT B-scan because the full photograph does not fit a T4 at 128 or 200.
- Training finished on 15 September 2026 (finest-scale sample timestamp 19:13). I scored on my Mac with the same `src/metrics.evaluate` as DIP. The finest-scale file is `_s5_`; it is already 2040 × 1356, so I did not resize it.
- Files: `outputs/sanity/div2k_filtered_stride_x4/batdiff/`. Decision image: `comparison.png`. Numbers: `scores.txt`.

The first Colab full run also finished (~104.5 min) but the virtual machine was recycled before I downloaded the zip. This scored run is the second training, with an immediate download. I did not re-run DIP.

## Numbers (official table)

Bicubic and DIP are the official Phase A scores from `outputs/sanity/div2k_filtered_stride_x4/scores.txt`. BATDiff is the Mac score of the `_s5_` sample.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample | 24.24 | 0.657 | 0.460 |
| DIP alone | 27.90 | 0.807 | 0.293 |
| BATDiff alone | 10.74 | 0.060 | 0.727 |

BATDiff versus bicubic: PSNR −13.50 dB, SSIM −0.597, LPIPS +0.267 (all worse).  
BATDiff versus DIP: PSNR −17.16 dB, SSIM −0.746, LPIPS +0.434 (all worse).

## What I see

`comparison.png` is the decision image. HR is a sharp penguin photograph. Bicubic is the same scene, softer. DIP is also a photograph of the same penguin, sharper than bicubic. The BATDiff panel is **high-frequency noise**. There is no penguin, no rock, and no colour structure. Coarser-scale samples (`_s0_` through `_s4_`) and `sample-1.png` look the same: snow, not a picture.

`running_loss.png` has about 40 points after a spike at the first step, then sits roughly between 0.05 and 0.15. Loss going down is not the same as recovering the photograph.

This is the same visual failure as the OCT BATDiff panels, now on an easy natural image where DIP already works.

## What this does and does not mean

Phase B **fails** the supervisor’s visual rule.

This does **not** mean DIP is broken. Phase A already passed on the identical LR/HR pair.

This does **not** mean I should go back to OCT yet. The 9 September order was: natural photograph first; only return to OCT when the methods look like photos. BATDiff still does not.

This does **not** license the DIP-as-`x_ref` contribution yet. That substitution is only meaningful after BATDiff alone can reconstruct a picture. Feeding DIP into a method that currently emits noise would mix two questions.

Three things can be true at once:

1. **The natural-image check did its job.** The OCT snow was not only “OCT is hard”. The same BATDiff settings also fail on a penguin photo.
2. **The T4 capacity story is no longer enough.** A 256 HR-crop identity run with the README defaults (`dim=200`, `ts=4`, `train_num_steps=101`, `sr_factor=1`) is also snow. So the failure is not only “full photo, dim=64”.
3. **I should not keep changing settings until a picture appears and then call that the protocol.** Further BATDiff debugging is an implementation probe, not a new OCT experiment. I must label it as such in the report.

## Extra check: the training target is already a penguin

I compared the BATDiff checkout to https://github.com/MaryamHeidari-1994/BATDiff. It is the same commit as `origin/main`. The only local edits are the DIP `--xref_image` patch and a Colab `pkg_resources` fallback. Phase B did not pass `--xref_image`, so `x_ref` is still the published line: bicubic enlargement of `lr.png`.

I then ran BATDiff’s own `create_img_scales()` on my Mac, with no training. The published `x_ref` from `lr.png` at ×4 is already a recognisable penguin (the same image as the bicubic upsample baseline). The per-scale training targets (`scale_0` … `scale_5`) are also penguins. BATDiff’s sampled output is snow at every scale, including the coarsest (`s0`). So the failure is not “we only gave it a tiny LR image, therefore snow”. A working sampler should have reconstructed something like that penguin-shaped `x_ref`. Figure: `outputs/sanity/div2k_filtered_stride_x4/batdiff/xref_check/training_target_vs_output.png`.

Feeding the original HR photograph as the input, with `sr_factor=1`, would make `x_ref` the sharp photo (about 32× more energy in the finest wavelet plane than the LR bicubic reference). I did not use `sr_factor=4` on full HR (that would try to invent an 8160×5424 image). The cheaper probe is the 256 crop below.

## Identity probe (256 HR crop, README defaults)

Input: centre 256×256 crop of the same penguin (`hr_crop256.png`). No downsampling. `sr_factor=1`, `dim=200`, `ts=4`, `train_num_steps=101`. Files: `outputs/sanity/div2k_filtered_stride_x4/batdiff/hr_identity/`.

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| BATDiff vs HR crop | 9.13 | 0.035 | 0.655 |

The `_s5_` sample, `_s0_` sample, and `sample-1.png` grid are all noise. There is no plumage, no flipper, no rock. The identity probe **fails**. So BATDiff, as I can run it from the public repo, does not copy a small photograph either. That is stronger than Phase B: it is not explained by LR being empty, by OCT speckle, or by reducing `dim` to 64.

## What I do next

1. Put the Phase B table, the identity probe, and both `comparison.png` figures in the thesis as the sanity-check finding.
2. Do **not** start OCT stride ×4 until the supervisor says to write the failure and move on, or until BATDiff can reconstruct a photograph.
3. Do **not** pass DIP into BATDiff on this penguin until BATDiff alone works.
4. Ask at the Wednesday meeting: is the honest result “I could not get the public BATDiff code to reconstruct a natural image”, or should remaining GPU time go into reading their sampling path / contacting the authors?
