# Supervisor meeting, 9 September 2026 (my notes)

I am writing this in the first person so I can paste it into the dissertation later.

## What I showed

I showed the OCT BATDiff comparison (`outputs/oct/batdiff/comparison.png`). The two BATDiff panels look like high-frequency noise, not like a retinal B-scan. My supervisor treated this as a failed reconstruction, not as a usable super-resolution result.

The protocol that produced those images was: bicubic downsampling by ×8 on a full OCT B-scan, then DIP and BATDiff.

## What my supervisor asked me to change

Three things were wrong with that protocol, and they should be fixed before I treat OCT as the scientific target again.

1. **×8 is too harsh.** A factor of 4 is usually enough for a first super-resolution test. ×8 throws away almost all of the pixels, so many very different high-resolution images can shrink to the same low-resolution image. The inverse problem is then too underdetermined for a fair method check.

2. **Do not create the low-resolution image with bicubic interpolation.** He treats bicubic as an interpolator, not a proper downsampler. The recipe is: **low-pass filter, then keep every Nth row and column** (×2 = every other; ×4 = every fourth). Skipping pixels with no filter aliases. I checked PyTorch/PIL bicubic: our code uses `F.interpolate(..., mode="bicubic", antialias=False)`, so it does **not** apply that anti-alias low-pass. PyTorch *can* set `antialias=True`, but that backward pass is not implemented on Mac MPS, so DIP could not train with it here. I therefore use torchvision's existing `gaussian_blur`, then stride, in `filtered_stride_downsample`.

3. **Do not debug the methods on OCT first.** OCT is noisy and structurally unusual. If BATDiff outputs noise on OCT, I cannot tell whether the code is broken or the OCT inverse problem is too hard. I should first run DIP alone and BATDiff alone on natural high-resolution photographs (DIV2K), with the corrected downsampling, and compare against the original photograph. Only when those outputs look like photographs — not like noise — should I return to OCT with the same assured protocol.

The DIP-inside-BATDiff contribution (replacing BATDiff’s bicubic `x_ref` with a DIP image) comes *after* both methods work on their own. I should not stack two untested methods.

## What stays in the thesis

The failed ×8 bicubic OCT BATDiff run is still a result. It is evidence that the original protocol was too aggressive, and it motivates the change of degradation. I will report it honestly as a negative / protocol result, then show the natural-image sanity check, then the corrected OCT experiments.

I must keep two words distinct in the write-up:

- **Downsampling** (HR → LR): Gaussian blur, then stride. Not bicubic.
- **Upsampling** (LR → an HR-sized guess): bicubic enlargement is still the cheap baseline I score against. BATDiff’s published code still builds its reference `x_ref` by bicubically *enlarging* the LR image. That is a different bicubic from the one my supervisor rejected.

DIP’s training loss must use the *same* downsample operator that made the LR image. If LR is blur-then-every-4th-pixel, DIP must do that to its estimate too.

## Experiment order from here

| Phase | What I run | Success looks like |
|---|---|---|
| A | DIV2K **full** photo, blur then stride ×4, **DIP alone** | Output looks like a photograph, not noise. SSIM should not collapse. |
| B | Same pair, **BATDiff alone** (published `x_ref`, `sr_factor=4`) | Same visual check. If this is still noise, the BATDiff settings/code are the problem, not OCT. |
| C | Same stride ×4 protocol on the OCT B-scan, DIP alone then BATDiff alone | Now a failure is about OCT, because the methods already worked on photographs. |
| D | Only then: DIP as BATDiff `x_ref` (the contribution) | Interpretation is allowed only after A–C. |

## How I make the full-image ×4 pair

```bash
source venv/bin/activate
python scripts/prepare_natural_sanity.py
python scripts/run_dip.py \
  --output-dir outputs/sanity/div2k_filtered_stride_x4 \
  --scale 4 \
  --downsample filtered_stride
```

`degradation_comparison.png` shows the full photograph with **no bicubic downsample panel**. ×2 kept too much detail, so the official factor is ×4.

## Methods paragraph (for the report)

To synthesise a low-resolution input I do not use bicubic interpolation. Bicubic is an interpolator; in this codebase PyTorch bicubic downsampling also runs with `antialias=False`, so it does not implement a proper anti-aliasing low-pass. Instead I apply a Gaussian blur with torchvision's `gaussian_blur` (standard deviation \(\sigma = s/2\)) and then keep every \(s\)-th row and column. For the natural-image sanity check, \(s=4\) on the full DIV2K validation image `0801.png` (2040\(\times\)1356 \(\rightarrow\) 510\(\times\)339). The factor 4 was chosen after \(\times 2\) still looked close to the original photograph. The same operator is used inside DIP's training loss, so the network is fitted to the degradation that actually produced the LR image. Bicubic interpolation is used only later, as a cheap **upsampling** baseline and as BATDiff's published `x_ref`, not to create the LR image.

BATDiff still runs on Colab GPU. Use the same `lr.png` / `hr.png` pair, set `--sr_factor 4`, and run only the published (bicubic-`x_ref`) configuration until Phase B passes. A T4 may run out of memory on the full 2040×1356 image; if it does, that is a later practical problem, not a reason to go back to bicubic ×8 for making LR.

## Why this is worth writing up

A benchmark is not only the final numbers. Showing that I diagnosed a protocol failure, changed the degradation to something my supervisor considers fair, verified the implementations on natural images, and only then returned to OCT is part of the scientific story. It also protects the contribution: DIP-as-`x_ref` is only meaningful if BATDiff itself can reconstruct structure.
