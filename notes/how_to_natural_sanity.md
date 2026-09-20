# How I run the natural-image sanity check myself

Step-by-step for me (and for the methods chapter). Two places of work:

- **My Mac terminal** — make the small image, run DIP
- **Google Colab** — run BATDiff (it needs an NVIDIA GPU; my Mac cannot run the unmodified BATDiff code)

Do DIP first. Do BATDiff second. Do not skip to OCT, and do not pass a DIP image into BATDiff yet.

---

## Part A — DIP on my Mac

### A1. Open a Terminal on my Mac

1. I open Terminal.
2. I type commands there and press **Return** after each line.

If the prompt does not already sit in the project folder, I go there:

```bash
cd /Users/yoyowu/Projects/3D-OCT-Image-SuperResolution-Benchmark
```

### A2. Turn on the Python environment

```bash
source venv/bin/activate
```

The prompt should start with `(venv)`. If it does not, I am using the wrong Python.

### A3. Make the low-resolution photo the new way

This uses the **original** DIV2K `0801.png` (no 256 crop). It blurs, then keeps
every fourth row and column (`::4`, scale ×4). It does **not** use bicubic to
shrink the photo. ×2 left too much detail, so the image did not look low-resolution.

```bash
python scripts/prepare_natural_sanity.py
```

What I should see:

- `HR: (2040, 1356)`
- `LR (filtered_stride ×4): (510, 339)`
- files under `outputs/sanity/div2k_filtered_stride_x4/`

To look at them in Finder:

```bash
open outputs/sanity/div2k_filtered_stride_x4
```

Open `degradation_comparison.png` first. Three panels only: original → skip pixels with no blur → official blur then `::4`. There is **no bicubic downsample** in this figure. The skip-only panel is shown with nearest-neighbour enlarge so the small grid looks blocky (that is display, not how LR was made).

### A4. Run DIP alone

This can take **much longer** than the 256 crop (full photograph, ~2040×1356).
Keep the Mac plugged in.

```bash
python scripts/run_dip.py \
  --output-dir outputs/sanity/div2k_filtered_stride_x4 \
  --scale 4 \
  --downsample filtered_stride
```

The two flags must match how I made `lr.png`: **scale 4** and **filtered_stride**.

A progress bar `DIP training: 45%` is normal. When it finishes, the Terminal prints PSNR / SSIM / LPIPS.

### A5. What I open to judge “does DIP work?”

```bash
open outputs/sanity/div2k_filtered_stride_x4/comparison.png
open outputs/sanity/div2k_filtered_stride_x4/scores.txt
```

| File | Question it answers |
|---|---|
| `comparison.png` | Does the DIP panel look like a photograph, or like noise? |
| `scores.txt` | Numbers vs bicubic **upsample** (enlarging the small image, not shrinking) |
| `loss_curve.png` | Did the LR error fall? (matching the small image ≠ recovering the sharp photo) |

**Pass for Phase A:** DIP looks like the animal/photo, not like snow. Numbers can still lose to bicubic on some metrics; that is allowed. Noise is not allowed.

---

## Part B — BATDiff alone on Google Colab

BATDiff’s code says `device = cuda:...`. It will not run on my Mac GPU (MPS). Colab is the same place I used for the OCT run.

### B1. Open the notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in with Google.
2. **File → Upload notebook**.
3. Choose `notebooks/batdiff_natural.ipynb` from this project (Finder: the `notebooks` folder).

Do **not** re-use `batdiff_oct.ipynb` for this check. That notebook is ×8 OCT and also runs the DIP-inside-BATDiff contribution. This sanity check is **BATDiff alone** at ×4 on the photograph.

### B2. Switch on a GPU

1. **Runtime → Change runtime type**
2. Hardware accelerator: **T4 GPU**
3. Save

### B3. Upload the two images from Part A

In the notebook there is an **Upload** cell. When it asks, I choose:

- `outputs/sanity/div2k_filtered_stride_x4/lr.png`
- `outputs/sanity/div2k_filtered_stride_x4/hr.png`

I do **not** upload `dip.png`. This run must not see DIP.

### B4. Run the cells in order

1. Leave `SMOKE_TEST = True` and run from the top through the BATDiff cell. A few minutes. The picture may look like noise — that only tests that the code runs.
2. If that finished without a red error: set `SMOKE_TEST = False`, **Runtime → Restart runtime**, run from the top again. This is the real check (about 1 hour).

### B5. What “BATDiff works” looks like

The comparison figure should look like a photograph, not like the OCT snow. If BATDiff is still noise on this easy photo, the problem is the BATDiff settings/code, not OCT.

Download the zip the last cell offers, and put it in `outputs/sanity/div2k_filtered_stride_x4/batdiff/` on my Mac.

---

## What I do not do yet

- Do not pass `--xref_image` / DIP into BATDiff.
- Do not go back to OCT until both Part A and Part B look like photographs.
- Do not use `scripts/make_lr.py` for this check — that script is the **old** bicubic ×8 pair.

---

## Later: the same recipe on OCT (Phase C)

Only after A and B pass. Then I will make an OCT LR by stride ×4 (512×1024 → 128×256) and run DIP, then BATDiff, with `--scale 4` / `--sr_factor 4`. That is a later session.
