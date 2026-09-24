Organised copy of the 8 September 2026 Colab BATDiff runs.

Source dump (kept, not deleted):
  batdiff_oct_results (2)/

Layout:
  comparison.png          five-panel figure used for the dissertation
  scores.csv / scores.txt same four-method table
  bicubic_ref/            Run A — BATDiff with bicubic x_ref (upstream)
  dip_ref/                Run B — BATDiff with DIP x_ref (this project)

Each run folder contains:
  running_loss.png        training loss
  sample.png              BATDiff sample-1.png (all scales concatenated)
  sr_finest_scale.png     final_samples *_s5_*  (finest wavelet scale; the SR output)
  final_samples/          s0–s5, as written by BATDiff

Weights (model-1.pt) stay in the original dump only.
