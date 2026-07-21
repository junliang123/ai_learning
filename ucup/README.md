# UCup RSRP prediction

Working code for the 2026 Universal Cup Summer Online Challenge task on RSRP prediction.

The intended model treats each base-station-to-query-point path as one sample. It combines
engineering parameters, propagation geometry, point-cloud obstruction features, and a learned
residual model. Validation is always grouped by base station or physical site.

The original 32 GB dataset lives in `TrainingData.26UCupSummer/` and is intentionally ignored
by Git.

## Current pipeline

```bash
.venv/bin/python scripts/build_rasters.py
.venv/bin/python scripts/train_geometry_baseline.py --use-path-features \
  --output artifacts/path_baseline
.venv/bin/python scripts/train_and_submit.py
.venv/bin/python scripts/apply_same_sector_transfer.py
.venv/bin/python scripts/train_sparse_unet.py
.venv/bin/python scripts/train_path_field.py
```

The strict site-held-out MAE is 8.8387 dB for geometry-only features and 8.4232 dB after
adding point-cloud path features. Same-site, same-sector transfer is separately validated on
three bidirectional 800 MHz/2.1 GHz training pairs. Its leave-one-pair-out MAE improves from
7.5323 to 7.3292 dB. Transfer is not extrapolated to 3.5 GHz because no matching training pair
is available.

Submission archives:

- `submissions/path_baseline_v1/output.zip`: path-feature baseline.
- `submissions/same_sector_transfer_v1/output.zip`: baseline plus validated 800 MHz/2.1 GHz
  same-sector transfer.

## Sparse 2D U-Net experiment

`train_sparse_unet.py` compresses each point cloud to a 512x512 raster with 14 channels,
including surface/ground height statistics, point density, occupancy, propagation geometry,
frequency, antenna parameters, and a geometry-only RSRP baseline. A 121,969-parameter U-Net
is trained on 128x128 patches with loss applied only at measured pixels.

On all 752,724 points from the same strictly held-out sites, the U-Net reaches 9.2848 dB MAE,
versus 8.7478 dB for the path-feature model. Their validation-tuned blend reaches 8.7356 dB,
an improvement of only 0.0123 dB. The result is retained as an experiment but is not used in
the recommended submission because the gain is too small relative to validation uncertainty.

## Conditional path-field experiment

`path_field.py` implements a shared conditional continuous field without a learned cell ID.
Each query remains one base-station-to-receiver path rather than one whole scene. The model
supports five cumulative ablations:

1. `sequence`: an ordered 96-sample obstruction profile encoded by a 1D CNN;
2. `bev`: a small CNN builds two BEV feature planes that are bilinearly queried along the path;
3. `fourier`: low-frequency encoding of base-station-relative receiver coordinates;
4. `moe`: four softly gated propagation experts;
5. `gradient`: environment-weighted local Taylor consistency.

All variants predict a residual over the existing path LightGBM model. Run the comparable
site-held-out ablation with:

```bash
.venv/bin/python scripts/train_path_field.py \
  --output artifacts/path_field_ablation_v1 \
  --variants sequence,bev,fourier,moe,gradient \
  --max-train-points-per-cell 5000 \
  --max-valid-points-per-cell 5000
```

For a valid residual experiment, first generate physical-site OOF LightGBM predictions using
only the strict training cells:

```bash
.venv/bin/python scripts/build_path_oof.py \
  --output artifacts/path_oof_strict_v1 \
  --folds 5 \
  --max-train-points-per-cell 5000
```

Each fold keeps every co-sited cell together. Fold models are trained on a cell-balanced sample,
but predictions are written for every label row of the held-out cells. Train the path field on
these OOF residuals with all available strict-training labels by setting the point limit to zero:

```bash
.venv/bin/python scripts/train_path_field.py \
  --oof-dir artifacts/path_oof_strict_v1 \
  --output artifacts/path_field_oof_v1 \
  --variants sequence,gradient \
  --max-train-points-per-cell 0 \
  --max-valid-points-per-cell 0
```

`--oof-dir` is applied only to training cells. Held-out validation cells continue to use the
strict-split LightGBM artifact, so the validation sites cannot leak into neural training. For a
later production fit over all 75 cells, build a separate artifact with `--all-cells` and normally
raise `--max-train-points-per-cell` to 10000 to match the production LightGBM model.

The 63,562-point ablation produced:

| Variant | Parameters | Point MAE (dB) | Macro MAE (dB) |
|---|---:|---:|---:|
| Path LightGBM | - | 8.4232 | 8.4388 |
| Sequence | 35,841 | 8.4255 | 8.4375 |
| + BEV | 47,145 | 8.4150 | 8.4260 |
| + Fourier | 48,425 | 8.4143 | 8.4265 |
| + MoE | 87,536 | 8.4122 | 8.4242 |
| + Gradient | 87,536 | 8.4112 | 8.4229 |

The final checkpoint was also evaluated on all 752,724 held-out points:

```bash
.venv/bin/python scripts/evaluate_path_field.py \
  --model-artifact artifacts/path_field_ablation_v1/gradient
```

An unscaled neural correction worsens point MAE from 8.7478 to 8.7729 dB, despite improving
the per-cell macro average from 8.4502 to 8.4335 dB. Validation tuning selects a conservative
residual scale of 0.30 and reaches 8.7409 dB, a 0.0069 dB point-level improvement. This is too
small and validation-dependent to enter the recommended submission. The implementation is
kept as a research branch; `submissions/same_sector_transfer_v1/output.zip` remains the safe
submission.

### Path-field leaderboard submissions

The five validation-trained ablations can be exported together with two residual strengths:

```bash
.venv/bin/python scripts/submit_path_field_variants.py \
  --residual-scales 0.30,1.00
```

Every generated `output.zip` contains exactly the 19 required CSV files and is validated for
column names, `point_id` order, and finite predictions. The directories are:

- `submissions/path_field_sequence_scale030_v1/`
- `submissions/path_field_sequence_scale100_v1/`
- `submissions/path_field_bev_scale030_v1/`
- `submissions/path_field_bev_scale100_v1/`
- `submissions/path_field_fourier_scale030_v1/`
- `submissions/path_field_fourier_scale100_v1/`
- `submissions/path_field_moe_scale030_v1/`
- `submissions/path_field_moe_scale100_v1/`
- `submissions/path_field_gradient_scale030_v1/`
- `submissions/path_field_gradient_scale100_v1/`

These checkpoints were trained on the strict-split training sites, not retrained on all 75
cells. They are intended as low-cost leaderboard probes. Prefer the `scale030` versions first;
the `scale100` versions deliberately expose the complete neural correction and are riskier.
