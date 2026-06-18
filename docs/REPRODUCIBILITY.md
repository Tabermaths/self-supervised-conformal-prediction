# Reproducibility notes

## Datasets

The paper experiments use DIV2K crops. The clean scripts expect either:

1. existing DeepInv HDF5 datasets in the configured output directory; or
2. `generate_dataset: true` in a custom launcher that creates the measurements from DIV2K.

For public release, do not commit DIV2K or generated HDF5 files. Instead, document the generation command and
store only checksums/metadata.

## Checkpoints

Checkpoints are intentionally ignored by git. Recommended layout:

```text
outputs/<experiment>/checkpoints/<method>/ckp_*.pth.tar
```

For a collaborator-only release, place checkpoints on OneDrive/Google Drive/Hugging Face and give the URL in this file.
For a public release, consider uploading only final lightweight model weights if licensing permits it.

## Expected experiment settings

### Gaussian/SURE paper

- Denoising: DIV2K RGB crops, 128x128, Gaussian sigma = 0.1, DRUNet, SURE loss, 900 noisy measurements.
- Deblurring: DIV2K RGB crops, 256x256, diagonal Gaussian blur with major/minor bandwidth approximately `(2, 0.3)`, angle 30 degrees, Gaussian sigma = 0.01, Polyblur estimator.
- Evaluation: 200 measurement/ground-truth pairs.

### Poisson/PURE paper

- Denoising: DIV2K RGB crops, 256x256, Poisson observation model with gamma = 4, unrolled PGD network, 4 iterations, U-Net denoiser with 2 scales, no weight tying.
- Deblurring: DIV2K grayscale crops, 256x256, Gaussian blur sigma = 2 px, Poisson gamma = 60, SwinIR-style estimator trained with a PURE-based self-supervised loss.
- Evaluation: 900 calibration/training measurements and 200 measurement/ground-truth pairs.

## Reference results

Small saved arrays and figures from the uploaded archives are in:

```text
results/reference/gaussian_sure/
results/reference/poisson_pure/
```

These are useful for quick sanity checks and figure generation without retraining.
