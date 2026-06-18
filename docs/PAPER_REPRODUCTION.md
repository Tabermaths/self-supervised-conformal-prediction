# Paper reproduction guide

This file records the intended public reproduction pathway. The repository separates:

1. core mathematical code, which runs without large files;
2. paper experiment scripts, which need DIV2K-derived HDF5 datasets and trained checkpoints.

## Gaussian/SURE paper

### Gaussian denoising

- Dataset: DIV2K RGB crops.
- Crop size: 128 x 128.
- Noise: additive Gaussian noise with `sigma = 0.1`.
- Estimator: DRUNet trained with a SURE loss.
- Calibration/training measurements: 900 noisy images.
- Evaluation: 200 measurement/ground-truth pairs.

Config:

```bash
configs/gaussian_denoising_sure.yaml
```

Commands:

```bash
python scripts/train_sure.py --config configs/gaussian_denoising_sure.yaml
python scripts/calibrate_conformal.py --config configs/gaussian_denoising_sure.yaml \
  --checkpoint outputs/gaussian_denoising_div2k/checkpoints/sure/ckp_*.pth.tar
python scripts/evaluate_coverage.py --config configs/gaussian_denoising_sure.yaml \
  --checkpoint outputs/gaussian_denoising_div2k/checkpoints/sure/ckp_*.pth.tar \
  --qhat outputs/gaussian_denoising_div2k/conformal/qhat_self_supervised.txt
```

### Gaussian deblurring

- Dataset: DIV2K RGB crops.
- Crop size: 256 x 256.
- Blur: diagonal Gaussian blur with bandwidths approximately `(2.0, 0.3)` and angle `30` degrees.
- Noise: additive Gaussian noise with `sigma = 0.01`.
- Estimator: Polyblur model-driven restoration method.
- Calibration measurements: 900 blurred/noisy images.
- Evaluation: 200 measurement/ground-truth pairs.

Config:

```bash
configs/gaussian_deblurring_sure.yaml
```

## Poisson/PURE paper

### Poisson denoising

- Dataset: DIV2K RGB crops.
- Crop size: 256 x 256.
- Observation: Poisson denoising with `A = I` and `gamma = 4`.
- DeepInv scaled convention: `gain = 1/gamma = 0.25`.
- Estimator: unrolled PGD network with 4 iterations, U-Net denoiser with 2 scales, no weight tying.
- Training: PURE self-supervised loss.
- Calibration/training measurements: 900 noisy images.
- Evaluation: 200 measurement/ground-truth pairs.

Config:

```bash
configs/poisson_denoising_pure.yaml
```

Commands:

```bash
python scripts/train_pure.py --config configs/poisson_denoising_pure.yaml
python scripts/calibrate_conformal.py --config configs/poisson_denoising_pure.yaml \
  --checkpoint outputs/poisson_denoising_div2k/checkpoints/pure/ckp_*.pth.tar
python scripts/evaluate_coverage.py --config configs/poisson_denoising_pure.yaml \
  --checkpoint outputs/poisson_denoising_div2k/checkpoints/pure/ckp_*.pth.tar \
  --qhat outputs/poisson_denoising_div2k/conformal/qhat_self_supervised.txt
```

### Poisson deblurring

- Dataset: DIV2K grayscale crops.
- Crop size: 256 x 256.
- Blur: isotropic Gaussian blur with standard deviation `sigma = 2` pixels.
- Noise: Poisson noise with `gamma = 60`.
- Estimator: SwinIR/P4IP-style estimator trained with a PURE-based self-supervised loss.
- Calibration/training measurements: 900 blurred/noisy images.
- Evaluation: 200 measurement/ground-truth pairs.

Config:

```bash
configs/poisson_deblurring_pure.yaml
```

The public release currently keeps this as a template because the original SwinIR/P4IP checkpoint is large and was not included in the uploaded archive. The core PURE/conformal functions are model-agnostic, so the final builder can be added once the checkpoint release policy is fixed.

## Reference figures without retraining

To regenerate coverage figures from the small saved arrays:

```bash
python scripts/reproduce_reference_figures.py
```

Outputs are written to:

```text
outputs/reference_figures/
```


## Exact paper figures copied from Overleaf

The directory `paper_assets/` contains the active figure files referenced by the final `main.tex` files in the uploaded Overleaf archives. These are archived to make it easy to compare regenerated outputs against the published/compiled figures.

```text
paper_assets/sure_ssvm2025/
├── reconstructions.png
├── coveragedenoisingwithoracle.png
├── coveragedeblurringwithoracle.png
├── histogramdenoising.png
└── histogramdeblurring.png

paper_assets/poisson_ssp2025/
├── poisson_reconstruction_reviewed.png
├── coveragepoissondenoisingwithoracle.png
└── coveragepoissondeblurringwithoracle.png
```

`paper_assets/MANIFEST.json` records the original LaTeX path and SHA256 checksum for each copied figure. Commented-out `\includegraphics` entries were not copied into this folder.
