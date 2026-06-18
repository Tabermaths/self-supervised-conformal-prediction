# Self-supervised Conformal Prediction for Imaging

[![Tests](https://github.com/<OWNER>/<REPO>/actions/workflows/tests.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Clean research code for the papers:

1. **SURE-based self-supervised conformal prediction** for Gaussian imaging inverse problems.
2. **PURE-based self-supervised conformal prediction** for Poisson imaging inverse problems.

The repository focuses on the two operations most useful to collaborators:

- train a reconstruction network with a self-supervised risk estimator, SURE or PURE;
- calibrate and evaluate conformal prediction sets without ground-truth calibration data.

The core package is deliberately model-agnostic: any estimator `y -> x_hat(y)` can be used, as long as a forward operator `A` is available. The paper reproduction scripts use [DeepInv](https://deepinv.github.io/deepinv/) as an optional dependency.

## Repository layout

```text
configs/                         YAML configs for paper-style experiments
docs/                            Reproducibility, checkpoint and formula notes
examples/                        Small examples that run without DeepInv
notebooks/                       Lightweight demo notebook
scripts/                         Training, calibration and plotting entry points
src/sscp/losses/                 SURE, PURE and Hutchinson trace estimators
src/sscp/conformal/              Scores, conformal quantiles, implicit sets and coverage
src/sscp/experiments/            Optional DeepInv experiment builders
src/sscp/data/                   Tiny synthetic datasets for examples/tests
tests/                           Unit tests that do not require DeepInv
results/reference/               Small arrays/figures extracted from the old runs
paper_assets/                    Exact active Overleaf figures used in the papers
legacy_notes/                    Audit notes about the original project archives
```

Large checkpoints, downloaded datasets, `.git` folders, cache folders, W&B logs and Python bytecode are deliberately excluded from the public release.

## Installation

For the core package and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/check_install.py
pytest
```

For paper reproduction with DeepInv:

```bash
conda env create -f environment.yml
conda activate sscp
pip install -e .[deepinv,dev]
```

If you need to mimic the older environment more closely, see `requirements-deepinv-legacy.txt`.

## Minimal examples

Run a fully self-contained Gaussian toy example:

```bash
python examples/minimal_toy_gaussian.py
```

Regenerate the small reference coverage figures from saved score arrays:

```bash
python scripts/reproduce_reference_figures.py
```

## Main mathematical API

### Gaussian SURE

```python
from sscp.losses.sure import gaussian_sure
from sscp.conformal.scores import make_measurement_estimator

# estimator: y -> x_hat(y)
# physics: DeepInv physics object or any object exposing A(x)
h_fn = make_measurement_estimator(estimator, physics)  # h(y)=A x_hat(y)
sure_values = gaussian_sure(y, h_fn=h_fn, sigma=0.1, num_trace_samples=1)
```

### Poisson PURE

```python
from sscp.losses.pure import poisson_pure_deepinv, poisson_pure_counts

# DeepInv scaled-Poisson convention: Y = gain * Poisson(Ax/gain)
pure_values = poisson_pure_deepinv(y, h_fn=h_fn, gain=0.25, num_trace_samples=1)

# Paper raw-count convention: Y | X=x ~ Poisson(gamma A x)
pure_values_counts = poisson_pure_counts(y_counts, h_fn=h_fn, gamma=4.0, num_trace_samples=1)
```

### Conformal calibration and implicit sets

```python
from sscp.conformal.quantiles import conformal_quantile
from sscp.conformal.sets import ImplicitConformalSet

q_hat = conformal_quantile(sure_or_pure_scores, confidence=0.95)
C_hat = ImplicitConformalSet(q_hat=q_hat, estimator=estimator, physics_or_A=physics)
covered = C_hat.contains(x_test, y_test)
```

The prediction set is represented implicitly by

```text
C_hat(y) = { x : || A x - A x_hat(y) ||_2^2 / m <= q_hat }.
```

## Paper-style commands

Generic training wrapper:

```bash
python scripts/train.py --config configs/poisson_denoising_pure.yaml
```

Gaussian SURE training:

```bash
python scripts/train_sure.py --config configs/gaussian_denoising_sure.yaml
```

Poisson PURE training:

```bash
python scripts/train_pure.py --config configs/poisson_denoising_pure.yaml
```

Calibrate self-supervised conformal scores:

```bash
python scripts/calibrate_conformal.py --config configs/poisson_denoising_pure.yaml \
  --checkpoint outputs/poisson_denoising_div2k/checkpoints/pure/ckp_399.pth.tar
```

Evaluate empirical coverage:

```bash
python scripts/evaluate_coverage.py --config configs/poisson_denoising_pure.yaml \
  --checkpoint outputs/poisson_denoising_div2k/checkpoints/pure/ckp_399.pth.tar \
  --qhat outputs/poisson_denoising_div2k/conformal/qhat_self_supervised.txt
```

Create figures from already saved scores:

```bash
python scripts/make_figures_from_scores.py \
  --calibration-scores results/reference/poisson_pure/denoising/pure_scores_unsupervised.npy \
  --test-scores results/reference/poisson_pure/denoising/mse_DenoisingPoisson_div2k_unsupervised.npy \
  --out-dir outputs/reference_poisson_denoising_figures
```

## Important convention: confidence vs alpha

The cleaned API uses `confidence`, for example `0.95` for a 95% prediction set. This is `1 - alpha` in the mathematical notation. The legacy scripts used a variable named `alpha` in several places, sometimes as a confidence level and sometimes as a significance level. That ambiguity has been removed.

## What is reproducible without checkpoints?

The uploaded checkpoints were mostly removed because they are large. Therefore:

- core SURE/PURE/conformal unit tests run without checkpoints;
- calibration can be rerun after placing a trained checkpoint at the path supplied to the scripts;
- `results/reference/` contains small saved arrays and figures from earlier runs so collaborators can regenerate coverage/histogram plots without retraining.

See `docs/PAPER_REPRODUCTION.md`, `docs/DATASETS.md`, `docs/CHECKPOINTS.md`, and `docs/CODE_PAPER_ALIGNMENT.md` for the public-release policy and the mapping between the code and the two papers.

## Paper assets

`paper_assets/` contains the exact active figure files referenced by the final Overleaf `main.tex` files for the two papers. The previous `SSCP1.png` page screenshot was removed because it was not a reproducibility asset. Numerical arrays and regenerated plots remain in `results/reference/`.

## Citation

A `CITATION.cff` file is included. Please update the repository URL/DOI fields when the public GitHub and archival record are final.
