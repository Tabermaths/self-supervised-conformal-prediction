# Self-supervised Conformal Prediction for Imaging

[![Tests](https://github.com/Tabermaths/self-supervised-conformal-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/Tabermaths/self-supervised-conformal-prediction/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Public research code for **self-supervised conformal prediction** in imaging inverse problems. The repository implements conformal calibration without ground-truth calibration images by replacing the supervised calibration score with self-supervised risk estimates:

- **SURE** for Gaussian imaging problems;
- **PURE** for Poisson imaging problems.

The code is model-agnostic: any reconstruction method `y -> x_hat(y)` can be used as long as a forward operator `A` is available. Optional DeepInv builders are included for paper-style experiments.

## Associated papers

Please cite the relevant paper if you use this code.

1. Jasper M. Everink, Bernardin Tamo Amougou, and Marcelo Pereyra. **Self-supervised Conformal Prediction for Uncertainty Quantification in Imaging Problems.** SSVM 2025, LNCS 15667, pp. 108–118. DOI: `10.1007/978-3-031-92366-1_9`.

2. Bernardin Tamo Amougou, Marcelo Pereyra, and Barbara Pascal. **Self-supervised conformal prediction for uncertainty quantification in Poisson imaging problems.** IEEE Statistical Signal Processing Workshop, SSP 2025. DOI: `10.1109/SSP64130.2025.11073283`.

## Repository layout

```text
configs/              YAML configs for Gaussian SURE and Poisson PURE experiments
docs/                 Reproducibility notes, datasets, checkpoints, formula mapping
examples/             Small examples that run without DeepInv
scripts/              Training, calibration, coverage and figure-generation scripts
src/sscp/             Python package: losses, conformal sets, experiment utilities
tests/                Unit tests for the core package
results/reference/    Small saved arrays and figures from earlier runs
paper_assets/         Exact active Overleaf figures used in the papers
legacy_notes/         Audit notes from the original project archives
```

Large checkpoints, datasets, W&B logs, nested Git histories, cache folders and private cluster paths are deliberately excluded.

## Installation

For the core package and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/check_install.py
pytest
```

For paper-style experiments with DeepInv:

```bash
conda env create -f environment.yml
conda activate sscp
pip install -e .[deepinv,dev]
```

## Quick examples

Run a self-contained Gaussian toy example:

```bash
python examples/minimal_toy_gaussian.py
```

Run a self-contained Poisson toy example:

```bash
python examples/minimal_toy_poisson.py
```

Regenerate reference figures from saved score arrays:

```bash
python scripts/reproduce_reference_figures.py
```

## Main API

Gaussian SURE:

```python
from sscp.losses.sure import gaussian_sure
from sscp.conformal.scores import make_measurement_estimator

h_fn = make_measurement_estimator(estimator, physics)  # h(y) = A x_hat(y)
sure_values = gaussian_sure(y, h_fn=h_fn, sigma=0.1, num_trace_samples=1)
```

Poisson PURE:

```python
from sscp.losses.pure import poisson_pure_counts, poisson_pure_deepinv

# Paper raw-count convention: Y | X=x ~ Poisson(gamma A x)
pure_values = poisson_pure_counts(y_counts, h_fn=h_fn, gamma=4.0, num_trace_samples=1)

# DeepInv scaled-Poisson convention: Y = gain * Poisson(Ax/gain)
pure_values_deepinv = poisson_pure_deepinv(y, h_fn=h_fn, gain=0.25, num_trace_samples=1)
```

Conformal calibration:

```python
from sscp.conformal.quantiles import conformal_quantile
from sscp.conformal.sets import ImplicitConformalSet

q_hat = conformal_quantile(self_supervised_scores, confidence=0.95)
C_hat = ImplicitConformalSet(q_hat=q_hat, estimator=estimator, physics_or_A=physics)
covered = C_hat.contains(x_test, y_test)
```

The implicit prediction set is

```text
C_hat(y) = { x : || A x - A x_hat(y) ||_2^2 / m <= q_hat }.
```

## Paper-style commands

```bash
python scripts/train_sure.py --config configs/gaussian_denoising_sure.yaml
python scripts/train_pure.py --config configs/poisson_denoising_pure.yaml
python scripts/calibrate_conformal.py --config configs/poisson_denoising_pure.yaml --checkpoint PATH_TO_CHECKPOINT
python scripts/evaluate_coverage.py --config configs/poisson_denoising_pure.yaml --checkpoint PATH_TO_CHECKPOINT --qhat PATH_TO_QHAT
```

The cleaned API uses `confidence`, for example `0.95` for a 95% prediction set. This corresponds to `1 - alpha` in the papers.

## Reproducibility notes

The uploaded release does not include large trained checkpoints or full DIV2K-derived datasets. The repository therefore supports:

- running all core unit tests without external data;
- retraining models after downloading/preparing the datasets;
- recalibrating conformal sets after placing checkpoints at the paths supplied to the scripts;
- regenerating small reference plots from arrays in `results/reference/`.

See `docs/PAPER_REPRODUCTION.md`, `docs/CODE_PAPER_ALIGNMENT.md`, `docs/DATASETS.md`, and `docs/CHECKPOINTS.md` for details.

## Citation

A `CITATION.cff` file is included. It cites the software release and lists the two associated papers.
