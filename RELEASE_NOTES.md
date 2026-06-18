# Release notes

## v0.1.0 public-release candidate

This release candidate turns the original project archives into a shareable research-code repository.

### Added

- Reusable `sscp` Python package.
- Gaussian SURE and Poisson PURE estimators with Hutchinson/autodiff trace estimation.
- Model-agnostic measurement-domain conformal scores and implicit conformal sets.
- Split-conformal and leave-one-out quantile utilities.
- DeepInv experiment builders for the Gaussian denoising/deblurring and Poisson denoising experiments.
- Clear placeholder for Poisson deblurring, pending the final SwinIR/P4IP checkpoint release policy.
- Paper-style YAML configs.
- Training, calibration, coverage and figure-generation scripts.
- Minimal examples that run without DeepInv or datasets.
- Unit tests for core APIs.
- Exact active Overleaf paper figures under `paper_assets/`, with a manifest and checksums.
- `docs/CODE_PAPER_ALIGNMENT.md` mapping the cleaned code to the paper formulas.
- GitHub Actions workflow, pre-commit config, Dockerfile, Makefile and release checklist.

### Excluded intentionally

- Large checkpoints.
- DIV2K and generated HDF5 datasets.
- W&B runs and cache folders.
- Nested `.git` histories from third-party forks.
- Private absolute cluster paths.
- Non-reproducibility webpage screenshots such as the former `paper_assets/SSCP1.png`.

### Validation in this workspace

- `pytest -q`: 10 passed.
- `examples/minimal_toy_gaussian.py`: ran successfully.
- `examples/minimal_toy_poisson.py`: ran successfully.
- `scripts/reproduce_reference_figures.py`: ran successfully with saved reference arrays.
- `ruff check`: not run in this workspace because `ruff` was not installed in the base runtime; the repository includes the ruff configuration and GitHub Actions workflow.
