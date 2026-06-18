# Audit of uploaded legacy material

This repository was produced from the uploaded project archives and papers. The raw code had useful
scientific content but was not appropriate to share directly because it mixed experiments, cached data,
absolute filesystem paths, W&B runs, `.git` folders, checkpoints, generated figures, and one-off debugging
scripts.

## Source archives inspected

- `ConformalPredictionSure.zip` — Gaussian/SURE conformal code, DeepInv datasets, Polyblur code,
  figures, arrays, and one large tomography checkpoint.
- `ConformalPredictionUnsure.zip` — Poisson/PURE and UNSURE experiments, Scale-Equivariant-Imaging
  fork, P4IP/SwinIR-style deblurring code, figures, arrays, W&B artifacts, and nested `.git` history.
- `SURE_based_Conformal_Prediction_for_Uncertainty_Quantification_in_Image_Restoration_Problems__SSVM2025_.zip`
  — paper source and figures for the Gaussian/SURE paper.
- `PURE_based_Conformal_Prediction_for_Uncertainty_Quantification_in_Low_Photon_Image_Restoration.zip`
  — paper source and figures for the Poisson/PURE paper.
- `requirements.txt`, `requirements_pure_sure`, `environment_sure_pure` — environment snapshots.

## Main issues found in the legacy code

1. **Hard-coded absolute paths** such as `/users/bt2027/sharedscratch/...` made the scripts non-portable.
2. **Ambiguous variable naming**: `alpha` was used both as significance level and as confidence level.
3. **Mixed concerns**: training, calibration, plotting and debugging were often in one script.
4. **Duplicated SURE/PURE implementations** across Gaussian and Poisson folders.
5. **Checkpoint-dependent execution** without a clean failure message or public checkpoint policy.
6. **Cache and private artifacts**: `.git`, `.cache`, `wandb`, `__pycache__`, downloaded datasets, and large
   checkpoints were present in the uploaded archives.
7. **Debug prints and memory workarounds** inside core routines, making the code noisy for collaborators.
8. **No unit tests** for conformal quantiles or score shapes.

## What has been cleaned

- Core math has been moved to `src/sscp/losses` and `src/sscp/conformal`.
- SURE/PURE formulas are implemented as reusable functions with typed arguments and docstrings.
- Calibration and evaluation are now separate scripts.
- Config files replace hard-coded experiment parameters.
- The finite-sample conformal quantile is implemented explicitly and tested.
- Small reference arrays/figures are preserved in `results/reference/`.
- Large/private artifacts are excluded by `.gitignore` and not placed in the clean zip.

## Legacy-to-clean mapping

| Legacy file/function | Clean location |
| --- | --- |
| `ConformalPredictionSure/helper_fns.py::sure` and `surenew` | `src/sscp/losses/sure.py::gaussian_sure` |
| `ConformalPredictionSure/helper_fns.py::score` | `src/sscp/conformal/scores.py::measurement_score` |
| `ConformalPredictionSure/helper_fns.py::calibrate_q` | `src/sscp/conformal/quantiles.py::conformal_quantile` |
| `ConformalPredictionUnsure/unsure/helper_fn.py::sure_poissonnew` | `src/sscp/losses/pure.py::poisson_pure_deepinv` |
| `ConformalPredictionUnsure/unsure/helper_fn.py::approximation_weighted_divergence` | `src/sscp/losses/hutchinson.py::weighted_hutchinson_trace` |
| `main.py`, `main_conformal.py` | `scripts/calibrate_conformal.py` and `scripts/evaluate_coverage.py` |
| `train_denoiser_sure.py` | `scripts/train_sure.py` |
| Poisson training snippets in `unsure/main.py` and helpers | `scripts/train_pure.py` |

## Remaining assumptions

- Full reproduction of numerical values requires the original generated HDF5 datasets and trained checkpoints.
- The Poisson deblurring paper used a more specialised SwinIR/P4IP-style estimator. The clean package keeps the
  conformal and PURE logic general; a project-specific SwinIR builder can be added once the final checkpoint/model
  policy is decided.
- The exact DeepInv API can differ across versions. The included builders target the APIs used in the legacy code
  and are isolated from the core mathematical package.
