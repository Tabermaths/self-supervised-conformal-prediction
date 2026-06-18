# Code/paper alignment notes

This repository is a cleaned public-release candidate based on the uploaded legacy code archives and the two papers. The goal is to preserve the paper methodology while removing hard-coded cluster paths, duplicated notebooks, private outputs, caches, large checkpoints, and nested third-party histories.

## What matches the papers

| Paper component | Paper notation | Clean code location | Notes |
|---|---:|---|---|
| Gaussian measurement-domain score | `||Ax - A x_hat(y)||_2^2 / m` | `sscp.conformal.scores.measurement_score` | Used for supervised scores and coverage checks. |
| Gaussian SURE self-calibration | `||y - A x_hat(y)||_2^2/m - sigma^2 + 2 sigma^2 div(A x_hat(y))/m` | `sscp.losses.sure.gaussian_sure` | The divergence is estimated by Hutchinson/autodiff. |
| Poisson measurement-domain score | `||Ax - A x_hat(y)||_2^2 / m` | `sscp.conformal.scores.measurement_score` | Same score shape, different self-supervised risk estimate. |
| Poisson PURE, raw-count convention | `Y|X=x ~ Poisson(gamma A x)` | `sscp.losses.pure.poisson_pure_counts` | Implements the formula in the Poisson paper. |
| Poisson PURE, DeepInv scaled convention | `Y = gain * Poisson(Ax/gain)` | `sscp.losses.pure.poisson_pure_deepinv` | Useful for the legacy DeepInv code where `gain = 1/gamma`. |
| Split conformal quantile | top empirical quantile with finite-sample correction | `sscp.conformal.quantiles.conformal_quantile` | API uses `confidence = 1 - alpha` to avoid the legacy ambiguity. |
| Leave-one-out diagnostic quantiles | `Q_alpha^(i)` | `sscp.conformal.quantiles.leave_one_out_quantiles` | Useful for calibration-set diagnostics. |
| Implicit prediction set | `{x : score(x,y) <= q_hat}` | `sscp.conformal.sets.ImplicitConformalSet` | Avoids materialising a huge high-dimensional set. |

## What is intentionally not bit-for-bit identical

- The public package is cleaner and more modular than the original scripts; names, paths, and command-line interfaces were standardised.
- The code removes ambiguous legacy uses of `alpha` by using `confidence` for `1 - alpha`.
- Large checkpoints and generated datasets are not included. Reproduction scripts expect the user to provide them in documented paths.
- Poisson deblurring remains a documented template because the uploaded material did not include a compact public SwinIR/P4IP checkpoint. The generic PURE and conformal code is still valid for that experiment once the estimator/checkpoint is connected.

## Paper assets

The folder `paper_assets/` contains the active figures copied from the final Overleaf `main.tex` files. It is for visual comparison and archiving. Numerical data used to regenerate reference plots is kept in `results/reference/`.
