# API overview

## Core loss/risk estimators

- `sscp.losses.gaussian_sure(y, h_fn, sigma, ...)`
- `sscp.losses.poisson_pure_deepinv(y, h_fn, gain, ...)`
- `sscp.losses.poisson_pure_counts(y_counts, h_fn, gamma, ...)`
- `sscp.losses.hutchinson_trace(y, h_fn, ...)`
- `sscp.losses.weighted_hutchinson_trace(y, h_fn, weight, ...)`

Here `h_fn(y)` must return predicted measurements, i.e. `A x_hat(y)`, not necessarily the reconstructed image.

## Core conformal tools

- `sscp.conformal.measurement_score(x, y, estimator, physics_or_A)`
- `sscp.conformal.conformal_quantile(scores, confidence)`
- `sscp.conformal.calibrate_self_supervised(dataloader, risk_estimator, confidence, device)`
- `sscp.conformal.calibrate_supervised(dataloader, estimator, physics_or_A, confidence, device)`
- `sscp.conformal.empirical_coverage(dataloader, estimator, physics_or_A, q_hat, device)`

## Recommended coding pattern

```python
from sscp.conformal.scores import make_measurement_estimator
from sscp.losses.pure import poisson_pure_deepinv

estimator = lambda y: model(y, physics)       # y -> x_hat(y)
h_fn = make_measurement_estimator(estimator, physics)  # y -> A x_hat(y)
pure_scores = poisson_pure_deepinv(y, h_fn=h_fn, gain=0.25)
```
