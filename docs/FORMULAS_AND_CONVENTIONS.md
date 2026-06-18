# Formulas and conventions

## Score used by the conformal sets

Both papers use the measurement-domain score

```text
s(x, y) = || A x - A x_hat(y) ||_2^2 / m,
```

where `m` is the number of measurements and `x_hat(y)` is the reconstruction. The conformal region is the implicit sublevel set

```text
C_hat(y) = { x : || A x - A x_hat(y) ||_2^2 / m <= q_hat }.
```

## Gaussian SURE convention

For Gaussian observations

```text
Y | X=x ~ N(Ax, sigma^2 I),
```

and `h(y) = A x_hat(y)`, the SURE score estimator implemented in `sscp.losses.gaussian_sure` is

```text
SURE(y) = || y - h(y) ||_2^2 / m - sigma^2 + 2 sigma^2 div h(y) / m.
```

The divergence is estimated using Hutchinson probes and automatic differentiation.

## Poisson PURE conventions

The Poisson paper writes the raw-count model as

```text
Y | X=x ~ Poisson(gamma A x).
```

For this convention, use `sscp.losses.poisson_pure_counts`.

Some DeepInv experiments use the scaled convention

```text
Y = gain * Poisson(Ax / gain),
```

where approximately `gain = 1 / gamma`. For this convention, use `sscp.losses.poisson_pure_deepinv`.

## Finite-sample conformal quantile

The public API uses `confidence = 1 - alpha`. With `M` calibration scores, the default split-conformal order statistic is

```text
k = ceil((M + 1) * confidence).
```

If `k > M`, the quantile is returned as `+inf`, which is valid but conservative.
