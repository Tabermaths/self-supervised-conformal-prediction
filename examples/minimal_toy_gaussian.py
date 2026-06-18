"""Minimal end-to-end toy example without DeepInv.

This example is not an imaging benchmark; it demonstrates the API on a small
identity denoising problem.
"""
from __future__ import annotations

import torch

from sscp.conformal.quantiles import conformal_quantile
from sscp.conformal.scores import measurement_score
from sscp.losses.sure import gaussian_sure


def main():
    torch.manual_seed(0)
    sigma = 0.1
    confidence = 0.5
    n_cal = 100
    n_test = 50

    # Use a fixed zero signal so the SURE and exact score distributions are close
    # even in this very small toy problem.
    x_cal = torch.zeros(n_cal, 1, 8, 8)
    y_cal = x_cal + sigma * torch.randn_like(x_cal)
    x_test = torch.zeros(n_test, 1, 8, 8)
    y_test = x_test + sigma * torch.randn_like(x_test)

    # Identity estimator. This makes the SURE score approximately sigma^2.
    estimator = lambda y: y
    h_fn = estimator  # A is identity in this toy example.

    sure_scores = gaussian_sure(y_cal, h_fn=h_fn, sigma=sigma, num_trace_samples=1)
    q_hat = conformal_quantile(sure_scores, confidence=confidence)
    test_scores = measurement_score(x_test, y_test, estimator, physics_or_A=None)
    coverage = (test_scores <= q_hat).float().mean()

    print(f"q_hat={float(q_hat):.4f}")
    print(f"fraction of toy test scores below q_hat={float(coverage):.3f}")


if __name__ == "__main__":
    main()
