"""Minimal end-to-end Poisson PURE example without DeepInv."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from sscp.conformal.calibration import calibrate_self_supervised
from sscp.conformal.coverage import empirical_coverage
from sscp.conformal.scores import make_measurement_estimator
from sscp.data.synthetic import PoissonIdentityDataset
from sscp.losses.pure import poisson_pure_counts


def main():
    gamma = 8.0
    calibration = PoissonIdentityDataset(num_samples=64, gamma=gamma, seed=0, image_size=8)
    test = PoissonIdentityDataset(num_samples=32, gamma=gamma, seed=1, image_size=8)

    # Toy shrinkage estimator from raw counts to intensity domain.
    estimator = lambda y_counts: 0.8 * (y_counts / gamma)
    h_fn = make_measurement_estimator(estimator, physics_or_A=None)
    risk = lambda y_counts: poisson_pure_counts(y_counts, h_fn=h_fn, gamma=gamma)

    result = calibrate_self_supervised(DataLoader(calibration, batch_size=8), risk, confidence=0.9)
    cov = empirical_coverage(DataLoader(test, batch_size=8), estimator, None, result.q_hat)

    print(f"q_hat={float(result.q_hat):.4f}")
    print(f"empirical_coverage={cov:.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
