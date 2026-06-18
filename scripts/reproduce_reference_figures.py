#!/usr/bin/env python
"""Regenerate all reference coverage curves from the saved score arrays.

This is useful for reviewers/collaborators who want to inspect the published
curves without retraining the reconstruction networks.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    (
        ROOT / "results/reference/gaussian_sure/denoising/sure_scores_unsupervised.npy",
        ROOT / "results/reference/gaussian_sure/denoising/mse_denoising_unsupervised.npy",
        ROOT / "results/reference/gaussian_sure/denoising",
        ROOT / "outputs/reference_figures/gaussian_sure_denoising",
    ),
    (
        ROOT / "results/reference/gaussian_sure/deblurring/sure_scores_unsupervised.npy",
        ROOT / "results/reference/gaussian_sure/deblurring/mse_MildlyBlurred_DIV2K_unsupervised.npy",
        ROOT / "results/reference/gaussian_sure/deblurring",
        ROOT / "outputs/reference_figures/gaussian_sure_deblurring",
    ),
    (
        ROOT / "results/reference/poisson_pure/denoising/pure_scores_unsupervised.npy",
        ROOT / "results/reference/poisson_pure/denoising/mse_DenoisingPoisson_div2k_unsupervised.npy",
        ROOT / "results/reference/poisson_pure/denoising",
        ROOT / "outputs/reference_figures/poisson_pure_denoising",
    ),
    (
        ROOT / "results/reference/poisson_pure/deblurring/pure_scores_unsupervised.npy",
        ROOT / "results/reference/poisson_pure/deblurring/mse_PoissonDeblurring_div2k_unsupervised.npy",
        ROOT / "results/reference/poisson_pure/deblurring",
        ROOT / "outputs/reference_figures/poisson_pure_deblurring",
    ),
]


def main():
    for cal_scores, test_scores, _src_dir, out_dir in JOBS:
        if not cal_scores.exists() or not test_scores.exists():
            print(f"Skipping missing reference pair: {cal_scores}, {test_scores}")
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts/make_figures_from_scores.py"),
            "--calibration-scores",
            str(cal_scores),
            "--test-scores",
            str(test_scores),
            "--out-dir",
            str(out_dir),
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
