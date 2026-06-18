#!/usr/bin/env python
"""Create coverage and score-comparison figures from saved NumPy arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sscp.conformal.coverage import empirical_coverage_curve
from sscp.conformal.quantiles import conformal_quantile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--reference-scores", type=Path, default=None, help="Optional supervised/MSE scores for histogram comparison.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cal_scores = torch.as_tensor(np.load(args.calibration_scores)).flatten()
    test_scores = torch.as_tensor(np.load(args.test_scores)).flatten()
    confidence = torch.linspace(0.01, 0.99, 99)
    q = torch.stack([conformal_quantile(cal_scores, float(c)) for c in confidence])
    coverage = empirical_coverage_curve(test_scores, q)

    plt.figure(figsize=(5, 4))
    plt.plot(confidence.numpy(), coverage, label="Empirical")
    plt.plot(confidence.numpy(), confidence.numpy(), linestyle="--", label="Expected")
    plt.xlabel("Desired confidence level")
    plt.ylabel("Empirical coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "coverage_curve.png", dpi=300)
    plt.close()

    if args.reference_scores is not None:
        ref_scores = np.load(args.reference_scores).reshape(-1)
        plt.figure(figsize=(5, 4))
        plt.hist(cal_scores.numpy(), bins=25, alpha=0.6, label="Self-supervised")
        plt.hist(ref_scores, bins=25, alpha=0.6, label="Supervised / exact")
        plt.xlabel("Score value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "score_histogram.png", dpi=300)
        plt.close()

    np.save(args.out_dir / "confidence_grid.npy", confidence.numpy())
    np.save(args.out_dir / "empirical_coverage.npy", coverage)
    np.save(args.out_dir / "quantiles.npy", q.numpy())
    print(f"Saved figures and arrays to {args.out_dir}")


if __name__ == "__main__":
    main()
