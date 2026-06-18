#!/usr/bin/env python
"""Evaluate empirical coverage from a calibrated q-hat."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from sscp.conformal.coverage import empirical_coverage
from sscp.experiments.deepinv_builders import build_experiment
from sscp.experiments.runtime import make_estimator, maybe_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qhat", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bundle = build_experiment(**cfg["experiment"])
    maybe_load_checkpoint(bundle.model, args.checkpoint or cfg.get("checkpoint"), bundle.device)
    q_hat = float(args.qhat.read_text().strip())

    loader = torch.utils.data.DataLoader(
        bundle.test_dataset,
        batch_size=cfg["conformal"].get("batch_size", 1),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )
    estimator = make_estimator(bundle.model, bundle.physics, bundle.metadata)
    coverage = empirical_coverage(loader, estimator, bundle.physics, q_hat=q_hat, device=bundle.device)

    out_dir = bundle.output_dir / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "empirical_coverage.npy", np.array([coverage]))
    print(f"Empirical coverage: {coverage:.4f}")


if __name__ == "__main__":
    main()
