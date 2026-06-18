#!/usr/bin/env python
"""Calibrate self-supervised conformal sets using SURE or PURE scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from sscp.conformal.calibration import calibrate_self_supervised, calibrate_supervised
from sscp.conformal.scores import make_measurement_estimator
from sscp.experiments.deepinv_builders import build_experiment
from sscp.experiments.runtime import make_estimator, maybe_load_checkpoint
from sscp.losses.pure import poisson_pure_deepinv
from sscp.losses.sure import gaussian_sure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--supervised", action="store_true", help="Use ground truth scores for calibration.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    confidence = float(cfg["conformal"].get("confidence", 0.95))
    bundle = build_experiment(**cfg["experiment"])
    maybe_load_checkpoint(bundle.model, args.checkpoint or cfg.get("checkpoint"), bundle.device)

    loader = torch.utils.data.DataLoader(
        bundle.train_dataset,
        batch_size=cfg["conformal"].get("batch_size", 1),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )
    estimator = make_estimator(bundle.model, bundle.physics, bundle.metadata)

    if args.supervised:
        result = calibrate_supervised(
            loader,
            estimator=estimator,
            physics_or_A=bundle.physics,
            confidence=confidence,
            device=bundle.device,
        )
    else:
        h_fn = make_measurement_estimator(estimator, bundle.physics)
        if bundle.metadata["noise"] == "gaussian":
            risk = lambda y: gaussian_sure(
                y,
                h_fn=h_fn,
                sigma=bundle.metadata["sigma"],
                num_trace_samples=cfg["conformal"].get("num_trace_samples", 1),
                probe_distribution=cfg["conformal"].get("probe_distribution", "rademacher"),
            )
        elif bundle.metadata["noise"] == "poisson":
            risk = lambda y: poisson_pure_deepinv(
                y,
                h_fn=h_fn,
                gain=bundle.metadata["gain"],
                num_trace_samples=cfg["conformal"].get("num_trace_samples", 1),
                probe_distribution=cfg["conformal"].get("probe_distribution", "gaussian"),
            )
        else:
            raise ValueError(f"Unsupported noise model {bundle.metadata['noise']!r}")
        result = calibrate_self_supervised(
            loader,
            risk_estimator=risk,
            confidence=confidence,
            device=bundle.device,
        )

    out_dir = bundle.output_dir / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "supervised" if args.supervised else "self_supervised"
    np.save(out_dir / f"scores_{suffix}.npy", result.scores.numpy())
    (out_dir / f"qhat_{suffix}.txt").write_text(str(float(result.q_hat)))
    print(f"Saved scores and q-hat to {out_dir}")
    print(f"confidence={result.confidence:.3f}, q_hat={float(result.q_hat):.8g}")


if __name__ == "__main__":
    main()
