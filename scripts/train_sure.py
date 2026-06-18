#!/usr/bin/env python
"""Train a self-supervised network with Gaussian SURE.

This script is intentionally small. It delegates experiment construction to
``sscp.experiments.deepinv_builders`` and uses DeepInv's trainer/loss when
available, while the mathematical SURE estimator used for calibration lives in
``sscp.losses.sure``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from sscp.experiments.deepinv_builders import build_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    try:
        import deepinv as dinv
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install the optional DeepInv dependencies before training") from exc

    bundle = build_experiment(**cfg["experiment"])
    train_loader = torch.utils.data.DataLoader(
        bundle.train_dataset,
        batch_size=bundle.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
    )
    test_loader = torch.utils.data.DataLoader(
        bundle.test_dataset,
        batch_size=bundle.batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    sigma = bundle.metadata["sigma"]
    losses = [dinv.loss.SureGaussianLoss(sigma=sigma, tau=cfg["training"].get("tau", 1e-4))]
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=cfg["training"].get("lr", 5e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(cfg["training"].get("epochs", 100) * 0.8)),
    )

    save_path = bundle.output_dir / "checkpoints" / "sure"
    save_path.mkdir(parents=True, exist_ok=True)

    trainer = dinv.Trainer(
        losses=losses,
        model=bundle.model,
        physics=bundle.physics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=[train_loader],
        eval_dataloader=[test_loader],
        device=bundle.device,
        epochs=cfg["training"].get("epochs", 100),
        ckp_interval=cfg["training"].get("ckp_interval", 10),
        eval_interval=cfg["training"].get("eval_interval", 10),
        save_path=str(save_path),
        online_measurements=False,
        verbose_individual_losses=True,
        metrics=[dinv.loss.PSNR()],
        plot_images=cfg["training"].get("plot_images", False),
        wandb_vis=cfg["training"].get("wandb", False),
    )
    trainer.train()


if __name__ == "__main__":
    main()
