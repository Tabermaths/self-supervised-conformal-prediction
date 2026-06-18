#!/usr/bin/env python
"""Generate DeepInv HDF5 measurement datasets from a paper config.

This script is intentionally conservative because DeepInv's dataset-generation
API has changed across versions. It checks that the expected function is present
and reports a clear error if the installed DeepInv version is incompatible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sscp.experiments.deepinv_builders import build_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HDF5 measurement datasets.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    exp_cfg = dict(cfg["experiment"])
    exp_cfg["generate_dataset"] = True
    bundle = build_experiment(**exp_cfg)

    try:
        import deepinv as dinv
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install DeepInv before generating paper datasets.") from exc

    if not hasattr(dinv.datasets, "generate_dataset"):
        raise RuntimeError(
            "This installed DeepInv version does not expose dinv.datasets.generate_dataset. "
            "Generate the HDF5 files with your DeepInv version and place them under the "
            "experiment root documented in docs/DATASETS.md."
        )

    data_cfg = cfg.get("data", {})
    kwargs = dict(
        train_dataset=bundle.train_dataset,
        test_dataset=bundle.test_dataset,
        physics=bundle.physics,
        device=bundle.device,
        save_dir=str(bundle.output_dir),
        train_datapoints=int(data_cfg.get("num_train", 900)),
        test_datapoints=int(data_cfg.get("num_test", 200)),
    )
    try:
        dinv.datasets.generate_dataset(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "DeepInv's generate_dataset signature differs from the one expected by this script. "
            "Please adapt this wrapper to your installed DeepInv version. The intended arguments are:\n"
            f"{kwargs}"
        ) from exc

    print(f"Generated dataset under {bundle.output_dir}")


if __name__ == "__main__":
    main()
