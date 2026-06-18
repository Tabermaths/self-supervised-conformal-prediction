#!/usr/bin/env python
"""Generic training entry point for the paper configs."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train SURE/PURE self-supervised estimators.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["experiment"]["name"]
    target = "scripts.train_pure" if "poisson" in name else "scripts.train_sure"
    sys.argv = [target.replace(".", "/") + ".py", "--config", str(args.config)]
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
