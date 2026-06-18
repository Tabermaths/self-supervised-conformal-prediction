from __future__ import annotations

import importlib.util

import torch

import sscp


def check_install() -> None:
    """Print a minimal installation report."""
    print(f"sscp import: ok ({sscp.__name__})")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"deepinv installed: {importlib.util.find_spec('deepinv') is not None}")
