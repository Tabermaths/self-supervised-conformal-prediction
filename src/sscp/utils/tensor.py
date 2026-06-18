from __future__ import annotations

from collections.abc import Callable

import torch


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions.

    A 3D tensor ``(C,H,W)`` is treated as a batch of size one. A 4D tensor
    ``(B,C,H,W)`` is treated as a batch of size ``B``.
    """
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.dim() == 1:
        return x.reshape(1, -1)
    if x.dim() == 3:
        return x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


def batch_mean_square(x: torch.Tensor) -> torch.Tensor:
    """Return ``||x||_2^2 / m`` for each batch element."""
    return flatten_batch(x).pow(2).mean(dim=1)


def batch_sum(x: torch.Tensor) -> torch.Tensor:
    """Return the sum over non-batch dimensions for each batch element."""
    return flatten_batch(x).sum(dim=1)


def batch_mean(x: torch.Tensor) -> torch.Tensor:
    """Return the mean over non-batch dimensions for each batch element."""
    return flatten_batch(x).mean(dim=1)


def get_batch_size(x: torch.Tensor) -> int:
    if x.dim() <= 3:
        return 1
    return x.shape[0]


def num_measurements(y: torch.Tensor) -> int:
    """Number of measurements per sample."""
    return flatten_batch(y).shape[1]


def call_estimator(estimator: Callable, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Call an estimator with a small compatibility layer.

    This helper exists because the legacy scripts used two signatures:
    ``model(y, sigma_or_gain)`` for DRUNet/PhDNet-like models and
    ``model(y, physics)`` for DeepInv unfolded models. New code should prefer
    passing a closure with signature ``estimator(y)``.
    """
    if args or kwargs:
        return estimator(y, *args, **kwargs)
    return estimator(y)
