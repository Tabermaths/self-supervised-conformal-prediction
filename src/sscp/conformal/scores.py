from __future__ import annotations

from collections.abc import Callable

import torch

from sscp.utils.tensor import batch_mean_square


def apply_forward_operator(physics_or_A, x: torch.Tensor) -> torch.Tensor:
    """Apply a forward operator compatible with DeepInv or plain callables."""
    if physics_or_A is None:
        return x
    if hasattr(physics_or_A, "A"):
        return physics_or_A.A(x)
    if callable(physics_or_A):
        return physics_or_A(x)
    raise TypeError("physics_or_A must be None, callable, or expose an A(x) method")


def measurement_score(
    x: torch.Tensor,
    y: torch.Tensor,
    estimator: Callable[[torch.Tensor], torch.Tensor],
    physics_or_A=None,
) -> torch.Tensor:
    r"""Compute the measurement-domain conformal score.

    .. math:: s(x,y)=\|Ax-A\hat x(y)\|_2^2/m.

    Returns one score per batch element.
    """
    with torch.no_grad():
        x_hat = estimator(y)
        ax = apply_forward_operator(physics_or_A, x)
        ax_hat = apply_forward_operator(physics_or_A, x_hat)
        return batch_mean_square(ax - ax_hat)


def make_measurement_estimator(
    estimator: Callable[[torch.Tensor], torch.Tensor],
    physics_or_A=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``h(y)=A estimator(y)`` as a closure."""

    def h_fn(y: torch.Tensor) -> torch.Tensor:
        return apply_forward_operator(physics_or_A, estimator(y))

    return h_fn
