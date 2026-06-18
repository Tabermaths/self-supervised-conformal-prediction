from __future__ import annotations

from collections.abc import Callable

import torch

from .hutchinson import hutchinson_trace
from sscp.utils.tensor import batch_mean_square, num_measurements


def gaussian_sure(
    y: torch.Tensor,
    h_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float | torch.Tensor,
    num_trace_samples: int = 1,
    probe_distribution: str = "rademacher",
    create_graph: bool = False,
) -> torch.Tensor:
    r"""Stein's unbiased risk estimate for Gaussian measurements.

    The paper uses the measurement-domain score

    .. math:: s(x,y)=\|Ax-A\hat x(y)\|_2^2/m

    and estimates it by

    .. math:: \|y-h(y)\|_2^2/m - \sigma^2 + 2\sigma^2 \operatorname{div}h(y)/m,

    where ``h(y)=A x_hat(y)``.

    Returns one SURE value per batch element.
    """
    h_y = h_fn(y)
    m = num_measurements(y)
    sigma_t = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    residual = batch_mean_square(y - h_y)
    div = hutchinson_trace(
        y,
        h_fn,
        num_samples=num_trace_samples,
        distribution=probe_distribution,
        create_graph=create_graph,
    )
    return residual - sigma_t**2 + (2.0 * sigma_t**2 / m) * div
