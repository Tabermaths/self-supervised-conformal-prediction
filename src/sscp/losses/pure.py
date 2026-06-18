from __future__ import annotations

from collections.abc import Callable

import torch

from .hutchinson import weighted_hutchinson_trace
from sscp.utils.tensor import batch_mean, batch_mean_square, num_measurements


def poisson_pure_counts(
    y_counts: torch.Tensor,
    h_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float | torch.Tensor,
    num_trace_samples: int = 1,
    probe_distribution: str = "gaussian",
    create_graph: bool = False,
) -> torch.Tensor:
    r"""Poisson unbiased risk estimate for raw count measurements.

    This implements the notation of the Poisson paper:

    .. math:: Y\mid X=x \sim \mathcal P(\gamma A x),

    with estimator ``h(y)=A x_hat(y)``. The estimated score is

    .. math:: \|y/\gamma-h(y)\|_2^2/m - \mathbf 1^T y/(\gamma^2 m)
              + 2 y^T \operatorname{diag}(J_h(y))/(\gamma m).

    Use this when ``y`` contains raw Poisson counts.
    """
    gamma_t = torch.as_tensor(gamma, dtype=y_counts.dtype, device=y_counts.device)
    h_y = h_fn(y_counts)
    m = num_measurements(y_counts)
    weighted_trace = weighted_hutchinson_trace(
        y_counts,
        h_fn,
        weight=y_counts,
        num_samples=num_trace_samples,
        distribution=probe_distribution,
        create_graph=create_graph,
    )
    residual = batch_mean_square(y_counts / gamma_t - h_y)
    offset = -batch_mean(y_counts) / (gamma_t**2)
    divergence = (2.0 / (gamma_t * m)) * weighted_trace
    return residual + offset + divergence


def poisson_pure_deepinv(
    y: torch.Tensor,
    h_fn: Callable[[torch.Tensor], torch.Tensor],
    gain: float | torch.Tensor,
    num_trace_samples: int = 1,
    probe_distribution: str = "gaussian",
    create_graph: bool = False,
) -> torch.Tensor:
    r"""PURE under the DeepInv-style scaled Poisson convention.

    DeepInv's ``PoissonNoise(gain=g)`` convention is commonly used as
    ``Y = g * Poisson(Ax/g)``. In that convention the paper's ``gamma`` is
    approximately ``1/gain`` and the PURE estimator becomes

    .. math:: \|y-h(y)\|_2^2/m - g\,\bar y + 2g\, y^T\operatorname{diag}(J_h(y))/m.

    This is the convention used in the legacy code: ``gain=0.25`` corresponds
    to the paper's Poisson denoising experiment with ``gamma=4``.
    """
    gain_t = torch.as_tensor(gain, dtype=y.dtype, device=y.device)
    h_y = h_fn(y)
    m = num_measurements(y)
    weighted_trace = weighted_hutchinson_trace(
        y,
        h_fn,
        weight=y,
        num_samples=num_trace_samples,
        distribution=probe_distribution,
        create_graph=create_graph,
    )
    residual = batch_mean_square(y - h_y)
    offset = -gain_t * batch_mean(y)
    divergence = (2.0 * gain_t / m) * weighted_trace
    return residual + offset + divergence
