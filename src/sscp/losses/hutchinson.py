from __future__ import annotations

from collections.abc import Callable

import torch

from sscp.utils.tensor import flatten_batch


def _sample_probe_like(x: torch.Tensor, distribution: str = "rademacher") -> torch.Tensor:
    if distribution == "rademacher":
        return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    if distribution == "gaussian":
        return torch.randn_like(x)
    raise ValueError("distribution must be either 'rademacher' or 'gaussian'")


def hutchinson_trace(
    y: torch.Tensor,
    h_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 1,
    distribution: str = "rademacher",
    create_graph: bool = False,
) -> torch.Tensor:
    """Estimate ``trace(J_h(y))`` with Hutchinson's identity.

    Parameters
    ----------
    y:
        Measurement tensor. The function internally clones and enables gradients.
    h_fn:
        Function returning the predicted measurements ``h(y)=A x_hat(y)`` with the
        same batch dimension as ``y``.
    num_samples:
        Number of Hutchinson probes. One probe is often sufficient for high-dimensional
        images and matches the paper experiments.
    distribution:
        ``'rademacher'`` is the default because it usually has lower variance for trace
        estimation; ``'gaussian'`` matches the notation used in the papers.
    create_graph:
        Set to ``True`` when using the estimate as a training loss.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(batch,)`` containing an unnormalised trace estimate.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    y_req = y.detach().clone().requires_grad_(True)
    h_y = h_fn(y_req)
    estimates = []

    for sample_id in range(num_samples):
        probe = _sample_probe_like(h_y, distribution)
        inner = (h_y * probe).sum()
        grad = torch.autograd.grad(
            inner,
            y_req,
            retain_graph=sample_id < num_samples - 1,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        estimates.append(flatten_batch(grad * probe).sum(dim=1))

    return torch.stack(estimates, dim=0).mean(dim=0)


def weighted_hutchinson_trace(
    y: torch.Tensor,
    h_fn: Callable[[torch.Tensor], torch.Tensor],
    weight: torch.Tensor | None = None,
    num_samples: int = 1,
    distribution: str = "gaussian",
    create_graph: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate ``trace(D J_h(y))`` where ``D=diag(weight)``.

    For Poisson PURE, choosing ``weight=y`` estimates
    ``y^T diag(J_h(y))``. Probes are generated as
    ``sqrt(max(weight,0)) * epsilon`` with either Gaussian or Rademacher
    ``epsilon``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    y_req = y.detach().clone().requires_grad_(True)
    h_y = h_fn(y_req)
    if weight is None:
        weight = y_req
    weight = torch.clamp(weight.detach(), min=0.0) + eps

    estimates = []
    for sample_id in range(num_samples):
        probe = _sample_probe_like(h_y, distribution) * torch.sqrt(weight)
        inner = (h_y * probe).sum()
        grad = torch.autograd.grad(
            inner,
            y_req,
            retain_graph=sample_id < num_samples - 1,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        estimates.append(flatten_batch(grad * probe).sum(dim=1))

    return torch.stack(estimates, dim=0).mean(dim=0)
