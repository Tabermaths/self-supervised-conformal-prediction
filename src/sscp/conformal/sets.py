from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch

from .scores import measurement_score


@dataclass(frozen=True)
class ImplicitConformalSet:
    r"""Implicit measurement-domain conformal prediction set.

    The set is not enumerated in image space. It is represented by the test

    .. math:: \|Ax-A\hat x(y)\|_2^2/m \leq \widehat q.

    This is the representation used in the SURE/PURE papers and is the most
    convenient one for high-dimensional images.
    """

    q_hat: float | torch.Tensor
    estimator: Callable[[torch.Tensor], torch.Tensor]
    physics_or_A: object | None = None

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the measurement-domain non-conformity score for ``(x, y)``."""
        return measurement_score(x, y, self.estimator, self.physics_or_A)

    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a boolean tensor saying whether each ``x`` is in ``C_hat(y)``."""
        q = self.q_hat
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=y.dtype, device=y.device)
        else:
            q = q.to(dtype=y.dtype, device=y.device)
        return self.score(x, y) <= q
