from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .scores import measurement_score


def empirical_coverage(
    dataloader: Iterable,
    estimator: Callable[[torch.Tensor], torch.Tensor],
    physics_or_A,
    q_hat: float | torch.Tensor,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate empirical coverage on a paired test set."""
    if torch.is_tensor(q_hat):
        q_value = q_hat.detach().to(device)
    else:
        q_value = torch.tensor(q_hat, device=device)

    total = 0
    covered = 0
    for batch in tqdm(dataloader, desc="Evaluating empirical coverage"):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("coverage evaluation requires batches of (x, y)")
        x, y = batch[0].to(device), batch[1].to(device)
        scores = measurement_score(x, y, estimator=estimator, physics_or_A=physics_or_A)
        covered += (scores <= q_value).sum().item()
        total += scores.numel()
    return covered / max(total, 1)


def empirical_coverage_curve(
    test_scores: torch.Tensor | np.ndarray,
    quantiles: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Compute a coverage curve from precomputed test scores and quantiles."""
    test = torch.as_tensor(test_scores).flatten()
    qs = torch.as_tensor(quantiles).flatten()
    return torch.stack([(test <= q).float().mean() for q in qs]).cpu().numpy()
