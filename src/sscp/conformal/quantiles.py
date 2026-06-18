from __future__ import annotations

import math

import torch


def _as_1d_tensor(scores: torch.Tensor | list[float]) -> torch.Tensor:
    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.detach().flatten()
    if scores.numel() == 0:
        raise ValueError("scores must contain at least one element")
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        raise ValueError("scores contain no finite values")
    return scores


def conformal_quantile(
    scores: torch.Tensor | list[float],
    confidence: float,
    finite_sample_correction: bool = True,
) -> torch.Tensor:
    """Return the split-conformal quantile for a target confidence level.

    Parameters
    ----------
    scores:
        Calibration non-conformity scores, e.g. SURE/PURE scores or supervised
        measurement errors.
    confidence:
        Desired coverage probability, e.g. ``0.95`` for a 95% prediction set.
        This is ``1-alpha`` in the papers.
    finite_sample_correction:
        If ``True``, uses the standard order statistic
        ``ceil((M+1)*confidence)``. If this exceeds ``M`` the returned value is
        ``+inf`` so the prediction set is valid but conservative.

    Notes
    -----
    This avoids the common naming ambiguity in the legacy scripts where the
    variable named ``alpha`` was sometimes used as a confidence level and
    sometimes as a significance level.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    scores_t = _as_1d_tensor(scores)
    sorted_scores = torch.sort(scores_t).values
    m = sorted_scores.numel()

    if finite_sample_correction:
        k = math.ceil((m + 1) * confidence)
        if k > m:
            return torch.tensor(float("inf"), dtype=sorted_scores.dtype, device=sorted_scores.device)
        return sorted_scores[k - 1]

    # Nearest-higher empirical quantile without the +1 finite-sample correction.
    k = max(1, min(m, math.ceil(m * confidence)))
    return sorted_scores[k - 1]


def leave_one_out_quantiles(scores: torch.Tensor | list[float], confidence: float) -> torch.Tensor:
    """Leave-one-out conformal quantiles for calibration-set diagnostics."""
    scores_t = _as_1d_tensor(scores)
    quantiles = []
    for i in range(scores_t.numel()):
        mask = torch.ones(scores_t.numel(), dtype=torch.bool, device=scores_t.device)
        mask[i] = False
        quantiles.append(conformal_quantile(scores_t[mask], confidence))
    return torch.stack(quantiles)
