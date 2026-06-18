import math

import torch

from sscp.conformal.quantiles import conformal_quantile, leave_one_out_quantiles


def test_conformal_quantile_uses_finite_sample_order_statistic():
    scores = torch.tensor([0.1, 0.4, 0.2, 0.3])
    # ceil((4+1)*0.6)=3, sorted score 3 is 0.3
    assert torch.isclose(conformal_quantile(scores, 0.6), torch.tensor(0.3))


def test_conformal_quantile_returns_inf_when_confidence_too_high_for_m():
    scores = torch.tensor([0.1, 0.2, 0.3])
    assert math.isinf(float(conformal_quantile(scores, 0.95)))


def test_leave_one_out_quantiles_shape():
    scores = torch.linspace(0.1, 1.0, 10)
    q = leave_one_out_quantiles(scores, 0.8)
    assert q.shape == scores.shape
