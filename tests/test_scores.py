import torch

from sscp.conformal.scores import measurement_score


def test_measurement_score_identity_operator():
    x = torch.tensor([[[[1.0, 2.0]]]])
    y = x + 0.1
    estimator = lambda yy: yy - 0.1
    score = measurement_score(x, y, estimator, physics_or_A=None)
    assert torch.allclose(score, torch.zeros_like(score))


def test_measurement_score_callable_forward_operator():
    x = torch.tensor([[[[1.0, 2.0]]]])
    y = x
    estimator = lambda yy: yy + 1.0
    A = lambda z: 2.0 * z
    score = measurement_score(x, y, estimator, physics_or_A=A)
    assert torch.allclose(score, torch.tensor([4.0]))
