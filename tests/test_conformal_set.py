import torch

from sscp.conformal.sets import ImplicitConformalSet


def test_implicit_conformal_set_contains_identity_case():
    x = torch.tensor([[[[1.0, 2.0]]]])
    y = x + 0.1
    estimator = lambda yy: yy - 0.1
    cset = ImplicitConformalSet(q_hat=1e-8, estimator=estimator)
    assert cset.contains(x, y).item()
