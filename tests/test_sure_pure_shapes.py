import torch

from sscp.losses.pure import poisson_pure_deepinv
from sscp.losses.sure import gaussian_sure


def test_gaussian_sure_returns_batch_shape_for_linear_h():
    torch.manual_seed(0)
    y = torch.randn(2, 1, 4, 4)
    h_fn = lambda z: 0.5 * z
    out = gaussian_sure(y, h_fn, sigma=0.1, num_trace_samples=1)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_poisson_pure_deepinv_returns_batch_shape_for_linear_h():
    torch.manual_seed(0)
    y = torch.rand(2, 1, 4, 4)
    h_fn = lambda z: 0.5 * z
    out = poisson_pure_deepinv(y, h_fn, gain=0.25, num_trace_samples=1)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
