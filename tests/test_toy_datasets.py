import torch

from sscp.data.synthetic import GaussianIdentityDataset, PoissonIdentityDataset


def test_gaussian_identity_dataset_shapes():
    dataset = GaussianIdentityDataset(num_samples=3, image_size=4)
    x, y = dataset[0]
    assert x.shape == y.shape == (1, 4, 4)


def test_poisson_identity_dataset_counts_are_nonnegative():
    dataset = PoissonIdentityDataset(num_samples=3, image_size=4)
    _, y = dataset[0]
    assert torch.all(y >= 0)
