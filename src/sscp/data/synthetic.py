from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSpec:
    num_samples: int = 64
    channels: int = 1
    height: int = 8
    width: int = 8
    seed: int = 0


def _make_clean_images(spec: SyntheticSpec) -> torch.Tensor:
    generator = torch.Generator().manual_seed(spec.seed)
    x = torch.rand(
        spec.num_samples,
        spec.channels,
        spec.height,
        spec.width,
        generator=generator,
    )
    return x.clamp(0.0, 1.0)


class GaussianIdentityDataset(Dataset):
    """Small paired dataset for identity Gaussian denoising tests/examples."""

    def __init__(
        self,
        num_samples: int = 64,
        sigma: float = 0.1,
        seed: int = 0,
        image_size: int = 8,
    ):
        spec = SyntheticSpec(num_samples=num_samples, height=image_size, width=image_size, seed=seed)
        generator = torch.Generator().manual_seed(seed + 1)
        self.x = _make_clean_images(spec)
        self.y = self.x + sigma * torch.randn(self.x.shape, generator=generator)
        self.sigma = sigma

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class PoissonIdentityDataset(Dataset):
    """Small paired dataset for the paper's raw-count Poisson convention."""

    def __init__(
        self,
        num_samples: int = 64,
        gamma: float = 4.0,
        seed: int = 0,
        image_size: int = 8,
    ):
        spec = SyntheticSpec(num_samples=num_samples, height=image_size, width=image_size, seed=seed)
        generator = torch.Generator().manual_seed(seed + 1)
        self.x = _make_clean_images(spec)
        self.y_counts = torch.poisson(gamma * self.x, generator=generator)
        self.gamma = gamma

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_counts[idx]
