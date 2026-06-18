from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms


@dataclass
class ExperimentBundle:
    name: str
    model: Any
    physics: Any
    train_dataset: Any
    test_dataset: Any
    batch_size: int
    device: torch.device
    output_dir: Path
    metadata: dict[str, Any]


def _require_deepinv():
    try:
        import deepinv as dinv  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DeepInv is required for the paper experiment builders. Install with "
            "`pip install -e .[deepinv]` or use the environment file."
        ) from exc
    return dinv


def get_device(device: str | None = None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hdf5_or_download_div2k(
    dinv,
    output_dir: Path,
    transform,
    generate_dataset: bool,
    train_h5: str = "dinv_dataset0.h5",
):
    if generate_dataset:
        train = dinv.datasets.DIV2K(root=output_dir, mode="train", transform=transform, download=True)
        test = dinv.datasets.DIV2K(root=output_dir, mode="val", transform=transform, download=True)
        return train, test
    train_path = output_dir / train_h5
    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected pre-generated DeepInv dataset at {train_path}. "
            "Run scripts/generate_measurements.py first, or set generate_dataset=True "
            "inside your own launcher."
        )
    return (
        dinv.datasets.HDF5Dataset(path=str(train_path), train=True),
        dinv.datasets.HDF5Dataset(path=str(train_path), train=False),
    )


def build_gaussian_denoising_div2k(
    root: str | Path = "outputs/gaussian_denoising_div2k",
    sigma: float = 0.1,
    image_size: int = 128,
    batch_size: int = 7,
    generate_dataset: bool = False,
    device: str | None = None,
) -> ExperimentBundle:
    """DIV2K Gaussian denoising experiment from the SURE paper."""
    dinv = _require_deepinv()
    device_t = get_device(device)
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((image_size, image_size))])
    physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma)).to(device_t)
    train, test = _load_hdf5_or_download_div2k(dinv, output_dir, transform, generate_dataset)
    model = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download").to(device_t)

    return ExperimentBundle(
        name="gaussian_denoising_div2k",
        model=model,
        physics=physics,
        train_dataset=train,
        test_dataset=test,
        batch_size=batch_size,
        device=device_t,
        output_dir=output_dir,
        metadata={"noise": "gaussian", "sigma": sigma, "image_size": image_size},
    )


def build_gaussian_deblurring_div2k(
    root: str | Path = "outputs/gaussian_deblurring_div2k",
    sigma_noise: float = 0.01,
    blur_sigma: tuple[float, float] = (2.0, 0.3),
    blur_angle: float = 30.0,
    image_size: int = 256,
    batch_size: int = 1,
    generate_dataset: bool = False,
    device: str | None = None,
) -> ExperimentBundle:
    """DIV2K Gaussian deblurring experiment from the SURE paper.

    This uses the same model-driven Polyblur reconstruction as the legacy code.
    """
    dinv = _require_deepinv()
    try:
        from polyblur import polyblur_deblurring  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("polyblur is required for the Gaussian deblurring experiment") from exc

    device_t = get_device(device)
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((image_size, image_size))])
    kernel = dinv.physics.blur.gaussian_blur(sigma=blur_sigma, angle=blur_angle)
    physics = dinv.physics.Blur(
        kernel,
        device=device_t,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
        padding="circular",
    ).to(device_t)
    train, test = _load_hdf5_or_download_div2k(dinv, output_dir, transform, generate_dataset)

    def model(y, physics_arg=physics):
        return polyblur_deblurring(y, n_iter=2, alpha=6, beta=1, c=0.352, b=0.768)

    return ExperimentBundle(
        name="gaussian_deblurring_div2k",
        model=model,
        physics=physics,
        train_dataset=train,
        test_dataset=test,
        batch_size=batch_size,
        device=device_t,
        output_dir=output_dir,
        metadata={
            "noise": "gaussian",
            "sigma": sigma_noise,
            "blur_sigma": blur_sigma,
            "blur_angle": blur_angle,
            "image_size": image_size,
        },
    )


def build_poisson_denoising_div2k(
    root: str | Path = "outputs/poisson_denoising_div2k",
    gamma: float = 4.0,
    image_size: int = 256,
    batch_size: int = 1,
    generate_dataset: bool = False,
    device: str | None = None,
) -> ExperimentBundle:
    """DIV2K Poisson denoising experiment from the PURE paper.

    The paper uses ``gamma=4``. In DeepInv's scaled-Poisson convention this is
    represented by ``gain=1/gamma``.
    """
    dinv = _require_deepinv()
    device_t = get_device(device)
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    gain = 1.0 / gamma
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((image_size, image_size))])
    physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonNoise(gain=gain)).to(device_t)
    train, test = _load_hdf5_or_download_div2k(
        dinv, output_dir, transform, generate_dataset, train_h5=f"dinv_dataset_gain_{gain:g}.h5"
    )

    model = build_unrolled_pgd(dinv, channels=3, device=device_t, iterations=4, scales=2, weight_tied=False)

    return ExperimentBundle(
        name="poisson_denoising_div2k",
        model=model,
        physics=physics,
        train_dataset=train,
        test_dataset=test,
        batch_size=batch_size,
        device=device_t,
        output_dir=output_dir,
        metadata={"noise": "poisson", "gamma": gamma, "gain": gain, "image_size": image_size},
    )


def build_unrolled_pgd(dinv, channels: int, device: torch.device, iterations: int = 4, scales: int = 2, weight_tied: bool = False):
    """Build the unrolled PGD network used in the Poisson denoising experiment."""
    if weight_tied:
        prior = dinv.optim.PnP(denoiser=dinv.models.UNet(in_channels=channels, out_channels=channels, scales=scales))
    else:
        prior = [
            dinv.optim.PnP(denoiser=dinv.models.UNet(in_channels=channels, out_channels=channels, scales=scales))
            for _ in range(iterations)
        ]
    return dinv.unfolded.unfolded_builder(
        "PGD",
        params_algo={"stepsize": [1.0] * iterations, "g_param": [0.01] * iterations, "lambda": 1.0},
        trainable_params=["lambda", "stepsize", "g_param"],
        data_fidelity=dinv.optim.L2(),
        max_iter=iterations,
        prior=prior,
        verbose=False,
    ).to(device)


def build_poisson_deblurring_div2k(*args, **kwargs) -> ExperimentBundle:
    """Template hook for the Poisson deblurring experiment.

    The paper used a specialised SwinIR/P4IP-style deblurring estimator. The
    uploaded project did not include a small public checkpoint, so this builder
    deliberately raises a clear error instead of silently constructing the wrong
    model. The conformal/PURE code is model-agnostic: add the estimator here
    once the final checkpoint/model release policy is fixed.
    """
    raise NotImplementedError(
        "Poisson deblurring needs the specialised SwinIR/P4IP estimator and its "
        "checkpoint. Use the generic PURE/conformal functions with your estimator, "
        "or implement this builder once the model weights are available."
    )


EXPERIMENTS = {
    "gaussian_denoising_div2k": build_gaussian_denoising_div2k,
    "gaussian_deblurring_div2k": build_gaussian_deblurring_div2k,
    "poisson_denoising_div2k": build_poisson_denoising_div2k,
    "poisson_deblurring_div2k": build_poisson_deblurring_div2k,
}


def build_experiment(name: str, **kwargs) -> ExperimentBundle:
    try:
        return EXPERIMENTS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown experiment {name!r}. Available: {sorted(EXPERIMENTS)}") from exc
