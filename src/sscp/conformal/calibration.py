from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from tqdm import tqdm

from .quantiles import conformal_quantile
from .scores import measurement_score


@dataclass(frozen=True)
class CalibrationResult:
    """Container returned by calibration routines."""

    scores: torch.Tensor
    confidence: float
    q_hat: torch.Tensor
    method: str


def _to_device_pair(batch, device: torch.device | str):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0].to(device), batch[1].to(device)
    return None, batch.to(device)


def calibrate_self_supervised(
    dataloader: Iterable,
    risk_estimator: Callable[[torch.Tensor], torch.Tensor],
    confidence: float,
    device: torch.device | str = "cpu",
    description: str = "Calibrating self-supervised scores",
) -> CalibrationResult:
    """Calibrate from measurements only using SURE/PURE scores."""
    values = []
    for batch in tqdm(dataloader, desc=description):
        _, y = _to_device_pair(batch, device)
        score = risk_estimator(y)
        values.append(score.detach().cpu().flatten())
    scores = torch.cat(values)
    q_hat = conformal_quantile(scores, confidence)
    return CalibrationResult(scores=scores, confidence=confidence, q_hat=q_hat, method="self_supervised")


def calibrate_supervised(
    dataloader: Iterable,
    estimator: Callable[[torch.Tensor], torch.Tensor],
    physics_or_A,
    confidence: float,
    device: torch.device | str = "cpu",
    description: str = "Calibrating supervised scores",
) -> CalibrationResult:
    """Calibrate from paired ground-truth/measurement data for comparison."""
    values = []
    for batch in tqdm(dataloader, desc=description):
        x, y = _to_device_pair(batch, device)
        if x is None:
            raise ValueError("supervised calibration requires batches of (x, y)")
        score = measurement_score(x, y, estimator=estimator, physics_or_A=physics_or_A)
        values.append(score.detach().cpu().flatten())
    scores = torch.cat(values)
    q_hat = conformal_quantile(scores, confidence)
    return CalibrationResult(scores=scores, confidence=confidence, q_hat=q_hat, method="supervised")
