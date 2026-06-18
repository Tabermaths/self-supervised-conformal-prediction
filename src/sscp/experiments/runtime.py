from __future__ import annotations

from pathlib import Path

import torch


def maybe_load_checkpoint(model, checkpoint: str | Path | None, device):
    """Load a PyTorch checkpoint if provided.

    Supports both plain state dictionaries and DeepInv trainer checkpoints with a
    ``state_dict`` entry.
    """
    if checkpoint is None:
        return
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if hasattr(model, "eval"):
        model.eval()


def make_estimator(model, physics, metadata):
    """Return a one-argument estimator closure compatible with legacy call styles."""

    def estimator(y):
        try:
            return model(y)
        except TypeError:
            try:
                return model(y, physics)
            except TypeError:
                if "sigma" in metadata:
                    return model(y, metadata["sigma"])
                if "gain" in metadata:
                    return model(y, metadata["gain"])
                raise

    return estimator
