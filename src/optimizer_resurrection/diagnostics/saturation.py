from __future__ import annotations

import torch


@torch.no_grad()
def activation_statistics(activation: torch.Tensor, kind: str) -> dict[str, float]:
    if kind == "logistic":
        saturated = (activation < 0.05) | (activation > 0.95)
    elif kind == "tanh":
        saturated = activation.abs() > 0.95
    else:
        raise ValueError(f"unknown activation: {kind}")
    return {
        "mean": float(activation.mean()),
        "std": float(activation.std()),
        "saturation_fraction": float(saturated.float().mean()),
    }

