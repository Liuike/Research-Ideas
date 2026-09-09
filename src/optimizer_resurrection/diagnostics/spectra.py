from __future__ import annotations

import math

import torch


@torch.no_grad()
def matrix_statistics(matrix: torch.Tensor) -> dict[str, float]:
    values = torch.linalg.svdvals(matrix.float())
    total = values.sum().clamp_min(1e-12)
    probabilities = values / total
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    minimum = values.min()
    return {
        "spectral_norm": float(values.max()),
        "singular_min": float(minimum),
        "condition_number": float(values.max() / minimum.clamp_min(1e-12)),
        "effective_rank": float(effective_rank),
    }


@torch.no_grad()
def stiefel_residual(matrix: torch.Tensor, scale: float = 1.0) -> float:
    q = matrix.float() / scale
    if q.shape[0] < q.shape[1]:
        q = q.T
    identity = torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
    return float((q.T @ q - identity).norm() / math.sqrt(q.shape[1]))

