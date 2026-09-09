from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


@torch.no_grad()
def matrix_sign_svd(matrix: torch.Tensor) -> torch.Tensor:
    """Exact polar/matrix-sign factor U V^T."""
    original_dtype = matrix.dtype
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (u @ vh).to(original_dtype)


@torch.no_grad()
def matrix_sign_ns(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon Newton--Schulz quintic approximation to the polar factor."""
    if matrix.ndim != 2:
        raise ValueError("matrix_sign_ns expects a 2D tensor")
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(Optimizer):
    """Muon for a collection of matrix parameters only."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        backend: str = "ns",
    ) -> None:
        if backend not in {"ns", "svd"}:
            raise ValueError("backend must be 'ns' or 'svd'")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.ns_steps = ns_steps
        self.backend = backend

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon parameters must be matrices")
                state = self.state[p]
                momentum = state.setdefault("momentum_buffer", torch.zeros_like(p))
                momentum.mul_(group["momentum"]).add_(p.grad)
                update = p.grad.add(momentum, alpha=group["momentum"]) if group["nesterov"] else momentum
                direction = (
                    matrix_sign_ns(update, self.ns_steps)
                    if self.backend == "ns"
                    else matrix_sign_svd(update)
                )
                # Reference Muon corrects rectangular layers so per-output-unit
                # update scale matches square matrices.
                direction.mul_(math.sqrt(max(1.0, p.shape[0] / p.shape[1])))
                p.add_(direction, alpha=-group["lr"])
        return loss
