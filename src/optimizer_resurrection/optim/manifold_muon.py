from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .muon import matrix_sign_svd


def _tall(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    transpose = matrix.shape[0] < matrix.shape[1]
    return (matrix.T if transpose else matrix), transpose


@torch.no_grad()
def retract_stiefel(matrix: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    tall, transposed = _tall(matrix / scale)
    q = matrix_sign_svd(tall)
    result = q.T if transposed else q
    return result.mul(scale)


@torch.no_grad()
def manifold_muon_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    dual_lr: float = 0.01,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    scale: float = 1.0,
) -> tuple[torch.Tensor, float, int]:
    """Published dual-ascent Stiefel Muon direction.

    Returns a descent direction in the original orientation, its normalized
    tangent residual, and the number of inner iterations used.
    """
    w, transposed = _tall(weight / scale)
    g, _ = _tall(gradient * scale)
    lam = -0.25 * (w.T @ g + g.T @ w)
    residual = float("inf")
    direction = torch.zeros_like(w)
    used = max_iterations
    for iteration in range(max_iterations):
        direction = matrix_sign_svd(g + 2 * w @ lam)
        tangent_error = w.T @ direction + direction.T @ w
        residual = float(tangent_error.norm() / math.sqrt(tangent_error.numel()))
        if residual < tolerance:
            used = iteration + 1
            break
        lam.sub_(tangent_error, alpha=dual_lr * (1 - iteration / max_iterations))
    if residual >= tolerance:
        raise RuntimeError(
            "Manifold Muon dual ascent failed to reach tangent tolerance: "
            f"residual={residual:.3e}, tolerance={tolerance:.3e}, "
            f"iterations={max_iterations}"
        )
    # The reference takes W <- W - eta*A, so A is the gradient-like direction.
    if transposed:
        direction = direction.T
    return direction, residual, used


class ManifoldMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        dual_lr: float = 0.01,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        scale: float = 1.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, scale=scale)
        super().__init__(params, defaults)
        self.dual_lr = dual_lr
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                momentum = state.setdefault("momentum_buffer", torch.zeros_like(p))
                momentum.mul_(group["momentum"]).add_(p.grad)
                grad = p.grad.add(momentum, alpha=group["momentum"]) if group["nesterov"] else momentum
                direction, residual, iterations = manifold_muon_direction(
                    p,
                    grad,
                    dual_lr=self.dual_lr,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    scale=group["scale"],
                )
                # W=sQ, so a learning-rate step in Q coordinates is s*lr in W.
                p.add_(direction, alpha=-group["lr"] * group["scale"])
                p.copy_(retract_stiefel(p, group["scale"]))
                state["tangent_residual"] = residual
                state["inner_iterations"] = iterations
        return loss
