from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .muon import matrix_sign_svd


def _as_matrix(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """View a matrix or convolution kernel in the Stiefel matrix geometry.

    Convolution filters use one row per output channel and flatten their input
    channels and spatial kernel dimensions. Two-dimensional inputs are
    returned unchanged so the existing matrix path keeps the same operations.
    """
    if tensor.ndim == 2:
        return tensor, tensor.shape
    if tensor.ndim == 4:
        return tensor.reshape(tensor.shape[0], -1), tensor.shape
    raise ValueError("ManifoldMuon parameters must be 2D matrices or 4D convolution kernels")


def _tall(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    transpose = matrix.shape[0] < matrix.shape[1]
    return (matrix.T if transpose else matrix), transpose


@torch.no_grad()
def retract_stiefel(matrix: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    matrix_view, original_shape = _as_matrix(matrix)
    tall, transposed = _tall(matrix_view / scale)
    q = matrix_sign_svd(tall)
    result = q.T if transposed else q
    return result.mul(scale).reshape(original_shape)


@torch.no_grad()
def manifold_muon_direction(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    dual_lr: float = 0.01,
    max_iterations: int = 10,
    scale: float = 1.0,
) -> tuple[torch.Tensor, float, int]:
    """Published dual-ascent Stiefel Muon direction.

    Returns a descent direction in the original orientation, its normalized
    tangent residual, and the fixed number of inner iterations used.
    """
    if weight.shape != gradient.shape:
        raise ValueError("weight and gradient must have the same shape")
    weight_view, original_shape = _as_matrix(weight)
    gradient_view, _ = _as_matrix(gradient)
    w, transposed = _tall(weight_view / scale)
    g, _ = _tall(gradient_view * scale)
    lam = -0.25 * (w.T @ g + g.T @ w)
    direction = torch.zeros_like(w)
    for iteration in range(max_iterations):
        direction = matrix_sign_svd(g + 2 * w @ lam)
        tangent_error = w.T @ direction + direction.T @ w
        lam.sub_(tangent_error, alpha=dual_lr * (1 - iteration / max_iterations))
    # Only the final residual is returned. Converting every intermediate
    # residual to float needlessly synchronizes CUDA on each DA-10 iteration.
    residual = float(tangent_error.norm() / math.sqrt(tangent_error.numel()))
    # The reference takes W <- W - eta*A, so A is the gradient-like direction.
    if transposed:
        direction = direction.T
    return direction.reshape(original_shape), residual, max_iterations


class ManifoldMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        dual_lr: float = 0.01,
        max_iterations: int = 10,
        scale: float = 1.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, scale=scale)
        super().__init__(params, defaults)
        self.dual_lr = dual_lr
        self.max_iterations = max_iterations

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["momentum"] == 0:
                    # No history is created or consumed by the no-momentum ablation.
                    grad = p.grad
                else:
                    momentum = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    momentum.mul_(group["momentum"]).add_(p.grad)
                    grad = p.grad.add(momentum, alpha=group["momentum"]) if group["nesterov"] else momentum
                direction, residual, iterations = manifold_muon_direction(
                    p,
                    grad,
                    dual_lr=self.dual_lr,
                    max_iterations=self.max_iterations,
                    scale=group["scale"],
                )
                # W=sQ, so a learning-rate step in Q coordinates is s*lr in W.
                p.add_(direction, alpha=-group["lr"] * group["scale"])
                p.copy_(retract_stiefel(p, group["scale"]))
                state["tangent_residual"] = residual
                state["inner_iterations"] = iterations
        return loss
