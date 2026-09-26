"""FP32 PyTorch port of the Moonlight-scaled Muon update.

This applies Muon only to matrix parameters supplied by the caller. In a
hybrid setup, leave vector parameters and output heads to a separate AdamW
optimizer, matching Moonlight's parameter assignment convention.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

SOURCE = (
    "https://github.com/MoonshotAI/Moonlight/blob/"
    "c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py"
)


@torch.no_grad()
def moonlight_matrix_direction(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the polar direction using Moonlight's quintic iteration.

    The official implementation converts the input to bfloat16 before the
    iteration. This project uses FP32 for its primary comparisons, so this
    version deliberately performs all orthogonalization operations in FP32.
    """
    if matrix.ndim != 2:
        raise ValueError("moonlight_matrix_direction expects a 2D tensor")
    if not matrix.is_floating_point():
        raise TypeError("Muon requires floating-point matrices")
    if steps < 1:
        raise ValueError("steps must be positive")

    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        # Keep the source expression's operation order in FP32. It is
        # algebraically equal to c * (gram @ gram), with slightly different
        # round-off behavior.
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def moonlight_update_scale(shape: torch.Size | tuple[int, ...]) -> float:
    """Return Moonlight's shape-dependent learning-rate multiplier."""
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Moonlight Muon expects non-empty 2D parameter shapes")
    rows, cols = shape
    return 0.2 * math.sqrt(max(rows, cols))


class MoonlightMuon(Optimizer):
    """Muon with Moonlight's update scale and decoupled weight decay.

    Pass only the matrices assigned to Muon. Moonlight uses AdamW for vectors
    and output/embedding heads; a caller can combine this optimizer with a
    separate AdamW through ``OptimizerBundle``.

    ``weight_decay`` defaults to the Moonlight toy implementation's 0.1.
    Comparative recipes in this repository can pass 0.0 when matching its
    zero-decay primary protocol.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
    ) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError("ns_steps must be a positive integer")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Moonlight Muon parameters must be matrices")
                if parameter.grad.is_sparse:
                    raise ValueError("Moonlight Muon does not support sparse gradients")

                grad = parameter.grad
                state = self.state[parameter]
                momentum_buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(parameter)
                )
                momentum_buffer.mul_(group["momentum"]).add_(grad)
                update = (
                    grad.add(momentum_buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
                direction = moonlight_matrix_direction(update, group["ns_steps"])

                lr = group["lr"]
                if group["weight_decay"]:
                    # Moonlight decays using the base lr, before applying the
                    # shape-adjusted Muon step.
                    parameter.mul_(1 - lr * group["weight_decay"])
                adjusted_lr = lr * moonlight_update_scale(parameter.shape)
                parameter.add_(direction, alpha=-adjusted_lr)

        return loss
