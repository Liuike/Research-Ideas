from __future__ import annotations

import torch
from torch.optim import Optimizer

from .manifold_muon import retract_stiefel


class RiemannianSGD(Optimizer):
    """Projected SGD with polar retraction on scaled Stiefel matrices."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, scale: float = 1.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, scale=scale))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = group["scale"]
                q = p / scale
                grad_q = p.grad * scale
                symmetric = 0.5 * (q.T @ grad_q + grad_q.T @ q)
                tangent = grad_q - q @ symmetric
                if group["momentum"]:
                    buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                    buf.mul_(group["momentum"]).add_(tangent)
                    tangent = buf
                q.add_(tangent, alpha=-group["lr"])
                p.copy_(retract_stiefel(q, 1.0).mul(scale))
        return loss

