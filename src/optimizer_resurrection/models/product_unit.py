"""Single-hidden-layer product-unit network with learned real exponents.

The hidden unit follows Durbin and Rumelhart (1989), ``prod_i x_i ** v_ji``.
For signed nonzero inputs, ``real_complex`` takes the real part of the
principal complex power, as specified by Engelbrecht and Gouldie (2024).
The latter is a real-valued network, not a trainable complex-weight model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ProductUnitNetwork(nn.Module):
    """Learned-exponent hidden units followed by a linear output layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        output_dim: int = 1,
        *,
        input_domain: str = "real_complex",
        init_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_units, output_dim) < 1:
            raise ValueError("all dimensions must be positive")
        if input_domain not in {"positive", "real_complex"}:
            raise ValueError("input_domain must be 'positive' or 'real_complex'")
        if not math.isfinite(init_bound) or init_bound <= 0:
            raise ValueError("init_bound must be finite and positive")
        self.input_dim = input_dim
        self.input_domain = input_domain
        self.init_bound = init_bound
        self.exponents = nn.Parameter(torch.empty(hidden_units, input_dim))
        self.output = nn.Linear(hidden_units, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.exponents.uniform_(-self.init_bound, self.init_bound)
            self.output.weight.uniform_(-self.init_bound, self.init_bound)
            self.output.bias.uniform_(-self.init_bound, self.init_bound)

    def product_activations(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected [batch, {self.input_dim}] inputs")
        if not torch.isfinite(x).all():
            raise ValueError("product-unit inputs must be finite")
        if self.input_domain == "positive":
            if (x <= 0).any():
                raise ValueError("positive product units require strictly positive inputs")
            return torch.exp(torch.log(x) @ self.exponents.T)
        if (x == 0).any():
            raise ValueError("real_complex product units require nonzero inputs")
        log_magnitude = torch.log(x.abs()) @ self.exponents.T
        phase = (x < 0).to(x.dtype) @ self.exponents.T
        return torch.exp(log_magnitude) * torch.cos(math.pi * phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.product_activations(x))
