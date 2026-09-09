from __future__ import annotations

import math

import torch
from torch import nn


class VanillaRNN(nn.Module):
    """A deliberately plain tanh RNN with an exposed recurrent matrix."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int = 2,
        init: str = "historical",
        orthogonal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.b_y = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters(init, orthogonal_scale)

    def reset_parameters(self, init: str, scale: float = 1.0) -> None:
        with torch.no_grad():
            if init == "orthogonal":
                nn.init.orthogonal_(self.W_hh)
                self.W_hh.mul_(scale)
            elif init == "historical":
                bound = 1.0 / math.sqrt(self.hidden_size)
                self.W_hh.uniform_(-bound, bound)
            else:
                raise ValueError(f"unknown initialization: {init}")
            self.W_xh.uniform_(-1 / math.sqrt(self.input_size), 1 / math.sqrt(self.input_size))
            self.W_hy.uniform_(-1 / math.sqrt(self.hidden_size), 1 / math.sqrt(self.hidden_size))
            self.b_h.zero_()
            self.b_y.zero_()

    @property
    def compared_weights(self) -> list[nn.Parameter]:
        return [self.W_hh]

    @property
    def standard_muon_weights(self) -> list[nn.Parameter]:
        """Secondary assignment adds the input-to-hidden matrix."""
        return [self.W_hh, self.W_xh]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, timesteps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        states: list[torch.Tensor] = []
        if lengths is None:
            lengths = torch.full((batch,), timesteps, device=x.device, dtype=torch.long)
        for t in range(timesteps):
            proposed = torch.tanh(x[:, t] @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            active = (t < lengths).unsqueeze(1)
            h = torch.where(active, proposed, h)
            if return_states:
                states.append(h)
        logits = h @ self.W_hy.T + self.b_y
        return (logits, torch.stack(states, dim=1)) if return_states else logits
