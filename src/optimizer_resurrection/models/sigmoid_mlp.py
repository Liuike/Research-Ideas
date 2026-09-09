from __future__ import annotations

import math

import torch
from torch import nn


class SigmoidMLP(nn.Module):
    """Plain dense MLP whose ``depth`` is the number of hidden layers."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        num_classes: int,
        activation: str = "logistic",
        init: str = "historical",
        orthogonal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        if activation not in {"logistic", "tanh"}:
            raise ValueError("activation must be 'logistic' or 'tanh'")
        self.activation_name = activation
        self.hidden = nn.ModuleList(
            [nn.Linear(input_dim, width)]
            + [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        self.output = nn.Linear(width, num_classes)
        self.reset_parameters(init, orthogonal_scale)

    def reset_parameters(self, init: str, scale: float = 1.0) -> None:
        with torch.no_grad():
            for index, layer in enumerate([*self.hidden, self.output]):
                if init == "historical":
                    bound = 1.0 / math.sqrt(layer.in_features)
                    layer.weight.uniform_(-bound, bound)
                elif init == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                    # Scaled-Stiefel is an intervention on repeated hidden
                    # transforms only; input/output scaling must not confound it.
                    if 0 < index < len(self.hidden):
                        layer.weight.mul_(scale)
                else:
                    raise ValueError(f"unknown initialization: {init}")
                layer.bias.zero_()

    @property
    def compared_weights(self) -> list[nn.Parameter]:
        """Repeated hidden transforms W_2..W_L used by the primary intervention."""
        return [layer.weight for layer in self.hidden[1:]]

    @property
    def standard_muon_weights(self) -> list[nn.Parameter]:
        """Every eligible hidden 2D weight in the standard Muon convention."""
        return [layer.weight for layer in self.hidden]

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.flatten(1)
        activations: list[torch.Tensor] = []
        fn = torch.sigmoid if self.activation_name == "logistic" else torch.tanh
        for layer in self.hidden:
            x = fn(layer(x))
            if return_activations:
                activations.append(x)
        logits = self.output(x)
        return (logits, activations) if return_activations else logits
