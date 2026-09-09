from __future__ import annotations

import torch


def input_output_jacobian_spectrum(
    model: torch.nn.Module, sample: torch.Tensor, max_values: int = 16
) -> list[float]:
    """Exact single-example input/logit Jacobian spectrum for checkpoints."""
    was_training = model.training
    model.eval()
    x = sample[:1].detach().requires_grad_(True)

    def f(value: torch.Tensor) -> torch.Tensor:
        output = model(value)
        return output.flatten()

    jacobian = torch.autograd.functional.jacobian(f, x, vectorize=True)
    matrix = jacobian.reshape(jacobian.shape[0], -1).float()
    values = torch.linalg.svdvals(matrix)[:max_values].detach().cpu().tolist()
    model.train(was_training)
    return [float(value) for value in values]

