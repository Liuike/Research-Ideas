import math

import pytest
import torch

from optimizer_resurrection.optim.moonlight_muon import (
    MoonlightMuon,
    moonlight_matrix_direction,
    moonlight_update_scale,
)


def _official_fp32_direction(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Reference the official Moonlight update, retaining this repo's FP32."""
    x = matrix.float()
    if x.size(0) > x.size(1):
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if matrix.size(0) > matrix.size(1):
        x = x.T
    return x.to(matrix.dtype)


@pytest.mark.parametrize(
    "shape, expected",
    [((1, 1), 0.2), ((3, 2), 0.2 * math.sqrt(3)), ((2, 3), 0.2 * math.sqrt(3))],
)
def test_moonlight_shape_learning_rate_scale(shape, expected):
    assert moonlight_update_scale(torch.Size(shape)) == pytest.approx(expected)


def test_direction_matches_moonlight_quintic_in_fp32_for_tall_matrix():
    matrix = torch.tensor([[0.2, -0.3], [0.5, 0.7], [-0.1, 0.8]], dtype=torch.float32)
    actual = moonlight_matrix_direction(matrix)
    expected = _official_fp32_direction(matrix)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("shape", [(1, 1), (3, 2)])
def test_parameter_step_uses_adjusted_lr_and_base_lr_weight_decay(shape):
    parameter = torch.nn.Parameter(torch.full(shape, 1.25))
    gradient = torch.arange(1, parameter.numel() + 1, dtype=torch.float32).reshape(shape)
    parameter.grad = gradient.clone()
    initial = parameter.detach().clone()
    lr, weight_decay = 0.01, 0.2
    expected_direction = _official_fp32_direction(gradient)

    optimizer = MoonlightMuon(
        [parameter],
        lr=lr,
        momentum=0.0,
        nesterov=False,
        weight_decay=weight_decay,
    )
    optimizer.step()

    expected = initial * (1 - lr * weight_decay)
    expected -= lr * moonlight_update_scale(shape) * expected_direction
    assert torch.allclose(parameter, expected, atol=1e-7, rtol=1e-7)


def test_momentum_and_nesterov_match_official_buffer_update_order():
    parameter = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = MoonlightMuon([parameter], lr=0.01, weight_decay=0.0)
    beta = 0.95
    momentum = torch.zeros_like(parameter)
    expected_parameter = parameter.detach().clone()

    for values in (
        [[0.2, -0.3], [0.5, 0.7], [-0.1, 0.8]],
        [[-0.4, 0.9], [0.1, -0.2], [0.6, 0.3]],
    ):
        gradient = torch.tensor(values)
        parameter.grad = gradient.clone()
        momentum.mul_(beta).add_(gradient)
        nesterov_update = gradient.add(momentum, alpha=beta)
        expected_parameter.add_(
            _official_fp32_direction(nesterov_update),
            alpha=-0.01 * moonlight_update_scale(parameter.shape),
        )

        optimizer.step()

    assert torch.allclose(optimizer.state[parameter]["momentum_buffer"], momentum)
    assert torch.allclose(parameter, expected_parameter, atol=2e-7, rtol=2e-7)


def test_vector_parameter_is_rejected_when_it_has_a_gradient():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.ones_like(parameter)
    optimizer = MoonlightMuon([parameter])
    with pytest.raises(ValueError, match="must be matrices"):
        optimizer.step()

