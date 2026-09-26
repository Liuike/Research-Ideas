import copy
import random

import numpy as np
import pytest
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from optimizer_resurrection.models import ProductUnitNetwork
from optimizer_resurrection.punn_sampling import (
    evaluate_vectors,
    orthonormal_directions,
    sample_global,
    sample_slice,
)


def _task_data(input_dim: int, output_dim: int = 1, *, seed: int = 123):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(19, input_dim, generator=generator) * 2.0 - 1.0
    x[x == 0] = 0.25
    y = torch.randn(19, output_dim, generator=generator)
    return x, y


@pytest.mark.parametrize("input_dim,hidden_units", [(1, 1), (2, 3)])
def test_batched_vector_evaluation_matches_product_unit_forward_and_preserves_model(
    input_dim, hidden_units
):
    model = ProductUnitNetwork(input_dim, hidden_units, input_domain="real_complex")
    train_x, train_y = _task_data(input_dim)
    generator = torch.Generator(device="cpu").manual_seed(56)
    vectors = torch.randn(7, parameters_to_vector(model.parameters()).numel(), generator=generator)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}

    actual = evaluate_vectors(model, train_x, train_y, vectors, batch_size=3)
    expected = []
    reference = copy.deepcopy(model)
    for vector in vectors:
        vector_to_parameters(vector.clone(), reference.parameters())
        with torch.no_grad():
            expected.append((reference(train_x) - train_y).square().mean())

    assert torch.allclose(actual, torch.stack(expected), rtol=1e-5, atol=1e-7)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


def test_slice_has_exact_center_and_does_not_clip_points():
    model = ProductUnitNetwork(1, 1, input_domain="real_complex")
    train_x, train_y = _task_data(1)
    center = torch.tensor([1.25, -0.4, 0.1])
    directions = orthonormal_directions(3, seed=7)
    result = sample_slice(model, train_x, train_y, center,
                          directions=directions, radius=1.0, points=5)

    assert torch.equal(result["grid_coordinates"][2, 2], center)
    expected_center = evaluate_vectors(model, train_x, train_y, center[None])[0]
    assert torch.equal(result["mse"][2, 2], expected_center)
    assert result["metadata"]["bounds_clipped"] is False
    assert torch.any(result["grid_coordinates"].abs() > 1.0)


def _capture_rng_states():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def _assert_rng_states_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    if left["torch_cuda"] is not None:
        assert len(left["torch_cuda"]) == len(right["torch_cuda"])
        assert all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))


def test_global_sampling_repeats_and_preserves_all_global_rng_states():
    model = ProductUnitNetwork(2, 3, input_domain="real_complex")
    train_x, train_y = _task_data(2)
    kwargs = dict(seed=4, bound=1.0, prw_steps=11, mrw_steps=13,
                  uniform_per_dim=3, dispersion_samples=17, batch_size=4)
    before = _capture_rng_states()
    first = sample_global(model, train_x, train_y, **kwargs)
    middle = _capture_rng_states()
    second = sample_global(model, train_x, train_y, **kwargs)
    after = _capture_rng_states()

    _assert_rng_states_equal(before, middle)
    _assert_rng_states_equal(middle, after)
    assert first["metadata"] == second["metadata"]
    assert first["summary"] == second["summary"]
    for name in first["samples"]:
        for key in ("vectors", "mse", "nonfinite"):
            assert torch.equal(first["samples"][name][key], second["samples"][name][key])


def test_walks_are_bounded_and_follow_declared_step_rules():
    model = ProductUnitNetwork(1, 1, input_domain="real_complex")
    train_x, train_y = _task_data(1)
    result = sample_global(model, train_x, train_y, seed=8, bound=1.0,
                           prw_steps=20, mrw_steps=30, uniform_per_dim=5,
                           dispersion_samples=16)
    samples = result["samples"]
    for name in ("corners", "prw_micro", "prw_macro", "mrw", "uniform", "dispersion"):
        vectors = samples[name]["vectors"]
        assert torch.all(vectors >= -1.0)
        assert torch.all(vectors <= 1.0)

    micro_vectors = samples["prw_micro"]["vectors"]
    macro_vectors = samples["prw_macro"]["vectors"]
    assert torch.all((micro_vectors[:, 1:] - micro_vectors[:, :-1]).abs() <= 0.02 + 1e-6)
    assert torch.all((macro_vectors[:, 1:] - macro_vectors[:, :-1]).abs() <= 0.2 + 1e-6)
    mrw_vectors = samples["mrw"]["vectors"]
    mrw_steps = mrw_vectors[:, 1:] - mrw_vectors[:, :-1]
    assert mrw_vectors.shape[0] == sum(parameter.numel() for parameter in model.parameters())
    assert torch.all((mrw_steps != 0).sum(dim=2) == 1)
    assert torch.all(mrw_steps.abs() <= 0.02 + 1e-6)
    assert result["summary"]["mrw_gradient_estimates"]["valid_step_count"] == mrw_steps.shape[0] * 30


def test_orthonormal_directions_are_seeded_and_orthogonal():
    first = orthonormal_directions(10, seed=9)
    second = orthonormal_directions(10, seed=9)
    assert torch.equal(first, second)
    assert torch.allclose(first @ first.T, torch.eye(2), atol=1e-6)


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda m, x, y: evaluate_vectors(m, x, y, torch.zeros(1, 99)), "shape"),
        (lambda m, x, y: evaluate_vectors(m, x, y, torch.full((1, 3), float("nan"))), "finite"),
        (lambda m, x, y: evaluate_vectors(m, x, y, torch.zeros(1, 3), batch_size=0), "batch_size"),
        (lambda m, x, y: sample_global(m, x, y, seed=-1, bound=1.0), "seed"),
        (lambda m, x, y: sample_global(m, x, y, seed=1, bound=float("inf")), "bound"),
        (lambda m, x, y: sample_global(m, x, y, seed=1, bound=1.0, prw_steps=0), "prw_steps"),
        (lambda m, x, y: sample_global(m, x, y, seed=1, bound=1.0, device="mps"), "CPU or CUDA"),
        (lambda m, x, y: sample_slice(m, x, y, torch.zeros(3), directions=torch.ones(2, 3), points=4), "odd"),
        (lambda m, x, y: sample_slice(m, x, y, torch.zeros(3), directions=torch.ones(2, 3), radius=0), "radius"),
        (lambda m, x, y: sample_slice(m, x, y, torch.zeros(2), directions=torch.ones(2, 3)), "center"),
    ],
)
def test_invalid_sampling_inputs_raise_value_error(call, match):
    model = ProductUnitNetwork(1, 1, input_domain="real_complex")
    x, y = _task_data(1)
    with pytest.raises(ValueError, match=match):
        call(model, x, y)


def test_sampler_rejects_zero_inputs_for_real_complex_product_units():
    model = ProductUnitNetwork(1, 1, input_domain="real_complex")
    x, y = _task_data(1)
    x[0, 0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        evaluate_vectors(model, x, y, torch.zeros(1, 3))
