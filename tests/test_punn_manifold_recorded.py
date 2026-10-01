from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from optimizer_resurrection import punn_manifold_recorded as recorded
from optimizer_resurrection.models.product_unit import ProductUnitNetwork
from optimizer_resurrection.optim.manifold_muon import retract_stiefel
from optimizer_resurrection.punn_architecture_data import TASK_SPECS
from optimizer_resurrection.punn_sampling import orthonormal_directions


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/product_unit/manifold_landscape_gpu.yaml"


def _args(task: str, architecture: str, *, data_seed: int = 7, device: str = "cpu"):
    input_dim, small_hidden, oversized_hidden, output_dim = TASK_SPECS[task]
    hidden = small_hidden if architecture == "small" else oversized_hidden
    regularization = 0.0001 if architecture == "regularized" else 0.0
    seed = 2020007
    return recorded.parse_args([
        "--protocol", recorded.PROTOCOL,
        "--task", task,
        "--architecture", architecture,
        "--method", "manifold_muon_da10",
        "--seed", str(seed),
        "--data-seed", str(data_seed),
        "--epochs", "2",
        "--regularization-lambda", str(regularization),
        "--learning-rate", "0.01",
        "--momentum", "0.95",
        "--weight-decay", "0.0",
        "--aux-learning-rate", "0.001",
        "--secondary", "false",
        "--manifold-scale", "1.0",
        "--dual-learning-rate", "0.01",
        "--dual-iterations", "10",
        "--log-every", "1",
        "--device", device,
        "--stage", "test",
        "--run-group", "test-recorded-da10",
        "--landscape-device", device,
        "--landscape-seed-offset", "2000000",
        "--slice-points", "3",
        "--slice-radius", "0.1",
        "--slice-every", "1",
        "--landscape-batch-size", "8",
        "--landscape-views", "ambient", "manifold",
    ])


def _model(task: str = "f1", architecture: str = "small"):
    input_dim, small_hidden, oversized_hidden, output_dim = TASK_SPECS[task]
    hidden = small_hidden if architecture == "small" else oversized_hidden
    torch.manual_seed(123)
    model = ProductUnitNetwork(input_dim, hidden, output_dim,
                               input_domain="real_complex", init_bound=1.0)
    with torch.no_grad():
        model.exponents.copy_(retract_stiefel(model.exponents, 1.0))
    return model


def _dataset(input_dim: int, output_dim: int):
    generator = torch.Generator().manual_seed(71)
    train_x = torch.rand((9, input_dim), generator=generator) + 0.2
    train_y = torch.randn((9, output_dim), generator=generator) * 0.1
    return train_x, train_y


def test_registered_plan_expands_all_480_runs_without_requiring_gpu():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    commands = recorded.expand_config(config)
    parsed = [recorded.parse_args(command[3:]) for command in commands]

    assert len(commands) == len(parsed) == 480
    assert all(command[:3] == [recorded.sys.executable, "-m", "optimizer_resurrection.punn_manifold_recorded"]
               for command in commands)
    assert len({item.condition_id for item in parsed}) == 480
    assert {item.protocol for item in parsed} == {recorded.PROTOCOL}
    assert {item.device for item in parsed} == {"cuda"}
    assert {item.landscape_device for item in parsed} == {"cuda"}
    assert {item.seed for item in parsed} == set(range(30))
    assert {(item.task, item.architecture) for item in parsed} == {
        (task, architecture)
        for task in TASK_SPECS
        for architecture in recorded.stability.ARCHITECTURES
        if architecture != "regularized" or task in recorded.stability.CLASSIFICATION_TASKS
    }


def test_condition_identity_includes_recorded_protocol_views_and_landscape_settings():
    original = _args("f4", "oversized")
    changed = recorded.parse_args([
        "--protocol", recorded.PROTOCOL, "--task", "f4", "--architecture", "oversized",
        "--method", "manifold_muon_da10", "--seed", "2020007", "--data-seed", "7",
        "--epochs", "2", "--regularization-lambda", "0", "--learning-rate", "0.01",
        "--momentum", "0.95", "--weight-decay", "0", "--aux-learning-rate", "0.001",
        "--secondary", "false", "--manifold-scale", "1", "--dual-learning-rate", "0.01",
        "--dual-iterations", "10", "--device", "cpu", "--stage", "test",
        "--run-group", "test-recorded-da10", "--landscape-device", "cpu",
        "--landscape-seed-offset", "2000000", "--slice-points", "3", "--slice-radius", "0.2",
        "--slice-every", "1", "--landscape-batch-size", "8", "--landscape-views", "ambient", "manifold",
    ])
    recorded_values = recorded._condition_values(original, original)
    assert original.condition_id == recorded._condition_id(recorded_values)
    assert original.condition_id != changed.condition_id
    assert recorded_values["protocol"] == recorded.PROTOCOL
    assert recorded_values["training_protocol"] == recorded.TRAINING_PROTOCOL


@pytest.mark.parametrize(
    ("task", "architecture"),
    [("f1", "small"), ("f4", "oversized"), ("wine", "small")],
)
def test_ambient_directions_reuse_legacy_draws_and_manifold_directions_are_tangent(task, architecture):
    model = _model(task, architecture)
    center = parameters_to_vector(model.parameters()).detach()
    seed = 2000107
    directions = recorded._directions(model, seed, center)
    expected_ambient = orthonormal_directions(center.numel(), seed)

    assert torch.equal(directions["ambient"], expected_ambient)
    assert torch.allclose(directions["ambient"] @ directions["ambient"].T,
                          torch.eye(2), atol=2e-6)
    feasible = directions["manifold"]
    assert torch.isfinite(feasible).all()
    assert torch.allclose(feasible @ feasible.T, torch.eye(2), atol=2e-6)

    exponent_count = model.exponents.numel()
    q = model.exponents.detach().double()
    if q.shape[0] < q.shape[1]:
        q = q.T
        exponent_dirs = feasible[:, :exponent_count].reshape(2, *model.exponents.shape).double().transpose(1, 2)
        tangent_residual = q.T.unsqueeze(0) @ exponent_dirs + exponent_dirs.transpose(1, 2) @ q.unsqueeze(0)
    else:
        exponent_dirs = feasible[:, :exponent_count].reshape(2, *model.exponents.shape).double()
        tangent_residual = q.T.unsqueeze(0) @ exponent_dirs + exponent_dirs.transpose(1, 2) @ q.unsqueeze(0)
    assert torch.max(torch.abs(tangent_residual)) < 2e-6
    if task == "f1":
        assert directions["metadata"]["exponent_tangent_dimensions"]["zero_dimensional_exponent_tangent"]
        assert feasible[:, :exponent_count].abs().max() == 0
        assert feasible[:, exponent_count:].norm(dim=1).min() > 0


@pytest.mark.parametrize(
    ("task", "architecture", "view", "regularization_lambda"),
    [
        ("f1", "small", "ambient", 0.0),
        ("f1", "small", "manifold", 0.0),
        ("wine", "small", "manifold", 0.0001),
    ],
)
def test_vectorized_slice_centers_and_corners_match_scalar_model_reference(
    task, architecture, view, regularization_lambda,
):
    model = _model(task, architecture)
    train_x, train_y = _dataset(model.input_dim, model.output.out_features)
    center = parameters_to_vector(model.parameters()).detach().clone()
    directions = recorded._directions(model, 2000007, center)[view]
    result = recorded.evaluate_slice(
        model, train_x, train_y, center, directions,
        view=view, points=3, radius=0.1, batch_size=4, device="cpu",
        regularization_lambda=regularization_lambda,
    )
    vectors = recorded._candidate_vectors(
        center, directions, result["coordinates_1d"], view=view,
        hidden=model.exponents.shape[0], input_dim=model.exponents.shape[1],
        scale=1.0, device=torch.device("cpu"),
    )
    scalar = []
    scalar_regularization = []
    scalar_model = copy.deepcopy(model)
    for vector in vectors:
        vector_to_parameters(vector, scalar_model.parameters())
        with torch.no_grad():
            scalar.append((scalar_model(train_x) - train_y).square().mean())
            scalar_regularization.append(
                regularization_lambda * sum(parameter.square().sum()
                                            for parameter in scalar_model.parameters())
            )
    scalar_grid = torch.stack(scalar).reshape(3, 3)
    scalar_regularization_grid = torch.stack(scalar_regularization).reshape(3, 3)

    assert torch.allclose(result["mse"], scalar_grid, rtol=2e-5, atol=2e-6)
    assert torch.allclose(result["regularization"], scalar_regularization_grid, rtol=2e-5, atol=2e-7)
    assert torch.allclose(result["objective"], scalar_grid + scalar_regularization_grid,
                          rtol=2e-5, atol=2e-6)
    assert result["mse"][1, 1].item() == pytest.approx(float(scalar_grid[1, 1]), rel=1e-6, abs=1e-7)
    assert result["realized_displacement"][1, 1].item() == pytest.approx(0.0, abs=1e-7)
    if view == "manifold":
        assert torch.isfinite(result["stiefel_residual"]).all()
        assert result["stiefel_residual"].max().item() < 2e-6
    else:
        assert torch.isnan(result["stiefel_residual"]).all()


def test_nonfinite_center_is_retained_with_explicit_masks_without_retraction():
    model = _model()
    train_x, train_y = _dataset(model.input_dim, model.output.out_features)
    center = parameters_to_vector(model.parameters()).detach().clone()
    center[0] = float("nan")
    directions = orthonormal_directions(center.numel(), 2000007)
    result = recorded.evaluate_slice(
        model, train_x, train_y, center, directions,
        view="manifold", points=3, radius=0.1, batch_size=4, device="cpu",
        regularization_lambda=0.0,
    )

    assert result["status"] == "invalid_center_nonfinite"
    assert torch.isnan(result["mse"]).all()
    assert result["mse_nonfinite"].all()
    assert not result["finite_parameter_mask"].any()
    assert torch.isnan(result["realized_displacement"]).all()
    assert torch.isnan(result["stiefel_residual"]).all()


def test_off_manifold_center_disables_only_the_feasible_view_and_keeps_json_safe_metadata():
    model = _model()
    with torch.no_grad():
        model.exponents.fill_(2.0)
    center = parameters_to_vector(model.parameters()).detach()
    directions = recorded._directions(model, 2000007, center)

    assert not torch.isfinite(directions["manifold"]).all()
    assert directions["metadata"]["exponent_tangent_dimensions"]["center_feasible"] is False
    assert directions["metadata"]["exponent_tangent_dimensions"]["center_stiefel_residual"] == pytest.approx(3.0)
    import json

    json.dumps(directions["metadata"], allow_nan=False)
    train_x, train_y = _dataset(model.input_dim, model.output.out_features)
    result = recorded.evaluate_slice(
        model, train_x, train_y, center, directions["manifold"],
        view="manifold", points=3, radius=0.1, batch_size=4, device="cpu",
        regularization_lambda=0.0,
    )
    assert result["status"] == "invalid_center_off_manifold"
    assert result["surface_evaluated"] is False
    assert not result["candidate_evaluated_mask"].any()



def test_cpu_continuation_keeps_all_cells_and_device_distinct_ids():
    from optimizer_resurrection.punn_manifold_resume import _validate_continuation_recipes
    source = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cpu = yaml.safe_load((CONFIG.parent / "manifold_landscape_cpu_continuation.yaml").read_text(encoding="utf-8"))
    gpu_args = [recorded.parse_args(command[3:]) for command in recorded.expand_config(source)]
    cpu_args = [recorded.parse_args(command[3:]) for command in recorded.expand_config(cpu)]
    assert len(cpu_args) == 480
    assert {arg.device for arg in cpu_args} == {"cpu"}
    assert {arg.landscape_device for arg in cpu_args} == {"cpu"}
    assert {arg.condition_id for arg in gpu_args}.isdisjoint(arg.condition_id for arg in cpu_args)
    _validate_continuation_recipes(
        {arg.condition_id: vars(arg) for arg in gpu_args},
        {arg.condition_id: vars(arg) for arg in cpu_args},
    )
