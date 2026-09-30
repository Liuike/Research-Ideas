from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from optimizer_resurrection import punn_architecture_stability as stability
from optimizer_resurrection.punn_architecture_data import TASK_SPECS
from optimizer_resurrection.punn_landscape import Dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/product_unit/manifold_architecture_stability.yaml"
CUDA_CONFIG_PATH = ROOT / "configs/product_unit/manifold_architecture_stability_cuda.yaml"


def _synthetic_dataset_pair(task: str, data_seed: int) -> tuple[Dataset, dict[str, int]]:
    """Small deterministic in-memory data for tests that must not access UCI/W&B."""
    input_dim, _, _, output_dim = TASK_SPECS[task]
    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    train_x = torch.rand((12, input_dim), generator=generator) + 0.25
    train_y = torch.randn((12, output_dim), generator=generator) * 0.1
    test_x = torch.rand((5, input_dim), generator=generator) + 0.25
    test_y = torch.randn((5, output_dim), generator=generator) * 0.1
    digest = hashlib.sha256(
        b"".join(tensor.contiguous().numpy().tobytes()
                 for tensor in (train_x, train_y, test_x, test_y))
    ).hexdigest()
    return Dataset(train_x, train_y, test_x, test_y, digest), {"data_seed": data_seed}


def _baseline_args(task: str, architecture: str, seed: int, epochs: int = 2):
    regularization_lambda = 0.0001 if architecture == "regularized" else 0.0
    return stability.parse_args([
        "--task", task,
        "--architecture", architecture,
        "--seed", str(seed),
        "--data-seed", str(seed + 100000),
        "--epochs", str(epochs),
        "--regularization-lambda", str(regularization_lambda),
        "--run-group", "test-manifold-initialization-baseline",
        "--stage", "test",
    ])


def _manifold_args(task: str, architecture: str, seed: int, epochs: int = 2):
    regularization_lambda = 0.0001 if architecture == "regularized" else 0.0
    args = [
        "--protocol", stability.MANIFOLD_PROTOCOL,
        "--task", task,
        "--architecture", architecture,
        "--method", stability.MANIFOLD_METHOD,
        "--seed", str(seed),
        "--data-seed", str(seed + 100000),
        "--epochs", str(epochs),
        "--regularization-lambda", str(regularization_lambda),
        "--run-group", "test-manifold-architecture-stability",
        "--stage", "test",
    ]
    for key, value in stability.MANIFOLD_RECIPE.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        args.extend(["--" + key.replace("_", "-"), rendered])
    return stability.parse_args(args)


def test_manifold_plan_has_480_unique_da10_conditions_across_six_tasks():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    commands = stability.expand_config(config)

    assert len(commands) == 16 * 30 == 480
    parsed = [stability.parse_args(command[3:]) for command in commands]
    assert len({args.condition_id for args in parsed}) == 480
    assert {args.task for args in parsed} == set(TASK_SPECS)
    assert {args.method for args in parsed} == {stability.MANIFOLD_METHOD}
    assert {args.protocol for args in parsed} == {stability.MANIFOLD_PROTOCOL}
    assert {args.seed for args in parsed} == set(range(30))
    assert all(args.data_seed == args.seed + 100000 for args in parsed)
    assert all(
        {key: getattr(args, key) for key in stability.MANIFOLD_RECIPE}
        == stability.MANIFOLD_RECIPE
        for args in parsed
    )
    assert all(args.regularization_lambda == 0.0001
               for args in parsed if args.architecture == "regularized")
    assert all(args.regularization_lambda == 0.0
               for args in parsed if args.architecture != "regularized")


def test_cuda_manifold_plan_preserves_cells_and_has_distinct_condition_ids():
    cpu = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cuda = yaml.safe_load(CUDA_CONFIG_PATH.read_text(encoding="utf-8"))
    cpu_args = [stability.parse_args(command[3:]) for command in stability.expand_config(cpu)]
    cuda_args = [stability.parse_args(command[3:]) for command in stability.expand_config(cuda)]
    assert len(cuda_args) == 480
    assert all(args.device == "cuda" for args in cuda_args)
    assert {(args.task, args.architecture, args.seed) for args in cpu_args} == {
        (args.task, args.architecture, args.seed) for args in cuda_args
    }
    assert {args.condition_id for args in cpu_args}.isdisjoint(
        args.condition_id for args in cuda_args
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("learning_rate", 0.02),
        ("momentum", 0.9),
        ("weight_decay", 0.01),
        ("aux_learning_rate", 0.002),
        ("secondary", True),
        ("manifold_scale", 0.5),
        ("dual_learning_rate", 0.02),
        ("dual_iterations", 9),
    ],
)
def test_manifold_plan_rejects_any_change_to_frozen_da10_recipe(field, changed):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    altered = {
        **config,
        "optimizer_settings": {
            stability.MANIFOLD_METHOD: {**stability.MANIFOLD_RECIPE, field: changed},
        },
    }

    with pytest.raises(ValueError, match="fixed registered recipe"):
        stability.expand_config(altered)


@pytest.mark.parametrize(
    ("task", "architecture"),
    [("f1", "small"), ("f4", "oversized"), ("iris", "regularized")],
)
def test_manifold_cpu_trial_projects_same_seed_initialization_and_stays_finite(
    monkeypatch, task, architecture
):
    from optimizer_resurrection.optim import manifold_muon

    seed = 7
    dataset_pair = _synthetic_dataset_pair(task, seed + 100000)
    baseline = stability.run_trial(
        _baseline_args(task, architecture, seed), dataset_pair=dataset_pair
    )

    captured_models = []
    primary_parameter_ids = []
    auxiliary_parameter_ids = []
    original_network = stability.ProductUnitNetwork
    original_manifold_muon = manifold_muon.ManifoldMuon
    original_adamw = torch.optim.AdamW

    class CapturedNetwork(original_network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_models.append(self)

    class CapturedManifoldMuon(original_manifold_muon):
        def __init__(self, params, *args, **kwargs):
            params = list(params)
            primary_parameter_ids.append({id(parameter) for parameter in params})
            super().__init__(params, *args, **kwargs)

    def captured_adamw(params, *args, **kwargs):
        params = list(params)
        auxiliary_parameter_ids.append({id(parameter) for parameter in params})
        return original_adamw(params, *args, **kwargs)

    monkeypatch.setattr(stability, "ProductUnitNetwork", CapturedNetwork)
    monkeypatch.setattr(manifold_muon, "ManifoldMuon", CapturedManifoldMuon)
    monkeypatch.setattr(torch.optim, "AdamW", captured_adamw)
    result = stability.run_trial(
        _manifold_args(task, architecture, seed), dataset_pair=dataset_pair
    )

    assert result["terminal_outcome"] == "completed"
    assert result["epochs_completed"] == 2
    assert result["method"] == stability.MANIFOLD_METHOD
    assert result["preprojection_initialization_digest"] == baseline["initialization_digest"]
    assert result["initialization_digest"]
    assert result["initial_stiefel_residual"] < 1e-5
    assert result["final_stiefel_residual"] < 1e-5
    assert result["optimizer_assignment"] == (
        "DA-10 Manifold Muon on projected exponents; AdamW on output weight and bias"
    )
    assert all(torch.isfinite(torch.tensor(result[key])) for key in (
        "initial_train_mse", "train_mse", "test_mse", "objective",
        "max_gradient_norm", "max_gradient_abs", "max_parameter_abs",
    ))
    assert len(captured_models) == 1
    model = captured_models[0]
    assert primary_parameter_ids == [{id(model.exponents)}]
    assert auxiliary_parameter_ids == [
        {id(parameter) for parameter in model.output.parameters()}
    ]
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def test_finite_da10_inputs_can_fail_inside_svd_and_are_scientific_failures(monkeypatch):
    from optimizer_resurrection.optim import manifold_muon

    # The FP64 gradient norm remains finite, but DA-10's FP32 matrix products
    # overflow and the SVD raises instead of returning an update.
    weight = torch.eye(8, dtype=torch.float32)
    gradient = weight * 3e38
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(gradient.double().norm()))
    with pytest.raises(torch.linalg.LinAlgError):
        manifold_muon.manifold_muon_direction(weight, gradient)

    original_step = manifold_muon.ManifoldMuon.step

    def failed_svd(self, closure=None):
        raise torch.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(manifold_muon.ManifoldMuon, "step", failed_svd)
    result = stability.run_trial(
        _manifold_args("f4", "oversized", seed=7),
        dataset_pair=_synthetic_dataset_pair("f4", 100007),
    )
    assert result["terminal_outcome"] == "numerical_failure"
    assert result["failure_reason"] == "manifold_svd_failure"
    assert result["failure_phase"] == "optimizer_update"
    assert result["failed_epoch"] == 1
    assert result["examples_processed"] == 1
    monkeypatch.setattr(manifold_muon.ManifoldMuon, "step", original_step)
