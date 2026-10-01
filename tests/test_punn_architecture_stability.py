from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from optimizer_resurrection import punn_architecture_stability as stability
from optimizer_resurrection.punn_architecture_data import make_architecture_dataset
from optimizer_resurrection.punn_landscape import Dataset


ROOT = Path(__file__).resolve().parents[1]


def _args(task: str = "xor", architecture: str = "oversized", seed: int = 4,
          regularization_lambda: float = 0.0, epochs: int = 2):
    return stability.parse_args([
        "--task", task,
        "--architecture", architecture,
        "--seed", str(seed),
        "--data-seed", str(seed + 100000),
        "--epochs", str(epochs),
        "--regularization-lambda", str(regularization_lambda),
        "--run-group", "test-architecture-stability",
        "--stage", "test",
    ])


def _adaptive_args(method: str, task: str = "xor", architecture: str = "oversized",
                   seed: int = 4, regularization_lambda: float = 0.0, epochs: int = 2):
    recipe = stability.ADAPTIVE_RECIPES[method]
    values = [
        "--protocol", stability.ADAPTIVE_PROTOCOL,
        "--task", task,
        "--architecture", architecture,
        "--method", method,
        "--seed", str(seed),
        "--data-seed", str(seed + 100000),
        "--epochs", str(epochs),
        "--regularization-lambda", str(regularization_lambda),
        "--run-group", "test-adaptive-architecture-stability",
        "--stage", "test",
    ]
    for key, value in recipe.items():
        values.extend(["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)])
    return stability.parse_args(values)


def test_architecture_stability_plan_has_480_unique_frozen_conditions():
    config = yaml.safe_load((ROOT / "configs/product_unit/architecture_stability.yaml").read_text())
    commands = stability.expand_config(config)
    assert len(commands) == 16 * 30 == 480
    parsed = [stability.parse_args(command[3:]) for command in commands]
    assert len({args.condition_id for args in parsed}) == 480
    assert all(args.method == "sgd" and args.learning_rate == 0.1 and args.momentum == 0.0
               for args in parsed)
    assert all(args.data_seed == args.seed + 100000 for args in parsed)
    regularized = [args for args in parsed if args.architecture == "regularized"]
    assert len(regularized) == 120
    assert all(args.task in stability.CLASSIFICATION_TASKS and args.regularization_lambda == 0.0001
               for args in regularized)
    assert all(args.regularization_lambda == 0.0
               for args in parsed if args.architecture != "regularized")


def test_adaptive_architecture_stability_plan_has_960_paired_conditions():
    config = yaml.safe_load((ROOT / "configs/product_unit/adaptive_architecture_stability.yaml").read_text())
    commands = stability.expand_config(config)
    assert len(commands) == 16 * 30 * 2 == 960
    parsed = [stability.parse_args(command[3:]) for command in commands]
    assert len({args.condition_id for args in parsed}) == 960
    assert {args.method for args in parsed} == set(stability.ADAPTIVE_METHODS)
    for method in stability.ADAPTIVE_METHODS:
        method_runs = [args for args in parsed if args.method == method]
        assert len(method_runs) == 480
        assert all(args.data_seed == args.seed + 100000 for args in method_runs)
        assert all(
            {key: getattr(args, key) for key in stability.ADAPTIVE_RECIPE_FIELDS}
            == stability.ADAPTIVE_RECIPES[method]
            for args in method_runs
        )
        regularized = [args for args in method_runs if args.architecture == "regularized"]
        assert len(regularized) == 120
        assert all(args.task in stability.CLASSIFICATION_TASKS and args.regularization_lambda == 0.0001
                   for args in regularized)
        assert all(args.regularization_lambda == 0.0
                   for args in method_runs if args.architecture != "regularized")


def test_config_rejects_duplicate_conditions_and_regularized_regression():
    config = yaml.safe_load((ROOT / "configs/product_unit/architecture_stability.yaml").read_text())
    duplicate = {**config, "conditions": [*config["conditions"], config["conditions"][0]]}
    with pytest.raises(ValueError, match="duplicate"):
        stability.expand_config(duplicate)
    invalid = {**config, "conditions": [{"task": "f1", "architecture": "regularized"}]}
    with pytest.raises(ValueError, match="not registered"):
        stability.expand_config(invalid)


def test_adaptive_config_rejects_recipe_drift_and_regularized_regression():
    config = yaml.safe_load((ROOT / "configs/product_unit/adaptive_architecture_stability.yaml").read_text())
    altered_recipe = {**config, "optimizer_settings": {
        **config["optimizer_settings"],
        "adamw": {**config["optimizer_settings"]["adamw"], "learning_rate": 0.002},
    }}
    with pytest.raises(ValueError, match="frozen gradient-defaults"):
        stability.expand_config(altered_recipe)
    invalid = {**config, "conditions": [{"task": "f1", "architecture": "regularized"}]}
    with pytest.raises(ValueError, match="not registered"):
        stability.expand_config(invalid)


def test_adaptive_optimizer_builder_preserves_adamw_and_muon_parameter_assignment():
    from optimizer_resurrection.optim.moonlight_muon import MoonlightMuon
    from optimizer_resurrection.punn_gradients import build_optimizer

    model = stability.ProductUnitNetwork(2, 2, 1)
    adamw_args = _adaptive_args("adamw")
    adamw = build_optimizer(model, adamw_args)
    assert isinstance(adamw.primary, torch.optim.AdamW)
    assert adamw.auxiliary is None
    assert {id(parameter) for parameter in adamw.primary.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    assert adamw.primary.param_groups[0]["lr"] == 0.001
    assert adamw.primary.param_groups[0]["weight_decay"] == 0.01
    assert adamw.primary.param_groups[0]["betas"] == (0.9, 0.999)

    muon_args = _adaptive_args("muon_moonlight")
    muon = build_optimizer(model, muon_args)
    assert isinstance(muon.primary, MoonlightMuon)
    assert isinstance(muon.auxiliary, torch.optim.AdamW)
    assert muon.primary.param_groups[0]["lr"] == 0.02
    assert muon.primary.param_groups[0]["momentum"] == 0.95
    assert muon.primary.param_groups[0]["weight_decay"] == 0.0
    assert muon.auxiliary.param_groups[0]["lr"] == 0.001
    assert muon.auxiliary.param_groups[0]["betas"] == (0.9, 0.95)
    assert {id(parameter) for parameter in muon.primary.param_groups[0]["params"]} == {id(model.exponents)}
    assert {id(parameter) for parameter in muon.auxiliary.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.output.parameters()
    }


@pytest.mark.parametrize("method", ["adamw", "muon_moonlight"])
def test_adaptive_trial_uses_recipe_and_completes_with_same_data_initialization_and_order(method):
    dataset_pair = make_architecture_dataset("xor", 100004)
    args = _adaptive_args(method)
    result = stability.run_trial(args, dataset_pair=dataset_pair)
    assert result["method"] == method
    assert result["protocol"] == stability.ADAPTIVE_PROTOCOL
    assert result["terminal_outcome"] == "completed"
    assert result["epochs_completed"] == 2
    assert result["dataset_digest"] == stability._dataset_digest(dataset_pair[0])
    assert result["initialization_digest"]
    assert result["order_seed"] == 1_000_007
    assert result["learning_rate"] == stability.ADAPTIVE_RECIPES[method]["learning_rate"]
    assert result["weight_decay"] == stability.ADAPTIVE_RECIPES[method]["weight_decay"]
    assert result["secondary"] == stability.ADAPTIVE_RECIPES[method]["secondary"]
    assert result["optimizer_assignment"]


def test_model_state_observer_covers_each_epoch_without_changing_training():
    dataset_pair = make_architecture_dataset("xor", 100004)
    args = _adaptive_args("adamw")
    reference = stability.run_trial(args, dataset_pair=dataset_pair)
    states = []

    def observe(model, metadata):
        states.append((dict(metadata), [p.detach().clone() for p in model.parameters()]))

    recorded = stability.run_trial(args, dataset_pair=dataset_pair, on_model_state=observe)
    assert [(metadata["phase"], metadata["epoch"]) for metadata, _ in states] == [
        ("initialization", 0), ("epoch", 1), ("epoch", 2), ("terminal", 2),
    ]
    assert states[-1][0]["terminal_outcome"] == "completed"
    for key in reference.keys() - {"wall_seconds"}:
        assert recorded[key] == reference[key]
    assert all(torch.equal(before, after) for before, after in zip(states[-2][1], states[-1][1]))


def test_adaptive_methods_and_architecture_variants_share_paired_inputs_and_initialization():
    dataset_pair = make_architecture_dataset("xor", 100004)
    results = {}
    for method in stability.ADAPTIVE_METHODS:
        for architecture in ("oversized", "regularized"):
            reg = 0.0001 if architecture == "regularized" else 0.0
            key = (method, architecture)
            results[key] = stability.run_trial(
                _adaptive_args(method, architecture=architecture, regularization_lambda=reg),
                dataset_pair=dataset_pair,
            )
    paired = [result for result in results.values()]
    assert len({result["dataset_digest"] for result in paired}) == 1
    assert len({result["initialization_digest"] for result in paired}) == 1
    assert len({result["order_seed"] for result in paired}) == 1


@pytest.mark.parametrize(
    ("method", "optimizer_kind", "expected_state_prefix"),
    [
        ("adamw", "primary_adamw", "optimizer_state.primary."),
        ("muon_moonlight", "primary_muon", "optimizer_state.primary."),
        ("muon_moonlight", "auxiliary_adamw", "optimizer_state.auxiliary."),
    ],
)
def test_adaptive_trial_detects_nonfinite_state_in_each_optimizer(
    monkeypatch, method, optimizer_kind, expected_state_prefix
):
    from optimizer_resurrection.optim.moonlight_muon import MoonlightMuon

    dataset_pair = make_architecture_dataset("xor", 100004)
    observed_gradients = []
    original_network = stability.ProductUnitNetwork
    created_models = []

    class CapturedNetwork(original_network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_models.append(self)

    monkeypatch.setattr(stability, "ProductUnitNetwork", CapturedNetwork)
    if optimizer_kind.endswith("adamw"):
        original_step = torch.optim.AdamW.step

        def step_then_corrupt_state(optimizer, *args, **kwargs):
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.fill_(2e21)
                            observed_gradients.append(parameter.grad.clone())
            outcome = original_step(optimizer, *args, **kwargs)
            return outcome

        monkeypatch.setattr(torch.optim.AdamW, "step", step_then_corrupt_state)
    else:
        original_step = MoonlightMuon.step

        def step_then_corrupt_state(optimizer, *args, **kwargs):
            outcome = original_step(optimizer, *args, **kwargs)
            state = next(iter(optimizer.state.values()))
            state["momentum_buffer"].fill_(float("inf"))
            return outcome

        monkeypatch.setattr(MoonlightMuon, "step", step_then_corrupt_state)

    result = stability.run_trial(_adaptive_args(method), dataset_pair=dataset_pair)
    assert result["terminal_outcome"] == "numerical_failure"
    assert result["failure_reason"] == "nonfinite_optimizer_state"
    assert result["failure_phase"] == "optimizer_update"
    assert result["failed_epoch"] == 1
    assert result["failure_pattern"]["tensor_names"][0].startswith(expected_state_prefix)
    assert created_models and all(bool(torch.isfinite(parameter).all())
                                  for parameter in created_models[-1].parameters())
    if optimizer_kind.endswith("adamw"):
        assert observed_gradients and all(bool(torch.isfinite(gradient).all()) for gradient in observed_gradients)
        assert all(float(gradient.abs().max()) == pytest.approx(2e21) for gradient in observed_gradients)
        assert "exp_avg_sq" in result["failure_pattern"]["tensor_names"][0]


def test_selected_penalty_is_lambda_times_all_parameter_squares_including_bias():
    model = stability.ProductUnitNetwork(1, 1, 1)
    with torch.no_grad():
        model.exponents.fill_(1.0)
        model.output.weight.fill_(2.0)
        model.output.bias.fill_(3.0)
    observed = stability._regularization_term(model, 0.0001)
    assert float(observed.detach()) == pytest.approx(0.0001 * (1.0 + 4.0 + 9.0))


def test_zero_penalty_skips_squaring_extremely_large_parameters():
    model = stability.ProductUnitNetwork(1, 1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1e30)
    assert float(stability._regularization_term(model, 0.0)) == 0.0


def test_oversized_and_regularized_share_data_order_and_bitwise_initialization():
    dataset_pair = make_architecture_dataset("xor", 100004)
    oversized = stability.run_trial(_args("xor", "oversized"), dataset_pair=dataset_pair)
    regularized = stability.run_trial(
        _args("xor", "regularized", regularization_lambda=0.0001),
        dataset_pair=dataset_pair,
    )
    assert oversized["initialization_digest"] == regularized["initialization_digest"]
    assert oversized["dataset_digest"] == regularized["dataset_digest"]
    assert oversized["order_seed"] == regularized["order_seed"] == 1_000_007
    assert oversized["epochs_completed"] == regularized["epochs_completed"] == 2
    assert oversized["terminal_outcome"] == regularized["terminal_outcome"] == "completed"
    assert regularized["regularization_term"] > 0
    assert regularized["objective"] == pytest.approx(
        regularized["train_mse"] + regularized["regularization_term"]
    )


def test_nonfinite_initial_mse_is_a_recorded_failure_at_epoch_zero():
    dataset, metadata = make_architecture_dataset("xor", 100004)
    targets = dataset.train_y.clone()
    targets[0, 0] = float("nan")
    malformed = Dataset(dataset.train_x, targets, dataset.test_x, dataset.test_y, "")
    digest = stability._dataset_digest(malformed)
    malformed = replace(malformed, digest=digest)
    result = stability.run_trial(_args(), dataset_pair=(malformed, metadata))
    assert result["numerical_failure"] is True
    assert result["terminal_outcome"] == "numerical_failure"
    assert result["failure_reason"] == "nonfinite_train_mse"
    assert result["failure_phase"] == "initialization"
    assert result["failed_epoch"] == 0
    assert result["examples_processed"] == 0


def test_nonfinite_training_prediction_records_first_sample_location(monkeypatch):
    original = stability.ProductUnitNetwork

    class NonfiniteAfterInitialEvaluation(original):
        calls = 0

        def forward(self, x):
            type(self).calls += 1
            prediction = super().forward(x)
            return prediction if type(self).calls == 1 else prediction * float("nan")

    monkeypatch.setattr(stability, "ProductUnitNetwork", NonfiniteAfterInitialEvaluation)
    dataset_pair = make_architecture_dataset("xor", 100004)
    result = stability.run_trial(_args(), dataset_pair=dataset_pair)
    assert result["failure_reason"] == "nonfinite_prediction"
    assert result["failure_phase"] == "training_forward"
    assert result["failed_epoch"] == 1
    assert result["examples_processed"] == 1
    assert result["failure_examples_processed"] == 1
    assert result["failure_sample_index"] is not None
    assert result["failure_within_epoch_position"] == 1


def test_nonfinite_gradient_and_parameter_updates_are_distinct_failure_types(monkeypatch):
    dataset_pair = make_architecture_dataset("xor", 100004)

    original_class = stability.ProductUnitNetwork
    created = []

    class CapturedNetwork(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(stability, "ProductUnitNetwork", CapturedNetwork)
    original_backward = torch.Tensor.backward

    def backward_then_corrupt_gradient(tensor, *args, **kwargs):
        outcome = original_backward(tensor, *args, **kwargs)
        created[-1].exponents.grad.view(-1)[0] = float("nan")
        return outcome

    monkeypatch.setattr(torch.Tensor, "backward", backward_then_corrupt_gradient)
    gradient_result = stability.run_trial(_args(), dataset_pair=dataset_pair)
    assert gradient_result["failure_reason"] == "nonfinite_gradient"
    assert gradient_result["failure_phase"] == "backward"

    monkeypatch.setattr(torch.Tensor, "backward", original_backward)
    original_step = torch.optim.SGD.step

    def step_then_corrupt_parameter(optimizer, *args, **kwargs):
        outcome = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].view(-1)[0] = float("nan")
        return outcome

    monkeypatch.setattr(torch.optim.SGD, "step", step_then_corrupt_parameter)
    parameter_result = stability.run_trial(_args(), dataset_pair=dataset_pair)
    assert parameter_result["failure_reason"] == "nonfinite_parameter"
    assert parameter_result["failure_phase"] == "optimizer_update"


def test_failure_helper_keeps_bounded_bad_tensor_locations():
    pattern = stability._tensor_pattern([
        ("linear.weight", torch.tensor([[1.0, float("inf")], [float("nan"), 2.0]]))
    ])
    assert pattern is not None
    assert pattern["tensor_names"] == ["linear.weight"]
    assert pattern["nonfinite_values"] == 2
    assert pattern["examples"] == [
        {"tensor": "linear.weight", "index": [0, 1]},
        {"tensor": "linear.weight", "index": [1, 0]},
    ]


def test_terminal_summary_survives_delayed_history_metric_reduction():
    class DelayedHistoryRun:
        def __init__(self):
            self.summary = {}
            self.history = []

        def log(self, values, step):
            self.history.append((dict(values), step))

        def reduce_history(self):
            # W&B's delayed summary reduction writes the latest logged value
            # for each history key after training has set the final summary.
            for key, value in self.history[-1][0].items():
                self.summary[key] = value

    run = DelayedHistoryRun()
    stability._log_training_epoch(run, {
        "epoch": 1,
        "train_mse": 31.7,
        "objective": 31.7,
        "examples_processed": 37,
        "max_gradient_norm": 1.6e6,
    })
    terminal = {
        "terminal_outcome": "numerical_failure",
        "failure_phase": "training_forward",
        "failed_epoch": 3,
        "failure_examples_processed": 85,
        "examples_processed": 85,
        "train_mse": 22.4,
        "objective": 22.4,
        "max_gradient_norm": 2.3e8,
    }
    stability._record_terminal_result(run, terminal)
    run.reduce_history()

    history, step = run.history[0]
    assert step == history["epoch"] == 1
    assert "train_mse" not in history and "examples_processed" not in history
    assert history["training/train_mse"] == 31.7
    assert history["training/examples_processed"] == 37
    assert run.summary["terminal_result"] == terminal
    for key in ("examples_processed", "train_mse", "objective", "max_gradient_norm",
                "failure_examples_processed", "failure_phase", "failed_epoch"):
        assert run.summary[key] == terminal[key]
