from pathlib import Path

import torch
import yaml

from optimizer_resurrection.experiment import expand_config
from optimizer_resurrection.models import ProductUnitNetwork
from optimizer_resurrection.punn_landscape import TASKS, make_dataset, parse_args, run_trial


ROOT = Path(__file__).resolve().parents[1]


def test_paper_optimal_architecture_parameter_counts() -> None:
    for task, expected_count in (("f1", 3), ("f4", 10)):
        inputs, hidden, _ = TASKS[task]
        model = ProductUnitNetwork(inputs, hidden)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_count


def test_paper_synthetic_functions_and_paired_split() -> None:
    for task, train_size, test_size in (("f1", 37, 13), ("f4", 225, 75)):
        data = make_dataset(task, 123)
        assert len(data.train_x) == train_size
        assert len(data.test_x) == test_size
        x = data.train_x
        expected = x[:, :1].square() if task == "f1" else x[:, 1:2].pow(7) * x[:, :1].pow(3) - 0.5 * x[:, :1].pow(6)
        assert torch.equal(data.train_y, expected)
        assert data.digest == make_dataset(task, 123).digest


def test_sgd_engineering_step_and_plan_expansion() -> None:
    result = run_trial(task="f1", method="sgd", seed=3, data_seed=100003, epochs=2)
    assert result.epochs_completed <= 2
    assert result.pattern_evaluations > 0
    assert result.dataset_digest == make_dataset("f1", 100003).digest
    config = yaml.safe_load((ROOT / "configs/product_unit/landscape_smoke.yaml").read_text())
    commands = expand_config(config)
    assert len(commands) == 6
    assert {parse_args(command[3:]).task for command in commands} == {"f1", "f4"}
    assert {parse_args(command[3:]).method for command in commands} == {"sgd", "pso", "de"}


def test_population_training_end_to_end() -> None:
    for method in ("pso", "de"):
        result = run_trial(task="f1", method=method, seed=2, data_seed=100002,
                           epochs=2, population_size=8)
        assert result.full_objective_evaluations == 24
        assert result.pattern_evaluations == 24 * 37
        assert result.train_mse is not None
        assert result.train_mse <= result.initial_train_mse
        assert result.terminal_outcome == "completed"


def test_online_sgd_numerical_failure_is_recorded_at_incomplete_epoch() -> None:
    result = run_trial(task="f1", method="sgd", seed=0, data_seed=100000, epochs=2)
    assert result.numerical_failure
    assert result.failed_epoch == 1
    assert result.epochs_completed == 0
    assert 0 < result.pattern_evaluations <= 37
    assert result.terminal_outcome == "numerical_failure"
