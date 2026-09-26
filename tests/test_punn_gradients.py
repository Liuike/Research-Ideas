from pathlib import Path

import torch
import yaml
import pytest

from optimizer_resurrection.experiment import expand_config
from optimizer_resurrection.punn_gradients import parse_args, run_trial
from optimizer_resurrection.punn_landscape import run_trial as historical_trial
from optimizer_resurrection.punn_analysis import summarize_records
from optimizer_resurrection.tracking import RunRecord


def args(method, **overrides):
    values = {"task": "f1", "method": method, "seed": 2, "data_seed": 100002,
              "epochs": 2, "learning_rate": 0.001, "momentum": 0.9,
              "run_group": "unit-test", **overrides}
    command = [token for key, value in values.items()
               for token in ("--" + key.replace("_", "-"), str(value))]
    return parse_args(command)


def test_adam_and_zero_decay_adamw_are_identical_and_paired():
    adam = run_trial(args("adam"))
    adamw = run_trial(args("adamw"))
    assert adam["terminal_outcome"] == adamw["terminal_outcome"] == "completed"
    for key in ("train_mse", "test_mse", "initial_train_mse", "dataset_digest", "initialization_digest", "order_seed"):
        assert adam[key] == adamw[key]


def test_sgd_preserves_historical_initialization_order_and_update():
    current = run_trial(args("sgd", learning_rate=0.1))
    previous = historical_trial(task="f1", method="sgd", seed=2, data_seed=100002, epochs=2)
    assert current["train_mse"] == previous.train_mse
    assert current["initial_train_mse"] == previous.initial_train_mse
    assert current["pattern_evaluations"] == previous.pattern_evaluations


def test_gradient_config_expands_all_requested_methods():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/product_unit/gradient_defaults.yaml").read_text())
    commands = expand_config(config)
    assert len(commands) == 80
    resolved = [parse_args(command[3:]) for command in commands]
    assert {item.method for item in resolved} == {"sgd", "adam", "adamw", "muon_moonlight"}
    assert all(item.secondary for item in resolved if item.method == "adamw")
    assert all(item.weight_decay == 0 for item in resolved if item.method != "adamw")


def test_analysis_rejects_initialization_and_order_mismatches():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/product_unit/gradient_smoke.yaml").read_text())
    records = []
    for index, command in enumerate(expand_config(config)):
        resolved = parse_args(command[3:])
        summary = {"terminal_outcome": "completed", "numerical_failure": False,
                   "epochs_completed": 2, "train_mse": 0.1, "test_mse": 0.1,
                   "initial_train_mse": 1.0, "dataset_digest": resolved.task,
                   "initialization_digest": resolved.task, "order_seed": 1000003}
        records.append(RunRecord(str(index), "finished", vars(resolved), summary, ""))
    assert summarize_records(config, records)["observed_cells"] == 8
    records[0].summary["initialization_digest"] = "wrong"
    with pytest.raises(ValueError, match="different initializations"):
        summarize_records(config, records)
    records[0].summary["initialization_digest"] = "f1"
    records[0].summary["order_seed"] = 0
    with pytest.raises(ValueError, match="order seed mismatch"):
        summarize_records(config, records)
