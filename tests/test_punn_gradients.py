import random
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("method", ["adamw", "muon_moonlight"])
def test_state_observer_is_copy_only_and_preserves_training_rng(method):
    recipe = {"learning_rate": .02, "momentum": .95} if method == "muon_moonlight" else {}
    trial_args = args(method, epochs=2, log_every=1, **recipe)
    expected = run_trial(trial_args)
    expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    states = []

    def observer(epoch, kind, vector, population, losses):
        states.append((epoch, kind, vector.clone(), population, losses))
        random.random()
        np.random.random()
        torch.rand(1)
        vector.fill_(123)

    actual = run_trial(trial_args, on_state=observer)
    actual_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert actual == expected
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1][0] == expected_rng[1][0]
    assert np.array_equal(actual_rng[1][1], expected_rng[1][1])
    assert actual_rng[1][2:] == expected_rng[1][2:]
    assert torch.equal(actual_rng[2], expected_rng[2])
    assert [(epoch, kind) for epoch, kind, *_ in states] == [
        (0, "initial"), (1, "epoch"), (2, "epoch"), (2, "final")
    ]
    assert all(population is None and losses is None for _, _, _, population, losses in states)
    assert not torch.equal(states[0][2], states[-1][2])
    assert torch.equal(states[-2][2], states[-1][2])


def test_state_observer_receives_terminal_state_after_numerical_failure():
    trial_args = args("sgd", learning_rate=1e8, epochs=2)
    states = []
    result = run_trial(trial_args, on_state=lambda *values: states.append(values))
    assert result["numerical_failure"]
    assert states[0][0:2] == (0, "initial")
    terminal_epoch = result["failed_epoch"] or result["epochs_completed"]
    assert states[-1][0] == terminal_epoch
    assert states[-1][1] == "numerical_failure"
    assert states[-1][2].numel() == 3


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
