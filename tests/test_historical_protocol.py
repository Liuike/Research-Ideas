from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from optimizer_resurrection.experiment import expand_config
from optimizer_resurrection.historical_latch import parse_args, resolved_config
from optimizer_resurrection.historical_protocol import select_latch_recipe
from optimizer_resurrection.tracking import RunRecord


def scout():
    return yaml.safe_load(Path("configs/historical/latch_scout.yaml").read_text())


def records(config):
    result = []
    for i, cmd in enumerate(expand_config(config)):
        c = resolved_config(parse_args(cmd[3:]))
        result.append(RunRecord(str(i), "finished", c, {
            "terminal_outcome": "completed", "presentations": c["presentations"],
            "validation_loss": abs(c["learning_rate"] - 0.01),
            "train_loss": abs(c["learning_rate"] - 0.01),
            "validation_accuracy": 1.0, "train_accuracy": 1.0,
        }, f"https://example.invalid/{i}"))
    return result


def test_plan_has_all_six_scout_cells_and_explicit_fields():
    commands = expand_config(scout())
    assert len(commands) == 6
    assert len({resolved_config(parse_args(c[3:]))["condition_id"] for c in commands}) == 6
    assert all(c[2] == "optimizer_resurrection.historical_latch" for c in commands)
    assert all("--presentations" in c and "--train-size" in c for c in commands)


def test_planner_rejects_unknown_missing_duplicate_fields():
    c = scout()
    c["momentum"] = 0.9
    with pytest.raises(ValueError, match="unknown"):
        expand_config(c)
    c = scout()
    del c["presentations"]
    with pytest.raises(ValueError, match="missing"):
        expand_config(c)
    c = scout()
    c["seeds"] = [1000, 1000]
    with pytest.raises(ValueError, match="unique"):
        expand_config(c)


def test_selection_requires_full_exact_budgeted_cells():
    c = scout()
    rows = records(c)
    report = select_latch_recipe(rows, c)
    assert report["learning_rate"] == 0.01
    assert report["easy_controls_pass"]
    with pytest.raises(ValueError, match="incomplete"):
        select_latch_recipe(rows[:-1], c)
    with pytest.raises(ValueError, match="duplicate"):
        select_latch_recipe(rows + [rows[0]], c)
    bad = deepcopy(rows)
    bad[0].summary["presentations"] -= 1
    with pytest.raises(ValueError, match="budget"):
        select_latch_recipe(bad, c)
    bad = deepcopy(rows)
    bad[0].config["max_length"] = 100
    with pytest.raises(ValueError, match="mismatch"):
        select_latch_recipe(bad, c)


def test_selection_does_not_release_failed_easy_controls():
    c = scout()
    rows = records(c)
    for row in rows:
        row.summary["validation_accuracy"] = 0.5
    assert not select_latch_recipe(rows, c)["easy_controls_pass"]


def test_only_held_out_short_task_can_select_recipe():
    c = scout()
    c["max_lengths"] = [100]
    with pytest.raises(ValueError, match="restricted"):
        select_latch_recipe([], c)


def test_sign_accuracy_alone_does_not_pass_easy_control():
    c = scout()
    rows = records(c)
    for row in rows:
        row.summary["train_loss"] = 0.32
    assert not select_latch_recipe(rows, c)["easy_controls_pass"]
