from pathlib import Path

import yaml
import pytest

from optimizer_resurrection.experiment import (
    _unique_training_runs,
    check_gates,
    expand_config,
    select_recipes,
)
from optimizer_resurrection.tracking import RunRecord


def _record(run_id, config, summary):
    return RunRecord(
        run_id=run_id,
        state="finished",
        config={"condition_id": run_id, **config},
        summary=summary,
        url=f"https://wandb.invalid/{run_id}",
    )


def test_mvp_configs_expand_to_declared_run_counts():
    mlp = yaml.safe_load(Path("configs/mlp/mvp.yaml").read_text())
    rnn = yaml.safe_load(Path("configs/rnn/mvp.yaml").read_text())
    assert len(expand_config(mlp)) == 160
    assert len(expand_config(rnn)) == 200


def test_calibration_has_equal_trial_budget_per_optimizer():
    config = yaml.safe_load(Path("configs/calibration.yaml").read_text())
    commands = expand_config(config)
    counts = {}
    for command in commands:
        optimizer = command[command.index("--optimizer") + 1]
        counts[optimizer] = counts.get(optimizer, 0) + 1
        assert command[command.index("--run-group") + 1] == "calibration-v1"
        assert command[command.index("--stage") + 1] == "calibration"
    assert len(set(counts.values())) == 1


def test_full_study_requires_and_consumes_frozen_recipes():
    config = yaml.safe_load(Path("configs/full_study.yaml").read_text())
    optimizers = config["optimizers"]
    recipes = {f"{track}.{optimizer}": 1e-3 for track in ("mlp", "rnn") for optimizer in optimizers}
    commands = expand_config(config, recipes)
    expected_mlp = 3 * 2 * 9 * 8 * 10
    expected_rnn = 3 * 3 * 8 * 8 * 20
    assert len(commands) == expected_mlp + expected_rnn


def test_reproduction_config_expands_every_required_gate_cell():
    config = yaml.safe_load(Path("configs/gates.yaml").read_text())
    commands = expand_config(config)
    expected = 2 * 5 + 3 * 3 * 20 + 3 * 5
    assert len(commands) == expected


def test_two_l40s_wave_counts_and_scaled_mm_flag():
    root = Path("configs/exploratory")
    smoke_mlp = yaml.safe_load((root / "smoke_mlp_2xl40s.yaml").read_text())
    smoke_rnn = yaml.safe_load((root / "smoke_rnn_2xl40s.yaml").read_text())
    gates = yaml.safe_load((root / "gates_2xl40s.yaml").read_text())
    scout = yaml.safe_load((root / "lr_scout_2xl40s.yaml").read_text())
    wave_one = [
        *expand_config(smoke_mlp),
        *expand_config(smoke_rnn),
        *expand_config(gates),
        *expand_config(scout),
    ]
    assert len(wave_one) == 62
    assert all("--device" in command and command[command.index("--device") + 1] == "cuda" for command in wave_one)

    scaled = [command for command in expand_config(gates) if command[command.index("--optimizer") + 1] == "scaled_mm"]
    assert len(scaled) == 2
    assert all(command[command.index("--manifold-scale") + 1] == "2.0" for command in scaled)

    recipes = {
        f"{track}.{optimizer}": 1e-3
        for track in ("mlp", "rnn")
        for optimizer in scout["optimizers"]
    }
    boundary = [
        *expand_config(yaml.safe_load((root / "boundary_mlp_2xl40s.yaml").read_text()), recipes),
        *expand_config(yaml.safe_load((root / "boundary_rnn_2xl40s.yaml").read_text()), recipes),
    ]
    assert len(boundary) == 32
    assert all(command[command.index("--stage") + 1] == "exploratory" for command in boundary)


def test_reduced_scout_selects_dynamic_optimizer_set(tmp_path):
    config = yaml.safe_load(Path("configs/exploratory/lr_scout_2xl40s.yaml").read_text())
    records = []
    for track in ("mlp", "rnn"):
        ranges = config[track]["learning_rate_ranges"]
        for optimizer in config["optimizers"]:
            family = "adam" if optimizer.startswith("adam") else "muon" if optimizer.startswith("muon") else optimizer
            low, high = ranges[family]
            for trial, learning_rate in enumerate((low, (low * high) ** 0.5, high)):
                run_id = f"{track}-{optimizer}-{trial}"
                records.append(_record(
                    run_id,
                    {
                        "track": track,
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "recipe_version": f"calibration-trial-{trial}",
                    },
                    {"success": trial == 1, "validation_accuracy": 0.9, "validation_loss": 0.1},
                ))
    selected = select_recipes(records, tmp_path / "recipes.json", config)
    assert len(selected) == 8


def test_oscar_submitter_separates_smoke_and_caps_gpu_arrays():
    submitter = Path("scripts/oscar_submit_exploratory.sh").read_text()
    runner = Path("scripts/oscar_plan.sbatch").read_text()

    assert 'SMOKE_ARRAY="0-7%2"' in submitter
    assert 'EXPLORATORY_ARRAY="8-61%2"' in submitter
    assert 'WAVE2_ARRAY="0-31%2"' in submitter
    assert 'SMOKE_VERIFIED:-0' in submitter
    assert "#SBATCH --gres=gpu:1" in runner
    assert "#SBATCH --constraint=l40s" in runner
    assert 'done < ".secrets/env"' not in runner
    assert "#SBATCH --array=0-159%2" in Path("scripts/oscar_mlp_mvp.sbatch").read_text()
    assert "#SBATCH --array=0-199%2" in Path("scripts/oscar_rnn_mvp.sbatch").read_text()


def test_duplicate_finished_condition_ids_are_rejected():
    first = _record("same", {"track": "rnn"}, {"success": True})
    second = RunRecord(
        run_id="retry",
        state="finished",
        config={"condition_id": "same", "track": "rnn"},
        summary={"success": True},
        url="https://wandb.invalid/retry",
    )
    with pytest.raises(ValueError, match="duplicate finished W&B runs"):
        _unique_training_runs([first, second])


def test_gate_c_requires_every_registered_optimizer_seed_cell():
    gate_config = yaml.safe_load(Path("configs/gates.yaml").read_text())
    records = []
    for optimizer in gate_config["gate_c"]["optimizers"]:
        for seed in gate_config["gate_c"]["seeds"]:
            run_id = f"{optimizer}-{seed}"
            records.append(_record(
                run_id,
                {
                    "recipe_version": "reproduction-gate-c",
                    "optimizer": optimizer,
                    "seed": seed,
                },
                {"max_stiefel_residual": 1e-6},
            ))

    complete = check_gates(records, gate_config)
    incomplete = check_gates(records[:-1], gate_config)

    assert complete["gate_c"] is True
    assert complete["gate_c_complete"] is True
    assert incomplete["gate_c"] is False
    assert incomplete["gate_c_complete"] is False
