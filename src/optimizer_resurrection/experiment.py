from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .tracking import RunRecord, fetch_wandb_runs, wandb_analysis_run


def _condition_family(condition: str) -> str:
    if condition.startswith("adam"):
        return "adam"
    if condition.startswith("sgd"):
        return "sgd"
    if condition.startswith("muon"):
        return "muon"
    return condition


def _base_command(values: dict[str, Any]) -> list[str]:
    # Portable plan files must run on both Windows and Oscar; local --run
    # replaces this executable with the interpreter running the orchestrator.
    command = ["python", "-m", "optimizer_resurrection.train"]
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        else:
            rendered = value.as_posix() if isinstance(value, Path) else str(value)
            command.extend([flag, rendered])
    return command


def _tag_commands(
    commands: list[list[str]], *, run_group: str, stage: str
) -> list[list[str]]:
    for command in commands:
        command.extend(["--run-group", run_group, "--stage", stage])
    return commands


def expand_config(
    config: dict[str, Any], recipes: dict[str, float] | None = None
) -> list[list[str]]:
    """Expand committed study configs into auditable one-run commands."""
    commands: list[list[str]] = []
    if config.get("regime") == "reproduction":
        gate_a = config["gate_a"]
        for depth, seed in itertools.product(gate_a["depths"], gate_a["seeds"]):
            commands.append(_base_command({
                "track": "mlp", "dataset": gate_a["dataset"],
                "activation": gate_a["activation"], "optimizer": gate_a["optimizer"],
                "depth": depth, "width": gate_a.get("width", 1000),
                "seed": seed, "data_seed": seed,
                "steps": gate_a["steps"], "batch_size": gate_a["batch_size"],
                "learning_rate": gate_a.get("learning_rate", 1e-3),
                "weight_decay": gate_a.get("weight_decay", 0.0),
                "device": gate_a.get("device", config.get("device")),
                "run_name": f"gate-a__depth{depth}__{gate_a['optimizer']}__seed{seed}",
                "recipe_version": "reproduction-gate-a",
            }))
        gate_b = config["gate_b"]
        for task, length, seed in itertools.product(gate_b["tasks"], gate_b["max_lengths"], gate_b["seeds"]):
            commands.append(_base_command({
                "track": "rnn", "task": task, "optimizer": gate_b["optimizer"],
                "hidden_size": gate_b["hidden_sizes"][task], "max_length": length,
                "seed": seed, "data_seed": seed, "steps": gate_b["steps"],
                "batch_size": gate_b["batch_size"],
                "learning_rate": gate_b.get("learning_rate", 1e-3),
                "weight_decay": gate_b.get("weight_decay", 0.0),
                "device": gate_b.get("device", config.get("device")),
                "run_name": f"gate-b__{task}__length{length}__seed{seed}",
                "recipe_version": "reproduction-gate-b",
            }))
        gate_c = config["gate_c"]
        for optimizer, seed in itertools.product(gate_c["optimizers"], gate_c["seeds"]):
            values = {
                "track": "rnn", "task": gate_c["task"], "optimizer": optimizer,
                "hidden_size": gate_c["hidden_size"], "max_length": gate_c["max_length"],
                "seed": seed, "data_seed": seed, "steps": gate_c["steps"],
                "batch_size": gate_c["batch_size"],
                "learning_rate": gate_c.get("learning_rate", 1e-3),
                "weight_decay": gate_c.get("weight_decay", 0.0),
                "device": gate_c.get("device", config.get("device")),
                "run_name": f"gate-c__{optimizer}__seed{seed}",
                "recipe_version": "reproduction-gate-c",
            }
            manifold_scale = gate_c.get("manifold_scales", {}).get(optimizer)
            if manifold_scale is not None:
                values["manifold_scale"] = manifold_scale
            commands.append(_base_command(values))
        return _tag_commands(
            commands,
            run_group=config.get("run_group", "reproduction-v1"),
            stage=config.get("stage", "reproduction"),
        )
    if "protocol_version" in config:
        if recipes is None:
            raise ValueError("full-study expansion requires calibration-selected --recipes")
        primary = config["primary"]
        mlp = config["mlp"]
        for dataset, activation, depth, optimizer, seed in itertools.product(
            mlp["datasets"], mlp["activations"], mlp["depths"], config["optimizers"], config["seeds"]["image"]
        ):
            commands.append(_base_command({
                "track": "mlp", "dataset": dataset, "activation": activation,
                "depth": depth, "width": mlp["width"], "optimizer": optimizer,
                "seed": seed, "data_seed": seed, "steps": mlp["steps"],
                "batch_size": mlp["batch_size"], "learning_rate": recipes[f"mlp.{optimizer}"],
                "weight_decay": primary["weight_decay"], "success_loss": mlp["success_loss"],
                "success_accuracy": mlp["success_accuracy"], "recipe_version": "frozen-v1",
            }))
        rnn = config["rnn"]
        for task, hidden, length, optimizer, seed in itertools.product(
            rnn["tasks"], rnn["matrix_hidden_sizes"], rnn["max_lengths"],
            config["optimizers"], config["seeds"]["synthetic_rnn"],
        ):
            commands.append(_base_command({
                "track": "rnn", "task": task, "hidden_size": hidden,
                "max_length": length, "optimizer": optimizer, "seed": seed,
                "data_seed": seed, "steps": rnn["steps"], "batch_size": rnn["batch_size"],
                "learning_rate": recipes[f"rnn.{optimizer}"], "weight_decay": primary["weight_decay"],
                "success_loss": rnn["success_loss"], "success_accuracy": rnn["success_accuracy"],
                "recipe_version": "frozen-v1",
            }))
        return _tag_commands(
            commands,
            run_group=config.get("run_group", "full-study-v1"),
            stage="full-study",
        )
    if config.get("regime") == "oracle":
        trials = int(config["trials_per_optimizer"])
        optimizers = config["optimizers"]
        for optimizer in optimizers:
            family = _condition_family(optimizer)
            for trial, lr in enumerate(np.geomspace(*config["learning_rate_ranges"][family], trials)):
                for dataset, activation, depth in itertools.product(config["mlp"]["datasets"], config["mlp"]["activations"], config["mlp"]["depths"]):
                    commands.append(_base_command({
                        "track": "mlp", "dataset": dataset, "activation": activation,
                        "depth": depth, "width": config["mlp"]["width"], "optimizer": optimizer,
                        "learning_rate": float(lr), "seed": 0, "data_seed": 0,
                        "steps": config["mlp"]["steps"], "batch_size": config["mlp"]["batch_size"],
                        "success_loss": config["mlp"]["success_loss"],
                        "success_accuracy": config["mlp"]["success_accuracy"],
                        "recipe_version": f"oracle-trial-{trial}",
                    }))
                for task, hidden, length in itertools.product(config["rnn"]["tasks"], config["rnn"]["hidden_sizes"], config["rnn"]["max_lengths"]):
                    commands.append(_base_command({
                        "track": "rnn", "task": task, "hidden_size": hidden,
                        "max_length": length, "optimizer": optimizer,
                        "learning_rate": float(lr), "seed": 0, "data_seed": 0,
                        "steps": config["rnn"]["steps"], "batch_size": config["rnn"]["batch_size"],
                        "success_loss": config["rnn"]["success_loss"],
                        "success_accuracy": config["rnn"]["success_accuracy"],
                        "recipe_version": f"oracle-trial-{trial}",
                    }))
        return _tag_commands(
            commands,
            run_group=config.get("run_group", "oracle-v1"),
            stage="oracle",
        )

    if config.get("regime") == "calibration":
        optimizers = config.get(
            "optimizers",
            ["sgd_x", "sgd_o", "adam_x", "adam_o", "muon_x", "muon_o", "rsgd", "mm"],
        )
        trials = int(config["trials_per_optimizer"])
        for track in ("mlp", "rnn"):
            section = config[track]
            axes = section["depths"] if track == "mlp" else section["max_lengths"]
            for optimizer in optimizers:
                low, high = section["learning_rate_ranges"][_condition_family(optimizer)]
                for trial, lr in enumerate(np.geomspace(low, high, trials)):
                    for axis in axes:
                        values = {
                            "track": track,
                            "optimizer": optimizer,
                            "learning_rate": float(lr),
                            "weight_decay": config["primary_weight_decay"],
                            "aux_learning_rate": config.get("aux_learning_rate", 1e-3),
                            "seed": config.get("seed", 0),
                            "data_seed": config.get("data_seed", config.get("seed", 0)),
                            "recipe_version": f"calibration-trial-{trial}",
                            "steps": section["steps"], "batch_size": section["batch_size"],
                            "success_loss": section["success_loss"],
                            "success_accuracy": section["success_accuracy"],
                            "device": section.get("device", config.get("device")),
                            "run_name": (
                                f"{track}-lr-scout__{optimizer}__axis{axis}__"
                                f"lr-{float(lr):.3g}__seed{config.get('seed', 0)}"
                            ),
                        }
                        if track == "mlp":
                            values.update(
                                dataset=section.get("dataset", "mnist"),
                                activation=section.get("activation", "logistic"),
                                depth=axis,
                                width=section.get("width", 1000),
                            )
                        else:
                            values.update(
                                task=section.get("task", "latch"),
                                hidden_size=section.get("hidden_size", 64),
                                max_length=axis,
                            )
                        commands.append(_base_command(values))
        return _tag_commands(
            commands,
            run_group=config.get("run_group", "calibration-v1"),
            stage=config.get("stage", "calibration"),
        )

    if config.get("stage") == "historical_reproduction":
        for task, length, seed in itertools.product(config["tasks"], config["max_lengths"], config["seeds"]):
            commands.append(_base_command({
                "track": "rnn", "task": task,
                "hidden_size": config["tasks"][task]["hidden_size"],
                "max_length": length, "optimizer": config["optimizer"],
                "seed": seed, "data_seed": seed, "weight_decay": config["weight_decay"],
                "recipe_version": "historical-b0",
            }))
        return _tag_commands(
            commands,
            run_group=config.get("run_group", "historical-b0"),
            stage="reproduction",
        )

    track = config.get("track")
    recipe_version = config.get("recipe_version", "mvp-pilot")
    if track == "mlp":
        for activation, depth, optimizer, seed in itertools.product(
            config["activations"], config["depths"], config["optimizers"], config["seeds"]
        ):
            learning_rate = (
                recipes[f"mlp.{optimizer}"]
                if recipes is not None
                else config["learning_rate"]
            )
            commands.append(_base_command({
                "track": "mlp", "dataset": config["dataset"], "activation": activation,
                "depth": depth, "width": config["width"], "optimizer": optimizer,
                "seed": seed, "data_seed": seed, "steps": config["steps"],
                "batch_size": config["batch_size"], "learning_rate": learning_rate,
                "aux_learning_rate": config["aux_learning_rate"], "weight_decay": config["weight_decay"],
                "success_loss": config["success_loss"], "success_accuracy": config["success_accuracy"],
                "device": config.get("device"),
                "run_name": (
                    f"mlp__{config['dataset']}__{activation}__depth{depth}__"
                    f"{optimizer}__seed{seed}"
                ),
                "recipe_version": recipe_version,
            }))
    elif track == "rnn":
        for hidden, length, optimizer, seed in itertools.product(
            config["hidden_sizes"], config["max_lengths"], config["optimizers"], config["seeds"]
        ):
            learning_rate = (
                recipes[f"rnn.{optimizer}"]
                if recipes is not None
                else config["learning_rate"]
            )
            commands.append(_base_command({
                "track": "rnn", "task": config["task"], "hidden_size": hidden,
                "max_length": length, "optimizer": optimizer, "seed": seed, "data_seed": seed,
                "steps": config["steps"], "batch_size": config["batch_size"],
                "learning_rate": learning_rate, "aux_learning_rate": config["aux_learning_rate"],
                "weight_decay": config["weight_decay"], "success_loss": config["success_loss"],
                "success_accuracy": config["success_accuracy"], "recipe_version": recipe_version,
                "device": config.get("device"),
                "run_name": (
                    f"rnn__{config['task']}__length{length}__hidden{hidden}__"
                    f"{optimizer}__seed{seed}"
                ),
            }))
    else:
        raise ValueError("config must describe calibration, historical reproduction, or one track")
    if track == "mlp":
        default_group = f"mlp-{config['dataset']}-mvp-v1"
    else:
        default_group = f"rnn-{config['task']}-mvp-v1"
    return _tag_commands(
        commands,
        run_group=config.get("run_group", default_group),
        stage=config.get("stage", "mvp"),
    )


def _unique_training_runs(records: Iterable[RunRecord]) -> list[RunRecord]:
    """Reject duplicate completed attempts before computing study statistics."""
    unique: dict[str, RunRecord] = {}
    for record in records:
        if str(record.state).lower() != "finished":
            continue
        condition_id = record.config.get("condition_id")
        if not condition_id:
            continue
        if condition_id in unique:
            raise ValueError(
                "duplicate finished W&B runs for condition_id "
                f"{condition_id}: {unique[condition_id].run_id}, {record.run_id}"
            )
        unique[condition_id] = record
    return list(unique.values())


def select_recipes(
    records: Iterable[RunRecord], destination: Path, calibration_config: dict[str, Any]
) -> dict[str, float]:
    """Select one LR per track/optimizer from complete W&B calibration runs."""
    grouped: dict[tuple[str, str, float], list[float]] = {}
    for record in _unique_training_runs(records):
        config = record.config
        result = record.summary
        if not str(config.get("recipe_version", "")).startswith("calibration-trial-"):
            continue
        key = (config["track"], config["optimizer"], float(config["learning_rate"]))
        score = 1000.0 * float(result["success"]) + result["validation_accuracy"] - result["validation_loss"]
        grouped.setdefault(key, []).append(score)
    selected: dict[str, float] = {}
    expected_trials = int(calibration_config["trials_per_optimizer"])
    for track, optimizer in sorted({(key[0], key[1]) for key in grouped}):
        candidates = [(np.mean(scores), lr) for (t, o, lr), scores in grouped.items() if (t, o) == (track, optimizer)]
        if len(candidates) != expected_trials:
            raise ValueError(f"incomplete calibration for {track}/{optimizer}: {len(candidates)} of {expected_trials} trials")
        expected_axes = len(calibration_config[track]["depths" if track == "mlp" else "max_lengths"])
        if any(len(grouped[(track, optimizer, lr)]) != expected_axes for _, lr in candidates):
            raise ValueError(f"incomplete calibration axes for {track}/{optimizer}")
        selected[f"{track}.{optimizer}"] = float(max(candidates)[1])
    expected_recipes = len(calibration_config.get("optimizers", [
        "sgd_x", "sgd_o", "adam_x", "adam_o", "muon_x", "muon_o", "rsgd", "mm"
    ])) * 2
    if len(selected) != expected_recipes:
        raise ValueError(f"calibration covers {len(selected)} of {expected_recipes} track/optimizer recipes")
    destination.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def check_gates(records: Iterable[RunRecord], gate_config: dict[str, Any]) -> dict[str, Any]:
    rows = [(record.config, record.summary) for record in _unique_training_runs(records)]
    a_spec = gate_config["gate_a"]
    sigmoid = [
        (c["depth"], c["seed"], r["validation_accuracy"], r["final_activation_saturation_mean"])
        for c, r in rows
        if c.get("recipe_version") == "reproduction-gate-a"
        and c.get("dataset") == a_spec["dataset"] and c.get("activation") == a_spec["activation"]
        and c.get("optimizer") == a_spec["optimizer"]
    ]
    a_complete = all(
        len({row[1] for row in sigmoid if row[0] == depth}) >= a_spec["min_seeds_per_depth"]
        for depth in a_spec["depths"]
    )
    shallow_a, deep_a = min(a_spec["depths"]), max(a_spec["depths"])
    gate_a = a_complete and (
        np.mean([row[2] for row in sigmoid if row[0] == shallow_a])
        > np.mean([row[2] for row in sigmoid if row[0] == deep_a])
    ) and (
        np.mean([row[3] for row in sigmoid if row[0] == deep_a])
        >= np.mean([row[3] for row in sigmoid if row[0] == shallow_a])
    )

    b_spec = gate_config["gate_b"]
    rnn = [
        (c["task"], c["max_length"], c["seed"], r["validation_accuracy"])
        for c, r in rows
        if c.get("recipe_version") == "reproduction-gate-b"
        and c.get("optimizer") == b_spec["optimizer"]
    ]
    b_complete = all(
        len({row[2] for row in rnn if row[0] == task and row[1] == length})
        >= b_spec["min_seeds_per_length"]
        for task in b_spec["tasks"] for length in b_spec["max_lengths"]
    )
    short_b, long_b = min(b_spec["max_lengths"]), max(b_spec["max_lengths"])
    gate_b = b_complete and all(
        np.mean([row[3] for row in rnn if row[0] == task and row[1] == short_b])
        - np.mean([row[3] for row in rnn if row[0] == task and row[1] == long_b])
        >= b_spec["minimum_accuracy_drop"]
        for task in b_spec["tasks"]
    )

    c_spec = gate_config["gate_c"]
    manifold_residuals: list[float] = []
    c_cells: set[tuple[str, int]] = set()
    for config, result in rows:
        if config.get("recipe_version") != "reproduction-gate-c" or config.get("optimizer") not in c_spec["optimizers"]:
            continue
        c_cells.add((config["optimizer"], int(config["seed"])))
        residual = result.get("max_stiefel_residual")
        if residual is not None:
            manifold_residuals.append(float(residual))
    expected_c_cells = {
        (optimizer, int(seed))
        for optimizer in c_spec["optimizers"] for seed in c_spec["seeds"]
    }
    c_complete = c_cells == expected_c_cells
    gate_c = c_complete and bool(manifold_residuals) and max(manifold_residuals) < c_spec["maximum_residual"]
    return {
        "gate_a": bool(gate_a), "gate_a_complete": a_complete,
        "gate_b": bool(gate_b), "gate_b_complete": b_complete,
        "gate_c": bool(gate_c), "gate_c_complete": c_complete,
        "gate_c_runs": len(c_cells),
        "maximum_stiefel_residual": max(manifold_residuals) if manifold_residuals else None,
        "headline_ready": bool(gate_a and gate_b and gate_c),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("config", type=Path, nargs="+")
    plan.add_argument("--recipes", type=Path, help="Calibration-selected recipes for full_study.yaml")
    plan.add_argument("--run", action="store_true")
    plan.add_argument("--max-runs", type=int)
    plan.add_argument("--plan-file", type=Path, help="Write one JSON command array per line for Oscar")
    select = sub.add_parser("select-recipes")
    select.add_argument("--output", type=Path, default=Path("configs/frozen_recipes.json"))
    select.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    select.add_argument("--run-group", help="Restrict completed W&B runs to one group")
    gates = sub.add_parser("check-gates")
    gates.add_argument("--config", type=Path, default=Path("configs/gates.yaml"))
    gates.add_argument("--run-group", help="Restrict completed W&B runs to one group")
    args = parser.parse_args(argv)
    if args.command == "plan":
        recipes = json.loads(args.recipes.read_text()) if args.recipes else None
        commands = []
        for config_path in args.config:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            commands.extend(expand_config(config, recipes))
        selected = commands[: args.max_runs] if args.max_runs is not None else commands
        if args.plan_file:
            args.plan_file.parent.mkdir(parents=True, exist_ok=True)
            args.plan_file.write_text("".join(json.dumps(command) + "\n" for command in selected), encoding="utf-8")
        for command in selected:
            print(json.dumps(command))
            if args.run:
                subprocess.run([sys.executable, *command[1:]], check=True)
        print(json.dumps({"planned_runs": len(commands), "selected_runs": len(selected)}))
    elif args.command == "select-recipes":
        calibration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        filters = {"group": args.run_group} if args.run_group else None
        records = fetch_wandb_runs(filters)
        selected = select_recipes(records, args.output, calibration)
        analysis_group = args.run_group or "calibration"
        with wandb_analysis_run(
            name=f"recipe-selection-{analysis_group}",
            group=analysis_group,
            job_type="recipe-selection",
            config={"calibration_config": calibration, "source_run_ids": [record.run_id for record in records]},
        ) as run:
            run.summary.update({"selected_recipes": selected})
        print(json.dumps(selected, sort_keys=True))
    else:
        gate_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        filters = {"group": args.run_group} if args.run_group else None
        records = fetch_wandb_runs(filters)
        report = check_gates(records, gate_config)
        analysis_group = args.run_group or "reproduction"
        with wandb_analysis_run(
            name=f"gate-check-{analysis_group}",
            group=analysis_group,
            job_type="gate-check",
            config={"gate_config": gate_config, "source_run_ids": [record.run_id for record in records]},
        ) as run:
            run.summary.update(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["headline_ready"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
