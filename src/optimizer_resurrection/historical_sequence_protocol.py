"""Strict expansion of reconstructed Two-Sequence and Parity records."""
from __future__ import annotations
import itertools
import argparse
import json
import math
from pathlib import Path
import yaml
from .tracking import RunRecord, fetch_wandb_runs, wandb_analysis_run
from typing import Any
from .historical_protocol import FIELDS
from .historical_sequences import parse_args, resolved_config


def expand_config(config: dict[str, Any]) -> list[list[str]]:
    missing, extra = FIELDS - config.keys(), config.keys() - FIELDS
    if missing or extra:
        raise ValueError(f"historical config missing={sorted(missing)} unknown={sorted(extra)}")
    axes = ("max_lengths", "noise_amplitudes", "seeds", "learning_rates")
    for key in axes:
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty unique list")
    commands = []
    identities = set()
    for length, noise, seed, lr in itertools.product(*(config[key] for key in axes)):
        values = {key: value for key, value in config.items() if key not in axes}
        values.update(max_length=length, noise_amplitude=noise, seed=seed,
                      data_seed=seed, learning_rate=lr,
                      run_name=f"{config['protocol']}__T{length}__noise{noise:g}__lr{lr:g}__seed{seed}")
        command = ["python", "-m", "optimizer_resurrection.historical_sequences"]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        identity = resolved_config(parse_args(command[3:]))["condition_id"]
        if identity in identities:
            raise ValueError("duplicate resolved condition")
        identities.add(identity)
        command.extend(["--condition-id", identity])
        commands.append(command)
    return commands


def select_recipe(records: list[RunRecord], config: dict[str, Any]) -> dict[str, Any]:
    """Require exactly the registered completed cells; never select on long tasks."""
    from .historical_sequences import parse_args, resolved_config, condition_id
    if config["max_lengths"] != [10] or config["seeds"] != [1000]:
        raise ValueError("Sequence calibration is restricted to T=10 and held-out seed 1000")
    expected = {}
    for command in expand_config(config):
        resolved = resolved_config(parse_args(command[3:]))
        expected[condition_id(resolved)] = resolved
    rows = {}
    for record in records:
        if record.config.get("protocol") != config["protocol"]:
            continue  # W&B analysis runs do not masquerade as training cells.
        cid = record.config.get("condition_id")
        if cid not in expected or cid in rows:
            raise ValueError(f"unexpected or duplicate historical condition: {cid}")
        if record.state != "finished" or record.summary.get("terminal_outcome") != "completed":
            raise ValueError(f"incomplete historical run: {record.run_id}")
        actual = {key: record.config.get(key) for key in expected[cid]}
        if actual != expected[cid]:
            raise ValueError(f"resolved config mismatch: {record.run_id}")
        if record.summary.get("presentations") != config["presentations"]:
            raise ValueError(f"budget mismatch: {record.run_id}")
        for metric in ("train_loss", "validation_loss", "validation_accuracy", "train_accuracy"):
            if not math.isfinite(float(record.summary.get(metric, float("nan")))):
                raise ValueError(f"nonfinite or missing {metric}: {record.run_id}")
        rows[cid] = record
    if rows.keys() != expected.keys():
        raise ValueError(f"incomplete scout: {len(rows)}/{len(expected)} conditions")
    scores = {}
    for lr in config["learning_rates"]:
        matching = [rows[cid] for cid, c in expected.items() if c["learning_rate"] == lr]
        scores[lr] = sum(float(r.summary["validation_loss"]) for r in matching) / len(matching)
    selected = min(scores, key=lambda lr: (scores[lr], lr))
    selected_rows = [rows[cid] for cid, c in expected.items() if c["learning_rate"] == selected]
    learned = all(float(r.summary["train_loss"]) <= 0.10 and
                  float(r.summary["validation_loss"]) <= 0.10 and
                  float(r.summary["train_accuracy"]) >= 0.95 and
                  float(r.summary["validation_accuracy"]) >= 0.95 for r in selected_rows)
    return {"learning_rate": selected, "easy_controls_pass": learned,
            "mean_validation_losses": {str(k): v for k, v in scores.items()},
            "source_run_ids": sorted(r.run_id for r in rows.values()),
            "expected_conditions": sorted(expected)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = select_recipe(fetch_wandb_runs({"group": config["run_group"]}), config)
    with wandb_analysis_run("historical-sequence-recipe-selection", config["run_group"],
                            "recipe-selection", {"scout_config": config,
                            "source_run_ids": report["source_run_ids"]}) as run:
        run.summary.update(report)
        report["selection_run_id"] = run.id
    print(json.dumps(report, sort_keys=True))
    if not report["easy_controls_pass"]:
        raise SystemExit("selected recipe failed easy controls; do not submit evaluation")


if __name__ == "__main__":
    main()
