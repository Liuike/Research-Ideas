"""Validate and summarize online W&B records for the frozen PURe adaptation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml

from .pure_cifar import PARAMETERS, expand_config, parse_args
from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run


def summarize_records(config, records):
    expected = {a.condition_id: a for command in expand_config(config)
                for a in [parse_args(command[3:])]}
    observed, revisions, digests, data = {}, set(), set(), set()
    for record in records:
        if record.config.get("protocol") != config["protocol"]:
            continue
        condition = record.config.get("condition_id")
        if condition not in expected or condition in observed:
            raise ValueError("unexpected or duplicate PURe condition")
        args = expected[condition]
        for key, value in vars(args).items():
            if key != "dry_run" and record.config.get(key) != value:
                raise ValueError(f"resolved config mismatch: {record.run_id} {key}")
        s = record.summary
        outcome = s.get("terminal_outcome")
        if outcome == "completed":
            if record.state != "finished" or s.get("numerical_failure") is not False:
                raise ValueError("invalid completed run")
            if s.get("epochs_completed") != args.epochs or s.get("test_examples_evaluated") != args.test_examples:
                raise ValueError("incomplete epoch/test budget")
            for key in ("test_accuracy", "test_loss", "training_seconds"):
                if not math.isfinite(float(s.get(key, float("nan")))):
                    raise ValueError(f"nonfinite/missing {key}")
            if not 0 <= s["test_accuracy"] <= 1:
                raise ValueError("invalid test accuracy")
        elif outcome == "numerical_failure":
            if s.get("numerical_failure") is not True or not s.get("failure_reason") or not s.get("failed_epoch"):
                raise ValueError("unrecorded numerical failure")
            completed, failed = s.get("epochs_completed"), s.get("failed_epoch")
            if (record.state not in {"failed", "crashed", "finished"}
                    or type(completed) is not int or type(failed) is not int
                    or not 0 <= completed <= args.epochs or not 1 <= failed <= args.epochs
                    or not (failed == completed + 1 or
                            (completed == failed == args.epochs and
                             s["failure_reason"] == "nonfinite_test_output"))):
                raise ValueError("invalid terminal numerical failure or epoch budget")
        else:
            raise ValueError(f"missing terminal outcome: {record.run_id}")
        if s.get("parameter_count") != PARAMETERS or not s.get("initialization_digest") or not s.get("dataset_digest"):
            raise ValueError("missing model/data identity")
        for key, value in (("train_examples", args.train_examples), ("test_examples", args.test_examples),
                           ("order_seed", args.data_seed + 1_000_003),
                           ("worker_seed", args.data_seed + 2_000_003),
                           ("augmentation_seed", args.data_seed + 3_000_003)):
            if s.get(key) != value:
                raise ValueError(f"data seed/budget mismatch: {key}")
        if args.stage not in {"engineering-smoke", "test"}:
            provenance = record.config.get("provenance", {})
            git, tree = provenance.get("git", {}), provenance.get("source_tree", {})
            if git.get("dirty") is not False or not git.get("revision") or not tree.get("digest"):
                raise ValueError("scientific source must be clean and identified")
            revisions.add(git["revision"])
            digests.add(tree["digest"])
        data.add(s["dataset_digest"])
        observed[condition] = record
    if set(observed) != set(expected):
        raise ValueError(f"incomplete PURe plan: {len(observed)}/{len(expected)}")
    if len(revisions) > 1 or len(digests) > 1 or len(data) != 1:
        raise ValueError("inconsistent source or datasets")
    rows = [{"seed": r.config["seed"], "run_id": r.run_id, "url": r.url,
             **{key: r.summary.get(key) for key in ("terminal_outcome", "test_accuracy", "test_loss",
                "training_seconds", "epochs_completed", "failure_reason", "failed_epoch",
                "peak_cuda_memory_bytes")}} for r in observed.values()]
    rows.sort(key=lambda row: row["seed"])
    good = [r for r in rows if r["terminal_outcome"] == "completed"]
    accuracy = [r["test_accuracy"] * 100 for r in good]
    return {"expected_cells": len(expected), "observed_cells": len(observed),
        "completed": len(good), "numerical_failures": len(rows) - len(good),
        "all_seeds_completed": len(good) == len(rows), "rows": rows,
        "mean_test_accuracy_percent": statistics.mean(accuracy) if accuracy else None,
        "sd_test_accuracy_percent": statistics.stdev(accuracy) if len(accuracy) > 1 else None,
        "mean_training_seconds": statistics.mean(r["training_seconds"] for r in good) if good else None,
        "parameter_count": PARAMETERS, "source_revision": next(iter(revisions), None),
        "source_tree_digest": next(iter(digests), None), "dataset_digest": next(iter(data)),
        "source_run_ids": sorted(r.run_id for r in observed.values()),
        "interpretation": "PURe ResNet-18 CIFAR adaptation; no published accuracy target; no tuning"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb
    runs = wandb.Api().runs(credentials["WANDB_ENTITY"] + "/" + credentials["WANDB_PROJECT"],
                            filters={"group": config["run_group"]})
    records = [RunRecord(r.id, r.state, dict(r.config), dict(r.summary), r.url) for r in runs]
    report = summarize_records(config, records)
    with wandb_analysis_run("pure-cifar-resnet18-analysis", config["run_group"], "analysis",
                            {"config": config, "source_run_ids": report["source_run_ids"]}) as run:
        run.summary.update(report)
        report["analysis_run_id"] = run.id
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
