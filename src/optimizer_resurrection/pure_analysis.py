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
        if getattr(args, "landscape", False):
            from .pure_landscape import probe_epochs
            schedule = list(probe_epochs(args))
            completed = s.get("epochs_completed", 0)
            expected_probes = [epoch for epoch in schedule if epoch <= completed]
            if (s.get("landscape_expected_probe_epochs") != schedule or
                    s.get("landscape_probe_epochs") != expected_probes):
                raise ValueError("incomplete landscape probe schedule")
            per_epoch = math.ceil(args.train_examples / args.batch_size)
            batches = s.get("landscape_gradient_batches")
            if (type(batches) is not int or batches < completed * per_epoch or
                    (outcome == "completed" and batches != completed * per_epoch) or
                    batches > min(args.epochs, completed + 1) * per_epoch):
                raise ValueError("incomplete landscape gradient budget")
            if (not s.get("landscape_probe_digest") or
                    s.get("landscape_probe_examples") != min(32, args.train_examples)):
                raise ValueError("missing fixed training probe identity")
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


def validate_landscape_history(config, summary, history):
    """Reject missing per-epoch diagnostics while retaining explicitly invalid probes."""
    from .pure_landscape import probe_epochs
    args = parse_args(expand_config(config)[0][3:])
    completed = summary["epochs_completed"]
    epochs = [row for row in history if row.get("epoch", 0) > 0 and "landscape/failure/batch" not in row]
    if [row["epoch"] for row in epochs] != list(range(1, completed + 1)):
        raise ValueError("missing or duplicated landscape training epochs")
    for row in epochs:
        count = math.ceil(args.train_examples / args.batch_size)
        if (row.get("landscape/train/minibatch_count") != count or
                row.get("landscape/train/gradient_valid_batches") != count or
                row.get("landscape/train/nonfinite_gradient_batches") != 0):
            raise ValueError("incomplete per-epoch gradient measurements")
        for metric in ("data_l2", "data_rms", "data_max_abs", "data_zero_fraction",
                       "effective_l2", "actual_update_l2", "data_grad_to_param_ratio"):
            for aggregate in ("mean", "max"):
                value = row.get(f"landscape/train/global/{metric}_{aggregate}")
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"missing/nonfinite gradient metric: {metric}_{aggregate}")
        for stage in range(1, 5):
            for block in range(2):
                prefix = f"landscape/activation/layer{stage}.{block}.conv2/"
                if prefix + "finite" not in row or prefix + "input_floor_fraction" not in row:
                    raise ValueError("missing product-layer activation diagnostics")
    probes = [row for row in history if "landscape/probe/epoch" in row]
    expected = [epoch for epoch in probe_epochs(args) if epoch <= completed]
    if [row["landscape/probe/epoch"] for row in probes] != expected:
        raise ValueError("missing or duplicated landscape history probes")
    allowed = {"valid", "invalid_numerical", "unsupported"}
    if any(row.get("landscape/probe/status") not in allowed or
           row.get("landscape/probe/hessian_status") not in allowed for row in probes):
        raise ValueError("missing diagnostic terminal status")
    for row in probes:
        if row["landscape/probe/status"] == "valid":
            for key in ("ce_loss", "objective", "accuracy", "logit_max_abs"):
                value = row.get("landscape/probe/" + key)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("missing valid probe measurements")
        elif not row.get("landscape/probe/reason"):
            raise ValueError("missing invalid probe reason")
        if row["landscape/probe/hessian_status"] == "valid":
            for key in ("hessian_signed_rayleigh", "hessian_trace_estimate", "hessian_residual_norm"):
                value = row.get("landscape/probe/" + key)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("missing valid curvature measurements")
        elif not row.get("landscape/probe/hessian_reason"):
            raise ValueError("missing invalid curvature reason")
        if row.get("landscape/probe/slice_status") == "valid":
            from .pure_landscape import SLICE_ALPHAS
            for direction in (1, 2):
                for alpha in SLICE_ALPHAS:
                    for kind in ("ce", "objective"):
                        value = row.get(f"landscape/probe/slice{direction}/{kind}_alpha_{alpha:+.2f}")
                        if value is None or not math.isfinite(float(value)):
                            raise ValueError("missing valid loss slice measurements")
            for alpha in SLICE_ALPHAS:
                for beta in SLICE_ALPHAS:
                    for kind in ("ce", "objective"):
                        value = row.get(f"landscape/probe/slice2d/{kind}_a_{alpha:+.2f}_b_{beta:+.2f}")
                        if value is None or not math.isfinite(float(value)):
                            raise ValueError("missing valid two-dimensional loss slice")
        elif row.get("landscape/probe/slice_status") not in allowed:
            raise ValueError("missing loss-slice status")
        from .pure_landscape import SLICE_ALPHAS
        point_keys = [f"landscape/probe/slice{direction}/valid_alpha_{alpha:+.2f}"
                      for direction in (1, 2) for alpha in SLICE_ALPHAS]
        point_keys += [f"landscape/probe/slice2d/valid_a_{alpha:+.2f}_b_{beta:+.2f}"
                       for alpha in SLICE_ALPHAS for beta in SLICE_ALPHAS]
        if any(type(row.get(key)) is not bool for key in point_keys):
            raise ValueError("missing loss-slice point validity masks")
        if row.get("landscape/probe/slice_status") == "valid" and not all(row[key] for key in point_keys):
            raise ValueError("inconsistent loss-slice validity")
    return {"training_epochs": len(epochs), "probe_epochs": expected,
            "invalid_probe_epochs": [row["landscape/probe/epoch"] for row in probes
                                     if row["landscape/probe/status"] != "valid"],
            "invalid_hessian_epochs": [row["landscape/probe/epoch"] for row in probes
                                       if row["landscape/probe/hessian_status"] != "valid"],
            "invalid_slice_epochs": [row["landscape/probe/epoch"] for row in probes
                                     if row["landscape/probe/slice_status"] != "valid"]}


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
    runs = list(runs)
    records = [RunRecord(r.id, r.state, dict(r.config), dict(r.summary), r.url) for r in runs]
    report = summarize_records(config, records)
    if config.get("landscape"):
        report["landscape_history"] = {
            r.id: validate_landscape_history(config, dict(r.summary), list(r.scan_history(page_size=200)))
            for r in runs if r.config.get("protocol") == config["protocol"]}
    with wandb_analysis_run("pure-cifar-resnet18-analysis", config["run_group"], "analysis",
                            {"config": config, "source_run_ids": report["source_run_ids"]}) as run:
        run.summary.update(report)
        report["analysis_run_id"] = run.id
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
