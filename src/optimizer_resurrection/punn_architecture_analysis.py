"""Strict online analysis of plain-SGD PUNN architecture stability."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run


def summarize_records(config: dict[str, Any], records: list[RunRecord]) -> dict[str, Any]:
    from .punn_architecture_stability import expand_config, parse_args

    expected = {}
    for command in expand_config(config):
        args = parse_args(command[3:])
        if args.condition_id in expected:
            raise ValueError("duplicate expected condition")
        expected[args.condition_id] = args
    observed = {}
    revisions, source_digests = set(), set()
    data, orders, initializations = defaultdict(set), defaultdict(set), defaultdict(set)
    groups = defaultdict(list)
    for record in records:
        if record.config.get("protocol") != config["protocol"]:
            continue
        key = record.config.get("condition_id")
        if key not in expected or key in observed:
            raise ValueError(f"unexpected or duplicate condition: {record.run_id}")
        args = expected[key]
        for field, value in vars(args).items():
            if field not in {"dry_run", "condition_id"} and record.config.get(field) != value:
                raise ValueError(f"resolved config mismatch: {record.run_id}: {field}")
        summary = record.summary
        outcome = summary.get("terminal_outcome")
        if outcome == "completed":
            if record.state != "finished" or summary.get("numerical_failure") is not False:
                raise ValueError(f"invalid completion: {record.run_id}")
            if summary.get("epochs_completed") != config["epochs"]:
                raise ValueError(f"incomplete budget: {record.run_id}")
            for field in ["initial_train_mse", "train_mse", "test_mse", "objective"]:
                if not math.isfinite(float(summary.get(field, float("nan")))):
                    raise ValueError(f"nonfinite/missing completion metric: {record.run_id}: {field}")
        elif outcome == "numerical_failure":
            if summary.get("numerical_failure") is not True or not summary.get("failure_reason"):
                raise ValueError(f"missing failure details: {record.run_id}")
            epoch = summary.get("failed_epoch")
            if not isinstance(epoch, int) or not 0 <= epoch <= config["epochs"]:
                raise ValueError(f"invalid failed epoch: {record.run_id}")
            if not summary.get("failure_phase") or record.state != "finished":
                raise ValueError(f"unfinished failure record: {record.run_id}")
        else:
            raise ValueError(f"missing terminal outcome: {record.run_id}")
        if config["stage"] not in {"engineering-smoke", "test"}:
            provenance = record.config.get("provenance", {})
            git = provenance.get("git", {})
            revision = git.get("revision")
            digest = provenance.get("source_tree", {}).get("digest")
            if git.get("dirty") is not False or not revision or not digest:
                raise ValueError(f"missing clean provenance: {record.run_id}")
            revisions.add(revision)
            source_digests.add(digest)
        if not summary.get("dataset_digest") or not summary.get("initialization_digest"):
            raise ValueError(f"missing identity digests: {record.run_id}")
        artifact = summary.get("dataset_artifact")
        metadata = record.config.get("data_metadata", {})
        if not artifact or record.config.get("dataset_artifact") != artifact:
            raise ValueError(f"missing or mismatched dataset artifact: {record.run_id}")
        for field, value in [("dataset_digest", summary["dataset_digest"]),
                             ("task", args.task), ("data_seed", args.data_seed)]:
            if metadata.get(field) != value:
                raise ValueError(f"dataset metadata mismatch: {record.run_id}: {field}")
        from .punn_architecture_data import RAW_SOURCES
        if args.task in RAW_SOURCES:
            source = RAW_SOURCES[args.task]
            recorded_source = metadata.get("raw_source", {})
            if recorded_source.get("sha256") != source.sha256 or recorded_source.get("url") != source.url:
                raise ValueError(f"unpinned dataset source: {record.run_id}")
        if summary.get("order_seed") != args.seed + 1_000_003:
            raise ValueError(f"invalid order seed: {record.run_id}")
        data[(args.task, args.seed)].add(summary["dataset_digest"])
        orders[(args.task, args.seed)].add(summary["order_seed"])
        if args.architecture in {"oversized", "regularized"}:
            initializations[(args.task, args.seed)].add(summary["initialization_digest"])
        groups[(args.task, args.architecture)].append(record)
        observed[key] = record
    if set(observed) != set(expected):
        raise ValueError(f"incomplete plan: {len(observed)}/{len(expected)}")
    if any(len(values) != 1 for axis in [data, orders, initializations] for values in axis.values()):
        raise ValueError("architecture pairing mismatch")
    if config["stage"] not in {"engineering-smoke", "test"} and (len(revisions) != 1 or len(source_digests) != 1):
        raise ValueError("inconsistent scientific source")
    rows = []
    for (task, architecture), runs in sorted(groups.items()):
        finite = [r for r in runs if r.summary["terminal_outcome"] == "completed"]
        failed = [r for r in runs if r.summary["terminal_outcome"] == "numerical_failure"]
        row = {"task": task, "architecture": architecture, "attempted": len(runs),
               "finite_completions": len(finite), "numerical_failures": len(failed),
               "failed_seeds": sorted(r.config["seed"] for r in failed),
               "failure_reasons": dict(Counter(r.summary["failure_reason"] for r in failed)),
               "failure_phases": dict(Counter(r.summary["failure_phase"] for r in failed)),
               "failed_epochs": {str(r.config["seed"]): r.summary["failed_epoch"] for r in failed},
               "run_ids": sorted(r.run_id for r in runs)}
        for field in ["train_mse", "test_mse"]:
            values = [float(r.summary[field]) for r in finite]
            row["mean_" + field] = statistics.mean(values) if values else None
            row["median_" + field] = statistics.median(values) if values else None
        rows.append(row)
    return {"expected_cells": len(expected), "observed_cells": len(observed),
            "source_revision": next(iter(revisions), None),
            "source_tree_digest": next(iter(source_digests), None),
            "pairing_verified": True, "rows": rows,
            "source_run_ids": sorted(r.run_id for r in observed.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb
    runs = wandb.Api(timeout=60).runs(
        f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}",
        filters={"group": config["run_group"]})
    records = [RunRecord(str(r.id), str(r.state), dict(r.config), dict(r.summary), str(r.url))
               for r in runs if r.config.get("protocol") == config["protocol"]]
    report = summarize_records(config, records)
    with wandb_analysis_run("punn-architecture-stability-analysis", config["run_group"],
                            "analysis", {"config": config, "source_run_ids": report["source_run_ids"]}) as run:
        run.summary.update(report)
        artifact = wandb.Artifact("punn-architecture-stability-" + str(run.id), type="analysis")
        with artifact.new_file("stability_report.json", mode="w") as handle:
            json.dump(report, handle, sort_keys=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tasks = list(dict.fromkeys(condition["task"] for condition in config["conditions"]))
        columns = min(3, len(tasks))
        figure, axes = plt.subplots(math.ceil(len(tasks) / columns), columns,
                                    figsize=(4.5 * columns, 3.8 * math.ceil(len(tasks) / columns)),
                                    layout="constrained", squeeze=False)
        for axis, task in zip(axes.flat, tasks):
            rows = sorted((row for row in report["rows"] if row["task"] == task),
                          key=lambda row: ["small", "oversized", "regularized"].index(row["architecture"]))
            positions = list(range(len(rows)))
            failures = [row["numerical_failures"] for row in rows]
            finite = [row["finite_completions"] for row in rows]
            axis.bar(positions, finite, color="#0072b2", label="Finite through final evaluation")
            axis.bar(positions, failures, bottom=finite, color="#d55e00", label="Numerical failure")
            for index, row in enumerate(rows):
                axis.text(index, row["attempted"] + .3,
                          str(row["numerical_failures"]) + "/" + str(row["attempted"]) + " failed",
                          ha="center", fontsize=9)
            axis.set_xticks(positions, [row["architecture"] for row in rows])
            axis.set_ylim(0, max(row["attempted"] for row in rows) + 4)
            axis.set_title(task)
            axis.set_ylabel("Attempted seeds")
        for axis in list(axes.flat)[len(tasks):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        figure.suptitle(f"PUNN numerical stability | plain SGD | {config['epochs']} epochs", fontsize=15)
        with artifact.new_file("numerical_stability.png", mode="wb") as handle:
            figure.savefig(handle, format="png", dpi=160)
        plt.close(figure)
        run.log_artifact(artifact)
        artifact.wait()
        run.summary["analysis_artifact"] = artifact.qualified_name
        report["analysis_run_id"] = str(run.id)
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
