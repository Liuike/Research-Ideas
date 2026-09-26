"""Strict W&B analysis of registered 2024 PUNN reconstruction cells."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .punn_landscape import parse_args
from .punn_protocol import expand_config
from .punn_recorded import RECORDED_PROTOCOLS
from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run


def summarize_records(config: dict[str, Any], records: list[RunRecord]) -> dict[str, Any]:
    expand, parse = expand_config, parse_args
    if config["protocol"] in RECORDED_PROTOCOLS:
        from .punn_recorded import expand_config as expand, parse_args as parse
    if config["protocol"] == "punn-gradient-defaults-v1":
        from .punn_gradients import expand_config as expand, parse_args as parse
    expected = {}
    for command in expand(config):
        args = parse(command[3:])
        if args.condition_id in expected:
            raise ValueError(f"duplicate expected condition: {args.condition_id}")
        expected[args.condition_id] = args
    observed: dict[str, RunRecord] = {}
    for record in records:
        if record.config.get("protocol") != config["protocol"]:
            continue
        condition_id = record.config.get("condition_id")
        if condition_id not in expected or condition_id in observed:
            raise ValueError(f"unexpected or duplicate PUNN condition: {condition_id}")
        args = expected[condition_id]
        for key, value in vars(args).items():
            if key in {"dry_run", "condition_id"}:
                continue
            if key == "landscape_cpu_reference" and key not in config and key not in record.config:
                continue
            if record.config.get(key) != value:
                raise ValueError(f"resolved config mismatch for {record.run_id}: {key}")
        outcome = record.summary.get("terminal_outcome")
        if config["protocol"] in RECORDED_PROTOCOLS:
            if record.summary.get("landscape_status") != "completed" or not record.summary.get("landscape_artifact"):
                raise ValueError(f"missing completed landscape artifact: {record.run_id}")
            if int(record.summary.get("landscape_slices", 0)) < 1:
                raise ValueError(f"missing landscape slices: {record.run_id}")
            if record.summary.get("landscape_seed") != args.data_seed + args.landscape_seed_offset:
                raise ValueError(f"landscape seed mismatch: {record.run_id}")
            if not record.summary.get("landscape_samples_digest"):
                raise ValueError(f"missing landscape samples digest: {record.run_id}")
            if args.landscape_cpu_reference == "true" and record.summary.get("landscape_cpu_reference") is not True:
                raise ValueError(f"missing CPU landscape reference: {record.run_id}")
        if outcome == "completed":
            if record.state != "finished" or record.summary.get("numerical_failure") is not False:
                raise ValueError(f"invalid completed run: {record.run_id}")
            if record.summary.get("epochs_completed") != config["epochs"]:
                raise ValueError(f"incomplete epoch budget: {record.run_id}")
            for metric in ("train_mse", "test_mse", "initial_train_mse"):
                if not math.isfinite(float(record.summary.get(metric, float("nan")))):
                    raise ValueError(f"missing or nonfinite {metric}: {record.run_id}")
        elif outcome == "numerical_failure":
            if record.summary.get("numerical_failure") is not True:
                raise ValueError(f"unflagged numerical failure: {record.run_id}")
        else:
            raise ValueError(f"missing terminal outcome: {record.run_id}")
        observed[condition_id] = record
    missing = set(expected) - set(observed)
    if missing:
        raise ValueError(f"incomplete PUNN plan: {len(observed)}/{len(expected)} cells")
    source_revisions: set[str] = set()
    source_digests: set[str] = set()
    if config["stage"] not in {"engineering-smoke", "test"}:
        for record in observed.values():
            provenance = record.config.get("provenance", {})
            git = provenance.get("git", {})
            revision = git.get("revision")
            digest = provenance.get("source_tree", {}).get("digest")
            if git.get("dirty") is not False or not revision or not digest:
                raise ValueError(f"missing clean source provenance: {record.run_id}")
            source_revisions.add(revision)
            source_digests.add(digest)
        if len(source_revisions) != 1 or len(source_digests) != 1:
            raise ValueError("PUNN cells used different source revisions or digests")
    digests: dict[tuple[str, int], set[str]] = defaultdict(set)
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for condition_id, record in observed.items():
        args = expected[condition_id]
        digests[(args.task, args.seed)].add(record.summary["dataset_digest"])
        groups[(args.task, args.method)].append(record)
    if any(len(values) != 1 for values in digests.values()):
        raise ValueError("paired methods used different datasets")
    if config["protocol"] in RECORDED_PROTOCOLS:
        landscape_digests = defaultdict(set)
        for condition_id, record in observed.items():
            args = expected[condition_id]
            landscape_digests[(args.task, args.seed)].add(record.summary["landscape_samples_digest"])
        if any(len(values) != 1 for values in landscape_digests.values()):
            raise ValueError("paired methods used different global landscape samples")
    if config["protocol"] in RECORDED_PROTOCOLS | {"punn-gradient-defaults-v1"}:
        initializations: dict[tuple[str, int], set[str]] = defaultdict(set)
        for condition_id, record in observed.items():
            args = expected[condition_id]
            if record.summary.get("order_seed") != args.seed + 1_000_003:
                raise ValueError("minibatch order seed mismatch")
            digest = record.summary.get("initialization_digest")
            if not digest:
                raise ValueError("missing initialization digest")
            initializations[(args.task, args.seed)].add(digest)
        if any(len(values) != 1 for values in initializations.values()):
            raise ValueError("paired methods used different initializations")
    rows = []
    for (task, method), group in sorted(groups.items()):
        good = [record for record in group if record.summary["terminal_outcome"] == "completed"]
        row: dict[str, Any] = {
            "task": task, "method": method,
            "expected": len(group), "completed": len(good),
            "numerical_failures": len(group) - len(good),
            "run_ids": sorted(record.run_id for record in group),
        }
        for metric in ("train_mse", "test_mse"):
            values = [float(record.summary[metric]) for record in good]
            row[f"mean_{metric}"] = statistics.mean(values) if values else None
            row[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else None
            row[f"median_{metric}"] = statistics.median(values) if values else None
        rows.append(row)
    paired_rows = []
    if {"sgd", "pso", "de"}.issubset(config["methods"]):
        by_pair = {
            (expected[condition_id].task, expected[condition_id].seed,
             expected[condition_id].method): record
            for condition_id, record in observed.items()
        }
        for task in config["tasks"]:
            counts = {"task": task, "seeds": len(config["seeds"]),
                      "sgd_numerical_failures": 0,
                      "sgd_failed_or_higher_test_mse_than_pso": 0,
                      "sgd_failed_or_higher_test_mse_than_de": 0,
                      "sgd_failed_or_higher_test_mse_than_both": 0}
            for seed in config["seeds"]:
                sgd = by_pair[(task, seed, "sgd")]
                failed = sgd.summary["terminal_outcome"] == "numerical_failure"
                counts["sgd_numerical_failures"] += int(failed)
                worse = {}
                for method in ("pso", "de"):
                    population = by_pair[(task, seed, method)]
                    if population.summary["terminal_outcome"] != "completed":
                        raise ValueError(f"population method failed in paired comparison: {population.run_id}")
                    worse[method] = failed or float(sgd.summary["test_mse"]) > float(population.summary["test_mse"])
                    counts[f"sgd_failed_or_higher_test_mse_than_{method}"] += int(worse[method])
                counts["sgd_failed_or_higher_test_mse_than_both"] += int(all(worse.values()))
            paired_rows.append(counts)
    return {"expected_cells": len(expected), "observed_cells": len(observed),
            "source_revision": next(iter(source_revisions), None),
            "source_tree_digest": next(iter(source_digests), None),
            "source_run_ids": sorted(record.run_id for record in observed.values()),
            "rows": rows, "paired_rows": paired_rows}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol"] in RECORDED_PROTOCOLS:
        from .punn_recorded import expand_config as expand_recorded
        expand_recorded(config)
    elif config["protocol"] == "punn-gradient-defaults-v1":
        from .punn_gradients import expand_config as expand_gradients
        expand_gradients(config)
    else:
        expand_config(config)
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb

    runs = wandb.Api().runs(
        f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}",
        filters={"group": config["run_group"]},
    )
    records = [RunRecord(str(run.id), str(run.state), dict(run.config),
                         dict(run.summary), str(run.url)) for run in runs
               if run.config.get("protocol") == config["protocol"]]
    report = summarize_records(config, records)
    with wandb_analysis_run("punn-landscape-basic-analysis", config["run_group"],
                            "analysis", {"config": config,
                                         "source_run_ids": report["source_run_ids"]}) as run:
        run.summary.update(report)
        report["analysis_run_id"] = str(run.id)
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
