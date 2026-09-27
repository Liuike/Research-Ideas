"""Compare adaptive-optimizer and plain-SGD numerical stability records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .punn_architecture_analysis import (
    _terminal_snapshot,
    summarize_adaptive_records,
    summarize_records,
)
from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run


ADAPTIVE_METHODS = ("adamw", "muon_moonlight")
COMPARISON_METHODS = ("sgd", *ADAPTIVE_METHODS)
METHOD_LABELS = {"sgd": "SGD", "adamw": "AdamW", "muon_moonlight": "Muon"}
ARCHITECTURES = ("small", "oversized", "regularized")


def _planned_cells(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    return {(condition["task"], condition["architecture"], seed)
            for condition in config["conditions"] for seed in config["seeds"]}


def build_comparison_report(
    adaptive_config: dict[str, Any],
    baseline_config: dict[str, Any],
    adaptive_records: list[RunRecord],
    baseline_records: list[RunRecord],
) -> dict[str, Any]:
    """Validate both complete plans, then compare paired numerical outcomes only."""
    if tuple(adaptive_config.get("methods", ())) != ADAPTIVE_METHODS:
        raise ValueError("comparison requires exactly the registered AdamW and Muon methods")
    for field in ("conditions", "seeds", "epochs", "data_seed_offset"):
        if adaptive_config.get(field) != baseline_config.get(field):
            raise ValueError(f"adaptive and plain-SGD plans differ in {field}")
    if baseline_config.get("protocol") != "punn-architecture-stability-v1":
        raise ValueError("baseline config is not the plain-SGD architecture-stability protocol")
    if adaptive_config.get("protocol") != "punn-adaptive-architecture-stability-v1":
        raise ValueError("adaptive config is not the registered adaptive architecture protocol")

    adaptive_report = summarize_adaptive_records(adaptive_config, adaptive_records)
    baseline_report = summarize_records(baseline_config, baseline_records)
    adaptive_cells, baseline_cells = _planned_cells(adaptive_config), _planned_cells(baseline_config)
    if adaptive_cells != baseline_cells:
        raise ValueError("adaptive and plain-SGD condition/seed cells differ")

    # Index only records from each registered protocol; analysis runs and unrelated
    # experiments returned by the W&B group query cannot satisfy plan cells.
    adaptive_by_cell: dict[tuple[str, str, str, int], tuple[RunRecord, dict[str, Any]]] = {}
    baseline_by_cell: dict[tuple[str, str, int], tuple[RunRecord, dict[str, Any]]] = {}
    for record in adaptive_records:
        if record.config.get("protocol") != adaptive_config["protocol"]:
            continue
        result = _terminal_snapshot(record.summary, record.run_id)
        cell = (record.config["task"], record.config["architecture"],
                record.config["method"], record.config["seed"])
        if cell in adaptive_by_cell:
            raise ValueError(f"duplicate adaptive comparison cell: {cell}")
        adaptive_by_cell[cell] = (record, result)
    for record in baseline_records:
        if record.config.get("protocol") != baseline_config["protocol"]:
            continue
        cell = (record.config["task"], record.config["architecture"], record.config["seed"])
        if cell in baseline_by_cell:
            raise ValueError(f"duplicate plain-SGD comparison cell: {cell}")
        baseline_by_cell[cell] = (record, record.summary)

    if len(adaptive_by_cell) != len(adaptive_cells) * len(ADAPTIVE_METHODS):
        raise ValueError("adaptive comparison has missing or unexpected cells")
    if set(baseline_by_cell) != baseline_cells:
        raise ValueError("plain-SGD comparison has missing or unexpected cells")

    dataset_digests: dict[tuple[str, int], set[str]] = defaultdict(set)
    order_seeds: dict[tuple[str, int], set[int]] = defaultdict(set)
    init_digests: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    regularized_init_digests: dict[tuple[str, int], set[str]] = defaultdict(set)
    for (task, architecture, seed), (record, summary) in baseline_by_cell.items():
        dataset_digests[(task, seed)].add(summary["dataset_digest"])
        order_seeds[(task, seed)].add(summary["order_seed"])
        init_digests[(task, architecture, seed)].add(summary["initialization_digest"])
        if architecture in {"oversized", "regularized"}:
            regularized_init_digests[(task, seed)].add(summary["initialization_digest"])
    for (task, architecture, method, seed), (record, result) in adaptive_by_cell.items():
        dataset_digests[(task, seed)].add(result["dataset_digest"])
        order_seeds[(task, seed)].add(result["order_seed"])
        init_digests[(task, architecture, seed)].add(result["initialization_digest"])
        if architecture in {"oversized", "regularized"}:
            regularized_init_digests[(task, seed)].add(result["initialization_digest"])
    axes = (dataset_digests, order_seeds, init_digests, regularized_init_digests)
    if any(len(values) != 1 for axis in axes for values in axis.values()):
        raise ValueError("adaptive/plain-SGD data, initialization, or order pairing mismatch")

    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for (task, architecture, seed), (record, summary) in baseline_by_cell.items():
        grouped[(task, architecture)].setdefault("sgd", [])
        grouped[(task, architecture)]["sgd"].append((seed, summary))
    for method in ADAPTIVE_METHODS:
        for (task, architecture, _method, seed), (record, result) in adaptive_by_cell.items():
            if _method != method:
                continue
            grouped[(task, architecture)].setdefault(method, [])
            grouped[(task, architecture)][method].append((seed, result))

    rows = []
    for (task, architecture), methods in sorted(grouped.items()):
        sgd_by_seed = dict(methods["sgd"])
        row: dict[str, Any] = {"task": task, "architecture": architecture}
        for method in COMPARISON_METHODS:
            values = sorted(methods[method], key=lambda item: item[0])
            failed = [(seed, value) for seed, value in values
                      if value.get("terminal_outcome") == "numerical_failure"]
            summary = {
                "attempted": len(values),
                "finite_completions": sum(value.get("terminal_outcome") == "completed"
                                           for _, value in values),
                "numerical_failures": len(failed),
                "failed_seeds": [seed for seed, _ in failed],
            }
            if method != "sgd":
                transitions = defaultdict(list)
                for seed, value in values:
                    sgd_failed = sgd_by_seed[seed]["terminal_outcome"] == "numerical_failure"
                    method_failed = value["terminal_outcome"] == "numerical_failure"
                    category = ("both_failed" if sgd_failed and method_failed else
                                "adaptive_only_failed" if method_failed else
                                "sgd_only_failed" if sgd_failed else "both_finite")
                    transitions[category].append(seed)
                summary["paired_with_sgd"] = {
                    "both_finite": len(transitions["both_finite"]),
                    "both_failed": len(transitions["both_failed"]),
                    "adaptive_only_failed": len(transitions["adaptive_only_failed"]),
                    "sgd_only_failed": len(transitions["sgd_only_failed"]),
                    "seeds": {key: sorted(value) for key, value in sorted(transitions.items())},
                }
            row[method] = summary
        rows.append(row)

    return {
        "protocol": "punn-adaptive-architecture-comparison-v1",
        "adaptive_expected_cells": adaptive_report["expected_cells"],
        "adaptive_observed_cells": adaptive_report["observed_cells"],
        "baseline_expected_cells": baseline_report["expected_cells"],
        "baseline_observed_cells": baseline_report["observed_cells"],
        "pairing_verified": True,
        "adaptive_source_revision": adaptive_report["source_revision"],
        "adaptive_source_tree_digest": adaptive_report["source_tree_digest"],
        "baseline_source_revision": baseline_report["source_revision"],
        "baseline_source_tree_digest": baseline_report["source_tree_digest"],
        "numerical_failure_scope": adaptive_report["numerical_failure_scope"],
        "adaptive_source_run_ids": adaptive_report["source_run_ids"],
        "baseline_source_run_ids": baseline_report["source_run_ids"],
        "rows": rows,
    }


def _plot_report(report: dict[str, Any], adaptive_config: dict[str, Any], artifact: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(condition["task"] for condition in adaptive_config["conditions"]))
    columns = min(3, len(tasks))
    row_count = (len(tasks) + columns - 1) // columns
    figure, axes = plt.subplots(row_count, columns,
                                figsize=(5.5 * columns, 3.8 * row_count),
                                layout="constrained", squeeze=False)
    by_task_arch = {(row["task"], row["architecture"]): row for row in report["rows"]}
    for axis, task in zip(axes.flat, tasks):
        architectures = [architecture for architecture in ARCHITECTURES
                         if (task, architecture) in by_task_arch]
        slots = []
        labels = []
        values = []
        for arch_index, architecture in enumerate(architectures):
            row = by_task_arch[(task, architecture)]
            for method_index, method in enumerate(COMPARISON_METHODS):
                slots.append(arch_index * len(COMPARISON_METHODS) + method_index)
                labels.append(METHOD_LABELS[method])
                values.append(row[method])
        finite = [item["finite_completions"] for item in values]
        failed = [item["numerical_failures"] for item in values]
        axis.bar(slots, finite, color="#0072b2", label="Finite through final evaluation")
        axis.bar(slots, failed, bottom=finite, color="#d55e00", label="Numerical failure")
        for slot, value in zip(slots, values):
            axis.text(slot, value["attempted"] + .3,
                      f'{value["numerical_failures"]}/{value["attempted"]}',
                      ha="center", fontsize=7)
        tick_labels = []
        for architecture in architectures:
            tick_labels.extend([f"{architecture}\n{METHOD_LABELS[method]}"
                                for method in COMPARISON_METHODS])
        axis.set_xticks(slots, tick_labels)
        axis.tick_params(axis="x", labelsize=8)
        max_attempted = max((value["attempted"] for value in values), default=0)
        axis.set_ylim(0, max_attempted + 4)
        axis.set_title(task)
        axis.set_ylabel("Attempted seeds")
    for axis in list(axes.flat)[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    figure.suptitle("PUNN numerical stability | SGD, AdamW, and Muon (CPU FP32; SGD momentum=0)\n"
                    "Frozen optimizer recipes; labels show failures/attempts",
                    fontsize=14)
    with artifact.new_file("numerical_stability_comparison.png", mode="wb") as handle:
        figure.savefig(handle, format="png", dpi=160)
    plt.close(figure)


def _fetch_group_records(api: Any, project: str, group: str, protocol: str) -> list[RunRecord]:
    runs = api.runs(project, filters={"group": group}, per_page=1000)
    return [RunRecord(str(run.id), str(run.state), dict(run.config), dict(run.summary), str(run.url))
            for run in runs if run.config.get("protocol") == protocol]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    args = parser.parse_args(argv)
    adaptive_config = yaml.safe_load(args.adaptive_config.read_text(encoding="utf-8"))
    baseline_config = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb

    api = wandb.Api(timeout=60)
    project = f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}"
    adaptive_records = _fetch_group_records(api, project, adaptive_config["run_group"],
                                            adaptive_config["protocol"])
    baseline_records = _fetch_group_records(api, project, baseline_config["run_group"],
                                            baseline_config["protocol"])
    report = build_comparison_report(adaptive_config, baseline_config,
                                     adaptive_records, baseline_records)
    with wandb_analysis_run("punn-adaptive-architecture-comparison",
                            adaptive_config["run_group"], "analysis",
                            {"adaptive_config": adaptive_config,
                             "baseline_config": baseline_config}) as run:
        artifact = wandb.Artifact("punn-adaptive-architecture-comparison-" + str(run.id),
                                  type="analysis")
        with artifact.new_file("comparison_report.json", mode="w") as handle:
            json.dump(report, handle, sort_keys=True)
        _plot_report(report, adaptive_config, artifact)
        run.log_artifact(artifact)
        artifact.wait()
        run.summary.update({key: value for key, value in report.items()
                            if key not in {"adaptive_source_run_ids", "baseline_source_run_ids", "rows"}})
        run.summary["analysis_artifact"] = artifact.qualified_name
        report["analysis_run_id"] = str(run.id)
        report["analysis_artifact"] = artifact.qualified_name
    printable = {key: value for key, value in report.items()
                 if key not in {"adaptive_source_run_ids", "baseline_source_run_ids"}}
    print(json.dumps(printable, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
