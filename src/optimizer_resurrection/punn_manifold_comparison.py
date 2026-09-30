"""Strict four-method PUNN numerical-stability comparison with Manifold Muon."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .punn_architecture_analysis import _terminal_snapshot
from .punn_architecture_comparison import (
    ADAPTIVE_METHODS,
    ARCHITECTURES,
    COMPARISON_METHODS,
    _fetch_group_records,
    build_comparison_report,
)
from .punn_architecture_stability import _dimensions
from .punn_architecture_stability import expand_config, parse_args
from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run


MANIFOLD_PROTOCOL = "punn-manifold-architecture-stability-v1"
MANIFOLD_METHOD = "manifold_muon_da10"
MANIFOLD_LABEL = "MM-DA10"
FOUR_METHODS = (*COMPARISON_METHODS, MANIFOLD_METHOD)
METHOD_LABELS = {"sgd": "SGD", "adamw": "AdamW", "muon_moonlight": "Muon",
                 MANIFOLD_METHOD: MANIFOLD_LABEL}
MANIFOLD_RECIPE = {
    "learning_rate": 0.01,
    "momentum": 0.95,
    "weight_decay": 0.0,
    "aux_learning_rate": 0.001,
    "secondary": False,
    "manifold_scale": 1.0,
    "dual_learning_rate": 0.01,
    "dual_iterations": 10,
}
EXPECTED_TASKS = {"f1", "f4", "xor", "iris", "wine", "diabetes"}
EXPECTED_CONDITIONS = {
    (task, architecture)
    for task in ("f1", "f4")
    for architecture in ("small", "oversized")
} | {
    (task, architecture)
    for task in ("xor", "iris", "wine", "diabetes")
    for architecture in ARCHITECTURES
}


def _expected_cells(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    cells = set()
    for condition in config["conditions"]:
        task, architecture = condition["task"], condition["architecture"]
        if task not in EXPECTED_TASKS or architecture not in ARCHITECTURES:
            raise ValueError(f"unexpected Manifold task/architecture: {task}/{architecture}")
        for seed in config["seeds"]:
            cell = (task, architecture, seed)
            if cell in cells:
                raise ValueError(f"duplicate expected Manifold cell: {cell}")
            cells.add(cell)
    return cells


def _registered_manifold_cells(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    cells = _expected_cells(config)
    conditions = {(task, architecture) for task, architecture, _ in cells}
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError("Manifold Muon plan must cover the registered 16 task/architecture conditions")
    if config.get("seeds") != list(range(30)):
        raise ValueError("Manifold Muon plan must use seeds 0 through 29")
    if config.get("epochs") != 500:
        raise ValueError("Manifold Muon plan must preserve the registered 500-epoch budget")
    if (config.get("data_seed_offset") != 100000
            or config.get("regularization_lambda") != 0.0001
            or config.get("device") not in {"cpu", "cuda"}):
        raise ValueError("Manifold Muon plan changed the registered data, regularization, or device recipe")
    return cells


def _check_plan_compatibility(
    manifold_config: dict[str, Any], adaptive_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> set[tuple[str, str, int]]:
    if manifold_config.get("protocol") != MANIFOLD_PROTOCOL:
        raise ValueError("config is not the registered Manifold Muon architecture-stability protocol")
    if tuple(manifold_config.get("methods", ())) != (MANIFOLD_METHOD,):
        raise ValueError("comparison requires exactly the registered Manifold Muon DA-10 method")
    settings = manifold_config.get("optimizer_settings", {}).get(MANIFOLD_METHOD)
    if settings != MANIFOLD_RECIPE:
        raise ValueError("Manifold Muon recipe must match the frozen plain DA-10 configuration")
    for field in ("conditions", "seeds", "epochs", "data_seed_offset", "regularization_lambda"):
        if manifold_config.get(field) != baseline_config.get(field):
            raise ValueError(f"Manifold Muon and plain-SGD plans differ in {field}")
        if manifold_config.get(field) != adaptive_config.get(field):
            raise ValueError(f"Manifold Muon and adaptive plans differ in {field}")
    cells = _registered_manifold_cells(manifold_config)
    if {task for task, _, _ in cells} != EXPECTED_TASKS:
        raise ValueError("Manifold Muon plan must cover all six registered PUNN environments")
    if cells != _expected_cells(baseline_config) or cells != _expected_cells(adaptive_config):
        raise ValueError("four-method condition/seed cells differ")
    return cells


def _validate_terminal_outcome(record: RunRecord, result: dict[str, Any], epochs: int) -> None:
    outcome = result.get("terminal_outcome")
    if outcome == "completed":
        if record.state != "finished" or result.get("numerical_failure") is not False:
            raise ValueError(f"invalid Manifold completion: {record.run_id}")
        if result.get("epochs_completed") != epochs:
            raise ValueError(f"incomplete Manifold budget: {record.run_id}")
        for field in ("initial_train_mse", "train_mse", "test_mse", "objective"):
            if not math.isfinite(float(result.get(field, float("nan")))):
                raise ValueError(f"nonfinite/missing Manifold metric: {record.run_id}: {field}")
    elif outcome == "numerical_failure":
        if record.state != "finished" or result.get("numerical_failure") is not True:
            raise ValueError(f"unfinished Manifold failure: {record.run_id}")
        if not result.get("failure_reason") or not result.get("failure_phase"):
            raise ValueError(f"missing Manifold failure details: {record.run_id}")
        epoch = result.get("failed_epoch")
        if type(epoch) is not int or not 0 <= epoch <= epochs:
            raise ValueError(f"invalid Manifold failed epoch: {record.run_id}")
    else:
        raise ValueError(f"missing Manifold terminal outcome: {record.run_id}")


def _validate_manifold_records(
    config: dict[str, Any], records: list[RunRecord],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], tuple[RunRecord, dict[str, Any]]]]:
    """Validate 480 Manifold records and return report plus records by cell."""
    if config.get("protocol") != MANIFOLD_PROTOCOL:
        raise ValueError("wrong Manifold protocol")
    if tuple(config.get("methods", ())) != (MANIFOLD_METHOD,):
        raise ValueError("unexpected Manifold method set")
    settings = config.get("optimizer_settings", {}).get(MANIFOLD_METHOD)
    if settings != MANIFOLD_RECIPE:
        raise ValueError("Manifold DA-10 resolved recipe mismatch")
    expected = _registered_manifold_cells(config)
    expected_run_count = len(expected)
    if expected_run_count != 480:
        raise ValueError(f"expected 480 registered Manifold runs, config expands to {expected_run_count}")
    expected_args = {}
    for command in expand_config(config):
        args = parse_args(command[3:])
        key = (args.task, args.architecture, args.seed)
        if key in expected_args:
            raise ValueError(f"duplicate expanded Manifold cell: {key}")
        expected_args[key] = args
    if set(expected_args) != expected:
        raise ValueError("Manifold resolved plan differs from its registered task/seed cells")

    observed: dict[tuple[str, str, int], tuple[RunRecord, dict[str, Any]]] = {}
    revisions: set[str] = set()
    source_digests: set[str] = set()
    datasets: dict[tuple[str, int], set[str]] = defaultdict(set)
    orders: dict[tuple[str, int], set[int]] = defaultdict(set)
    projected_large_pair: dict[tuple[str, int], set[str]] = defaultdict(set)

    from .punn_architecture_data import RAW_SOURCES

    for record in records:
        if record.config.get("protocol") != MANIFOLD_PROTOCOL:
            continue
        task = record.config.get("task")
        architecture = record.config.get("architecture")
        seed = record.config.get("seed")
        key = (task, architecture, seed)
        if key not in expected or key in observed:
            raise ValueError(f"unexpected or duplicate Manifold cell: {record.run_id}: {key}")
        args = expected_args[key]
        if record.config.get("condition_id") != args.condition_id:
            raise ValueError(f"Manifold condition id does not match frozen recipe: {record.run_id}")
        for field, value in vars(args).items():
            if field not in {"dry_run", "condition_id"} and record.config.get(field) != value:
                raise ValueError(f"Manifold resolved config mismatch: {record.run_id}: {field}")
        config_values = {
            "method": MANIFOLD_METHOD,
            "data_seed": config["data_seed_offset"] + seed,
            "epochs": config["epochs"],
            **settings,
        }
        for field, value in config_values.items():
            if record.config.get(field) != value:
                raise ValueError(f"Manifold resolved config mismatch: {record.run_id}: {field}")
        input_dim, hidden_units, output_dim = _dimensions(task, architecture)
        regularization = (config["regularization_lambda"]
                         if architecture == "regularized" else 0.0)
        for field, value in (("input_dim", input_dim), ("hidden_units", hidden_units),
                             ("output_dim", output_dim),
                             ("regularization_lambda", regularization)):
            if record.config.get(field) != value:
                raise ValueError(f"Manifold resolved shape mismatch: {record.run_id}: {field}")

        result = _terminal_snapshot(record.summary, record.run_id)
        identity = {
            "condition_id": record.config.get("condition_id"),
            "protocol": MANIFOLD_PROTOCOL,
            "task": task,
            "architecture": architecture,
            "method": MANIFOLD_METHOD,
            "seed": seed,
            "data_seed": config["data_seed_offset"] + seed,
            "epochs": config["epochs"],
            "input_dim": input_dim,
            "hidden_units": hidden_units,
            "output_dim": output_dim,
            "regularization_lambda": regularization,
            **settings,
        }
        for field, value in identity.items():
            if result.get(field) != value:
                raise ValueError(f"Manifold terminal snapshot identity/recipe mismatch: {record.run_id}: {field}")
        if not identity["condition_id"]:
            raise ValueError(f"missing Manifold condition id: {record.run_id}")
        _validate_terminal_outcome(record, result, config["epochs"])

        provenance = record.config.get("provenance", {})
        git = provenance.get("git", {})
        revision = git.get("revision")
        digest = provenance.get("source_tree", {}).get("digest")
        if config.get("stage") not in {"engineering-smoke", "test"}:
            if git.get("dirty") is not False or not revision or not digest:
                raise ValueError(f"missing clean Manifold provenance: {record.run_id}")
            revisions.add(revision)
            source_digests.add(digest)

        dataset_digest = result.get("dataset_digest")
        preprojection_digest = result.get("preprojection_initialization_digest")
        projected_digest = result.get("initialization_digest")
        dataset_artifact = result.get("dataset_artifact")
        if not dataset_digest or not preprojection_digest or not projected_digest:
            raise ValueError(f"missing Manifold identity digest: {record.run_id}")
        if projected_digest == preprojection_digest:
            raise ValueError(f"Manifold projection did not change the initialization: {record.run_id}")
        initial_residual = result.get("initial_stiefel_residual")
        final_residual = result.get("final_stiefel_residual")
        if not isinstance(initial_residual, (int, float)) or (
                not math.isfinite(initial_residual) or initial_residual > 1e-4):
            raise ValueError(f"invalid projected Manifold initialization: {record.run_id}")
        if result.get("terminal_outcome") == "completed" and (
                not isinstance(final_residual, (int, float)) or
                not math.isfinite(final_residual) or final_residual > 1e-4):
            raise ValueError(f"completed Manifold run left the Stiefel manifold: {record.run_id}")
        if result.get("manifold_nesterov") is not True:
            raise ValueError(f"Manifold DA-10 must use Nesterov momentum: {record.run_id}")
        if (result.get("optimizer_name") != "ManifoldMuon DA-10 plus AdamW auxiliary"
                or result.get("optimizer_assignment") !=
                "DA-10 Manifold Muon on projected exponents; AdamW on output weight and bias"
                or result.get("optimizer_betas") != {"auxiliary_adamw": [0.9, 0.95]}
                or result.get("optimizer_epsilon") != 1e-8):
            raise ValueError(f"Manifold DA-10 optimizer implementation mismatch: {record.run_id}")
        metadata = record.config.get("data_metadata", {})
        if (not dataset_artifact or record.config.get("dataset_artifact") != dataset_artifact
                or metadata.get("dataset_digest") != dataset_digest
                or metadata.get("task") != task
                or metadata.get("data_seed") != config["data_seed_offset"] + seed):
            raise ValueError(f"missing/mismatched Manifold dataset metadata: {record.run_id}")
        if task in RAW_SOURCES:
            source = RAW_SOURCES[task]
            raw_source = metadata.get("raw_source", {})
            if raw_source.get("sha256") != source.sha256 or raw_source.get("url") != source.url:
                raise ValueError(f"unpinned Manifold dataset source: {record.run_id}")
        expected_order_seed = seed + 1_000_003
        if result.get("order_seed") != expected_order_seed:
            raise ValueError(f"invalid Manifold order seed: {record.run_id}")

        datasets[(task, seed)].add(dataset_digest)
        orders[(task, seed)].add(expected_order_seed)
        if architecture in {"oversized", "regularized"}:
            projected_large_pair[(task, seed)].add(projected_digest)
        observed[key] = (record, result)

    if set(observed) != expected:
        raise ValueError(f"incomplete Manifold plan: {len(observed)}/{len(expected)}")
    if any(len(values) != 1 for axis in (datasets, orders, projected_large_pair)
           for values in axis.values()):
        raise ValueError("Manifold data, order, or postprojection initialization pairing mismatch")
    if config.get("stage") not in {"engineering-smoke", "test"} and (
            len(revisions) != 1 or len(source_digests) != 1):
        raise ValueError("inconsistent Manifold scientific source revision/digest")

    rows = []
    grouped: dict[tuple[str, str], list[tuple[RunRecord, dict[str, Any]]]] = defaultdict(list)
    for (task, architecture, _seed), run_result in observed.items():
        grouped[(task, architecture)].append(run_result)
    for (task, architecture), values in sorted(grouped.items()):
        failed = [(record, result) for record, result in values
                  if result["terminal_outcome"] == "numerical_failure"]
        rows.append({
            "task": task,
            "architecture": architecture,
            "attempted": len(values),
            "finite_completions": sum(result["terminal_outcome"] == "completed"
                                       for _, result in values),
            "numerical_failures": len(failed),
            "failed_seeds": sorted(record.config["seed"] for record, _ in failed),
            "failure_reasons": _count(result["failure_reason"] for _, result in failed),
            "failure_phases": _count(result["failure_phase"] for _, result in failed),
            "failed_epochs": {str(record.config["seed"]): result["failed_epoch"]
                              for record, result in failed},
            "run_ids": sorted(record.run_id for record, _ in values),
        })
    return ({
        "expected_cells": expected_run_count,
        "observed_cells": len(observed),
        "source_revision": next(iter(revisions), None),
        "source_tree_digest": next(iter(source_digests), None),
        "pairing_verified": True,
        "rows": rows,
        "source_run_ids": sorted(record.run_id for record, _ in observed.values()),
    }, observed)


def _count(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _manifold_transition(sgd_by_seed: dict[int, dict[str, Any]],
                         manifold_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    transitions: dict[str, list[int]] = defaultdict(list)
    for seed, result in sorted(manifold_by_seed.items()):
        sgd_failed = sgd_by_seed[seed]["terminal_outcome"] == "numerical_failure"
        method_failed = result["terminal_outcome"] == "numerical_failure"
        category = ("both_failed" if sgd_failed and method_failed else
                    "manifold_only_failed" if method_failed else
                    "sgd_only_failed" if sgd_failed else "both_finite")
        transitions[category].append(seed)
    return {
        "both_finite": len(transitions["both_finite"]),
        "both_failed": len(transitions["both_failed"]),
        "manifold_only_failed": len(transitions["manifold_only_failed"]),
        "sgd_only_failed": len(transitions["sgd_only_failed"]),
        "seeds": {key: seeds for key, seeds in sorted(transitions.items())},
    }


def build_four_way_comparison_report(
    manifold_config: dict[str, Any], adaptive_config: dict[str, Any],
    baseline_config: dict[str, Any], manifold_records: list[RunRecord],
    adaptive_records: list[RunRecord], baseline_records: list[RunRecord],
) -> dict[str, Any]:
    """Validate all four complete protocols and pair Manifold Muon to the SGD init."""
    cells = _check_plan_compatibility(manifold_config, adaptive_config, baseline_config)
    existing = build_comparison_report(adaptive_config, baseline_config,
                                       adaptive_records, baseline_records)
    manifold_report, manifold_by_cell = _validate_manifold_records(manifold_config, manifold_records)

    baseline_by_cell: dict[tuple[str, str, int], tuple[RunRecord, dict[str, Any]]] = {}
    for record in baseline_records:
        if record.config.get("protocol") != baseline_config["protocol"]:
            continue
        cell = (record.config["task"], record.config["architecture"], record.config["seed"])
        baseline_by_cell[cell] = (record, record.summary)
    if set(baseline_by_cell) != cells:
        raise ValueError("plain-SGD baseline cells differ from the registered Manifold plan")

    datasets: dict[tuple[str, int], set[str]] = defaultdict(set)
    orders: dict[tuple[str, int], set[int]] = defaultdict(set)
    projected_initializations: dict[tuple[str, int], set[str]] = defaultdict(set)
    baseline_initializations: dict[tuple[str, int], set[str]] = defaultdict(set)
    for (task, architecture, seed), (record, result) in manifold_by_cell.items():
        baseline = baseline_by_cell[(task, architecture, seed)][1]
        if result["preprojection_initialization_digest"] != baseline.get("initialization_digest"):
            raise ValueError(f"Manifold preprojection initialization does not match SGD: {task}/{architecture}/{seed}")
        if result["dataset_digest"] != baseline.get("dataset_digest"):
            raise ValueError(f"Manifold dataset does not match SGD: {task}/{architecture}/{seed}")
        if result["order_seed"] != baseline.get("order_seed"):
            raise ValueError(f"Manifold sample order does not match SGD: {task}/{architecture}/{seed}")
        datasets[(task, seed)].add(result["dataset_digest"])
        orders[(task, seed)].add(result["order_seed"])
        if architecture in {"oversized", "regularized"}:
            projected_initializations[(task, seed)].add(result["initialization_digest"])
            baseline_initializations[(task, seed)].add(baseline["initialization_digest"])
    if any(len(values) != 1 for axis in (datasets, orders, projected_initializations,
                                         baseline_initializations) for values in axis.values()):
        raise ValueError("Manifold data, order, or pre/postprojection pairing mismatch")

    manifold_rows = {(row["task"], row["architecture"]): row
                      for row in manifold_report["rows"]}
    baseline_snapshots = {(task, architecture, seed): result
                          for (task, architecture, seed), (_record, result)
                          in baseline_by_cell.items()}
    manifold_snapshots = {key: result for key, (_record, result) in manifold_by_cell.items()}
    rows = []
    for row in existing["rows"]:
        task, architecture = row["task"], row["architecture"]
        added = manifold_rows[(task, architecture)]
        manifold_summary = {field: added[field] for field in
                            ("attempted", "finite_completions", "numerical_failures",
                             "failed_seeds", "failure_reasons", "failure_phases", "failed_epochs")}
        sgd_by_seed = {seed: result for (t, a, seed), result in baseline_snapshots.items()
                       if t == task and a == architecture}
        manifold_by_seed = {seed: result for (t, a, seed), result in manifold_snapshots.items()
                            if t == task and a == architecture}
        manifold_summary["paired_with_sgd"] = _manifold_transition(sgd_by_seed, manifold_by_seed)
        row[MANIFOLD_METHOD] = manifold_summary
        rows.append(row)

    return {
        "protocol": "punn-manifold-architecture-comparison-v1",
        "pairing_verified": True,
        "device_by_method": {
            "sgd": baseline_config["device"],
            "adamw": adaptive_config["device"],
            "muon_moonlight": adaptive_config["device"],
            MANIFOLD_METHOD: manifold_config["device"],
        },
        "device_comparison_caveat": (
            "Manifold Muon ran on CUDA while SGD, AdamW, and Moonlight Muon ran on CPU; "
            "the displayed stability rates are descriptive and device is a confound."
            if manifold_config["device"] != baseline_config["device"] else None
        ),
        "baseline_expected_cells": existing["baseline_expected_cells"],
        "baseline_observed_cells": existing["baseline_observed_cells"],
        "adaptive_expected_cells": existing["adaptive_expected_cells"],
        "adaptive_observed_cells": existing["adaptive_observed_cells"],
        "manifold_expected_cells": manifold_report["expected_cells"],
        "manifold_observed_cells": manifold_report["observed_cells"],
        "baseline_source_revision": existing["baseline_source_revision"],
        "baseline_source_tree_digest": existing["baseline_source_tree_digest"],
        "adaptive_source_revision": existing["adaptive_source_revision"],
        "adaptive_source_tree_digest": existing["adaptive_source_tree_digest"],
        "manifold_source_revision": manifold_report["source_revision"],
        "manifold_source_tree_digest": manifold_report["source_tree_digest"],
        "manifold_recipe": MANIFOLD_RECIPE,
        "initialization_pairing": (
            "preprojection_initialization_digest is paired to plain-SGD initialization_digest; "
            "initialization_digest records post-Stiefel-projection weights and is expected to differ"),
        "cell_annotations": {
            "f1/small": (
                "Below-capacity architecture retained for numerical-stability coverage; "
                "its fit is capacity-limited and should not be interpreted as optimizer quality."),
        },
        "numerical_failure_scope": existing["numerical_failure_scope"],
        "manifold_additional_failure_scope": (
            "SVD failure from finite DA-10 inputs is classified as a scientific "
            "numerical failure at optimizer_update"),
        "methods": list(FOUR_METHODS),
        "baseline_source_run_ids": existing["baseline_source_run_ids"],
        "adaptive_source_run_ids": existing["adaptive_source_run_ids"],
        "manifold_source_run_ids": manifold_report["source_run_ids"],
        "rows": rows,
    }


def _plot_four_way_report(report: dict[str, Any], config: dict[str, Any], artifact: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(condition["task"] for condition in config["conditions"]))
    columns = min(3, len(tasks))
    row_count = (len(tasks) + columns - 1) // columns
    figure, axes = plt.subplots(row_count, columns,
                                figsize=(6.0 * columns, 4.7 * row_count),
                                layout="constrained", squeeze=False)
    by_task_arch = {(row["task"], row["architecture"]): row for row in report["rows"]}
    architecture_labels = {"small": "Small", "oversized": "Oversized",
                           "regularized": "L2-regularized"}
    for axis, task in zip(axes.flat, tasks):
        architectures = [architecture for architecture in ARCHITECTURES
                         if (task, architecture) in by_task_arch]
        slots, values = [], []
        for arch_index, architecture in enumerate(architectures):
            row = by_task_arch[(task, architecture)]
            for method_index, method in enumerate(FOUR_METHODS):
                slots.append(arch_index * len(FOUR_METHODS) + method_index)
                values.append(row[method])
        finite = [item["finite_completions"] for item in values]
        failed = [item["numerical_failures"] for item in values]
        attempted = [item["attempted"] for item in values]
        axis.bar(slots, finite, color="#0072b2", label="Finite through final evaluation")
        axis.bar(slots, failed, bottom=finite, color="#d55e00", label="Numerical failure")
        for slot, value in zip(slots, values):
            axis.text(slot, value["attempted"] + .3,
                      f'{value["numerical_failures"]}/{value["attempted"]}',
                      ha="center", fontsize=7)
        axis.set_xticks(slots, [METHOD_LABELS[method]
                                for _architecture in architectures
                                for method in FOUR_METHODS])
        axis.tick_params(axis="x", labelsize=7, pad=2)
        for arch_index, architecture in enumerate(architectures):
            first = arch_index * len(FOUR_METHODS)
            last = first + len(FOUR_METHODS) - 1
            center = (first + last) / 2
            axis.plot([first - .4, last + .4], [-.22, -.22],
                      transform=axis.get_xaxis_transform(), color="#777777",
                      linewidth=.7, clip_on=False)
            label = architecture_labels[architecture]
            if task == "f1" and architecture == "small":
                label += "†"
            axis.text(center, -.25, label,
                      transform=axis.get_xaxis_transform(), ha="center", va="top",
                      fontsize=8, clip_on=False)
        if task == "f1":
            axis.text(.02, .96, "† below-capacity cell; included for stability coverage",
                      transform=axis.transAxes, ha="left", va="top", fontsize=7)
        axis.set_ylim(0, max(attempted, default=0) + 4)
        axis.set_title(task)
        axis.set_ylabel("Attempted seeds")
    for axis in list(axes.flat)[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    devices = report["device_by_method"]
    device_note = (f"DA-10 {devices[MANIFOLD_METHOD].upper()}; other methods "
                   f"{devices['sgd'].upper()} FP32"
                   if devices[MANIFOLD_METHOD] != devices["sgd"]
                   else f"{devices['sgd'].upper()} FP32")
    figure.suptitle(
        "PUNN numerical stability | SGD, AdamW, Moonlight Muon, Manifold Muon DA-10\n"
        f"{device_note}; frozen recipes; labels show failures/attempts",
        fontsize=13,
    )
    with artifact.new_file("numerical_stability_comparison.png", mode="wb") as handle:
        figure.savefig(handle, format="png", dpi=160)
    plt.close(figure)


def _partition_manifold_records(
    records: list[RunRecord],
) -> tuple[list[RunRecord], list[dict[str, Any]]]:
    """Keep completed scientific runs and describe interrupted infrastructure jobs."""
    completed = []
    excluded = []
    for record in records:
        if record.state == "finished":
            completed.append(record)
        else:
            if record.state not in {"crashed", "failed", "killed"}:
                raise ValueError(f"Manifold W&B run is still active or unknown: {record.run_id}: {record.state}")
            nested = record.summary.get("terminal_result", {})
            if (record.summary.get("terminal_outcome") in {"completed", "numerical_failure"}
                    or record.summary.get("terminal_result.terminal_outcome")
                    in {"completed", "numerical_failure"}
                    or (isinstance(nested, dict)
                        and nested.get("terminal_outcome") in {"completed", "numerical_failure"})):
                raise ValueError(f"scientific terminal result is not finished: {record.run_id}")
            excluded.append({
                "run_id": record.run_id,
                "state": record.state,
                "condition_id": record.config.get("condition_id"),
                "task": record.config.get("task"),
                "architecture": record.config.get("architecture"),
                "seed": record.config.get("seed"),
            })
    return completed, excluded


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifold-config", type=Path, required=True)
    parser.add_argument("--adaptive-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    args = parser.parse_args(argv)
    manifold_config = yaml.safe_load(args.manifold_config.read_text(encoding="utf-8"))
    adaptive_config = yaml.safe_load(args.adaptive_config.read_text(encoding="utf-8"))
    baseline_config = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb

    api = wandb.Api(timeout=60)
    project = f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}"
    all_manifold_records = _fetch_group_records(
        api, project, manifold_config["run_group"], manifold_config["protocol"])
    # W&B retains interrupted infrastructure jobs. They are not scientific
    # outcomes; preserve their identities while validating exactly one
    # completed scientific record per registered condition.
    manifold_records, excluded_infrastructure = _partition_manifold_records(
        all_manifold_records)
    adaptive_records = _fetch_group_records(api, project, adaptive_config["run_group"],
                                             adaptive_config["protocol"])
    baseline_records = _fetch_group_records(api, project, baseline_config["run_group"],
                                             baseline_config["protocol"])
    report = build_four_way_comparison_report(
        manifold_config, adaptive_config, baseline_config,
        manifold_records, adaptive_records, baseline_records)
    report["excluded_infrastructure_runs"] = excluded_infrastructure
    report["excluded_infrastructure_count"] = len(excluded_infrastructure)
    with wandb_analysis_run("punn-manifold-architecture-comparison",
                            manifold_config["run_group"], "analysis",
                            {"manifold_config": manifold_config,
                             "adaptive_config": adaptive_config,
                             "baseline_config": baseline_config}) as run:
        artifact = wandb.Artifact("punn-manifold-architecture-comparison-" + str(run.id),
                                  type="analysis")
        with artifact.new_file("comparison_report.json", mode="w") as handle:
            json.dump(report, handle, sort_keys=True)
        _plot_four_way_report(report, manifold_config, artifact)
        run.log_artifact(artifact)
        artifact.wait()
        run.summary.update({key: value for key, value in report.items()
                            if key not in {"baseline_source_run_ids", "adaptive_source_run_ids",
                                           "manifold_source_run_ids", "rows"}})
        run.summary["analysis_artifact"] = artifact.qualified_name
        report["analysis_run_id"] = str(run.id)
        report["analysis_artifact"] = artifact.qualified_name
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"baseline_source_run_ids", "adaptive_source_run_ids",
                                     "manifold_source_run_ids"}}, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
