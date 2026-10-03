"""Comparison report and figures for the recorded no-momentum DA-10 study.

Data retrieval and immutable W&B artifact auditing happen before this module is
called.  The functions here consume those audited rows and write the report
and figures only through the supplied W&B artifact handle.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL = "punn-da10-no-momentum-comparison-v1"
REFERENCE_PROTOCOL = "punn-manifold-landscape-comparison-v1"
TASKS = ("f1", "f4", "xor", "iris", "wine", "diabetes")
CELL_ARCHITECTURES = {
    "f1": ("small", "oversized"),
    "f4": ("small", "oversized"),
    "xor": ("small", "oversized", "regularized"),
    "iris": ("small", "oversized", "regularized"),
    "wine": ("small", "oversized", "regularized"),
    "diabetes": ("small", "oversized", "regularized"),
}
SEEDS = tuple(range(30))
EXPECTED_CELLS = 480
MOMENTUM_METHOD = "manifold_muon_da10"
NO_MOMENTUM_METHOD = "manifold_muon_da10_no_momentum"
BASELINE_METHODS = ("sgd", "adamw", "muon_moonlight")
METHOD_LABELS = {
    MOMENTUM_METHOD: "DA-10, momentum 0.95",
    NO_MOMENTUM_METHOD: "DA-10, no momentum",
    "sgd": "SGD",
    "adamw": "AdamW",
    "muon_moonlight": "Moonlight Muon",
}
METHOD_COLORS = {
    MOMENTUM_METHOD: "#0072b2",
    NO_MOMENTUM_METHOD: "#d55e00",
}
METRICS = ("train_mse", "test_mse")
LANDSCAPE_METRIC = "absolute_p95_mse_delta"


def _condition(row: Mapping[str, Any]) -> Mapping[str, Any]:
    condition = row.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("every run row must include a condition mapping")
    missing = {"task", "architecture", "method", "seed", "data_seed"} - set(condition)
    if missing:
        raise ValueError(f"run condition is missing fields: {sorted(missing)}")
    return condition


def _logical_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    condition = _condition(row)
    return (str(condition["task"]), str(condition["architecture"]), int(condition["seed"]))


def _index_rows(rows: Iterable[Mapping[str, Any]], method: str) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        condition = _condition(row)
        if condition["method"] != method:
            raise ValueError(f"expected method {method!r}, found {condition['method']!r}")
        key = _logical_key(row)
        if key in indexed:
            raise ValueError(f"duplicate logical condition {key} for {method}")
        if key[0] not in CELL_ARCHITECTURES or key[1] not in CELL_ARCHITECTURES[key[0]] or key[2] not in SEEDS:
            raise ValueError(f"condition is outside the registered 480-cell plan: {key}")
        if int(condition["data_seed"]) != 100000 + key[2]:
            raise ValueError(f"unexpected data seed for paired condition {key}")
        outcome = row.get("terminal_outcome")
        if outcome not in {"completed", "numerical_failure"}:
            raise ValueError(f"unaccepted scientific outcome {outcome!r} for {key}")
        indexed[key] = row
    return indexed


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = [value for item in values if (value := _finite_number(item)) is not None]
    if not finite:
        return {"count": 0, "q25": None, "median": None, "q75": None}
    q25, median, q75 = np.quantile(np.asarray(finite, dtype=np.float64), [0.25, 0.5, 0.75])
    return {"count": len(finite), "q25": float(q25), "median": float(median), "q75": float(q75)}


def _terminal_slice(row: Mapping[str, Any], view: str) -> Mapping[str, Any] | None:
    matches = [
        item for item in row.get("slices", [])
        if item.get("view") == view and item.get("phase") == "terminal"
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate terminal {view} landscape for {_logical_key(row)}")
    return matches[0] if matches else None


def _summary_run(rows: Iterable[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    indexed = _index_rows(rows, method)
    outcomes = Counter(str(row["terminal_outcome"]) for row in indexed.values())
    completed = [row for row in indexed.values() if row["terminal_outcome"] == "completed"]
    return {
        "run_count": len(indexed),
        "terminal_outcomes": {name: outcomes.get(name, 0) for name in ("completed", "numerical_failure")},
        "training_devices": dict(sorted(Counter(
            str(_condition(row).get("training_device", "unknown")) for row in indexed.values()
        ).items())),
        "train_mse": _summary(row.get("verified_train_mse", row.get("train_mse")) for row in completed),
        "test_mse": _summary(row.get("verified_test_mse", row.get("test_mse")) for row in completed),
    }


def validate_baselines(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the restricted 60-run landscape cohort from the old report."""
    allowed_tasks = {"f1", "f4"}
    expected = {(task, "small", method, seed)
                for task in allowed_tasks for method in BASELINE_METHODS for seed in range(10)}
    observed: set[tuple[str, str, str, int]] = set()
    run_ids: set[str] = set()
    selected = list(rows)
    for row in selected:
        condition = _condition(row)
        key = (str(condition["task"]), str(condition["architecture"]),
               str(condition["method"]), int(condition["seed"]))
        if key not in expected:
            raise ValueError(f"baseline condition falls outside the registered f1/f4 small cohort: {key}")
        if key in observed:
            raise ValueError(f"duplicate baseline condition {key}")
        observed.add(key)
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError(f"missing or duplicate baseline run identity for {key}")
        run_ids.add(run_id)
        if int(condition["data_seed"]) != 100000 + key[3]:
            raise ValueError(f"baseline data seed is unpaired for {key}")
        pairing = row.get("pairing", {})
        if not isinstance(pairing, Mapping) or any(pairing.get(name) is not True for name in (
                "dataset_exact", "ambient_directions_exact", "preprojection_initialization_exact", "order_seed_exact")):
            raise ValueError(f"baseline data/initialization/direction pairing is unverified for {key}")
        outcome = row.get("terminal_outcome")
        if outcome not in {"completed", "numerical_failure"}:
            raise ValueError(f"baseline has an unaccepted outcome for {key}: {outcome!r}")
        if outcome == "completed":
            if any(_finite_number(row.get(metric)) is None for metric in METRICS):
                raise ValueError(f"completed baseline lacks finite terminal MSE values for {key}")
            terminal = _terminal_slice(row, "ambient")
            if terminal is None or int(terminal.get("epoch", -1)) != 500:
                raise ValueError(f"completed baseline lacks its terminal epoch-500 ambient slice for {key}")
    if observed != expected or len(selected) != 60:
        raise ValueError(f"baseline cohort must contain all 60 unique attempts; found {len(selected)}")
    return selected


def validate_historical_performance(report: Mapping[str, Any]) -> dict[str, Any]:
    """Check the schema and full cell coverage of the completed four-way report."""
    methods = ("sgd", "adamw", "muon_moonlight", MOMENTUM_METHOD)
    if report.get("protocol") != "punn-manifold-architecture-comparison-v1" or report.get("pairing_verified") is not True:
        raise ValueError("historical performance source must be the paired architecture comparison report")
    if (report.get("baseline_observed_cells") != 480 or report.get("adaptive_observed_cells") != 960
            or report.get("manifold_observed_cells") != 480):
        raise ValueError("historical performance artifact does not cover all registered attempts")
    recipe = report.get("manifold_recipe", {})
    expected_recipe = {"learning_rate": .01, "momentum": .95, "weight_decay": 0.0,
                       "aux_learning_rate": .001, "secondary": False,
                       "manifold_scale": 1.0, "dual_learning_rate": .01,
                       "dual_iterations": 10}
    if recipe != expected_recipe:
        raise ValueError("historical DA-10 performance recipe differs from the registered momentum study")
    source_rows = report.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError("historical performance report lacks cell rows")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in source_rows:
        task, architecture = str(row.get("task")), str(row.get("architecture"))
        if task not in CELL_ARCHITECTURES or architecture not in CELL_ARCHITECTURES[task]:
            raise ValueError(f"historical performance contains an unknown cell {(task, architecture)}")
        key = (task, architecture)
        if key in indexed:
            raise ValueError(f"duplicate historical performance cell {key}")
        if not set(methods).issubset(row):
            raise ValueError(f"historical performance cell {key} is missing a method")
        for method in methods:
            record = row[method]
            if record.get("attempted") != 30:
                raise ValueError(f"historical {method} has an unexpected attempt denominator in {key}")
            completed = record.get("final_performance", {}).get("completed", [])
            complete_seeds = [int(item["seed"]) for item in completed]
            failed_seeds = [int(seed) for seed in record.get("failed_seeds", [])]
            if (len(complete_seeds) != len(set(complete_seeds))
                    or len(failed_seeds) != len(set(failed_seeds))
                    or set(complete_seeds) & set(failed_seeds)
                    or set(complete_seeds) | set(failed_seeds) != set(SEEDS)
                    or int(record.get("numerical_failures", -1)) != len(failed_seeds)):
                raise ValueError(f"historical {method} outcomes do not partition 30 seeds in {key}")
            for item in completed:
                if _finite_number(item.get("train_mse")) is None or _finite_number(item.get("test_mse")) is None:
                    raise ValueError(f"historical {method} has a nonfinite completed MSE in {key}")
        indexed[key] = row
    expected_cells = {(task, architecture) for task, architectures in CELL_ARCHITECTURES.items()
                      for architecture in architectures}
    if set(indexed) != expected_cells or len(indexed) != 16:
        raise ValueError("historical performance report must contain exactly the registered 16 cells")
    devices = report.get("device_by_method", {})
    if any(devices.get(method) != "cpu" for method in methods):
        raise ValueError("historical six-environment comparison unexpectedly mixes performance devices")
    return {
        "protocol": report["protocol"],
        "pairing_verified": True,
        "methods": list(methods),
        "device_by_method": dict(devices),
        "baseline_source_revision": report.get("baseline_source_revision"),
        "adaptive_source_revision": report.get("adaptive_source_revision"),
        "manifold_source_revision": report.get("manifold_source_revision"),
        "baseline_source_tree_digest": report.get("baseline_source_tree_digest"),
        "adaptive_source_tree_digest": report.get("adaptive_source_tree_digest"),
        "manifold_source_tree_digest": report.get("manifold_source_tree_digest"),
        "rows": [dict(indexed[key]) for key in sorted(indexed)],
    }


def validate_recorded_pairing(
    momentum_rows: Iterable[Mapping[str, Any]],
    no_momentum_rows: Iterable[Mapping[str, Any]],
    *,
    old_cache: str | Path,
    no_momentum_cache: str | Path,
) -> dict[str, Any]:
    """Read back and compare the paired dataset, initial state and ambient basis.

    The initialization state is checked exactly when both training devices
    match. A mixed CPU/CUDA pair allows rtol=1e-5 and atol=1e-6 for the
    polar-projected initial parameters because FP32 SVD implementations can
    differ slightly. Preprojection initialization digests, datasets, update
    order seeds, and seeded CPU ambient direction arrays remain exact.
    """
    old_rows = _index_rows(momentum_rows, MOMENTUM_METHOD)
    new_rows = _index_rows(no_momentum_rows, NO_MOMENTUM_METHOD)
    if set(old_rows) != set(new_rows) or len(old_rows) != EXPECTED_CELLS:
        raise ValueError("raw readback pairing requires all 480 matching logical conditions")
    old_cache, no_momentum_cache = Path(old_cache), Path(no_momentum_cache)
    checked = []
    max_initial_abs_error = 0.0
    max_postprojection_tolerance_ratio = 0.0
    cross_device_pairs = 0
    for key in sorted(old_rows):
        old, new = old_rows[key], new_rows[key]
        if (old.get("preprojection_initialization_digest") is None
                or old.get("preprojection_initialization_digest") != new.get("preprojection_initialization_digest")
                or old.get("order_seed") != new.get("order_seed")
                or _condition(old).get("data_seed") != _condition(new).get("data_seed")):
            raise ValueError(f"pretraining initialization/data/order pairing differs for {key}")
        old_id, new_id = str(old["run_id"]), str(new["run_id"])
        old_directory = old_cache / "runs" / old_id
        new_directory = no_momentum_cache / "runs" / new_id
        old_dataset_path, new_dataset_path = old_directory / "dataset" / "dataset.npz", new_directory / "dataset" / "dataset.npz"
        old_payload_path, new_payload_path = old_directory / "recorded_landscapes.npz", new_directory / "recorded_landscapes.npz"
        for path in (old_dataset_path, new_dataset_path, old_payload_path, new_payload_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing audited pairing readback for {key}: {path}")
        with np.load(old_dataset_path, allow_pickle=False) as left_dataset, np.load(new_dataset_path, allow_pickle=False) as right_dataset:
            for name in ("train_x", "train_y", "test_x", "test_y"):
                if not np.array_equal(left_dataset[name], right_dataset[name]):
                    raise ValueError(f"dataset array {name} differs for paired cell {key}")
            left_digest = str(left_dataset["dataset_digest"].item())
            right_digest = str(right_dataset["dataset_digest"].item())
            if left_digest != right_digest or old.get("dataset_digest") != new.get("dataset_digest"):
                raise ValueError(f"dataset digest differs for paired cell {key}")
        with np.load(old_payload_path, allow_pickle=False) as left_payload, np.load(new_payload_path, allow_pickle=False) as right_payload:
            left_initial = np.asarray(left_payload["trajectory/parameters"][0])
            right_initial = np.asarray(right_payload["trajectory/parameters"][0])
            if left_initial.shape != right_initial.shape:
                raise ValueError(f"initial parameter vector shape differs for {key}")
            if not np.isfinite(left_initial).all() or not np.isfinite(right_initial).all():
                raise ValueError(f"nonfinite projected initialization for paired cell {key}")
            old_device = str(_condition(old).get("training_device", "unknown"))
            new_device = str(_condition(new).get("training_device", "unknown"))
            if old_device not in {"cpu", "cuda"} or new_device != "cpu":
                raise ValueError(f"unexpected training-device pairing for {key}: {old_device}/{new_device}")
            cross_device = old_device != new_device
            if cross_device:
                cross_device_pairs += 1
                initial_equal = np.allclose(left_initial, right_initial, rtol=1e-5, atol=1e-6, equal_nan=True)
            else:
                initial_equal = np.array_equal(left_initial, right_initial, equal_nan=True)
            if not initial_equal:
                max_error = float(np.nanmax(np.abs(left_initial.astype(np.float64) - right_initial.astype(np.float64))))
                raise ValueError(f"projected initial parameters differ for {key}; max abs error {max_error}")
            finite = np.isfinite(left_initial) & np.isfinite(right_initial)
            if finite.any():
                difference = np.abs(left_initial[finite].astype(np.float64)
                                    - right_initial[finite].astype(np.float64))
                max_initial_abs_error = max(max_initial_abs_error, float(np.max(difference)))
                allowed = 1e-6 + 1e-5 * np.abs(left_initial[finite].astype(np.float64)) if cross_device else np.full_like(difference, 1e-15)
                max_postprojection_tolerance_ratio = max(
                    max_postprojection_tolerance_ratio,
                    float(np.max(difference / allowed)) if difference.size else 0.0,
                )
            if not np.array_equal(left_payload["directions/ambient"], right_payload["directions/ambient"]):
                raise ValueError(f"ambient slice direction basis differs for paired cell {key}")
        checked.append({"task": key[0], "architecture": key[1], "seed": key[2],
                        "dataset_exact": True, "preprojection_initialization_exact": True,
                        "order_seed_exact": True, "projected_initial_parameters_equal_with_device_tolerance": True,
                        "ambient_directions_exact": True, "cross_device": cross_device})
    return {"passed": True, "paired_cells_checked": len(checked),
            "dataset_exact": True, "preprojection_initialization_exact": True,
            "order_seed_exact": True, "ambient_directions_exact": True,
            "cross_device_initialization_pairs": cross_device_pairs,
            "projected_initial_max_absolute_error": max_initial_abs_error,
            "projected_initial_max_tolerance_ratio": max_postprojection_tolerance_ratio,
            "mixed_device_initialization_tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "same_device_initialization_comparison": "bitwise exact",
            "ambient_directions_comparison": "bitwise exact",
            "all_cells": checked}


def build_report(
    momentum_report: Mapping[str, Any],
    no_momentum_result: Mapping[str, Any],
    *,
    historical_performance: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Build a paired report from two independently audited W&B collections.

    ``no_momentum_result`` is the result of ``punn_no_momentum_check.check_group``:
    it supplies audited ``rows``, rejected/incomplete ``excluded`` attempts,
    and any currently ``missing`` or ``active`` conditions.  When
    ``require_complete`` is true, incomplete collections cannot be presented as
    the final 480-condition comparison.
    """
    if momentum_report.get("protocol") != REFERENCE_PROTOCOL:
        raise ValueError("the DA-10 reference must be the audited recorded-landscape report")
    momentum_rows = momentum_report.get("runs")
    new_rows = no_momentum_result.get("rows")
    if not isinstance(momentum_rows, list) or not isinstance(new_rows, list):
        raise ValueError("both DA-10 groups must supply audited run rows")
    if int(momentum_report.get("verified_recorded_outcomes", len(momentum_rows))) != EXPECTED_CELLS:
        raise ValueError("the reference report does not contain all 480 verified outcomes")
    historical = validate_historical_performance(historical_performance) if historical_performance is not None else None
    if require_complete and historical is None:
        raise ValueError("final all-six performance comparison needs the checksum-verified four-method report")

    momentum = _index_rows(momentum_rows, MOMENTUM_METHOD)
    no_momentum = _index_rows(new_rows, NO_MOMENTUM_METHOD)
    expected_keys = {
        (task, architecture, seed)
        for task, architectures in CELL_ARCHITECTURES.items()
        for architecture in architectures
        for seed in SEEDS
    }
    missing = set(no_momentum_result.get("missing", []))
    active = list(no_momentum_result.get("active", []))
    if require_complete and (set(momentum) != expected_keys or set(no_momentum) != expected_keys or missing or active):
        raise ValueError(
            "final comparison needs 480 unique outcomes in both groups, with no active or pending conditions "
            f"(reference={len(momentum)}, no_momentum={len(no_momentum)}, missing={len(missing)}, active={len(active)})"
        )
    if require_complete and set(momentum) != set(no_momentum):
        raise ValueError("the two DA-10 groups do not cover the same logical conditions")

    paired: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key in sorted(set(momentum) & set(no_momentum)):
        left, right = momentum[key], no_momentum[key]
        left_condition, right_condition = _condition(left), _condition(right)
        checks = {
            "data_seed_exact": left_condition["data_seed"] == right_condition["data_seed"],
            "dataset_digest_exact": left.get("dataset_digest") is not None
            and left.get("dataset_digest") == right.get("dataset_digest"),
        }
        for field in ("preprojection_initialization_digest", "order_seed"):
            left_value, right_value = left.get(field), right.get(field)
            checks[field + "_exact"] = left_value is not None and left_value == right_value
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"paired DA-10 runs differ in seed/data initialization: {key}: {failed}")
        paired[key] = {"momentum": left, "no_momentum": right, "pairing": checks}

    baselines = validate_baselines(momentum_report.get("baselines", []))
    excluded_new = list(no_momentum_result.get("excluded", []))
    excluded_old = list(momentum_report.get("excluded_attempts", []))
    cell_summaries: list[dict[str, Any]] = []

    for task, architectures in CELL_ARCHITECTURES.items():
        for architecture in architectures:
            keys = [(task, architecture, seed) for seed in SEEDS]
            rows_by_method = {
                method: [indexed[key] for key in keys if key in indexed]
                for method, indexed in (("momentum", momentum), ("no_momentum", no_momentum))
            }
            method_summaries: dict[str, Any] = {}
            landscape_summaries: dict[str, Any] = {}
            for method, selected in rows_by_method.items():
                method_summaries[method] = {
                    "outcomes": dict(Counter(row["terminal_outcome"] for row in selected)),
                    "completed": sum(row["terminal_outcome"] == "completed" for row in selected),
                    "numerical_failures": sum(row["terminal_outcome"] == "numerical_failure" for row in selected),
                    "missing": 30 - len(selected),
                    "train_mse": _summary(
                        row.get("verified_train_mse", row.get("train_mse"))
                        for row in selected if row["terminal_outcome"] == "completed"
                    ),
                    "test_mse": _summary(
                        row.get("verified_test_mse", row.get("test_mse"))
                        for row in selected if row["terminal_outcome"] == "completed"
                    ),
                }
            for view in ("ambient", "manifold"):
                landscape_summaries[view] = {}
                for method in ("momentum", "no_momentum"):
                    selected_slices = [
                        _terminal_slice(row, view)
                        for row in rows_by_method[method]
                        if row["terminal_outcome"] == "completed"
                    ]
                    valid = [item for item in selected_slices if item is not None]
                    landscape_summaries[view][method] = {
                        "sensitivity_p95_mse_delta": _summary(item.get(LANDSCAPE_METRIC) for item in valid),
                        "realized_displacement_p95": _summary(item.get("actual_displacement_p95") for item in valid),
                        "finite_grid_count": sum(
                            1 for item in valid if int(item.get("mse_nonfinite_count", 0)) == 0
                        ),
                        "nonfinite_grid_count": sum(
                            1 for item in valid if int(item.get("mse_nonfinite_count", 0)) > 0
                        ),
                        "unavailable_grid_count": 30 - len(valid),
                    }
                paired_ratios, zero_references = [], 0
                for key in keys:
                    if key not in paired:
                        continue
                    left_row, right_row = paired[key]["momentum"], paired[key]["no_momentum"]
                    if left_row["terminal_outcome"] != "completed" or right_row["terminal_outcome"] != "completed":
                        continue
                    left, right = _terminal_slice(left_row, view), _terminal_slice(right_row, view)
                    left_value = _finite_number(left.get(LANDSCAPE_METRIC)) if left else None
                    right_value = _finite_number(right.get(LANDSCAPE_METRIC)) if right else None
                    if left_value is not None and right_value is not None:
                        if left_value > 0 and right_value > 0:
                            paired_ratios.append(right_value / left_value)
                        elif left_value == 0:
                            zero_references += 1
                landscape_summaries[view]["paired_ratio_no_momentum_over_momentum"] = _summary(paired_ratios)
                landscape_summaries[view]["paired_ratio_undefined_zero_reference_count"] = zero_references

            paired_metric_ratios: dict[str, Any] = {}
            for metric in METRICS:
                ratios, zero_references = [], 0
                for key in keys:
                    if key not in paired:
                        continue
                    left_row, right_row = paired[key]["momentum"], paired[key]["no_momentum"]
                    if left_row["terminal_outcome"] != "completed" or right_row["terminal_outcome"] != "completed":
                        continue
                    left = _finite_number(left_row.get("verified_" + metric, left_row.get(metric)))
                    right = _finite_number(right_row.get("verified_" + metric, right_row.get(metric)))
                    if left is not None and right is not None:
                        if left > 0:
                            ratios.append(right / left)
                        else:
                            zero_references += 1
                paired_metric_ratios[metric] = {
                    "no_momentum_over_momentum": _summary(ratios),
                    "undefined_zero_reference_count": zero_references,
                }

            cell_summaries.append({
                "task": task,
                "architecture": architecture,
                "expected_seeds": 30,
                "paired_seed_count": sum(key in paired for key in keys),
                "strategies": method_summaries,
                "paired_terminal_ratios": paired_metric_ratios,
                "landscapes": landscape_summaries,
            })

    no_momentum_counts = _summary_run(new_rows, method=NO_MOMENTUM_METHOD)
    momentum_counts = _summary_run(momentum_rows, method=MOMENTUM_METHOD)
    return {
        "protocol": PROTOCOL,
        "expected_logical_cells": EXPECTED_CELLS,
        "verified_paired_conditions": len(paired),
        "complete": (set(momentum) == expected_keys and set(no_momentum) == expected_keys
                     and len(paired) == EXPECTED_CELLS and not missing and not active
                     and historical is not None),
        "pairing_validation": None,
        "reference": {"method": MOMENTUM_METHOD, **momentum_counts,
                      "excluded_attempt_count": len(excluded_old), "excluded_attempts": excluded_old},
        "no_momentum": {"method": NO_MOMENTUM_METHOD, **no_momentum_counts,
                         "excluded_attempt_count": len(excluded_new), "excluded_attempts": excluded_new,
                         "missing_condition_count": len(missing), "active_condition_count": len(active)},
        "cells": cell_summaries,
        "runs": {"momentum": list(momentum_rows), "no_momentum": list(new_rows)},
        "baselines": baselines,
        "historical_performance": historical,
        "comparison_limits": [
            "Failure means a recorded numerical failure; infrastructure failures and interrupted attempts are listed separately.",
            "Performance distributions include completed numerical outcomes only; failures stay visible in the fixed 30-seed denominator.",
            "Loss landscapes are sampled two-dimensional slices, not full-space flatness estimates.",
            "Ambient directions can leave the exponent constraint; feasible directions are projected and retracted.",
            "Equal slice coordinates do not guarantee equal realized parameter displacement.",
            "The direct DA-10 comparison uses CPU in the new study; the completed reference includes 116 CUDA and 364 CPU runs.",
            "SGD, AdamW and Moonlight Muon landscapes are available only for f1/f4 small models and use different recipes.",
            "The f1-small exponent tangent space has zero dimension, so its feasible slice varies the head only.",
        ],
    }


def _artifact_png(figure: Any, artifact: Any, filename: str) -> None:
    with artifact.new_file(filename, mode="wb") as handle:
        figure.savefig(handle, format="png", dpi=165, facecolor="white", bbox_inches="tight")


def _pairing_verified(report: Mapping[str, Any]) -> bool:
    pairing = report.get("pairing_validation")
    return bool(
        isinstance(pairing, Mapping)
        and pairing.get("passed") is True
        and pairing.get("paired_cells_checked") == EXPECTED_CELLS
        and all(pairing.get(name) is True for name in (
            "dataset_exact", "preprojection_initialization_exact", "order_seed_exact", "ambient_directions_exact"
        ))
        and _finite_number(pairing.get("projected_initial_max_tolerance_ratio")) is not None
        and float(pairing["projected_initial_max_tolerance_ratio"]) <= 1.0000001
    )


def _cell_rows(report: Mapping[str, Any], method_key: str, task: str, architecture: str) -> list[Mapping[str, Any]]:
    return [row for row in report["runs"][method_key]
            if _logical_key(row)[:2] == (task, architecture)]


def _failure_figure(report: Mapping[str, Any], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, positions = [], []
    for cell_index, cell in enumerate(report["cells"]):
        labels.append(f"{cell['task'].upper()}\n{cell['architecture']}")
        positions.append(cell_index)
    figure, axis = plt.subplots(figsize=(16, 6.5))
    methods = (*BASELINE_METHODS, MOMENTUM_METHOD, NO_MOMENTUM_METHOD)
    colors = {**METHOD_COLORS, "sgd": "#009e73", "adamw": "#cc79a7", "muon_moonlight": "#e69f00"}
    width = .15
    historical = {(row["task"], row["architecture"]): row
                  for row in report["historical_performance"]["rows"]}
    max_y = 1
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        failures, missing = [], []
        for cell in report["cells"]:
            if method in BASELINE_METHODS or method == MOMENTUM_METHOD:
                source = historical[(cell["task"], cell["architecture"])][method]
                failures.append(int(source["numerical_failures"]))
                missing.append(30 - int(source["attempted"]))
            else:
                summary = cell["strategies"]["no_momentum"]
                failures.append(summary["numerical_failures"])
                missing.append(summary["missing"])
        max_y = max(max_y, *(failures or [0]))
        axis.bar(np.asarray(positions) + offset, failures, width=width, color=colors[method],
                 label=METHOD_LABELS[method])
        for x, count, absent in zip(np.asarray(positions) + offset, failures, missing):
            if count:
                axis.text(x, count + 0.04, str(count), ha="center", va="bottom", fontsize=7)
            if absent:
                axis.scatter([x], [max_y + 0.7], marker="x", color="#777777", s=25)
    excluded_ref = report["reference"]["excluded_attempt_count"]
    excluded_new = report["no_momentum"]["excluded_attempt_count"]
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("Recorded numerical failures (of 30 seeds)")
    axis.set_ylim(0, max_y + 1.8)
    axis.set_title("Numerical failures across optimizers", pad=56)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01))
    figure.text(.08, .015,
              f"Excluded attempts: recorded momentum DA-10 {excluded_ref}; no-momentum DA-10 {excluded_new}. "
              "Legacy four-way counts use their verified CPU runs.\nGray x marks a missing logical outcome. "
              "High finite loss is not a failure.",
              ha="left", va="bottom", fontsize=8)
    figure.subplots_adjust(bottom=.22, top=.77)
    filename = "da10_no_momentum_failures.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def _performance_figure(report: Mapping[str, Any], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    colors = {"momentum": METHOD_COLORS[MOMENTUM_METHOD],
              "no_momentum": METHOD_COLORS[NO_MOMENTUM_METHOD]}
    method_offsets = {"momentum": -0.17, "no_momentum": 0.17}
    ymax = []
    for axis, metric in zip(axes, METRICS):
        for cell_index, cell in enumerate(report["cells"]):
            task, architecture = cell["task"], cell["architecture"]
            indexed = {
                method: {_logical_key(row)[2]: row for row in _cell_rows(report, method, task, architecture)}
                for method in ("momentum", "no_momentum")
            }
            for seed in sorted(set(indexed["momentum"]) & set(indexed["no_momentum"])):
                left, right = indexed["momentum"][seed], indexed["no_momentum"][seed]
                values = []
                for row in (left, right):
                    value = _finite_number(row.get("verified_" + metric, row.get(metric)))
                    if row["terminal_outcome"] != "completed" or value is None or value < 0:
                        values.append(None)
                    else:
                        values.append(max(value, 1e-15))
                points = []
                for method, offset, value in zip(("momentum", "no_momentum"),
                                                 (method_offsets["momentum"], method_offsets["no_momentum"]), values):
                    if value is not None:
                        points.append((cell_index + offset, value))
                        axis.scatter([cell_index + offset], [value], color=colors[method], s=12, alpha=.5, zorder=3)
                if len(points) == 2:
                    axis.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]],
                              color="#888888", alpha=.22, linewidth=.6, zorder=1)
        for cell_index, cell in enumerate(report["cells"]):
            for method, offset in method_offsets.items():
                values = []
                for row in _cell_rows(report, method, cell["task"], cell["architecture"]):
                    if row["terminal_outcome"] != "completed":
                        continue
                    value = _finite_number(row.get("verified_" + metric, row.get(metric)))
                    if value is not None and value >= 0:
                        values.append(max(value, 1e-15))
                if values:
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    x = cell_index + offset
                    axis.vlines(x, q25, q75, color=colors[method], linewidth=2.2, zorder=4)
                    axis.hlines(median, x - .09, x + .09, color=colors[method], linewidth=2.6, zorder=5)
                    ymax.extend(values)
        axis.set_yscale("log")
        axis.set_ylabel("Training MSE" if metric == "train_mse" else "Held-out MSE")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=.6)
    axes[-1].set_xticks(range(len(report["cells"])),
                        [f"{cell['task'].upper()}\n{cell['architecture']}" for cell in report["cells"]],
                        fontsize=8)
    figure.suptitle("DA-10 terminal performance: momentum versus no momentum", fontsize=14)
    figure.text(.5, .015,
                "Dots are paired seeds; thin gray lines connect the same seed across the two runs. "
                "Colored marks show median and interquartile range among completed outcomes. "
                "Numerical failures are omitted from the loss distribution and counted in the separate failure figure.\n"
                "Reference DA-10: 116 CUDA + 364 CPU runs; no-momentum DA-10: 480 CPU runs.",
                ha="center", va="bottom", fontsize=8)
    figure.legend(handles=[
        plt.Line2D([0], [0], marker="o", linestyle="none", color=colors[key], label=METHOD_LABELS[
            MOMENTUM_METHOD if key == "momentum" else NO_MOMENTUM_METHOD])
        for key in colors
    ], loc="upper center", bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    figure.subplots_adjust(top=.88, bottom=.10, hspace=.14, left=.08, right=.99)
    filename = "da10_no_momentum_terminal_performance.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def _all_six_performance_figure(report: Mapping[str, Any], artifact: Any) -> str:
    """Overlay the new run on the validated four-method, all-six-task cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    historical = {(row["task"], row["architecture"]): row
                  for row in report["historical_performance"]["rows"]}
    methods = ("sgd", "adamw", "muon_moonlight", MOMENTUM_METHOD, NO_MOMENTUM_METHOD)
    colors = {**METHOD_COLORS, "sgd": "#009e73", "adamw": "#cc79a7", "muon_moonlight": "#e69f00"}
    figure, axes = plt.subplots(4, 3, figsize=(18, 15), squeeze=False)
    for metric_index, metric in enumerate(METRICS):
        for task_index, task in enumerate(TASKS):
            row_index, column_index = metric_index * 2 + task_index // 3, task_index % 3
            axis = axes[row_index, column_index]
            architectures = CELL_ARCHITECTURES[task]
            for arch_index, architecture in enumerate(architectures):
                center = arch_index * 6.0
                historical_cell = historical[(task, architecture)]
                for method_index, method in enumerate(methods):
                    x = center + (method_index - 2) * .62
                    points = []
                    if method == NO_MOMENTUM_METHOD:
                        selected = [row for row in report["runs"]["no_momentum"]
                                    if _logical_key(row)[:2] == (task, architecture)
                                    and row["terminal_outcome"] == "completed"]
                        points = [(int(_condition(run)["seed"]),
                                   _finite_number(run.get("verified_" + metric, run.get(metric))))
                                  for run in selected]
                    else:
                        selected = historical_cell[method]["final_performance"]["completed"]
                        points = [(int(run["seed"]), _finite_number(run.get(metric))) for run in selected]
                    points = [(seed, value) for seed, value in points if value is not None and value >= 0]
                    if not points:
                        continue
                    shown = [max(value, 1e-15) for _, value in points]
                    jittered_x = [x + ((seed * 7919 % 997) / 996.0 - .5) * .22 for seed, _ in points]
                    axis.scatter(jittered_x, shown, s=12, alpha=.55, color=colors[method], zorder=2)
                    q25, median, q75 = np.quantile(shown, [.25, .5, .75])
                    axis.vlines(x, q25, q75, color=colors[method], linewidth=1.9, zorder=4)
                    axis.hlines(median, x - .18, x + .18, color=colors[method], linewidth=2.3, zorder=5)
            axis.set_title(task.upper())
            axis.set_xticks([index * 6.0 for index in range(len(architectures))], architectures, fontsize=8)
            axis.set_xlim(-2.0, (len(architectures) - 1) * 6 + 2.0)
            axis.set_yscale("log")
            axis.set_ylabel("Training MSE" if metric == "train_mse" else "Held-out MSE")
            axis.grid(axis="y", which="both", color="#dddddd", linewidth=.6)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color=colors[method],
                          label=METHOD_LABELS[method]) for method in methods]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=5,
                  frameon=False, fontsize=8)
    figure.suptitle("Terminal performance across all six tasks", y=.995, fontsize=14)
    figure.text(.5, .02,
                "Dots are completed seeds; colored marks show median and IQR. The legacy SGD, AdamW, Moonlight Muon, "
                "and momentum DA-10 rows come from the independently verified four-method CPU study. "
                "No-momentum DA-10 is the new CPU study. Recipes differ across methods; failure counts and "
                "attempt denominators are shown in the companion failure graph. No significance test or general ranking is implied.",
                ha="center", va="bottom", fontsize=8, wrap=True)
    figure.subplots_adjust(top=.92, bottom=.12, hspace=.30, wspace=.23, left=.07, right=.99)
    filename = "five_method_terminal_performance_all_six_tasks.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def _landscape_figure(report: Mapping[str, Any], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    colors = {"momentum": METHOD_COLORS[MOMENTUM_METHOD],
              "no_momentum": METHOD_COLORS[NO_MOMENTUM_METHOD]}
    offsets = {"momentum": -.17, "no_momentum": .17}
    max_values = []
    zero_count = 0
    for axis, view in zip(axes, ("ambient", "manifold")):
        for index, cell in enumerate(report["cells"]):
            by_method: dict[str, dict[int, tuple[float, Mapping[str, Any]]]] = {}
            for method in ("momentum", "no_momentum"):
                by_seed = {}
                for row in _cell_rows(report, method, cell["task"], cell["architecture"]):
                    if row["terminal_outcome"] != "completed":
                        continue
                    surface = _terminal_slice(row, view)
                    if surface is None:
                        continue
                    value = _finite_number(surface.get(LANDSCAPE_METRIC))
                    if value is None or value < 0:
                        continue
                    if value == 0:
                        zero_count += 1
                    by_seed[_logical_key(row)[2]] = (max(value, 1e-15), surface)
                    max_values.append(max(value, 1e-15))
                by_method[method] = by_seed
            for seed in sorted(set(by_method["momentum"]) & set(by_method["no_momentum"])):
                left, right = by_method["momentum"][seed][0], by_method["no_momentum"][seed][0]
                axis.plot([index + offsets["momentum"], index + offsets["no_momentum"]], [left, right],
                          color="#888888", alpha=.2, linewidth=.55, zorder=1)
            for method, by_seed in by_method.items():
                x = index + offsets[method]
                values = [value for value, _ in by_seed.values()]
                if values:
                    axis.scatter([x + ((seed % 5) - 2) * .018 for seed in by_seed], values,
                                 color=colors[method], s=12, alpha=.5, zorder=2)
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    axis.vlines(x, q25, q75, color=colors[method], linewidth=2.2, zorder=4)
                    axis.hlines(median, x - .09, x + .09, color=colors[method], linewidth=2.6, zorder=5)
        axis.set_yscale("log")
        axis.set_ylabel("95th percentile |grid MSE - center|")
        axis.set_title("Ambient slice" if view == "ambient" else "Feasible manifold slice")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=.6)
    axes[-1].set_xticks(range(len(report["cells"])),
                        [f"{cell['task'].upper()}\n{cell['architecture']}" for cell in report["cells"]],
                        fontsize=8)
    figure.suptitle("Recorded terminal loss-landscape sensitivity by task and architecture", fontsize=14)
    figure.text(.5, .015,
                "Dots are per-seed terminal 31x31 slices; colored marks show median and IQR. "
                "Gray lines pair the same initialization and data seed. Coordinate radius is 1 in both views, "
                "but realized displacement can differ after projection and polar retraction. "
                f"Zero sensitivities floored at 1e-15 for the log axis: {zero_count}. Slices are not full-space flatness.\n"
                "Reference DA-10: 116 CUDA + 364 CPU runs; no-momentum DA-10: 480 CPU runs.",
                ha="center", va="bottom", fontsize=8)
    figure.legend(handles=[
        plt.Line2D([0], [0], marker="o", linestyle="none", color=colors[key], label=METHOD_LABELS[
            MOMENTUM_METHOD if key == "momentum" else NO_MOMENTUM_METHOD])
        for key in colors
    ], loc="upper center", bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    figure.subplots_adjust(top=.88, bottom=.10, hspace=.24, left=.08, right=.99)
    filename = "da10_no_momentum_landscape_sensitivity.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def _optimizer_context_figure(report: Mapping[str, Any], artifact: Any) -> str:
    """Show the f1/f4 small-model legacy methods with their restricted cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = (*BASELINE_METHODS, MOMENTUM_METHOD, NO_MOMENTUM_METHOD)
    labels = [METHOD_LABELS[method] for method in method_order]
    colors = {**METHOD_COLORS, "sgd": "#009e73", "adamw": "#cc79a7", "muon_moonlight": "#e69f00"}
    baseline_rows = [row for row in report.get("baselines", [])
                     if row.get("condition", {}).get("architecture") == "small"
                     and row.get("condition", {}).get("task") in {"f1", "f4"}
                     and int(row.get("condition", {}).get("seed", 100)) < 10]
    rows_by_method = {method: [row for row in report["runs"][key]
                               if _condition(row).get("architecture") == "small"
                               and _condition(row).get("task") in {"f1", "f4"}
                               and int(_condition(row)["seed"]) < 10]
                      for method, key in ((MOMENTUM_METHOD, "momentum"),
                                          (NO_MOMENTUM_METHOD, "no_momentum"))}
    rows_by_method.update({method: [row for row in baseline_rows
                                    if row.get("condition", {}).get("method") == method]
                           for method in BASELINE_METHODS})

    figure, axes = plt.subplots(3, 2, figsize=(15, 11), squeeze=False)
    x = np.arange(len(method_order))
    jitter = lambda seed: ((int(seed) * 7919 % 997) / 996.0 - .5) * .17
    for column, task in enumerate(("f1", "f4")):
        selected = {
            method: [row for row in rows_by_method[method]
                     if _condition(row).get("task") == task]
            for method in method_order
        }
        fail_axis, performance_axis, landscape_axis = (axes[row, column] for row in range(3))
        for index, method in enumerate(method_order):
            runs = selected[method]
            failures = sum(row.get("terminal_outcome") == "numerical_failure" for row in runs)
            recorded = len(runs)
            fail_axis.bar(index, failures, color=colors[method], width=.62)
            fail_axis.text(index, failures + .05, f"{failures}/{recorded if method in BASELINE_METHODS else 10}",
                           ha="center", va="bottom", fontsize=7)

            for metric, sign_offset, marker in (("train_mse", -.12, "o"),
                                                 ("test_mse", .12, "s")):
                metric_values = []
                for row in runs:
                    if row.get("terminal_outcome") != "completed":
                        continue
                    value = _finite_number(row.get("verified_" + metric, row.get(metric)))
                    if value is None or value < 0:
                        continue
                    seed = int(_condition(row)["seed"])
                    metric_values.append((seed, max(value, 1e-15)))
                    performance_axis.scatter(index + sign_offset + jitter(seed), max(value, 1e-15),
                                             marker=marker, s=17, color=colors[method], alpha=.65)
                if metric_values:
                    values = [value for _, value in metric_values]
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    pos = index + sign_offset
                    performance_axis.vlines(pos, q25, q75, color=colors[method], linewidth=2)
                    performance_axis.hlines(median, pos - .065, pos + .065, color=colors[method], linewidth=2.3)

            landscape_values = []
            for row in runs:
                if row.get("terminal_outcome") != "completed":
                    continue
                surface = _terminal_slice(row, "ambient")
                value = _finite_number(surface.get(LANDSCAPE_METRIC)) if surface else None
                if value is not None and value >= 0:
                    seed = int(_condition(row)["seed"])
                    landscape_values.append((seed, max(value, 1e-15)))
                    landscape_axis.scatter(index + jitter(seed), max(value, 1e-15), s=18,
                                           color=colors[method], alpha=.65)
            if landscape_values:
                values = [value for _, value in landscape_values]
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                landscape_axis.vlines(index, q25, q75, color=colors[method], linewidth=2)
                landscape_axis.hlines(median, index - .09, index + .09, color=colors[method], linewidth=2.4)

        fail_axis.set_title(f"{task.upper()}: numerical failures among first 10 seeds")
        fail_axis.set_ylabel("Failed runs")
        fail_axis.set_ylim(0, max(1, max((sum(r.get("terminal_outcome") == "numerical_failure"
                                               for r in selected[m]) for m in method_order), default=0)) + .8)
        fail_axis.grid(axis="y", color="#dddddd", linewidth=.6)
        performance_axis.set_yscale("log")
        performance_axis.set_ylabel("Terminal MSE (circles train, squares held-out)")
        performance_axis.set_title(f"{task.upper()}: completed terminal performance")
        performance_axis.grid(axis="y", which="both", color="#dddddd", linewidth=.6)
        landscape_axis.set_yscale("log")
        landscape_axis.set_ylabel("Ambient p95 |grid MSE - center|")
        landscape_axis.set_title(f"{task.upper()}: terminal ambient slice sensitivity")
        landscape_axis.grid(axis="y", which="both", color="#dddddd", linewidth=.6)
        for axis in (fail_axis, performance_axis, landscape_axis):
            axis.set_xticks(x, labels, rotation=18, ha="right", fontsize=7.5)

    figure.suptitle("Small-model optimizer context on f1/f4 (seeds 0-9)", fontsize=14)
    figure.text(.5, .015,
                "SGD, AdamW and Moonlight Muon have only these 10 seeds and use different recipes. "
                "Old DA-10 f1/f4 landscape runs were trained on CUDA; the new no-momentum run and legacy baseline "
                "reference grids are CPU. Failure counts use the 10-seed attempt denominator; MSE and landscape "
                "summaries use completed outcomes. Baselines contain ambient slices only. The plotted slices are not full-space flatness.",
                ha="center", va="bottom", fontsize=8, wrap=True)
    figure.subplots_adjust(top=.92, bottom=.16, hspace=.48, wspace=.22, left=.08, right=.99)
    filename = "da10_no_momentum_optimizer_context.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def _load_diagram_grid(row: Mapping[str, Any], view: str, cache: Path) -> np.ndarray:
    run_id = str(row["run_id"])
    path = cache / "runs" / run_id / "recorded_landscapes.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing audited landscape payload for {run_id}: {path}")
    slices = row.get("slices", [])
    indexes = [index for index, item in enumerate(slices)
               if item.get("view") == view and item.get("phase") == "terminal"]
    if len(indexes) != 1:
        raise ValueError(f"expected one terminal {view} slice for {run_id}")
    with np.load(path, allow_pickle=False) as payload:
        grid = np.asarray(payload[f"slices/{indexes[0]:04d}/mse"]).copy()
    if grid.shape != (31, 31):
        raise ValueError(f"unexpected terminal grid shape for {run_id}: {grid.shape}")
    return grid


def _surface_figure(report: Mapping[str, Any], artifact: Any, old_cache: Path, no_momentum_cache: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    momentum = {_logical_key(row): row for row in report["runs"]["momentum"]}
    no_momentum = {_logical_key(row): row for row in report["runs"]["no_momentum"]}
    figure, axes = plt.subplots(2, 4, figsize=(14, 9), squeeze=False)
    figure.subplots_adjust(left=.06, right=.90, top=.87, bottom=.18, wspace=.20, hspace=.55)
    selected = []
    for row_index, task in enumerate(("f1", "f4")):
        keys = sorted(
            key for key in momentum.keys() & no_momentum.keys()
            if key[0] == task and key[1] == "small"
            and momentum[key]["terminal_outcome"] == "completed"
            and no_momentum[key]["terminal_outcome"] == "completed"
        )
        if not keys:
            raise ValueError(f"no shared completed small-model seed for the {task} surface examples")
        key = keys[0]
        selected.append(f"{task.upper()} seed {key[2]}")
        grids = [
            _load_diagram_grid(momentum[key], "ambient", old_cache),
            _load_diagram_grid(momentum[key], "manifold", old_cache),
            _load_diagram_grid(no_momentum[key], "ambient", no_momentum_cache),
            _load_diagram_grid(no_momentum[key], "manifold", no_momentum_cache),
        ]
        positives = np.concatenate([grid[np.isfinite(grid) & (grid > 0)] for grid in grids])
        if positives.size == 0:
            raise ValueError(f"all terminal surface values are nonpositive/nonfinite for {task}")
        floor, ceiling = float(positives.min()) / 1.05, float(positives.max()) * 1.05
        if ceiling <= floor:
            ceiling = floor * 1.01
        norm = LogNorm(vmin=floor, vmax=ceiling)
        last_image = None
        for axis, grid, label in zip(axes[row_index], grids,
                                     ("Momentum DA-10\nambient", "Momentum DA-10\nfeasible",
                                      "No-momentum DA-10\nambient", "No-momentum DA-10\nfeasible")):
            image = axis.imshow(np.ma.masked_where(~np.isfinite(grid), np.maximum(grid, floor)).T,
                                extent=(-1, 1, -1, 1), origin="lower", aspect="equal",
                                interpolation="nearest", cmap="viridis", norm=norm)
            last_image = image
            axis.scatter([0], [0], s=26, marker="o", facecolors="none", edgecolors="white", linewidths=1.2)
            axis.set_title(label, fontsize=9)
            axis.set_xlabel("Direction coordinate 1")
            axis.set_ylabel(f"{task.upper()} / coordinate 2")
            axis.text(.02, .02, f"Center {grid[15, 15]:.4g}\nNonfinite {(~np.isfinite(grid)).sum()}/961",
                      transform=axis.transAxes, color="white", fontsize=7,
                      bbox={"facecolor": "black", "alpha": .55, "edgecolor": "none"})
        position = axes[row_index, -1].get_position()
        color_axis = figure.add_axes([.925, position.y0, .014, position.height])
        figure.colorbar(last_image, cax=color_axis, label="Training MSE (log scale)")
    figure.suptitle("Actual terminal loss slices: momentum and no-momentum DA-10", fontsize=14)
    figure.text(.5, .055,
                "Lowest shared completed seed selected before inspecting the surfaces: " + ", ".join(selected) + ". "
                "Each task uses one color scale across four panels; white rings mark centers and nonfinite samples are masked.\n"
                "Old f1/f4 DA-10 training used CUDA; no-momentum DA-10 training used CPU. "
                "Feasible axes use tangent projection and polar retraction.\n"
                "The f1-small feasible exponent slice varies only the output head. These are sampled 2D slices, not full-space flatness.",
                ha="center", va="bottom", fontsize=8, wrap=True)
    filename = "da10_no_momentum_terminal_surfaces.png"
    _artifact_png(figure, artifact, filename)
    plt.close(figure)
    return filename


def plot_report(
    report: Mapping[str, Any],
    artifact: Any,
    *,
    old_cache: str | Path | None = None,
    no_momentum_cache: str | Path | None = None,
) -> list[str]:
    """Write the failure, performance, sensitivity and optional surface plots."""
    if report.get("protocol") != PROTOCOL:
        raise ValueError("unexpected no-momentum comparison report protocol")
    if report.get("complete") is not True or not _pairing_verified(report):
        raise ValueError("final plots require a complete report with raw 480-cell pairing readback")
    if (old_cache is None) != (no_momentum_cache is None):
        raise ValueError("supply both audited payload-cache roots to plot actual surfaces")
    names = [_failure_figure(report, artifact),
             _performance_figure(report, artifact),
             _landscape_figure(report, artifact),
             _all_six_performance_figure(report, artifact),
             _optimizer_context_figure(report, artifact)]
    if old_cache is not None:
        names.append(_surface_figure(report, artifact, Path(old_cache), Path(no_momentum_cache)))
    return names


def write_report_artifact(
    report: Mapping[str, Any],
    artifact: Any,
    *,
    old_cache: str | Path | None = None,
    no_momentum_cache: str | Path | None = None,
) -> list[str]:
    """Add the validated JSON report and figure bundle to a W&B artifact."""
    import json

    if (report.get("protocol") != PROTOCOL or report.get("complete") is not True
            or not _pairing_verified(report)):
        raise ValueError("refusing to publish an incomplete or unregistered comparison")
    with artifact.new_file("comparison_report.json", mode="w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
    return plot_report(report, artifact, old_cache=old_cache, no_momentum_cache=no_momentum_cache)
