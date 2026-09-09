from __future__ import annotations

import argparse
import json

import numpy as np

from optimizer_resurrection.experiment import _unique_training_runs
from optimizer_resurrection.tracking import fetch_wandb_runs, wandb_analysis_run


def _nested_value(mapping: dict, dotted_key: str):
    value = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fit_boundary(x: np.ndarray, success: np.ndarray, target: float = 0.5) -> float:
    """Fit decreasing logistic success probability and solve for target.

    Returns +/- infinity for all-success/all-failure rather than inventing a
    transition outside the measured region.
    """
    x = np.asarray(x, dtype=float)
    success = np.asarray(success, dtype=float)
    if np.all(success == 1):
        return float("inf")
    if np.all(success == 0):
        return float("-inf")
    from scipy.optimize import minimize

    def objective(theta: np.ndarray) -> float:
        intercept, raw_slope = theta
        slope = -np.exp(raw_slope)
        logits = np.clip(intercept + slope * x, -30, 30)
        return float(np.logaddexp(0, logits).sum() - (success * logits).sum())

    result = minimize(objective, np.array([0.0, -1.0]), method="BFGS")
    intercept, raw_slope = result.x
    slope = -np.exp(raw_slope)
    return float((np.log(target / (1 - target)) - intercept) / slope)


def bootstrap_boundaries(
    x: np.ndarray,
    success: np.ndarray,
    target: float = 0.5,
    samples: int = 2000,
    seed: int = 0,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    estimate = fit_boundary(x, success, target)
    rng = np.random.default_rng(seed)
    values = []
    unique_clusters = np.unique(clusters) if clusters is not None else None
    for _ in range(samples):
        if unique_clusters is None:
            indices = rng.integers(0, len(x), len(x))
        else:
            sampled = rng.choice(unique_clusters, len(unique_clusters), replace=True)
            indices = np.concatenate([np.nonzero(clusters == cluster)[0] for cluster in sampled])
        value = fit_boundary(x[indices], success[indices], target)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=["depth", "max_length"], required=True)
    parser.add_argument("--log-axis", action="store_true")
    parser.add_argument("--group-by", nargs="+", help="Config fields defining a scientifically homogeneous curve")
    parser.add_argument("--expect-axis-values", nargs="+", type=float)
    parser.add_argument("--expect-seeds", nargs="+", type=int)
    parser.add_argument("--strict", action="store_true", help="Fail when expected cells are missing")
    parser.add_argument("--run-group", help="Restrict completed W&B runs to one group")
    args = parser.parse_args()
    group_fields = args.group_by or (
        ["dataset", "activation", "optimizer", "width", "assignment", "manifold_scale", "learning_rate", "aux_learning_rate", "weight_decay", "muon_backend", "steps", "batch_size", "success_loss", "success_accuracy", "recipe_version", "provenance.source_tree.digest"]
        if args.axis == "depth"
        else ["task", "optimizer", "hidden_size", "assignment", "manifold_scale", "learning_rate", "aux_learning_rate", "weight_decay", "muon_backend", "steps", "batch_size", "success_loss", "success_accuracy", "recipe_version", "provenance.source_tree.digest"]
    )
    grouped: dict[str, list[tuple[float, bool, int]]] = {}
    filters = {"group": args.run_group} if args.run_group else None
    for record in _unique_training_runs(fetch_wandb_runs(filters)):
        config = record.config
        result = record.summary
        if args.axis not in config or "success" not in result:
            continue
        group_values = {field: _nested_value(config, field) for field in group_fields}
        key = "__".join(f"{field}={group_values[field]}" for field in group_fields)
        grouped.setdefault(key, []).append((config[args.axis], result["success"], config["seed"]))
    report = {}
    for optimizer, rows in grouped.items():
        raw_x = np.array([row[0] for row in rows], dtype=float)
        x = raw_x.copy()
        if args.log_axis:
            x = np.log(x)
        y = np.array([row[1] for row in rows], dtype=float)
        seeds = np.array([row[2] for row in rows])
        d50 = bootstrap_boundaries(x, y, 0.5, clusters=seeds)
        d80 = bootstrap_boundaries(x, y, 0.8, clusters=seeds)
        if args.log_axis:
            d50 = tuple(float(np.exp(v)) if np.isfinite(v) else v for v in d50)
            d80 = tuple(float(np.exp(v)) if np.isfinite(v) else v for v in d80)
        unique = np.unique(raw_x)
        rates = np.array([y[raw_x == value].mean() for value in unique])
        log_unique = np.log(unique)
        auc = float(np.trapezoid(rates, log_unique) / (log_unique[-1] - log_unique[0])) if len(unique) > 1 else float("nan")
        missing = []
        if args.expect_axis_values and args.expect_seeds:
            observed = {(float(row[0]), int(row[2])) for row in rows}
            missing = [
                {"axis": axis, "seed": seed}
                for axis in args.expect_axis_values for seed in args.expect_seeds
                if (float(axis), int(seed)) not in observed
            ]
        report[optimizer] = {
            "boundary_50": d50,
            "boundary_80": d80,
            "success_rates": {str(value): float(rate) for value, rate in zip(unique, rates)},
            "auc_success_vs_log_axis": auc,
            "runs": len(rows),
            "missing_cells": missing,
        }
    analysis_group = args.run_group or "boundary-analysis"
    with wandb_analysis_run(
        name=f"boundary-{args.axis}-{analysis_group}",
        group=analysis_group,
        job_type="boundary-analysis",
        config={
            "axis": args.axis,
            "log_axis": args.log_axis,
            "group_fields": group_fields,
            "expected_axis_values": args.expect_axis_values,
            "expected_seeds": args.expect_seeds,
            "strict": args.strict,
        },
    ) as run:
        run.summary.update({"boundary_report": report})
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and any(group["missing_cells"] for group in report.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
