from __future__ import annotations

from io import BytesIO

import pytest

from optimizer_resurrection.punn_no_momentum_report import (
    MOMENTUM_METHOD,
    NO_MOMENTUM_METHOD,
    PROTOCOL,
    build_report,
    plot_report,
)
from optimizer_resurrection.punn_manifold_landscape_plot import CELL_ARCHITECTURES


def _run(task, architecture, seed, method, *, outcome="completed", loss=1.0):
    return {
        "condition": {
            "task": task,
            "architecture": architecture,
            "method": method,
            "seed": seed,
            "data_seed": 100000 + seed,
            "training_device": "cpu",
        },
        "terminal_outcome": outcome,
        "verified_train_mse": loss if outcome == "completed" else None,
        "verified_test_mse": loss * 1.2 if outcome == "completed" else None,
        "train_mse": loss if outcome == "completed" else None,
        "test_mse": loss * 1.2 if outcome == "completed" else None,
        "preprojection_initialization_digest": f"init-{task}-{architecture}-{seed}",
        "initialization_digest": f"projected-{task}-{architecture}-{seed}",
        "dataset_digest": f"dataset-{task}-{seed}",
        "order_seed": 1000003 + seed,
        "run_id": f"{method}-{task}-{architecture}-{seed}",
        "slices": [
            {"view": view, "phase": "terminal", "epoch": 500,
             "absolute_p95_mse_delta": loss * (2 if view == "ambient" else .5),
             "actual_displacement_p95": 1.1 if view == "ambient" else .8,
             "mse_nonfinite_count": 0}
            for view in ("ambient", "manifold")
        ] if outcome == "completed" else [],
    }


def _fixture():
    momentum_rows, no_momentum_rows = [], []
    for task, architectures in CELL_ARCHITECTURES.items():
        for architecture in architectures:
            for seed in range(30):
                momentum_rows.append(_run(task, architecture, seed, MOMENTUM_METHOD,
                                          outcome="numerical_failure" if (task, architecture, seed) == ("f1", "small", 0) else "completed",
                                          loss=1 + seed / 10))
                no_momentum_rows.append(_run(task, architecture, seed, NO_MOMENTUM_METHOD,
                                             outcome="numerical_failure" if (task, architecture, seed) == ("f1", "small", 1) else "completed",
                                             loss=1.2 + seed / 10))
    baseline_rows = []
    for method in ("sgd", "adamw", "muon_moonlight"):
        for task in ("f1", "f4"):
            for seed in range(10):
                outcome = "numerical_failure" if (method, task, seed) == ("sgd", "f1", 0) else "completed"
                baseline_rows.append({
                    "condition": {"task": task, "architecture": "small", "method": method,
                                  "seed": seed, "data_seed": 100000 + seed},
                    "run_id": f"baseline-{method}-{task}-{seed}",
                    "terminal_outcome": outcome,
                    "train_mse": .2 if outcome == "completed" else None,
                    "test_mse": .3 if outcome == "completed" else None,
                    "pairing": {key: True for key in ("dataset_exact", "ambient_directions_exact",
                                                      "preprojection_initialization_exact", "order_seed_exact")},
                    "slices": ([{"view": "ambient", "phase": "terminal", "epoch": 500,
                                 "absolute_p95_mse_delta": .2}]
                               if outcome == "completed" else []),
                })
    historical_rows = []
    for task, architectures in CELL_ARCHITECTURES.items():
        for architecture in architectures:
            cell = {"task": task, "architecture": architecture}
            for method in ("sgd", "adamw", "muon_moonlight", MOMENTUM_METHOD):
                cell[method] = {
                    "attempted": 30,
                    "failed_seeds": [],
                    "numerical_failures": 0,
                    "final_performance": {
                        "completed": [{"seed": seed, "train_mse": 1 + seed / 10,
                                       "test_mse": 1.2 + seed / 10} for seed in range(30)]
                    },
                }
            historical_rows.append(cell)
    historical_performance = {
        "protocol": "punn-manifold-architecture-comparison-v1",
        "pairing_verified": True,
        "baseline_observed_cells": 480,
        "adaptive_observed_cells": 960,
        "manifold_observed_cells": 480,
        "manifold_recipe": {"learning_rate": .01, "momentum": .95, "weight_decay": 0.0,
                            "aux_learning_rate": .001, "secondary": False,
                            "manifold_scale": 1.0, "dual_learning_rate": .01,
                            "dual_iterations": 10},
        "device_by_method": {name: "cpu" for name in ("sgd", "adamw", "muon_moonlight", MOMENTUM_METHOD)},
        "rows": historical_rows,
    }
    return ({"protocol": "punn-manifold-landscape-comparison-v1",
             "verified_recorded_outcomes": 480,
             "runs": momentum_rows,
             "baselines": baseline_rows,
             "excluded_attempts": [{"reason": "interrupted", "run_id": "old-interruption"}]},
            {"rows": no_momentum_rows,
             "excluded": [{"reason": "infrastructure_failure", "run_id": "retry-attempt"}],
             "missing": [], "active": []}, historical_performance)


def test_build_report_keeps_failures_out_of_performance_and_landscape_summaries():
    reference, no_momentum, historical = _fixture()

    report = build_report(reference, no_momentum, historical_performance=historical)

    assert report["protocol"] == PROTOCOL
    assert report["complete"] is True
    assert report["verified_paired_conditions"] == 480
    f1_small = next(cell for cell in report["cells"]
                    if (cell["task"], cell["architecture"]) == ("f1", "small"))
    assert f1_small["strategies"]["momentum"]["numerical_failures"] == 1
    assert f1_small["strategies"]["no_momentum"]["numerical_failures"] == 1
    assert f1_small["strategies"]["momentum"]["train_mse"]["count"] == 29
    assert f1_small["strategies"]["no_momentum"]["test_mse"]["count"] == 29
    assert report["reference"]["excluded_attempt_count"] == 1
    assert report["no_momentum"]["excluded_attempt_count"] == 1
    assert f1_small["paired_terminal_ratios"]["train_mse"]["no_momentum_over_momentum"]["count"] == 28


def test_build_report_rejects_changed_initialization_or_data_pairing():
    reference, no_momentum, historical = _fixture()
    no_momentum["rows"][0]["preprojection_initialization_digest"] = "different-init"

    with pytest.raises(ValueError, match="paired DA-10 runs differ"):
        build_report(reference, no_momentum, historical_performance=historical)


def test_build_report_refuses_a_partial_collection_as_final():
    reference, no_momentum, historical = _fixture()
    reference["runs"] = reference["runs"][:1]
    no_momentum["rows"] = no_momentum["rows"][:1]

    with pytest.raises(ValueError, match="final comparison needs 480 unique outcomes"):
        build_report(reference, no_momentum, historical_performance=historical)


def test_build_report_rejects_missing_or_duplicate_legacy_baseline_attempts():
    reference, no_momentum, historical = _fixture()
    reference["baselines"] = reference["baselines"][:-1]
    with pytest.raises(ValueError, match="all 60 unique attempts"):
        build_report(reference, no_momentum, historical_performance=historical)

    reference, no_momentum, historical = _fixture()
    reference["baselines"].append(reference["baselines"][0])
    with pytest.raises(ValueError, match="duplicate baseline condition"):
        build_report(reference, no_momentum, historical_performance=historical)


def test_build_report_rejects_unverified_legacy_baseline_pairing():
    reference, no_momentum, historical = _fixture()
    reference["baselines"][0]["pairing"]["ambient_directions_exact"] = False
    with pytest.raises(ValueError, match="pairing is unverified"):
        build_report(reference, no_momentum, historical_performance=historical)


class _MemoryArtifact:
    def __init__(self):
        self.files = {}

    class _FileContext:
        def __init__(self, parent, name):
            self.parent, self.name = parent, name
            self.handle = BytesIO()

        def __enter__(self):
            return self.handle

        def __exit__(self, *_):
            self.parent.files[self.name] = self.handle.getvalue()

    def new_file(self, name, mode="wb", **kwargs):
        return self._FileContext(self, name)


def test_plot_report_writes_failure_performance_landscape_and_context_graphs():
    reference, no_momentum, historical = _fixture()
    report = build_report(reference, no_momentum, historical_performance=historical)
    report["pairing_validation"] = {
        "passed": True, "paired_cells_checked": 480, "dataset_exact": True,
        "preprojection_initialization_exact": True, "order_seed_exact": True,
        "ambient_directions_exact": True, "projected_initial_max_tolerance_ratio": 0.0,
    }
    artifact = _MemoryArtifact()

    names = plot_report(report, artifact)

    assert names == [
        "da10_no_momentum_failures.png",
        "da10_no_momentum_terminal_performance.png",
        "da10_no_momentum_landscape_sensitivity.png",
        "five_method_terminal_performance_all_six_tasks.png",
        "da10_no_momentum_optimizer_context.png",
    ]
    assert set(names) == set(artifact.files)
    assert all(data.startswith(b"\x89PNG\r\n\x1a\n") for data in artifact.files.values())
