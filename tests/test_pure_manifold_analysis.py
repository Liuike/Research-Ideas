from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from optimizer_resurrection import pure_cifar as pc
from optimizer_resurrection.pure_analysis import (
    MANIFOLD_CONSTRAINT_TOLERANCE,
    summarize_records,
    validate_landscape_history,
)
from optimizer_resurrection.pure_landscape import probe_epochs
from optimizer_resurrection.tracking import RunRecord


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "pure_cifar" / "manifold_seed0.yaml").read_text(encoding="utf-8")
)
PU_NAMES = [f"layer{stage}.{block}.conv2.weight"
            for stage in range(1, 5) for block in range(2)]


def initial_constraint_values():
    return {
        f"manifold/constraint/{name}/{measure}": value
        for name in PU_NAMES
        for measure, value in (("rms", 8e-5), ("max_abs", 2e-4))
    }


def manifold_records():
    records = []
    for command in pc.expand_config(CONFIG):
        args = pc.parse_args(command[3:])
        summary = {
            "terminal_outcome": "completed",
            "numerical_failure": False,
            "epochs_completed": 160,
            "test_examples_evaluated": 10000,
            "test_accuracy": .84 if args.momentum else .82,
            "test_loss": .7 if args.momentum else .8,
            "training_seconds": 4100.0 if args.momentum else 3950.0,
            "parameter_count": pc.PARAMETERS,
            "initialization_digest": "projected-init-shared",
            "unprojected_initialization_digest": "torchvision-init-shared",
            "dataset_digest": "cifar-train-test-shared",
            "train_examples": 50000,
            "test_examples": 10000,
            "order_seed": 1100003,
            "worker_seed": 2100003,
            "augmentation_seed": 3100003,
            "manifold_parameter_names": list(PU_NAMES),
            "auxiliary_parameter_names": ["conv1.weight", "fc.weight", "fc.bias"],
            "manifold_sign_backend": "exact_svd",
            "manifold_nesterov": args.momentum > 0,
            "landscape_expected_probe_epochs": list(probe_epochs(args)),
            "landscape_probe_epochs": list(probe_epochs(args)),
            "landscape_gradient_batches": 160 * 391,
            "landscape_probe_digest": "fixed-training-probe-shared",
            "landscape_probe_examples": 32,
        }
        constraints = initial_constraint_values()
        if args.momentum:
            # The online W&B summary may flatten nested mappings into dot keys.
            summary.update({f"manifold_initial_constraints.{key}": value
                            for key, value in constraints.items()})
        else:
            summary["manifold_initial_constraints"] = constraints
        resolved = vars(args).copy()
        resolved["provenance"] = {
            "git": {"dirty": False, "revision": "clean-source-revision"},
            "source_tree": {"digest": "clean-source-tree-digest"},
        }
        records.append(RunRecord(
            run_id=f"run-momentum-{args.momentum:g}",
            state="finished",
            config=resolved,
            summary=summary,
            url=f"https://wandb.ai/test/run-momentum-{args.momentum:g}",
        ))
    return records


def test_manifold_analysis_accepts_paired_routes_and_flattened_constraints():
    records = manifold_records()
    report = summarize_records(CONFIG, records)

    assert report["expected_cells"] == report["observed_cells"] == 2
    assert report["completed"] == 2 and report["all_seeds_completed"]
    assert [row["momentum"] for row in report["rows"]] == [.95, 0.0]
    assert report["mean_test_accuracy_percent"] is None
    assert report["sd_test_accuracy_percent"] is None
    assert report["mean_training_seconds"] is None
    assert report["source_run_ids"] == ["run-momentum-0", "run-momentum-0.95"]
    assert "no pooling or SD across optimizers" in report["interpretation"]
    assert MANIFOLD_CONSTRAINT_TOLERANCE == 1e-3
    assert max(initial_constraint_values().values()) == 2e-4


def test_manifold_analysis_rejects_missing_initial_constraints():
    records = manifold_records()
    for key in list(records[0].summary):
        if key == "manifold_initial_constraints" or key.startswith("manifold_initial_constraints."):
            records[0].summary.pop(key)
    with pytest.raises(ValueError, match="initial Stiefel constraints"):
        summarize_records(CONFIG, records)


@pytest.mark.parametrize("digest", [
    "unprojected_initialization_digest",
    "initialization_digest",
    "landscape_probe_digest",
])
def test_manifold_analysis_rejects_pair_digest_mismatch(digest):
    records = manifold_records()
    records[1].summary[digest] = "different-pair-identity"
    with pytest.raises(ValueError, match="pair identity mismatch"):
        summarize_records(CONFIG, records)


def test_manifold_analysis_rejects_duplicate_condition():
    records = manifold_records()
    records.append(replace(records[0], run_id="duplicate"))
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        summarize_records(CONFIG, records)


def test_manifold_analysis_rejects_constraint_drift_over_frozen_tolerance():
    records = manifold_records()
    key = next(key for key in records[0].summary if key.startswith("manifold_initial_constraints."))
    records[0].summary[key] = MANIFOLD_CONSTRAINT_TOLERANCE + 1e-6
    with pytest.raises(ValueError, match="initial Stiefel constraints"):
        summarize_records(CONFIG, records)


def test_landscape_history_rejects_incorrect_da10_counter_first():
    count = 391  # ceil(50,000 / 128)
    row = {
        "epoch": 1,
        "manifold/minibatch_count": count,
        "manifold/dual_measurement_count": 8 * count - 1,
        "manifold/dual_iterations": 10,
    }
    with pytest.raises(ValueError, match="incomplete DA-10 update budget"):
        validate_landscape_history(CONFIG, {"epochs_completed": 1}, [row])
