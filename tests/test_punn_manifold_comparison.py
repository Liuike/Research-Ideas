from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from optimizer_resurrection.punn_architecture_stability import expand_config, parse_args
from optimizer_resurrection.punn_manifold_comparison import (
    MANIFOLD_METHOD,
    _partition_manifold_records,
    _validate_manifold_records,
    build_four_way_comparison_report,
)
from optimizer_resurrection.punn_architecture_data import RAW_SOURCES
from optimizer_resurrection.tracking import RunRecord


ROOT = Path(__file__).resolve().parents[1]


def test_interrupted_infrastructure_runs_are_reported_but_not_counted():
    completed = RunRecord(
        "completed", "finished", {"condition_id": "cell-1", "task": "diabetes",
                                   "architecture": "oversized", "seed": 0},
        {"terminal_outcome": "completed"}, "https://example.invalid/completed")
    crashed = RunRecord(
        "interrupted", "crashed", {"condition_id": "cell-1", "task": "diabetes",
                                  "architecture": "oversized", "seed": 0},
        {}, "https://example.invalid/interrupted")
    accepted, excluded = _partition_manifold_records([crashed, completed])
    assert accepted == [completed]
    assert excluded == [{
        "run_id": "interrupted", "state": "crashed",
        "reason": "interrupted_infrastructure", "condition_id": "cell-1",
        "task": "diabetes", "architecture": "oversized", "seed": 0,
    }]
    incomplete = RunRecord(
        "incomplete", "finished", completed.config,
        {"epoch": 400}, "https://example.invalid/incomplete")
    accepted, excluded = _partition_manifold_records([incomplete, completed])
    assert accepted == [completed]
    assert excluded[0]["reason"] == "missing_terminal_outcome"
    with pytest.raises(ValueError, match="still active"):
        _partition_manifold_records([
            RunRecord("bad", "running", completed.config, {}, completed.url),
        ])
    with pytest.raises(ValueError, match="not finished"):
        _partition_manifold_records([
            RunRecord("bad", "crashed", completed.config,
                      {"terminal_result": {"terminal_outcome": "completed"}},
                      completed.url),
        ])
    with pytest.raises(ValueError, match="not finished"):
        _partition_manifold_records([
            RunRecord("bad", "crashed", completed.config,
                      {"terminal_result.terminal_outcome": "numerical_failure"},
                      completed.url),
        ])


def _configs():
    def read(path):
        return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))

    return (
        read("configs/product_unit/manifold_architecture_stability.yaml"),
        read("configs/product_unit/adaptive_architecture_stability.yaml"),
        read("configs/product_unit/architecture_stability.yaml"),
    )


def _data_identity(task, seed):
    return f"data:{task}:{seed}"


def _initial_identity(task, architecture, seed):
    size = "small" if architecture == "small" else "large"
    return f"init:{task}:{size}:{seed}"


def _metadata(task, data_seed, digest, artifact):
    result = {"task": task, "data_seed": data_seed, "dataset_digest": digest}
    if task in RAW_SOURCES:
        source = RAW_SOURCES[task]
        result["raw_source"] = {"url": source.url, "sha256": source.sha256}
    return result


def _failure(result, *, failure_phase="optimizer_update"):
    result.update(terminal_outcome="numerical_failure", numerical_failure=True,
                  failed_epoch=1, failure_reason="nonfinite_optimizer_state",
                  failure_phase=failure_phase, initial_train_mse=None, train_mse=None,
                  test_mse=None, objective=None)


def _records_for_config(config, kind, revision):
    records = []
    for index, command in enumerate(expand_config(config)):
        args = parse_args(command[3:])
        resolved = vars(args).copy()
        resolved["provenance"] = {"git": {"dirty": False, "revision": revision},
                                  "source_tree": {"digest": f"digest:{revision}"}}
        data_digest = _data_identity(args.task, args.seed)
        artifact = f"dataset-{args.task}-{args.data_seed}:v0"
        resolved["dataset_artifact"] = artifact
        resolved["data_metadata"] = _metadata(args.task, args.data_seed, data_digest, artifact)
        if kind == "sgd":
            result = {
                "terminal_outcome": "completed", "numerical_failure": False,
                "epochs_completed": args.epochs, "initial_train_mse": 1.0,
                "train_mse": .2, "test_mse": .3, "objective": .21,
                "dataset_digest": data_digest, "dataset_artifact": artifact,
                "initialization_digest": _initial_identity(args.task, args.architecture, args.seed),
                "order_seed": args.seed + 1_000_003,
            }
        else:
            regularized = args.architecture == "regularized"
            large = args.architecture in {"oversized", "regularized"}
            result = {
                "condition_id": args.condition_id, "protocol": args.protocol,
                "task": args.task, "architecture": args.architecture,
                "method": args.method, "seed": args.seed, "data_seed": args.data_seed,
                "epochs": args.epochs, "input_dim": args.input_dim,
                "hidden_units": args.hidden_units, "output_dim": args.output_dim,
                "regularization_lambda": args.regularization_lambda,
                "terminal_outcome": "completed", "numerical_failure": False,
                "epochs_completed": args.epochs, "initial_train_mse": 1.0,
                "train_mse": .2, "test_mse": .3, "objective": .21,
                "failed_epoch": None, "failure_phase": None, "failure_reason": None,
                "dataset_digest": data_digest, "dataset_artifact": artifact,
                "initialization_digest": _initial_identity(args.task, args.architecture, args.seed),
                "order_seed": args.seed + 1_000_003,
            }
            if kind == "adaptive":
                result.update(learning_rate=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              aux_learning_rate=args.aux_learning_rate,
                              secondary=args.secondary)
            else:
                result.update(
                    {field: getattr(args, field) for field in
                     ("learning_rate", "momentum", "weight_decay", "aux_learning_rate",
                      "secondary", "manifold_scale", "dual_learning_rate", "dual_iterations")}
                )
                result["preprojection_initialization_digest"] = _initial_identity(
                    args.task, args.architecture, args.seed)
                result["initialization_digest"] = (
                    f"projected:{args.task}:{'large' if large else 'small'}:{args.seed}")
                result["manifold_nesterov"] = True
                result["initial_stiefel_residual"] = 1e-7
                result["final_stiefel_residual"] = 2e-7
                result["optimizer_name"] = "ManifoldMuon DA-10 plus AdamW auxiliary"
                result["optimizer_assignment"] = (
                    "DA-10 Manifold Muon on projected exponents; "
                    "AdamW on output weight and bias")
                result["optimizer_betas"] = {"auxiliary_adamw": [0.9, 0.95]}
                result["optimizer_epsilon"] = 1e-8
        records.append(RunRecord(str(index), "finished", resolved,
                                 result if kind == "sgd" else {"terminal_result": result}, "url"))
    return records


@pytest.fixture(scope="module")
def complete_fixture():
    manifold_config, adaptive_config, baseline_config = _configs()
    manifold = _records_for_config(manifold_config, "manifold", "manifold-r1")
    adaptive = _records_for_config(adaptive_config, "adaptive", "adaptive-r1")
    baseline = _records_for_config(baseline_config, "sgd", "sgd-r1")
    # Distinct paired failures exercise the four-way seed transition accounting.
    base = next(record for record in baseline
                if record.config["task"] == "f1" and record.config["architecture"] == "small"
                and record.config["seed"] == 0)
    _failure(base.summary, failure_phase="backward")
    adaptive_muon = next(record for record in adaptive
                         if record.config["task"] == "f1"
                         and record.config["architecture"] == "small"
                         and record.config["method"] == "muon_moonlight"
                         and record.config["seed"] == 1)
    _failure(adaptive_muon.summary["terminal_result"])
    manifold_only = next(record for record in manifold
                         if record.config["task"] == "f1"
                         and record.config["architecture"] == "small"
                         and record.config["seed"] == 2)
    _failure(manifold_only.summary["terminal_result"])
    return manifold_config, adaptive_config, baseline_config, manifold, adaptive, baseline


def test_four_way_report_validates_pairing_and_adds_manifold_outcomes(complete_fixture):
    manifold_config, adaptive_config, baseline_config, manifold, adaptive, baseline = complete_fixture
    report = build_four_way_comparison_report(
        manifold_config, adaptive_config, baseline_config, manifold, adaptive, baseline)
    assert report["pairing_verified"] is True
    assert report["manifold_expected_cells"] == report["manifold_observed_cells"] == 480
    assert report["methods"] == ["sgd", "adamw", "muon_moonlight", MANIFOLD_METHOD]
    assert report["manifold_source_revision"] == "manifold-r1"
    row = next(row for row in report["rows"]
               if row["task"] == "f1" and row["architecture"] == "small")
    assert row["sgd"]["numerical_failures"] == 1
    assert row["muon_moonlight"]["numerical_failures"] == 1
    assert row[MANIFOLD_METHOD]["numerical_failures"] == 1
    transitions = row[MANIFOLD_METHOD]["paired_with_sgd"]
    assert transitions["sgd_only_failed"] == 1
    assert transitions["manifold_only_failed"] == 1
    assert report["initialization_pairing"].startswith(
        "preprojection_initialization_digest is paired")
    assert "capacity-limited" in report["cell_annotations"]["f1/small"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "dirty", "recipe"])
def test_manifold_record_validation_rejects_incomplete_or_unpaired_cells(complete_fixture, mutation):
    manifold_config, _adaptive, _baseline, original, _a, _b = complete_fixture
    config = deepcopy(manifold_config)
    records = deepcopy(original)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "dirty":
        records[0].config["provenance"]["git"]["dirty"] = True
    elif mutation == "recipe":
        config["optimizer_settings"][MANIFOLD_METHOD]["dual_iterations"] = 9
    with pytest.raises(ValueError):
        _validate_manifold_records(config, records)


def test_four_way_report_rejects_a_manifold_initialization_that_does_not_match_sgd(complete_fixture):
    manifold_config, adaptive_config, baseline_config, original, adaptive, baseline = complete_fixture
    manifold = deepcopy(original)
    record = next(record for record in manifold
                  if record.config["task"] == "wine"
                  and record.config["architecture"] == "oversized"
                  and record.config["seed"] == 4)
    record.summary["terminal_result"]["preprojection_initialization_digest"] = "other-init"
    with pytest.raises(ValueError, match="preprojection initialization"):
        build_four_way_comparison_report(manifold_config, adaptive_config, baseline_config,
                                         manifold, adaptive, baseline)
